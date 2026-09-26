import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Dynamic Graph Attention Network v2 (GATv2) Layer
# Implements out-of-place message passing for 100% autograd compatibility
# ---------------------------------------------------------------------------
class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.head_dim = out_dim // heads

        self.lin_src = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]
        N = x.size(0)
        h_src = self.lin_src(x).view(N, self.heads, self.head_dim)
        h_dst = self.lin_dst(x).view(N, self.heads, self.head_dim)

        src_nodes, dst_nodes = edge_index[0], edge_index[1]

        # Dynamic attention score: a^T * LeakyReLU(W_src * x_i + W_dst * x_j)
        score = self.leaky_relu(h_src[src_nodes] + h_dst[dst_nodes])
        alpha = torch.clamp((score * self.att).sum(dim=-1), -10.0, 10.0) # [E, heads]
        exp_alpha = torch.exp(alpha)

        # Out-of-place scatter_reduce for normalization denominator.
        # include_self=True on a zero-initialised tensor is numerically identical
        # to False (adding zeros) and is the form the ONNX exporter accepts.
        denom_idx = dst_nodes.unsqueeze(1).expand(-1, self.heads)
        denom = torch.zeros(N, self.heads, device=x.device).scatter_reduce(
            0, denom_idx, exp_alpha, reduce="sum", include_self=True
        ) + 1e-9

        norm_alpha = self.dropout(exp_alpha / denom[dst_nodes])
        msg = h_src[src_nodes] * norm_alpha.unsqueeze(-1) # [E, heads, head_dim]

        # Out-of-place scatter_reduce for message aggregation (zeros init, so
        # include_self=True is numerically identical and ONNX-exportable)
        msg_idx = dst_nodes.view(-1, 1, 1).expand(-1, self.heads, self.head_dim)
        out = torch.zeros(N, self.heads, self.head_dim, device=x.device).scatter_reduce(
            0, msg_idx, msg, reduce="sum", include_self=True
        )

        # Isolated nodes (no incoming message: single-atom molecules) fall back to
        # the projected features. Derived from the edge list, not from the tensor
        # shape: a shape comparison yields a SymBool under torch.export and gets
        # baked in as a constant under jit tracing.
        edge_weights = torch.ones(dst_nodes.size(0), dtype=x.dtype, device=x.device)
        in_degree = torch.zeros(N, dtype=x.dtype, device=x.device).index_add(0, dst_nodes, edge_weights)
        isolated = (in_degree == 0).to(out.dtype).unsqueeze(-1)  # [N, 1]
        return out.view(N, self.out_dim) + h_src.view(N, self.out_dim) * isolated

# ---------------------------------------------------------------------------
# Substructure Cluster Pooling Layer
# Groups atom-level embeddings into K functional pharmacophore tokens
# ---------------------------------------------------------------------------
class SubstructurePooler(nn.Module):
    """Groups atom embeddings into K functional pharmacophore tokens.

    tokens[g, k] = sum_i w_ik * h_i / sum_i w_ik  (weighted mean over the atoms
    of molecule g), with w = softmax(assign_lin(h)). Written as matmuls rather
    than scatter_reduce so the token scale stays invariant to atom count and the
    graph survives ONNX export (ScatterND with a reduction is not portable).
    """
    def __init__(self, hidden_dim, num_substructures=4):
        super().__init__()
        self.num_substructures = num_substructures
        self.assign_lin = nn.Linear(hidden_dim, num_substructures)

    def forward(self, h_nodes, batch_mapping, num_graphs):
        # h_nodes: [total_nodes, D]
        # batch_mapping: [total_nodes]
        N, D = h_nodes.shape
        K = self.num_substructures

        # Soft assignment of atoms to K substructure clusters: [N, K]
        assign_weights = F.softmax(self.assign_lin(h_nodes), dim=-1)

        # One-hot molecule membership: [N, B].
        # Built from a comparison rather than F.one_hot because one_hot needs a
        # Python int for its class count, which tracing bakes in as a constant and
        # breaks dynamic batching in the exported graph.
        graph_ids = torch.arange(num_graphs, device=h_nodes.device)
        mask = (batch_mapping.unsqueeze(1) == graph_ids.unsqueeze(0)).to(h_nodes.dtype)
        mask_t = mask.t()  # [B, N]

        weighted = (assign_weights.unsqueeze(-1) * h_nodes.unsqueeze(1)).reshape(N, K * D)
        sums = mask_t @ weighted                                              # [B, K*D]
        counts = (mask_t @ assign_weights).clamp(min=1e-9)                    # [B, K]

        tokens = (sums / counts.unsqueeze(-1).expand(-1, -1, D).reshape(-1, K * D)).view(-1, K, D)
        return tokens, assign_weights

# ---------------------------------------------------------------------------
# Bi-Directional Cross-Attention Layer
# Models chemical interaction between substructure tokens of Drug A and Drug B
# ---------------------------------------------------------------------------
class BiDirectionalCrossAttention(nn.Module):
    """Cross-attention between the substructure tokens of Drug A and Drug B.

    One shared attention module (and one shared norm) serves both directions.
    Two separate modules would make the pair (attn_ab(A,B), attn_ba(B,A)) depend
    on the input order, which breaks f(A,B) == f(B,A) even with a commutative
    fusion head. Sharing is the encoder-side analogue of the Siamese weights.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens_a, tokens_b):
        # tokens_a: [B, K, D], tokens_b: [B, K, D]
        attended_a, weights_ab = self.attn(tokens_a, tokens_b, tokens_b)
        attended_b, weights_ba = self.attn(tokens_b, tokens_a, tokens_a)

        out_a = self.norm(tokens_a + attended_a)
        out_b = self.norm(tokens_b + attended_b)

        return out_a, out_b, weights_ab, weights_ba

# ---------------------------------------------------------------------------
# Complete Dual-Branch Molecular GNN with Cross-Attention Architecture
# ---------------------------------------------------------------------------
class MolecularGNN_DDI(nn.Module):
    def __init__(self, in_atom_features=24, hidden_dim=64, num_gnn_layers=3, num_substructures=4, dropout=0.15):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_substructures = num_substructures

        # 1. Atom Embedding Projection
        self.atom_proj = nn.Sequential(
            nn.Linear(in_atom_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )

        # 2. Dual-Branch Shared GATv2 Backbone
        self.gnn_layers = nn.ModuleList([
            GATv2Layer(hidden_dim, hidden_dim, heads=4, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # 3. Substructure Pharmacophore Pooling
        self.substructure_pooler = SubstructurePooler(hidden_dim, num_substructures=num_substructures)

        # 4. Bi-Directional Cross-Attention
        self.cross_attention = BiDirectionalCrossAttention(hidden_dim, num_heads=4, dropout=dropout)

        # 5. Prediction Head
        # Feature dimension: 2 molecules * num_substructures * hidden_dim
        total_dim = 2 * num_substructures * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def encode_molecular_graph(self, x, edge_index):
        h = self.atom_proj(x)
        for gnn, ln in zip(self.gnn_layers, self.layer_norms):
            h_res = h
            h = gnn(h, edge_index)
            h = ln(h + h_res)
            h = F.leaky_relu(h, 0.2)
        return h

    def forward(self, g1, g2):
        # 1. Encode both graphs via shared GATv2
        h1 = self.encode_molecular_graph(g1['x'], g1['edge_index'])
        h2 = self.encode_molecular_graph(g2['x'], g2['edge_index'])

        # 2. Extract K substructure tokens per drug: [B, K, D]
        #    Derived from the batch tensor rather than trusting a caller-supplied
        #    count: under DataParallel each replica only sees its shard, and a
        #    full-batch count would build an oversized membership mask.
        num_graphs = g1['batch'].max() + 1
        tokens_1, assign_1 = self.substructure_pooler(h1, g1['batch'], num_graphs)
        tokens_2, assign_2 = self.substructure_pooler(h2, g2['batch'], num_graphs)

        # 3. Bi-Directional Cross-Attention (Inter-Molecular Communication)
        attended_1, attended_2, cross_weights_12, cross_weights_21 = self.cross_attention(tokens_1, tokens_2)

        # 4. Symmetric (commutative) fusion of the interacted pharmacophore tokens.
        # f(A,B) == f(B,A) bitwise: swapping inputs swaps attended_1<->attended_2
        # (same ops, same weights, different order), and both product and
        # abs-difference are commutative. No extra parameters -> total_dim
        # unchanged (2 * K * D).
        flat_1 = attended_1.reshape(num_graphs, -1)
        flat_2 = attended_2.reshape(num_graphs, -1)
        pair_representation = torch.cat([flat_1 * flat_2, torch.abs(flat_1 - flat_2)], dim=-1)

        # 5. Classification
        logits = self.classifier(pair_representation).squeeze(-1)
        prob = torch.sigmoid(logits)

        return {
            "logits": logits,
            "prob": prob,
            "cross_weights": cross_weights_12,
            "assign_1": assign_1,
            "assign_2": assign_2,
            "h1": h1,
            "h2": h2
        }
