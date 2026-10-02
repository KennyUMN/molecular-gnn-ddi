import os
import sys
import torch
import torch.nn as nn

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MolecularGNN_DDI

INT8_SIZE_BUDGET_KB = 60.0


class ONNXExportWrapper(nn.Module):
    """
    Wrapper that accepts tensor inputs suitable for standard ONNX Runtime on mobile.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2):
        # Last molecule id + 1. Kept as a plain tensor expression: a Python
        # `if batch1.numel() > 0` gets baked in as a trace-time constant.
        num_graphs = batch1[-1] + 1

        g1 = {
            "x": x1,
            "edge_index": edge_index1,
            "batch": batch1,
            "num_graphs": num_graphs
        }
        g2 = {
            "x": x2,
            "edge_index": edge_index2,
            "batch": batch2,
            "num_graphs": num_graphs
        }
        out = self.model(g1, g2)
        return out["prob"]


def load_trained_model(model_checkpoint_path="models/best_model.pt", hidden_dim=64, num_gnn_layers=3):
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {model_checkpoint_path}")
    model = MolecularGNN_DDI(in_atom_features=24, hidden_dim=hidden_dim, num_gnn_layers=num_gnn_layers)
    state_dict = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Loaded trained weights from {model_checkpoint_path}")
    model.eval()
    return model


def emulate_int8_weights(model, per_channel=True, in_place=False):
    """Weight-only INT8 emulation (symmetric, per-output-channel).

    Mirrors what onnxruntime.quantization.quantize_dynamic does to MatMul/Gemm
    weights, so the attribution drift measured in torch matches the exported
    INT8 binary. ponytail: matmul weights only (dim >= 2), activations stay
    FP32 — that is exactly the dynamic-quantization scheme, not a shortcut.

    By default (in_place=False), deep-copies the model to preserve caller weights.
    """
    if not in_place:
        import copy
        model = copy.deepcopy(model)

    with torch.no_grad():
        for param in model.parameters():
            if param.dim() < 2:
                continue
            scale = param.abs().amax(dim=0 if not per_channel else tuple(range(1, param.dim())), keepdim=True)
            scale = (scale / 127.0).clamp(min=1e-8)
            param.copy_(torch.round(param / scale).clamp(-127, 127) * scale)
    return model


def quantize_onnx_int8(fp32_path, int8_path, per_channel=True):
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType

    # Sanitize ONNX model: torch exporter may include initializers in value_info,
    # causing shape inference mismatch when ORT transposes Gemm weights to MatMul.
    model = onnx.load(fp32_path)
    init_names = {init.name for init in model.graph.initializer}
    clean_value_info = [vi for vi in model.graph.value_info if vi.name not in init_names]
    model.graph.ClearField("value_info")
    model.graph.value_info.extend(clean_value_info)
    onnx.save(model, fp32_path)

    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )

    # Fail loudly instead of shipping a broken artifact: MatMulInteger load
    # errors only surface when onnxruntime builds the session.
    import onnxruntime as ort
    ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    size_kb = os.path.getsize(int8_path) / 1024.0
    if size_kb > INT8_SIZE_BUDGET_KB:
        raise RuntimeError(
            f"INT8 model is {size_kb:.1f} KB, over the {INT8_SIZE_BUDGET_KB:.0f} KB budget"
        )
    return size_kb


def export_model_to_onnx(model_checkpoint_path="models/best_model.pt", output_onnx_path="models/molecular_gnn_ddi.onnx",
                         hidden_dim=64, num_gnn_layers=3, quantize_int8=False, int8_output_path=None):
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)

    model = load_trained_model(model_checkpoint_path, hidden_dim=hidden_dim, num_gnn_layers=num_gnn_layers)

    wrapper = ONNXExportWrapper(model)
    wrapper.eval()

    # Dummy inputs representing a drug pair (e.g. Aspirin with 21 atoms, Warfarin with 30 atoms)
    dummy_x1 = torch.randn(21, 24)
    dummy_edge1 = torch.randint(0, 21, (2, 40))
    dummy_batch1 = torch.zeros(21, dtype=torch.long)

    dummy_x2 = torch.randn(30, 24)
    dummy_edge2 = torch.randint(0, 30, (2, 60))
    dummy_batch2 = torch.zeros(30, dtype=torch.long)

    print(f"Exporting model to ONNX: {output_onnx_path}...")
    try:
        # dynamo=True (torch.export-based) instead of jit tracing: the traced graph
        # baked shape comparisons in as constants, emitted 27 If subgraphs and was
        # not dynamic-batch safe, and onnxruntime's quantizer could not type its
        # weights. external_data=False keeps it a single self-contained file, the
        # form the edge/INT8 size budget is measured against.
        torch.onnx.export(
            wrapper,
            (dummy_x1, dummy_edge1, dummy_batch1, dummy_x2, dummy_edge2, dummy_batch2),
            output_onnx_path,
            export_params=True,
            # opset 18, not 17: the torch exporter otherwise bumps to 18 internally
            # and down-converts, which emitted ReduceMax with noop_with_empty_axes
            # and made the file unloadable in onnxruntime.
            opset_version=18,
            do_constant_folding=True,
            dynamo=True,
            external_data=False,
            input_names=['x1', 'edge_index1', 'batch1', 'x2', 'edge_index2', 'batch2'],
            output_names=['interaction_probability'],
            dynamic_shapes={
                'x1': {0: 'num_atoms_1'},
                'edge_index1': {1: 'num_bonds_1'},
                'batch1': {0: 'num_atoms_1'},
                'x2': {0: 'num_atoms_2'},
                'edge_index2': {1: 'num_bonds_2'},
                'batch2': {0: 'num_atoms_2'},
            }
        )
        file_size = os.path.getsize(output_onnx_path) / 1024
        print(f"✅ Model successfully exported to ONNX: {output_onnx_path} (FP32 size: {file_size:.1f} KB)")

        # Parity check: the artifact we serve must numerically match the
        # PyTorch model we test (audit finding: served-vs-tested gap).
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
        feed = {
            "x1": dummy_x1.numpy(), "edge_index1": dummy_edge1.numpy(), "batch1": dummy_batch1.numpy(),
            "x2": dummy_x2.numpy(), "edge_index2": dummy_edge2.numpy(), "batch2": dummy_batch2.numpy(),
        }
        onnx_out = np.asarray(sess.run(None, feed)[0]).ravel()
        with torch.no_grad():
            torch_out = wrapper(dummy_x1, dummy_edge1, dummy_batch1,
                                dummy_x2, dummy_edge2, dummy_batch2).numpy().ravel()
        parity = float(np.abs(onnx_out - torch_out).max())
        if parity > 1e-4:
            raise RuntimeError(f"ONNX parity check FAILED: max |delta| = {parity:.2e} > 1e-4")
        print(f"   ONNX parity vs PyTorch: max |delta| = {parity:.2e} -> PASS")

        if quantize_int8:
            int8_path = int8_output_path or output_onnx_path.replace(".onnx", "_int8.onnx")
            try:
                int8_kb = quantize_onnx_int8(output_onnx_path, int8_path)
            except Exception as e:
                # Never ship a broken/over-budget artifact: remove it and fail.
                if os.path.exists(int8_path):
                    os.remove(int8_path)
                print(f"❌ INT8 ONNX export FAILED (removed {int8_path}): {e}")
                # FP32 succeeded (parity passed above); only the INT8 variant failed.
                # Return a per-artifact status so callers don't conflate the two.
                return {"fp32": True, "int8": False, "int8_error": str(e)}
            print(f"✅ INT8 ONNX (dynamic, per-channel): {int8_path} (size: {int8_kb:.2f} KB)")
            print(f"   size budget <= {INT8_SIZE_BUDGET_KB:.0f} KB -> PASS")
            return {"fp32": True, "int8": True, "int8_path": int8_path}
        return {"fp32": True, "int8": None}
    except Exception as e:
        print(f"❌ ONNX Export FAILED: {e}")
        return {"fp32": False, "int8": None, "error": str(e)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/best_model.pt")
    ap.add_argument("--output", default="models/molecular_gnn_ddi.onnx")
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    export_model_to_onnx(args.checkpoint, args.output, hidden_dim=args.hidden_dim, quantize_int8=args.int8)
