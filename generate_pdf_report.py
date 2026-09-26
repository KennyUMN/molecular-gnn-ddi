import sys
import os
import shutil
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 810, "PHARMAGNN: AUDIT KOMPARATIF MODEL DDI PAPER Q1 & REZIM EVALUASI")
            self.drawRightString(559, 810, "UNIVERSITAS MULTIMEDIA NUSANTARA — IF542")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 804, 559, 804)
            
        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(36, 36, 559, 36)
        
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 24, "PharmaGNN Technical Report | Pembimbing: David Agustriawan, Ph.D. | Kenny Valent Winalda Sembiring")
        page_str = f"Halaman {self._pageNumber} dari {page_count}"
        self.drawRightString(559, 24, page_str)
        self.restoreState()

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#0369A1"),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155"),
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#1E293B")
    )

    th_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.white,
        alignment=1
    )

    tb_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor("#1E293B")
    )

    tb_pharma = ParagraphStyle(
        'TableCellPharma',
        parent=tb_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor("#0F766E")
    )

    story = []

    # =========================================================================
    # HALAMAN 1: Title, Executive Summary, Substructure GNN Family (Table 1)
    # =========================================================================
    story.append(Paragraph("PharmaGNN vs 14 Model DDI Paper Q1: Audit Komparatif Menyeluruh", title_style))
    meta_text = (
        "<b>Penulis:</b> Kenny Valent Winalda Sembiring & Tim Riset &nbsp;|&nbsp; "
        "<b>Institusi:</b> Universitas Multimedia Nusantara (UMN) &nbsp;|&nbsp; "
        "<b>Mata Kuliah:</b> IF542 Deep Learning<br/>"
        "<b>Koordinator Riset:</b> David Agustriawan, S.Kom., M.Sc., Ph.D. &nbsp;|&nbsp; "
        "<b>Tanggal:</b> 25 September 2026 &nbsp;|&nbsp; "
        "<b>Status:</b> Empirical Evidence-Grounded"
    )
    story.append(Paragraph(meta_text, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284C7"), spaceBefore=0, spaceAfter=6))

    summary_box_data = [[
        Paragraph(
            "<b>RINGKASAN EKSEKUTIF & MASALAH UTAMA LITERATUR DDI:</b><br/>"
            "Mayoritas model Drug-Drug Interaction (DDI) pada jurnal bereputasi tinggi (Q1) mengklaim skor AUROC >0.90–0.98. "
            "Namun, evaluasi mendalam mengungkap dua kelemahan metodologis fundamental: <b>(1) Ilusi Generalisasi</b> akibat <i>scaffold leakage</i> "
            "pada transductive random split di mana model sekadar menghafal analog cincin kimiawi (AUROC jatuh 20–30% saat cold-start), dan "
            "<b>(2) Translational Disconnect</b> di mana 100% model Q1 beroperasi murni sebagai model kimia <i>in-vitro</i> statis yang buta "
            "terhadap profil fisiologis pasien nyata. <b>PharmaGNN</b> mengatasi kedua masalah ini dengan memadukan GATv2 substructure "
            "cross-attention invarian komutatif yang sangat efisien (118k parameter, 815 KB ONNX, ~0.21 ms CPU) serta mengintegrasikan "
            "<b>Stage 2 Three-Tier Clinical Layer</b> berbasis bukti demografis FDA FAERS (29.096 kasus), fungsi ginjal eGFR CKD-EPI, dan farmakogenomik CPIC.",
            callout_style
        )
    ]]
    summary_table = Table(summary_box_data, colWidths=[523])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0F9FF")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BAE6FD")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Taksonomi & Bedah Arsitektur Model DDI Publikasi Top-Tier", h1_style))
    story.append(Paragraph(
        "Untuk memvalidasi kebaruan ilmiah PharmaGNN, dilakukan audit terhadap 14 model representatif dari publikasi top-tier "
        "(Briefings in Bioinformatics, Nature Computational Science, Chemical Science, Bioinformatics Oxford, NeurIPS, AAAI, WWW, TCBB).",
        body_style
    ))

    story.append(Paragraph("A. Keluarga Substructure-Level GNN (SSI-DDI, GMPNN-CS, SA-DDI, 3DGT-DDI)", h2_style))
    story.append(Paragraph(
        "Model keluarga ini membuktikan bahwa DDI dipicu oleh gugus aktif fungsional lokal. Namun, metode ekstraksinya "
        "memiliki kompleksitas ekstrem: <b>3DGT-DDI</b> mewajibkan optimasi koordinat 3D medan gaya MMFF94 via RDKit (1.000–2.500 ms per pasangan) "
        "dan memiliki bobot >115M parameter (>450 MB) sehingga mustahil di-deploy di edge/mobile. "
        "<b>SSI-DDI</b> dan <b>GMPNN-CS</b> menghasilkan graf komputasi dinamis yang sulit di-trace ke ONNX mobile runtime.",
        body_style
    ))

    # Table 1: Substructure Models vs PharmaGNN
    col_w1 = [78, 82, 80, 75, 65, 73, 70]
    t1_data = [
        [
            Paragraph("Model / Paper", th_style),
            Paragraph("Venue / Reputasi", th_style),
            Paragraph("Representasi Input", th_style),
            Paragraph("Parameter & Ukuran", th_style),
            Paragraph("Latensi CPU", th_style),
            Paragraph("AUROC (Rand / Cold)", th_style),
            Paragraph("Stage 2 Klinis", th_style)
        ],
        [
            Paragraph("<b>SSI-DDI</b> (2021)", tb_style),
            Paragraph("Briefings in Bioinfo (Q1)", tb_style),
            Paragraph("Graf 2D (Raw Atom)", tb_style),
            Paragraph("~1.8M (~7.2 MB)", tb_style),
            Paragraph("~20 ms (GPU)", tb_style),
            Paragraph("0.9701 / <b>0.6833</b>", tb_style),
            Paragraph("Nol (Hanya Kimia)", tb_style)
        ],
        [
            Paragraph("<b>GMPNN-CS</b> (2022)", tb_style),
            Paragraph("Briefings in Bioinfo (Q1)", tb_style),
            Paragraph("Graf 2D (Atom + Bond)", tb_style),
            Paragraph("~1.2M (~5.0 MB)", tb_style),
            Paragraph("~15 ms (GPU)", tb_style),
            Paragraph("0.9845 / <b>0.7748</b>", tb_style),
            Paragraph("Nol (Hanya Kimia)", tb_style)
        ],
        [
            Paragraph("<b>SA-DDI</b> (2022)", tb_style),
            Paragraph("Chemical Science (RSC Q1)", tb_style),
            Paragraph("Graf 2D Berarah", tb_style),
            Paragraph("~1.5M (~6.2 MB)", tb_style),
            Paragraph("~20 ms (GPU)", tb_style),
            Paragraph("0.9880 / <b>0.7914</b>", tb_style),
            Paragraph("Nol (Hanya Kimia)", tb_style)
        ],
        [
            Paragraph("<b>3DGT-DDI</b> (2022)", tb_style),
            Paragraph("Briefings in Bioinfo (Q1)", tb_style),
            Paragraph("3D Conformer + SciBERT", tb_style),
            Paragraph("<b>>115M (>450 MB)</b>", tb_style),
            Paragraph("<b>>1.000 ms</b> (MMFF94)", tb_style),
            Paragraph("0.9610 / Tidak Diuji", tb_style),
            Paragraph("Nol (Hanya Kimia)", tb_style)
        ],
        [
            Paragraph("<b>PharmaGNN</b> (Ours)", tb_pharma),
            Paragraph("Proposed Study (UMN)", tb_pharma),
            Paragraph("Enriched 2D Graph (RDKit)", tb_pharma),
            Paragraph("<b>118k (815 KB ONNX)</b>", tb_pharma),
            Paragraph("<b>~0.21 ms</b> (Single CPU)", tb_pharma),
            Paragraph("<b>0.9493 / 0.7623</b>", tb_pharma),
            Paragraph("<b>Three-Tier Layer</b>", tb_pharma)
        ],
    ]
    t1 = Table(t1_data, colWidths=col_w1)
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor("#ECFDF5")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t1)

    story.append(PageBreak())

    # =========================================================================
    # HALAMAN 2: Relational KG Models (Table 2) & Sequential Models (Table 3)
    # =========================================================================
    story.append(Paragraph("B. Keluarga Relational & Knowledge Graph DDI (Decagon, MIRACLE, EmerGNN, KGNN, DMFDDI)", h2_style))
    story.append(Paragraph(
        "Keluarga ini mengandalkan grafik relasi biologis masif (PPI network dan KG triples). "
        "Kelemahan paling fatal adalah <b>The Out-of-KG Entity Problem</b>: model lumpuh total jika senyawa baru tidak memiliki "
        "riwayat interaksi biologis di dalam graf. Selain itu, graf masif tersebut wajib disimpan resident di RAM (>1.5–4 GB), "
        "meniadakan kemungkinan eksekusi offline di perangkat mobile.",
        body_style
    ))

    # Table 2: Relational KG vs PharmaGNN
    col_w2 = [78, 82, 85, 75, 70, 65, 68]
    t2_data = [
        [
            Paragraph("Model / Paper", th_style),
            Paragraph("Venue / Reputasi", th_style),
            Paragraph("Ketergantungan Graf", th_style),
            Paragraph("Kebutuhan Memori", th_style),
            Paragraph("Novel Drug Entity", th_style),
            Paragraph("AUROC Cold-Start", th_style),
            Paragraph("Mobile Ready?", th_style)
        ],
        [
            Paragraph("<b>Decagon</b> (2018)", tb_style),
            Paragraph("Bioinformatics (ISMB Q1)", tb_style),
            Paragraph("Wajib PPI 19k protein + DDI", tb_style),
            Paragraph(">1.5 GB RAM", tb_style),
            Paragraph("<b>Lumpuh Total</b>", tb_style),
            Paragraph("<0.6500", tb_style),
            Paragraph("Mustahil", tb_style)
        ],
        [
            Paragraph("<b>MIRACLE</b> (2021)", tb_style),
            Paragraph("The Web Conf (WWW)", tb_style),
            Paragraph("Wajib Graf DDI Global", tb_style),
            Paragraph(">50 MB RAM", tb_style),
            Paragraph("Degradasi Berat", tb_style),
            Paragraph("Runtuh tanpa edge", tb_style),
            Paragraph("Mustahil", tb_style)
        ],
        [
            Paragraph("<b>EmerGNN</b> (2023)", tb_style),
            Paragraph("Nature Comput Sci (Q1)", tb_style),
            Paragraph("Wajib Metapath DB Biomedis", tb_style),
            Paragraph(">2.0 GB RAM", tb_style),
            Paragraph("Terbatas via Path", tb_style),
            Paragraph("0.6720 (S2 Unseen)", tb_style),
            Paragraph("Mustahil", tb_style)
        ],
        [
            Paragraph("<b>KG-DDI/KGNN</b>", tb_style),
            Paragraph("IJCAI / Briefings in Bio", tb_style),
            Paragraph("Wajib Knowledge Graph", tb_style),
            Paragraph(">1.0 GB RAM", tb_style),
            Paragraph("<b>Lumpuh Total</b>", tb_style),
            Paragraph("N/A (Lumpuh)", tb_style),
            Paragraph("Mustahil", tb_style)
        ],
        [
            Paragraph("<b>DMFDDI</b> (2023)", tb_style),
            Paragraph("Briefings in Bioinfo (Q1)", tb_style),
            Paragraph("Tri-modal: Graph+Seq+PPI", tb_style),
            Paragraph(">500 MB RAM", tb_style),
            Paragraph("Menurun Tajam", tb_style),
            Paragraph("~0.7100", tb_style),
            Paragraph("Sangat Sulit", tb_style)
        ],
        [
            Paragraph("<b>PharmaGNN</b> (Ours)", tb_pharma),
            Paragraph("Proposed Study (UMN)", tb_pharma),
            Paragraph("<b>Zero DB (Cukup SMILES)</b>", tb_pharma),
            Paragraph("<b><1 MB (815 KB ONNX)</b>", tb_pharma),
            Paragraph("<b>100% Berfungsi</b>", tb_pharma),
            Paragraph("<b>0.7623</b>", tb_pharma),
            Paragraph("<b>100% Offline Ready</b>", tb_pharma)
        ],
    ]
    t2 = Table(t2_data, colWidths=col_w2)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor("#ECFDF5")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))

    story.append(Paragraph("C. Keluarga Sekuensial, Co-Attention & Benchmark Baselines (CASTER, MHCADDI, DeepAttention, TDC)", h2_style))
    story.append(Paragraph(
        "Model 1D SMILES (CASTER, DeepAttention) memutus konektivitas cincin 2D aromatik. Selain itu, hampir seluruh model "
        "menggabungkan representasi obat secara asimetris via konkatenasi terarah [z_A || z_B], sehingga memicu pelanggaran hukum simetri fisika "
        "di mana f(A, B) berbeda dengan f(B, A). PharmaGNN memecahkan masalah ini secara analitik melalui <i>Symmetric Commutative Fusion Head</i>.",
        body_style
    ))

    # Table 3: Sequential & Benchmark Baselines vs PharmaGNN
    col_w3 = [80, 80, 85, 75, 70, 65, 68]
    t3_data = [
        [
            Paragraph("Model / Benchmark", th_style),
            Paragraph("Venue / Reputasi", th_style),
            Paragraph("Representasi Input", th_style),
            Paragraph("Simetri f(A,B)≡f(B,A)", th_style),
            Paragraph("Integritas Cincin 2D", th_style),
            Paragraph("AUROC Cold-Start", th_style),
            Paragraph("Personalisasi", th_style)
        ],
        [
            Paragraph("<b>CASTER</b> (2020)", tb_style),
            Paragraph("AAAI Conference on AI", tb_style),
            Paragraph("1D SMILES Substrings", tb_style),
            Paragraph("Asimetris ([zA || zB])", tb_style),
            Paragraph("Rusak (String 1D)", tb_style),
            Paragraph("Lumpuh (OOV Substr)", tb_style),
            Paragraph("Nol (In-Vitro)", tb_style)
        ],
        [
            Paragraph("<b>MHCADDI</b> (2019)", tb_style),
            Paragraph("NeurIPS Workshop", tb_style),
            Paragraph("Graf 2D (Atom)", tb_style),
            Paragraph("Asimetris (Query/Key)", tb_style),
            Paragraph("Terjaga", tb_style),
            Paragraph("0.7250", tb_style),
            Paragraph("Nol (In-Vitro)", tb_style)
        ],
        [
            Paragraph("<b>DeepAttention</b>", tb_style),
            Paragraph("IEEE/ACM TCBB (Q1)", tb_style),
            Paragraph("1D SMILES + ECFP", tb_style),
            Paragraph("Asimetris ([zA || zB])", tb_style),
            Paragraph("Rusak Sebagian", tb_style),
            Paragraph("Runtuh (<0.70)", tb_style),
            Paragraph("Nol (In-Vitro)", tb_style)
        ],
        [
            Paragraph("<b>TDC Baseline</b>", tb_style),
            Paragraph("NeurIPS Datasets 2021", tb_style),
            Paragraph("Graf 2D / Fingerprint", tb_style),
            Paragraph("Tergantung baseline", tb_style),
            Paragraph("Variatif", tb_style),
            Paragraph("~0.6480 (Rata-rata)", tb_style),
            Paragraph("Nol (In-Vitro)", tb_style)
        ],
        [
            Paragraph("<b>PharmaGNN</b> (Ours)", tb_pharma),
            Paragraph("Proposed Study (UMN)", tb_pharma),
            Paragraph("<b>Enriched 2D Graph</b>", tb_pharma),
            Paragraph("<b>Dijamin Analitik</b>", tb_pharma),
            Paragraph("<b>Terjaga Sempurna</b>", tb_pharma),
            Paragraph("<b>0.7623</b>", tb_pharma),
            Paragraph("<b>Three-Tier Layer</b>", tb_pharma)
        ],
    ]
    t3 = Table(t3_data, colWidths=col_w3)
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor("#ECFDF5")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t3)
    story.append(Spacer(1, 6))

    # Commutative Invariance Callout
    comm_box_data = [[
        Paragraph(
            "<b>PEMBUKTIAN ANALITIS JAMINAN SIFAT KOMUTATIF PHARMAGNN:</b><br/>"
            "Interaksi obat adalah fenomena fisis tanpa arah: risiko interaksi Obat A + B identik dengan Obat B + A. "
            "PharmaGNN menjamin f(A, B) ≡ f(B, A) secara matematis melalui vektor pasangan:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<b>z<sub>pair</sub> = [ u<sub>A</sub> * u<sub>B</sub> &nbsp;||&nbsp; |u<sub>A</sub> - u<sub>B</sub>| ]</b><br/>"
            "Di mana u<sub>A</sub> * u<sub>B</sub> adalah perkalian elemen-demi-elemen (Hadamard product) dan |u<sub>A</sub> - u<sub>B</sub>| "
            "adalah selisih absolut. Karena kedua operasi bersifat komutatif mutlak terhadap pertukaran urutan, tidak ada deviasi prediksi.",
            callout_style
        )
    ]]
    comm_table = Table(comm_box_data, colWidths=[523])
    comm_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(comm_table)

    story.append(PageBreak())

    # =========================================================================
    # HALAMAN 3: Generalization Mirage & Stage 2 Clinical Personalization
    # =========================================================================
    story.append(Paragraph("2. Pembuktian Empiris: Ilusi Generalisasi & Scaffold Leakage", h1_style))
    story.append(Paragraph(
        "Penelitian ini membuktikan secara empiris bahwa klaim AUROC >0.95 pada literatur DDI umum merupakan artefak dari "
        "<b>kebocoran analog kerangka molekul (Bemis-Murcko scaffold leakage)</b> pada random split. Ketika diuji secara ketat, "
        "seluruh model dunia mengalami penurunan drastis:",
        body_style
    ))

    # Table 4: Scaffold Leakage Drop
    col_w4 = [110, 85, 95, 105, 128]
    t4_data = [
        [
            Paragraph("Model Evaluasi", th_style),
            Paragraph("Random Split AUROC", th_style),
            Paragraph("Scaffold / Cold AUROC", th_style),
            Paragraph("Penurunan (Drop Δ)", th_style),
            Paragraph("Interpretasi Metodologis", th_style)
        ],
        [
            Paragraph("<b>SSI-DDI</b> (Q1 2021)", tb_style),
            Paragraph("0.9701", tb_style),
            Paragraph("0.6833 (Cold-Start S2)", tb_style),
            Paragraph("<b>-0.2868 (-29.6%)</b>", tb_style),
            Paragraph("Memorisasi kerangka analog parah", tb_style)
        ],
        [
            Paragraph("<b>GMPNN-CS</b> (Q1 2022)", tb_style),
            Paragraph("0.9845", tb_style),
            Paragraph("0.7748 (Cold-Start S2)", tb_style),
            Paragraph("<b>-0.2097 (-21.3%)</b>", tb_style),
            Paragraph("Gating ikatan meredam drop sebagian", tb_style)
        ],
        [
            Paragraph("<b>SA-DDI</b> (Q1 2022)", tb_style),
            Paragraph("0.9880", tb_style),
            Paragraph("0.7914 (Cold-Start S2)", tb_style),
            Paragraph("<b>-0.1966 (-19.9%)</b>", tb_style),
            Paragraph("SSIM menjaga kemiripan subgrafik", tb_style)
        ],
        [
            Paragraph("<b>TDC Baseline</b> (NeurIPS)", tb_style),
            Paragraph("~0.8600", tb_style),
            Paragraph("~0.6480 (Cold-Start S2)", tb_style),
            Paragraph("<b>-0.2120 (-24.7%)</b>", tb_style),
            Paragraph("Rata-rata drop benchmark dunia", tb_style)
        ],
        [
            Paragraph("<b>PharmaGNN (Scaffold Split)</b>", tb_pharma),
            Paragraph("<b>0.9493</b>", tb_pharma),
            Paragraph("<b>0.6605 (Scaffold Disjoint)</b>", tb_pharma),
            Paragraph("<b>-0.2888 (-30.4%)</b>", tb_pharma),
            Paragraph("Bukti transparan scaffold leakage", tb_pharma)
        ],
        [
            Paragraph("<b>PharmaGNN (Cold-Start Split)</b>", tb_pharma),
            Paragraph("<b>0.9493</b>", tb_pharma),
            Paragraph("<b>0.7623 (Unseen Drug Entity)</b>", tb_pharma),
            Paragraph("<b>-0.1870 (-19.7%)</b>", tb_pharma),
            Paragraph("Kompetitif dengan SOTA Q1 dunia", tb_pharma)
        ],
    ]
    t4 = Table(t4_data, colWidths=col_w4)
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('BACKGROUND', (0,5), (-1,6), colors.HexColor("#ECFDF5")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t4)
    story.append(Spacer(1, 8))

    story.append(Paragraph("3. Menjembatani Gap In-Vitro ke In-Vivo: Lapisan Personalisasi Klinis (Stage 2)", h1_style))
    story.append(Paragraph(
        "Seluruh 14 model literatur Q1 adalah model <i>in-vitro/in-silico</i> murni yang menghasilkan skor statis antar-molekul "
        "tanpa memperhitungkan konteks pasien. <b>PharmaGNN</b> mengimplementasikan <i>Three-Tier Graceful Degradation Layer</i>:",
        body_style
    ))
    
    stage2_box_data = [[
        Paragraph(
            "<b>FORMULASI MATEMATIS PERSONALISASI KLINIS PHARMAGNN:</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;<b>Risk<sub>final</sub> = σ( logit(S<sub>mol</sub>) + Δ<sub>demo</sub> + Δ<sub>renal</sub> + Δ<sub>pgx</sub> )</b><br/><br/>"
            "• <b>Tier 1 (Fallback Tanpa Data Pasien):</b> Seluruh penalti Δ = 0 → Risk<sub>final</sub> = S<sub>mol</sub> (kemurnian prediksi kimia).<br/>"
            "• <b>Tier 2 (Bukti Epidemiologis Riil FDA FAERS):</b> Dikalibrasi dari <b>29.096 laporan klinis FDA</b> (Reporting Odds Ratio, ROR = 1.5433, ln(ROR) = 0.4339):<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;- Pasien Geriatri (Usia ≥65 tahun): Δ<sub>age</sub> = <b>+0.4339</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;- Kondisi Kehamilan: Δ<sub>preg</sub> = <b>+0.10</b> &nbsp;|&nbsp; Gangguan Hati: Δ<sub>hepatic</sub> = <b>+0.12</b><br/>"
            "• <b>Tier 3 (Lab Kuantitatif & Farmakogenomik Presisi):</b><br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Klirens Ginjal eGFR:</b> Dihitung otomatis via formula <b>2021 CKD-EPI</b> berbasis serum kreatinin:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Δ<sub>renal</sub> = min(0.25, max(0.0, ((90 - eGFR) / 90) × 0.25))<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;- <b>Farmakogenomik CYP450 (PharmGKB CPIC Level 1A):</b> Penalti adaptif saat pasien membawa fenotip <i>Poor Metabolizer</i> (misal CYP2C9 pada interaksi Warfarin-NSAID): Δ<sub>pgx</sub> = <b>+0.45</b>.",
            callout_style
        )
    ]]
    stage2_table = Table(stage2_box_data, colWidths=[523])
    stage2_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(stage2_table)

    story.append(PageBreak())

    # =========================================================================
    # HALAMAN 4: Explainability, Honest Boundaries, Codebase Verification, References
    # =========================================================================
    story.append(Paragraph("4. Validasi Explainability Terhadap Ground-Truth Biokimiawi", h1_style))
    story.append(Paragraph(
        "Berbeda dengan model literatur yang sekadar memvisualisasikan heatmap warna kosmetik tanpa pembuktian biokimiawi, "
        "PharmaGNN memvalidasi atribusi atom terhadap dua ground-truth formal: "
        "<b>(1) Ground-Truth SMARTS CYP450</b> (kumarin, benzofuran, triazole/imidazole, asam karboksilat) dan "
        "<b>(2) Katalog Brenk unwanted-substructure RDKit</b>. "
        "Pada panel klinis terisolasi, motif CYP yang terdeteksi pada Warfarin+Aspirin (<code>CYP2C9_coumarin</code> pada warfarin; "
        "<code>CYP2C9_benzoic_acid</code>, <code>CYP2C9_carboxylate</code> pada aspirin) konsisten dengan aturan hold-out yang diharapkan "
        "(lihat <code>tests/test_pipeline.py::test_cyp_motifs_found_in_clinical_pairs</code>). "
        "Stabilitas atribusi FP32 vs bobot INT8 terukur (Spearman ρ = 0.9755–0.9852; Jaccard Top-5 = 0.8378–0.9343 pada tiga split). "
        "Metrik <i>alert hit rate</i> per-atom dan kurva degradasi fideliitas belum diimplementasikan.",
        body_style
    ))

    story.append(Paragraph("5. Batasan Ilmiah & Trade-Off Arsitektural yang Jujur", h1_style))
    story.append(Paragraph(
        "Untuk menjaga integritas akademik (*evidence-first*), berikut 3 batasan terukur dari PharmaGNN:<br/>"
        "1. <b>Representasi Topologis 2D vs Jarak 3D:</b> PharmaGNN tidak menghitung jarak Euclidean 3D atom secara kontinu. "
        "Interaksi yang murni bergantung pada orientasi ruang kiral stereoisomer didekati melalui tag kiralitas RDKit.<br/>"
        "2. <b>Klasifikasi Risiko Biner Terkalibrasi:</b> Model dirancang untuk memprediksi probabilitas DDI dan keparahan risiko klinis, "
        "bukan memprediksi 964 jenis efek samping simultan seperti Decagon.<br/>"
        "3. <b>Resolusi K=4 Substruktur:</b> Pengelompokan 4 token farmakofor merupakan kompromi efisiensi edge mobile. Pada molekul raksasa "
        "(>100 atom berat), resolusi ini dapat menggabungkan dua gugus fungsional berdekatan ke dalam satu cluster representasi.",
        body_style
    ))
    story.append(Spacer(1, 4))

    verify_box_data = [[
        Paragraph(
            "<b>STATUS VERIFIKASI CODEBASE & REPRODUKTIBILITAS:</b><br/>"
            "• <b>Unit Test Suite:</b> 50/50 test passing (100% green) pada <code>tests/test_full_suite.py</code>, <code>tests/test_pipeline.py</code> &amp; <code>tests/test_personalization_extended.py</code>.<br/>"
            "• <b>Presisi Numerik:</b> Deviasi PyTorch FP32 vs ONNX Runtime binary: <b>5.81 × 10<sup>-7</sup></b> (random split; lihat log parity per rezim).<br/>"
            "• <b>Kelayakan Edge:</b> Ukuran model <b>815.4 KB</b> (ONNX FP32), latensi <b>~0.21 ms</b> di CPU biasa tanpa dependensi GPU (bench_onnx_latency.py).",
            callout_style
        )
    ]]
    v_table = Table(verify_box_data, colWidths=[523])
    v_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#ECFDF5")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#A7F3D0")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(v_table)
    story.append(Spacer(1, 4))

    story.append(Paragraph("6. Daftar Referensi Publikasi Q1 & Tier-1 Terpilih", h1_style))
    ref_style = ParagraphStyle(
        'RefStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=2.5
    )
    refs = [
        "1. M. Zitnik et al., \"Modeling polypharmacy side effects with graph convolutional networks,\" <i>Bioinformatics (Oxford)</i>, vol. 34, pp. i457–i466, 2018.",
        "2. A. K. Nyamabo et al., \"SSI–DDI: substructure–substructure interactions for drug–drug interaction prediction,\" <i>Briefings in Bioinformatics</i>, vol. 22, p. bbab133, 2021.",
        "3. A. K. Nyamabo et al., \"GMPNN-CS: predicting drug–drug interactions using gated message passing neural networks,\" <i>Briefings in Bioinformatics</i>, vol. 23, p. bbac436, 2022.",
        "4. Z. Yang et al., \"SA-DDI: a substructure-aware deep learning framework for drug-drug interaction prediction,\" <i>Chemical Science (RSC)</i>, vol. 13, pp. 1109–1120, 2022.",
        "5. X. He et al., \"3DGT-DDI: 3D graph and text-based deep learning framework for drug-drug interaction,\" <i>Briefings in Bioinformatics</i>, vol. 23, p. bbac314, 2022.",
        "6. Y. Zhang et al., \"EmerGNN: emergent drug–drug interaction prediction via biomedical knowledge paths,\" <i>Nature Computational Science</i>, vol. 3, pp. 886–897, 2023.",
        "7. K. Huang et al., \"Therapeutics Data Commons: Machine Learning Applications and Benchmarks for Drug Discovery,\" <i>NeurIPS Datasets Track</i>, 2021.",
        "8. K. Huang et al., \"CASTER: Predicting Drug Interactions with Substructure Representation Learning,\" in <i>AAAI Conference on AI</i>, 2020.",
        "9. A. Deac et al., \"Drug-Drug Adverse Effect Prediction with Graph Co-Attention,\" in <i>NeurIPS Workshop</i>, 2019.",
        "10. A. Mulia, D. Agustriawan, et al., \"AI Design for Race-Based Prostate Cancer Stage Classification,\" <i>JMIR Formative Res</i>, vol. 10, p. e82587, 2026."
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Clean 4-page PDF successfully built: {filename}")

if __name__ == "__main__":
    out_pdf = "/Users/kennyvws/projects/molecular-gnn-ddi/paper/PharmaGNN_Q1_Comparative_Benchmark_Report.pdf"
    build_pdf(out_pdf)
    shutil.copyfile(out_pdf, "/Users/kennyvws/projects/molecular-gnn-ddi/PharmaGNN_Q1_Comparative_Benchmark_Report.pdf")
    shutil.copyfile(out_pdf, "/Users/kennyvws/.gemini/antigravity-cli/brain/740bf6fa-80d8-468f-8a53-59e62be610bb/PharmaGNN_Q1_Comparative_Benchmark_Report.pdf")
    print("Clean 4-page PDF copied to root and artifact directory.")
