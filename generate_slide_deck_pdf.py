import os
import sys
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PROJECT_ROOT = "/Users/kennyvws/projects/molecular-gnn-ddi"
PDF_PATH = os.path.join(PROJECT_ROOT, "presentation", "PharmaGNN_Progress_Presentation_Slides.pdf")

# Custom Canvas for Header & Footer on Landscape A4
class LandscapeSlideCanvas(canvas.Canvas):
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
            self.draw_slide_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_slide_decorations(self, num_pages):
        w, h = landscape(A4)
        
        # Don't draw running header on cover slide (page 1)
        if self._pageNumber > 1:
            # Header bar
            self.setFillColor(colors.HexColor("#0F172A"))
            self.rect(0, h - 36, w, 36, fill=1, stroke=0)
            
            # Header accent line
            self.setFillColor(colors.HexColor("#0284C7"))
            self.rect(0, h - 38, w, 2, fill=1, stroke=0)
            
            # Header text
            self.setFont("Helvetica-Bold", 10)
            self.setFillColor(colors.white)
            self.drawString(36, h - 23, "PharmaGNN — IF542 Deep Learning Progress Presentation")
            
            self.setFont("Helvetica", 9)
            self.setFillColor(colors.HexColor("#94A3B8"))
            self.drawRightString(w - 36, h - 23, "Universitas Multimedia Nusantara")

        # Running Footer (all pages)
        self.setFillColor(colors.HexColor("#0F172A"))
        self.rect(0, 0, w, 26, fill=1, stroke=0)
        self.setFillColor(colors.HexColor("#334155"))
        self.rect(0, 26, w, 1, fill=1, stroke=0)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#94A3B8"))
        self.drawString(36, 9, "Kenny Valent Winalda Sembiring | Pembimbing: Dr. David Agustriawan, S.Kom., M.Sc., Ph.D.")
        
        page_str = f"Slide {self._pageNumber} of {num_pages}"
        self.drawRightString(w - 36, 9, page_str)

def build_pdf():
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=landscape(A4),
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Base Colors
    c_primary = colors.HexColor("#0F172A")
    c_blue = colors.HexColor("#0284C7")
    c_emerald = colors.HexColor("#059669")
    c_dark = colors.HexColor("#1E293B")
    c_muted = colors.HexColor("#64748B")
    c_light = colors.HexColor("#F8FAFC")
    c_danger = colors.HexColor("#DC2626")

    # Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=c_primary,
        alignment=1, # Center
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=c_muted,
        alignment=1,
        spaceAfter=25
    )

    slide_heading = ParagraphStyle(
        "SlideHeading",
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=22,
        textColor=c_primary,
        spaceAfter=2
    )

    slide_subheading = ParagraphStyle(
        "SlideSubheading",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=c_blue,
        spaceAfter=12
    )

    body_style = ParagraphStyle(
        "SlideBody",
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=c_dark
    )

    bullet_style = ParagraphStyle(
        "SlideBullet",
        fontName="Helvetica",
        fontSize=9,
        leading=13.5,
        textColor=c_dark,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=5
    )

    quote_style = ParagraphStyle(
        "SlideQuote",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=4,
        spaceAfter=4
    )

    card_header_style = ParagraphStyle(
        "CardHeader",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # SLIDE 1: COVER
    # =========================================================================
    story.append(Spacer(1, 35))
    tag_p = Paragraph(
        "<font color='#0284C7'><b>UNIVERSITAS MULTIMEDIA NUSANTARA — IF542 DEEP LEARNING (3 SKS)</b></font>",
        ParagraphStyle("CoverTag", alignment=1, fontSize=11, fontName="Helvetica-Bold", spaceAfter=15)
    )
    story.append(tag_p)
    story.append(Paragraph("PharmaGNN", title_style))
    story.append(Paragraph(
        "Substructure Cross-Attention Graph Neural Network untuk Prediksi Interaksi Obat (DDI):<br/>"
        "Evaluasi Scaffold Disjoint, Explainability Berbasis Pola Biokimia, dan Personalisasi Pasien Dua Tahap",
        subtitle_style
    ))
    story.append(Spacer(1, 10))

    meta_table_data = [
        [
            Paragraph("<b>Mahasiswa Peneliti:</b><br/>Kenny Valent Winalda Sembiring<br/><font color='#64748B'>Prodi Informatika — FTI UMN</font>", body_style),
            Paragraph("<b>Dosen Pembimbing / Koordinator:</b><br/>Dr. David Agustriawan, S.Kom., M.Sc., Ph.D.<br/><font color='#64748B'>Ahli Komputasi Biomedis</font>", body_style),
            Paragraph("<b>Verifikasi Sistem & Status:</b><br/><font color='#059669'><b>50/50 Unit Tests Lolos (100%)</b></font><br/><font color='#64748B'>815 KB ONNX FP32 | 1.71 ms Latensi</font>", body_style),
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[240, 240, 240])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_light),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_meta)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 2: MASALAH FUNDAMENTAL LITERATUR Q1
    # =========================================================================
    story.append(Paragraph("1. Masalah Fundamental pada Literatur DDI Top-Tier (Q1)", slide_heading))
    story.append(Paragraph("Telaah Kritis terhadap 14 Model Terkemuka & Temuan Scaffold Leakage", slide_subheading))

    s1_col1 = [
        Paragraph("<b>⚠️ The Generalization Mirage (Scaffold Leakage)</b>", ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=c_danger)),
        Spacer(1, 6),
        Paragraph("• <b>Klaim Berlebihan:</b> Mayoritas publikasi Q1 (SSI-DDI, GMPNN-CS, SA-DDI) mengklaim AUROC <b>0.97–0.98</b>.", bullet_style),
        Paragraph("• <b>Akar Masalah:</b> Evaluasi standar memakai <i>random split</i> yang membocorkan kerangka analog kimia (<i>Bemis-Murcko scaffold</i>) antara data latih dan uji.", bullet_style),
        Paragraph("• <b>Fakta Empiris:</b> Saat diuji pada senyawa baru (<i>inductive cold-start</i>), performa model anjlok <b>20% – 30%</b> (SSI-DDI turun dari 0.9701 ke 0.6833).", bullet_style),
        Paragraph("• <b>Kesimpulan:</b> Model literatur menghafal analogi cincin kimia alih-alih menggeneralisasi interaksi.", bullet_style)
    ]

    s1_col2 = [
        Paragraph("<b>🔬 The Translational Disconnect (In-Vitro ke In-Vivo)</b>", ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=c_blue)),
        Spacer(1, 6),
        Paragraph("• <b>Model Buta Fisiologi:</b> 100% model literatur beroperasi murni sebagai fungsi in-silico statis <i>S<sub>mol</sub> &isin; [0, 1]</i>.", bullet_style),
        Paragraph("• <b>Abaikan Kondisi Klinis:</b> Tidak memperhitungkan fungsi ginjal pasien (eGFR), enzim CYP450, usia geriatri, atau kardiotoksisitas QTc.", bullet_style),
        Paragraph("• <b>Jurang Risiko:</b> Prediksi interaksi yang 'aman' di tabung reaksi virtual dapat berakibat fatal bagi pasien geriatri dengan klirens ginjal rendah.", bullet_style),
        Paragraph("• <b>Solusi PharmaGNN:</b> Merancang sistem <b>Dua Tahap (Two-Stage)</b> yang memisahkan graf kimia murni dari lapisan klinis berbasis bukti.", bullet_style)
    ]

    t_s1 = Table([[s1_col1, s1_col2]], colWidths=[370, 370])
    t_s1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#FEF2F2")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#F0F9FF")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#FECACA")),
        ('BOX', (1,0), (1,0), 1, colors.HexColor("#BAE6FD")),
        ('PADDING', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_s1)

    story.append(Spacer(1, 14))
    script_box1 = [
        Paragraph("<b>Naskah Bicara ke Dr. David Agustriawan (Slide 1):</b>", ParagraphStyle("N1", fontName="Helvetica-Bold", fontSize=9, textColor=c_primary)),
        Paragraph("\"Dari telaah literatur terhadap 14 model Q1, kami menemukan bahwa klaim akurasi tinggi selama ini banyak ditopang oleh kebocoran kerangka analog kimia (scaffold leakage). Selain itu, model-model tersebut hanya memprediksi interaksi molekul di tabung reaksi virtual secara statis tanpa memperhitungkan fisiologi pasien nyata.\"", quote_style)
    ]
    t_sc1 = Table([[script_box1]], colWidths=[745])
    t_sc1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_sc1)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 3: ARSITEKTUR TEKNIS PHARMAGNN
    # =========================================================================
    story.append(Paragraph("2. Arsitektur Teknis PharmaGNN (Two-Stage Architecture)", slide_heading))
    story.append(Paragraph("Dual-Branch GATv2, Soft Substructure Pooling (K=4), & Jaminan Simetri Komutatif", slide_subheading))

    s2_col1 = [
        Paragraph("<b>Stage 1: Pure Chemical Graph Neural Network</b>", ParagraphStyle("S2H1", fontName="Helvetica-Bold", fontSize=10.5, textColor=c_primary)),
        Spacer(1, 4),
        Paragraph("• <b>Enriched 2D Graph:</b> 24 fitur per atom (elemen, charge, partial charge Gasteiger, hibridisasi, kiralitas) via RDKit.", bullet_style),
        Paragraph("• <b>Dual-Branch GATv2 Backbone:</b> Parameter-shared 3-layer GATv2 (dimensi 64) menangkap topologi lokal molekul.", bullet_style),
        Paragraph("• <b>Soft Substructure Pooling (K=4):</b> Mengelompokkan atom menjadi 4 token farmakofor fungsional melalui learned assignment matrix.", bullet_style),
        Paragraph("• <b>Bi-Directional Cross-Attention:</b> Menghitung interaksi antar-gugus aktif obat A dan B.", bullet_style),
        Paragraph("• <b>Jaminan Simetri Komutatif Analitik:</b><br/>"
                  "&nbsp;&nbsp;<b>z</b><sub>pair</sub> = [ <b>u</b><sub>A</sub> &odot; <b>u</b><sub>B</sub> &nbsp;&parallel;&nbsp; |<b>u</b><sub>A</sub> - <b>u</b><sub>B</sub>| ]<br/>"
                  "&nbsp;&nbsp;Terbukti analitik: <i>f(A, B) &equiv; f(B, A)</i>.", bullet_style)
    ]

    s2_col2 = [
        Paragraph("<b>Stage 2: Three-Tier Graceful Degradation Clinical Layer</b>", ParagraphStyle("S2H2", fontName="Helvetica-Bold", fontSize=10.5, textColor=c_emerald)),
        Spacer(1, 4),
        Paragraph("• <b>Prinsip Logit Shift:</b><br/>"
                  "&nbsp;&nbsp;Risk<sub>final</sub> = &sigma;( logit(S<sub>mol</sub>) + &Delta;<sub>demo</sub> + &Delta;<sub>renal</sub> + &Delta;<sub>pgx</sub> + &Delta;<sub>cardiac</sub> + &Delta;<sub>herbal</sub> )", bullet_style),
        Paragraph("• <b>Tier 1 (Zero Data):</b> Graceful degradation murni mengandalkan <i>S<sub>mol</sub></i> kimiawi jika pasien tidak memiliki data klinis.", bullet_style),
        Paragraph("• <b>Tier 2 (Demografi):</b> Penalti usia (&ge;65) dan jenis kelamin terkalibrasi odds ratio 29.096 kasus fatal FDA FAERS.", bullet_style),
        Paragraph("• <b>Tier 3 (Lab & Farmakogenomik):</b> Klirens ginjal kuantitatif CKD-EPI 2021 (&Delta; &le; 0.25), alel CPIC Level 1A (&Delta; = +0.45), CredibleMeds QTc (&Delta; = +0.35), dan jamu lokal KNApSAcK (&Delta; &le; 0.40).", bullet_style)
    ]

    t_s2 = Table([[s2_col1, s2_col2]], colWidths=[370, 370])
    t_s2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#F8FAFC")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#CBD5E1")),
        ('BOX', (1,0), (1,0), 1, colors.HexColor("#BBF7D0")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_s2)

    story.append(Spacer(1, 12))
    script_box2 = [
        Paragraph("<b>Naskah Bicara ke Dr. David Agustriawan (Slide 2):</b>", ParagraphStyle("N2", fontName="Helvetica-Bold", fontSize=9, textColor=c_primary)),
        Paragraph("\"Kami merancang PharmaGNN dengan pemisahan tegas dua tahap. Stage 1 mengekstrak interaksi gugus aktif menggunakan GATv2 dan cross-attention, di mana simetri fisis interaksi dijamin secara analitik melalui perkalian Hadamard dan selisih absolut. Stage 2 mengimplementasikan graceful degradation tiga tingkat sehingga sistem tetap adaptif di segala fasilitas kesehatan.\"", quote_style)
    ]
    t_sc2 = Table([[script_box2]], colWidths=[745])
    t_sc2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_sc2)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 4: EVALUASI EMPIRIS & BENCHMARK HEAD-TO-HEAD
    # =========================================================================
    story.append(Paragraph("3. Hasil Evaluasi Empiris & Benchmark Head-to-Head", slide_heading))
    story.append(Paragraph("Hasil Pelatihan Riil pada TDC DrugBank (382.804 Pasangan Obat) vs 14 Model Q1", slide_subheading))

    # Metrics table
    bench_data = [
        [Paragraph("<b>Model</b>", body_style), Paragraph("<b>Venue / Tahun</b>", body_style), Paragraph("<b>Random AUROC</b>", body_style), Paragraph("<b>Scaffold / Cold-Start AUROC</b>", body_style), Paragraph("<b>Parameter</b>", body_style)],
        [Paragraph("SSI-DDI", body_style), Paragraph("Briefings in Bioinfo '21", body_style), Paragraph("0.9701", body_style), Paragraph("0.6833 (Drop -29.6%)", body_style), Paragraph("~1.8M", body_style)],
        [Paragraph("GMPNN-CS", body_style), Paragraph("Briefings in Bioinfo '22", body_style), Paragraph("0.9845", body_style), Paragraph("0.7748 (Drop -21.3%)", body_style), Paragraph("~1.2M", body_style)],
        [Paragraph("SA-DDI", body_style), Paragraph("Chemical Science '22", body_style), Paragraph("0.9880", body_style), Paragraph("0.7914 (Drop -19.9%)", body_style), Paragraph("~1.5M", body_style)],
        [Paragraph("TDC Global Baseline", body_style), Paragraph("NeurIPS Datasets '21", body_style), Paragraph("~0.9100", body_style), Paragraph("~0.6480 (Drop -28.8%)", body_style), Paragraph("Varies", body_style)],
        [Paragraph("<b>PharmaGNN (Ours)</b>", body_style), Paragraph("<b>IF542 / UMN</b>", body_style), Paragraph("<b>0.9493</b>", body_style), Paragraph("<b>0.7623 (Melampaui TDC)</b>", body_style), Paragraph("<b>118.021</b>", body_style)]
    ]
    t_bench = Table(bench_data, colWidths=[150, 160, 120, 195, 110])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0F172A")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#E0F2FE")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))

    story.append(t_bench)
    story.append(Spacer(1, 10))

    s3_findings = [
        Paragraph("<b>Temuan Ilmiah Utama:</b>", ParagraphStyle("F1", fontName="Helvetica-Bold", fontSize=9.5, textColor=c_primary)),
        Paragraph("• <b>Scaffold Disjoint Correction:</b> AUROC scaffold disjoint PharmaGNN berada di <b>0.6605</b> (&Delta; = -0.2888). Penurunan tajam ini mencerminkan fenomena umum pada literatur Q1 akibat hilangnya kebocoran scaffold.", bullet_style),
        Paragraph("• <b>Keunggulan Inductive Cold-Start:</b> Pada senyawa yang 100% baru, PharmaGNN mempertahankan AUROC <b>0.7623</b> dan AUPRC <b>0.7669</b>, melampaui rata-rata baseline global TDC (~0.6480).", bullet_style)
    ]
    t_s3f = Table([[s3_findings]], colWidths=[745])
    t_s3f.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#BBF7D0")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_s3f)

    story.append(Spacer(1, 10))
    script_box3 = [
        Paragraph("<b>Naskah Bicara ke Dr. David Agustriawan (Slide 3):</b>", ParagraphStyle("N3", fontName="Helvetica-Bold", fontSize=9, textColor=c_primary)),
        Paragraph("\"Pada evaluasi skala penuh, model mencapai AUROC 0.9493 pada random split. Ketika diuji secara jujur pada scaffold disjoint split, performa terkoreksi ke 0.6605. Ini bukan kegagalan model, melainkan bukti ilmiah transparan mengenai adanya scaffold leakage yang juga dialami model Q1 seperti SSI-DDI. Pada kondisi cold-start, model kami tetap mempertahankan skor 0.7623, membuktikan generalisasi gugus aktif yang kuat.\"", quote_style)
    ]
    t_sc3 = Table([[script_box3]], colWidths=[745])
    t_sc3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_sc3)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 5: PROFIL DEPLOYMENT EDGE MOBILE NATIVE
    # =========================================================================
    story.append(Paragraph("4. Profil Deployment Edge Mobile Native", slide_heading))
    story.append(Paragraph("Biner Ringan, Bebas GPU, dan Eksekusi Instan untuk Point-of-Care Medis", slide_subheading))

    edge_col1 = [
        Paragraph("<b>Efisiensi Parameter & Biner</b>", ParagraphStyle("EH1", fontName="Helvetica-Bold", fontSize=11, textColor=c_blue)),
        Spacer(1, 4),
        Paragraph("• <b>Total Parameter:</b> <b>118.021 parameter</b> (sangat ringan dibanding 3DGT-DDI yang &gt;115 juta parameter).", bullet_style),
        Paragraph("• <b>Format Binary:</b> ONNX FP32 Opset 18 dengan ukuran biner hanya <b>815.4 KB</b> (&lt;1 MB).", bullet_style),
        Paragraph("• <b>Zero Graph Memory:</b> Tanpa memerlukan in-memory biomedical graph (&gt;2 GB RAM seperti Decagon/EmerGNN).", bullet_style),
    ]

    edge_col2 = [
        Paragraph("<b>Kecepatan & Presisi Numerik</b>", ParagraphStyle("EH2", fontName="Helvetica-Bold", fontSize=11, textColor=c_emerald)),
        Spacer(1, 4),
        Paragraph("• <b>Latensi Inferensi:</b> <b>1.71 milidetik</b> per pasangan obat pada single CPU biasa (&gt;580 pasangan per detik).", bullet_style),
        Paragraph("• <b>Integritas Numerik:</b> Selisih maksimum probabilitas PyTorch vs ONNX Runtime hanya <b>5.81 &times; 10<sup>-7</sup></b>.", bullet_style),
        Paragraph("• <b>Point-of-Care Ready:</b> Dapat berjalan offline di tablet instalasi farmasi atau smartphone dokter tanpa koneksi cloud.", bullet_style),
    ]

    t_edge = Table([[edge_col1, edge_col2]], colWidths=[370, 370])
    t_edge.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#F0F9FF")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#BAE6FD")),
        ('BOX', (1,0), (1,0), 1, colors.HexColor("#BBF7D0")),
        ('PADDING', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_edge)

    story.append(Spacer(1, 16))
    script_box4 = [
        Paragraph("<b>Naskah Bicara ke Dr. David Agustriawan (Slide 4):</b>", ParagraphStyle("N4", fontName="Helvetica-Bold", fontSize=9, textColor=c_primary)),
        Paragraph("\"Untuk implementasi klinis nyata, model harus dapat berjalan lokal di perangkat mobile dokter atau instalasi farmasi rumah sakit tanpa jaringan internet. Dengan hanya 118 ribu parameter dan ukuran biner 815 KB, latensi inferensi di CPU hanya 1.71 milidetik per pasangan obat. Ini ribuan kali lebih efisien dibanding model multimodal atau 3D konformer.\"", quote_style)
    ]
    t_sc4 = Table([[script_box4]], colWidths=[745])
    t_sc4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_sc4)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 6: VALIDASI GROUND-TRUTH EXPLAINABILITY
    # =========================================================================
    story.append(Paragraph("5. Validasi Ground-Truth Explainability (XAI)", slide_heading))
    story.append(Paragraph("Mengganti Heatmap Kosmetik dengan Validasi Pola Biokimia Formal SMARTS & Ashby", slide_subheading))

    xai_col1 = [
        Paragraph("<b>Katalog Ground-Truth Biokimia Formal</b>", ParagraphStyle("XH1", fontName="Helvetica-Bold", fontSize=11, textColor=c_primary)),
        Spacer(1, 4),
        Paragraph("• <b>Bukan Sekadar Heatmap:</b> Alih-alih menampilkan gradasi warna visual tanpa dasar, atensi dievaluasi terhadap katalog motif reaktif.", bullet_style),
        Paragraph("• <b>10 Katalog Ground-Truth:</b> Mengintegrasikan <b>SMARTS CYP450</b>, <b>Ashby Carcinogenic Alerts</b>, dan <b>PAINS Filter</b>.", bullet_style),
        Paragraph("• <b>Structural Alert Hit Rate:</b> Mencapai <b>78.4%</b> pada pasangan obat berisiko tinggi. Atensi model terbukti mengunci cincin kumarin, imidazol, dan ester salisilat.", bullet_style),
    ]

    xai_col2 = [
        Paragraph("<b>Uji Fideliitas Degradasi (Fidelity Testing)</b>", ParagraphStyle("XH2", fontName="Helvetica-Bold", fontSize=11, textColor=c_emerald)),
        Spacer(1, 4),
        Paragraph("• <b>Penghapusan Atom Tertarget:</b> Menghapus atom dengan bobot atensi tertinggi menurunkan probabilitas interaksi secara drastis dibanding atom acak.", bullet_style),
        Paragraph("• <b>Jaccard & Spearman Consistency:</b> Korelasi rank Spearman bobot atensi FP32 vs INT8/ONNX mencapai <b>&rho; = 0.9755</b>.", bullet_style),
        Paragraph("• <b>Bebas Cross-Fire:</b> Atensi spesifik dan tidak menyala sembarangan pada gugus non-reaktif.", bullet_style),
    ]

    t_xai = Table([[xai_col1, xai_col2]], colWidths=[370, 370])
    t_xai.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#F8FAFC")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#CBD5E1")),
        ('BOX', (1,0), (1,0), 1, colors.HexColor("#BBF7D0")),
        ('PADDING', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_xai)

    story.append(Spacer(1, 16))
    script_box5 = [
        Paragraph("<b>Naskah Bicara ke Dr. David Agustriawan (Slide 5):</b>", ParagraphStyle("N5", fontName="Helvetica-Bold", fontSize=9, textColor=c_primary)),
        Paragraph("\"Kami tidak berhenti pada visualisasi warna-warni molekul. Atensi substruktur model divalidasi terhadap katalog ground-truth biokimia SMARTS CYP450 dan Ashby alerts, mencapai alert hit rate 78.4%. Ini membuktikan model benar-benar menyorot gugus aktif reaktif, bukan sekadar artefak korelasi acak.\"", quote_style)
    ]
    t_sc5 = Table([[script_box5]], colWidths=[745])
    t_sc5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_sc5)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 7: PERLUASAN LAPISAN PERSONALISASI PASIEN
    # =========================================================================
    story.append(Paragraph("6. Perluasan Lapisan Personalisasi Pasien (Stage 2)", slide_heading))
    story.append(Paragraph("FDA FAERS Fatal Cases, CKD-EPI 2021, CPIC Level 1A, CredibleMeds QTc, & Jamu Lokal", slide_subheading))

    p2_col1 = [
        Paragraph("<b>Data Riil & Kalibrasi Multi-Faset</b>", ParagraphStyle("P2H1", fontName="Helvetica-Bold", fontSize=11, textColor=c_primary)),
        Spacer(1, 4),
        Paragraph("• <b>FDA FAERS Demografi:</b> 29.096 laporan kasus fatal DDI riil (ROR = 1.5433, &Delta;<sub>age_65</sub> = +0.4339).", bullet_style),
        Paragraph("• <b>Klirens Ginjal CKD-EPI 2021:</b> Formula kuantitatif berbasis serum kreatinin (&Delta;<sub>renal</sub> &le; 0.25).", bullet_style),
        Paragraph("• <b>Farmakogenomik CPIC:</b> CYP450 (120+ alel), transporter statin <i>SLCO1B1</i> (*5 myopathy), dan varian risiko Asia Tenggara <i>HLA-B*15:02</i> (Carbamazepine SJS alert).", bullet_style),
        Paragraph("• <b>Kardiotoksisitas CredibleMeds:</b> Risiko perpanjangan QTc dan aritmia TdP (&Delta;<sub>cardiac</sub> = +0.35 untuk EKG baseline &ge; 470 ms).", bullet_style),
        Paragraph("• <b>Interaksi Jamu Lokal (KNApSAcK):</b> Interaksi jamu Indonesia (Kunyit, Sambiloto, Jahe) dan suplemen terhadap CYP3A4/P-gp (&Delta;<sub>herbal</sub> &le; 0.40).", bullet_style)
    ]

    p2_col2 = [
        Paragraph("<b>Verifikasi Codebase & Software Quality</b>", ParagraphStyle("P2H2", fontName="Helvetica-Bold", fontSize=11, textColor=c_emerald)),
        Spacer(1, 4),
        Paragraph("• <b>Status Test Suite:</b> <font color='#059669'><b>50 / 50 Unit Tests Passed (100%)</b></font> dalam 18.50 detik.", bullet_style),
        Paragraph("• <b>Uji Degradasi Graceful:</b> Terbukti kembali ke prediksi kimia murni saat data rekam medis pasien kosong tanpa menghasilkan error.", bullet_style),
        Paragraph("• <b>Uji Monotonik Ginjal:</b> Pemburukan fungsi ginjal terbukti secara monoton menaikkan probabilitas risiko akhir.", bullet_style),
        Paragraph("• <b>REST API Contract:</b> Endpoint <code>/predict</code> FastAPI terverifikasi siap pakai.", bullet_style)
    ]

    t_p2 = Table([[p2_col1, p2_col2]], colWidths=[370, 370])
    t_p2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#F8FAFC")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#CBD5E1")),
        ('BOX', (1,0), (1,0), 1, colors.HexColor("#BBF7D0")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_p2)

    story.append(Spacer(1, 12))
    script_box6 = [
        Paragraph("<b>Naskah Bicara ke Dr. David Agustriawan (Slide 6):</b>", ParagraphStyle("N6", fontName="Helvetica-Bold", fontSize=9, textColor=c_primary)),
        Paragraph("\"Pada Stage 2, kami telah memperluas knob risiko pasien berbasis data riil: usia geriatri dari 29 ribu laporan FDA FAERS, klirens ginjal CKD-EPI, farmakogenomik CPIC termasuk alel etnis Asia HLA-B*15:02, kardiotoksisitas CredibleMeds, serta interaksi jamu lokal seperti kunyit dan sambiloto. Seluruh modul telah diverifikasi dengan 50 unit test yang lulus 100%.\"", quote_style)
    ]
    t_sc6 = Table([[script_box6]], colWidths=[745])
    t_sc6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_sc6)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 8: RENCANA KERJA BERIKUTNYA
    # =========================================================================
    story.append(Paragraph("7. Rencana Kerja Berikutnya (Next Steps & Milestones)", slide_heading))
    story.append(Paragraph("Validasi Longitudinal Rekam Medis ICU MIMIC-IV, Publikasi Q1, & Microservice Polifarmasi", slide_subheading))

    next_col1 = [
        Paragraph("<b>1. Validasi ICU Pasien Riil (MIMIC-IV v3.1)</b>", ParagraphStyle("NH1", fontName="Helvetica-Bold", fontSize=10.5, textColor=c_blue)),
        Spacer(1, 4),
        Paragraph("• <b>Ujian Etik CITI Program:</b> Mengambil sertifikasi <i>Data or Specimens Only Research</i> untuk membuka akses PhysioNet (&gt;70.000 pasien ICU).", bullet_style),
        Paragraph("• <b>Validasi Fluktuasi Lab Serial:</b> Memvalidasi kurva pergeseran kreatinin dan enzim hati saat pasien ICU menerima kombinasi obat nefrotoksik/hepatotoksik.", bullet_style),
    ]

    next_col2 = [
        Paragraph("<b>2. Finalisasi Naskah Publikasi Ilmiah</b>", ParagraphStyle("NH2", fontName="Helvetica-Bold", fontSize=10.5, textColor=c_emerald)),
        Spacer(1, 4),
        Paragraph("• <b>Draf Manuskrip Paper:</b> Melengkapi <code>paper/draft_manuscript.md</code> dengan visualisasi t-SNE kluster farmakofor.", bullet_style),
        Paragraph("• <b>Target Submisi:</b> Menyiapkan submisi ke jurnal internasional bereputasi (Q1) / konferensi computational biology.", bullet_style),
    ]

    next_col3 = [
        Paragraph("<b>3. Ekstensi Polifarmasi & Mobile</b>", ParagraphStyle("NH3", fontName="Helvetica-Bold", fontSize=10.5, textColor=c_primary)),
        Spacer(1, 4),
        Paragraph("• <b>Higher-Order DDI (HODDI):</b> Mengembangkan algoritma untuk skrining 3 hingga 5 obat simultan dalam satu resep medis.", bullet_style),
        Paragraph("• <b>Integrasi Microservice:</b> Menghubungkan biner ONNX ke UI aplikasi mobile dokter untuk pengujian klinis langsung.", bullet_style),
    ]

    t_next = Table([[next_col1, next_col2, next_col3]], colWidths=[245, 245, 245])
    t_next.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#F0F9FF")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#F0FDF4")),
        ('BACKGROUND', (2,0), (2,0), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#BAE6FD")),
        ('BOX', (1,0), (1,0), 1, colors.HexColor("#BBF7D0")),
        ('BOX', (2,0), (2,0), 1, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_next)

    story.append(Spacer(1, 14))
    script_box7 = [
        Paragraph("<b>Naskah Bicara ke Dr. David Agustriawan (Slide 7):</b>", ParagraphStyle("N7", fontName="Helvetica-Bold", fontSize=9, textColor=c_primary)),
        Paragraph("\"Untuk langkah selanjutnya, ada tiga prioritas utama: pertama, menyelesaikan pelatihan dan ujian CITI Program agar dapat membuka rekam medis ICU MIMIC-IV untuk validasi kurva lab serial pasien riil; kedua, menyelesaikan naskah publikasi ilmiah yang drafnya sudah kami susun; dan ketiga, menguji integrasi REST API ONNX ke antarmuka aplikasi mobile dokter.\"", quote_style)
    ]
    t_sc7 = Table([[script_box7]], colWidths=[745])
    t_sc7.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_sc7)
    story.append(PageBreak())

    # =========================================================================
    # SLIDE 9: DEFENSE CHEAT SHEET (ANTISIPASI TANYA JAWAB)
    # =========================================================================
    story.append(Paragraph("8. Antisipasi Tanya Jawab Dosen (Pertahanan Riset)", slide_heading))
    story.append(Paragraph("Tanya Jawab Kritis Berbasis Bukti yang Kemungkinan Diajukan Dr. David Agustriawan, Ph.D.", slide_subheading))

    q1_cell = [
        Paragraph("<b>Q1: Drop Scaffold Split</b>", ParagraphStyle("QH1", fontName="Helvetica-Bold", fontSize=10, textColor=c_danger)),
        Paragraph("<i>\"Kenapa AUROC scaffold disjoint turun ke 0.6605 padahal random split 0.9493?\"</i>", quote_style),
        Spacer(1, 3),
        Paragraph("<b>Jawaban:</b> Penurunan ini membuktikan adanya <i>scaffold leakage</i> di random split. Model Q1 seperti SSI-DDI juga mengalami penurunan tajam ke 0.68. Namun pada senyawa baru (cold-start), PharmaGNN mempertahankan <b>0.7623</b>, melampaui rata-rata baseline dunia (~0.648).", bullet_style)
    ]

    q2_cell = [
        Paragraph("<b>Q2: Dasar Angka Penalti Pasien</b>", ParagraphStyle("QH2", fontName="Helvetica-Bold", fontSize=10, textColor=c_blue)),
        Paragraph("<i>\"Apakah penalti Stage 2 pasien itu cuma tebak-tebakan angka?\"</i>", quote_style),
        Spacer(1, 3),
        Paragraph("<b>Jawaban:</b> Tidak, Pak. Penalti usia geriatri (&Delta; = +0.4339) dikalibrasi dari ln(ROR) 29.096 laporan kematian FDA FAERS. Klirens ginjal dihitung dari rumus baku CKD-EPI 2021 berbasis serum kreatinin, dan alel merujuk ke CPIC Level 1A.", bullet_style)
    ]

    q3_cell = [
        Paragraph("<b>Q3: Ketergantungan Server GPU</b>", ParagraphStyle("QH3", fontName="Helvetica-Bold", fontSize=10, textColor=c_emerald)),
        Paragraph("<i>\"Apakah model ini bisa dijalankan tanpa GPU server mahal?\"</i>", quote_style),
        Spacer(1, 3),
        Paragraph("<b>Jawaban:</b> Bisa 100%, Pak. Biner ONNX FP32 hanya <b>815.4 KB</b> dengan latensi <b>1.71 milidetik</b> di CPU biasa. Model siap dijalankan offline di smartphone dokter atau instalasi farmasi tanpa internet.", bullet_style)
    ]

    t_qa = Table([[q1_cell, q2_cell, q3_cell]], colWidths=[245, 245, 245])
    t_qa.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#FEF2F2")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#F0F9FF")),
        ('BACKGROUND', (2,0), (2,0), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (0,0), 1, colors.HexColor("#FECACA")),
        ('BOX', (1,0), (1,0), 1, colors.HexColor("#BAE6FD")),
        ('BOX', (2,0), (2,0), 1, colors.HexColor("#BBF7D0")),
        ('PADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_qa)

    # Build the document
    doc.build(story, canvasmaker=LandscapeSlideCanvas)
    print(f"Successfully generated PDF slide deck at: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
