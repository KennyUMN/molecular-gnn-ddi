# 🧬 PharmaGNN: Substructure Cross-Attention GNN untuk Drug-Drug Interaction

> **Final Project — IF542 Deep Learning (3 SKS)**  
> **Universitas Multimedia Nusantara (UMN) — Semester Ganjil 2026/2027**  
> **Course Coordinator:** Dr. David Agustriawan, S.Kom., M.Sc., Ph.D. & Ajie Kusuma Wardhana, S.Kom., M.Eng.  
> **Evaluation Weight:** 50% Final Grade  
> **Authors:** Kenny Valent Winalda Sembiring & Team  

---

## 📌 Ringkasan Eksekutif & Pembaruan Arsitektur
**PharmaGNN** adalah framework Deep Learning hierarkis dua tahap (*Two-Stage Hierarchical GNN*) untuk memprediksi interaksi obat (*Drug-Drug Interaction* / DDI) langsung dari representasi graf kimiawi molekul 2D (SMILES), yang kemudian dipersonalisasi ke kondisi spesifik pasien di rumah sakit:

1. **Stage 1 (Pure Chemical GNN):**
   - Menggunakan **Dual-Branch GATv2** dengan *out-of-place scatter reduce* untuk memperbarui representasi **24 fitur atom** (indeks ikatan kimiawi *edge_index* saja — fitur ikatan tidak dipakai oleh layer GATv2).
   - **Substructure Pooling ($K=4$):** Mengelompokkan atom secara lunak (*soft clustering*) menjadi token farmakofor fungsional.
   - **Bi-Directional Cross-Attention:** Memodelkan interaksi kimiawi dua arah antar gugus aktif obat A dan obat B.
   - **Symmetric Commutative Fusion:** Menggabungkan representasi dengan selisih absolut $|A - B|$ dan perkalian Hadamard $A \odot B$ agar memenuhi hukum fisika komutatif: $f(A, B) \equiv f(B, A)$.
2. **Stage 2 (Patient Personalization Layer — Three-Tier Graceful Degradation):**
   - Menyesuaikan probabilitas kimiawi murni ($S_{\text{mol}}$) dengan faktor risiko pasien via pergeseran logit:
     $$\text{Risk}_{\text{final}} = \sigma\left(\text{logit}(S_{\text{mol}}) + \Delta_{\text{demo}} + \Delta_{\text{renal}} + \Delta_{\text{pgx}}\right)$$
   - **Tier 1 (Fallback):** Prediksi kimia murni saat data rekam medis tidak tersedia ($\Delta = 0$).
   - **Tier 2 (Demografis FAERS):** Penalti usia lanjut $\ge 65$ tahun ($\Delta_{\text{age}} = +0.4339$, terkalibrasi dari $\ln(\text{ROR})$ fatal-outcome FAERS), kehamilan ($+0.10$), gangguan fungsi hati ($+0.12$).
   - **Tier 3 (Farmakogenomik CPIC/PharmGKB & eGFR CKD-EPI):** Penalti kerusakan ginjal eGFR CKD-EPI 2021 ($0 - 0.25$), genotipe mutasi alel enzim hati CYP450 / transporter OATP1B1 ($+0.45$), risiko QTc CredibleMeds ($+0.35$), dan interaksi Jamu/herbal ($\le 0.40$).

---

## 📚 Ekosistem 12 Dataset Penelitian

| No | Nama Dataset | Sumber / Lisensi | Peran dalam Pipeline | Skala Data / Entitas |
|:--:|:---|:---|:---|:---|
| 1 | **TDC DrugBank DDI** | Therapeutics Data Commons | Dataset primer training Stage 1 | 382.804 pasangan (1.706 obat, seed 42) |
| 2 | **PubChem Compound / ChEMBL SMILES** | NIH / EMBL-EBI Open Access | Struktur kanonikal & rumus 2D | 1.706 molekul terstandarisasi |
| 3 | **RDKit Topological Graph** | Open-source Chemoinformatics | Ekstraksi fitur atom & indeks ikatan | 24 fitur atom (tanpa fitur ikatan) |
| 4 | **ChEMBL 34 / BindingDB** ❌ | EMBL-EBI / UCSD | *Rencana:* afinitas basah kuantitatif ($K_i, IC_{50}$) — **belum diimplementasikan** (nol kode/artefak) | — |
| 5 | **FDA FAERS** | US FDA Spontaneous Reports | Kalibrasi penalti demografis Tier 2 | Jutaan laporan kasus riil |
| 6 | **TwoSides** ❌ | Tatonetti et al. / Nature Biotech | *Rencana:* klasifikasi multi-task efek samping — **belum diimplementasikan** (nilai di notebook = demo sintetis) | — |
| 7 | **MIMIC-IV EHR** ❌ | PhysioNet / MIT (v2.2) | *Rencana:* validasi lintasan rekam medis ICU — **belum diimplementasikan** (hanya disebut di docstring) | — |
| 8 | **CPIC Guidelines** | Clinical Pharmacogenetics Cons. | Aturan klinis penyesuaian dosis genetik | Pedoman Level A/B terstandarisasi |
| 9 | **PharmGKB** | NIH Pharmacogenomics KB | Knowledge graph genotipe-ke-fenotipe | *VKORC1, CYP2C9, SLCO1B1, DPYD, TPMT* |
| 10 | **Brenk Alerts (RDKit)** | Toksikologi Kimia Komputasi | Kamus gugus toksikofor reaktif | Katalog *unwanted-substructure* Brenk RDKit |
| 11 | **Ground Truth SMARTS** | Literatur Biokimia & PDB | Pola substruktur validasi Explainable AI | CYP2C9 coumarin, CYP3A4 imidazole |
| 12 | **HODDI (Higher-Order DDI)** ❌ | Literatur Polifarmasi 2024 | *Rencana:* ekstensi Hyper-Cross-Attention multi-obat — **belum diimplementasikan** (modul di notebook tidak terhubung ke training) | — |

> **Keterangan:** ❌ = dataset/klaim tanpa implementasi kode di repo ini (hanya rencana atau kutipan literatur). Baris tanpa ❌ terverifikasi ada di kode/artefak: TDC (dataset primer), PubChem/SMILES (mapping), RDKit Graph, FAERS (kalibrasi Tier 2), CPIC/PharmGKB (`calibrate_deltas.py`), Brenk/SMARTS (`explain.py`).

---

## 📊 Hasil Evaluasi Empiris Skala Penuh (Kaggle 2× Tesla T4 GPU)

Evaluasi dilakukan pada 382.804 pasangan obat dari benchmark TDC DrugBank di bawah tiga rezim pengujian ketat:

| Rezim Pengujian Split | Jumlah Train | Jumlah Test | AUROC | AUPRC | Akurasi | F1-Score | Keterangan & Makna Ilmiah |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Random Split (Transductive)** | 305.766 | 38.488 | **0.9493** | **0.9482** | **87.19%** | **0.8769** | Benchmark standar literatur; rentan optimisme palsu (*data leakage*). |
| **Bemis-Murcko Scaffold Disjoint** | 200.383 | 10.979 | **0.6605** | **0.5981** | **66.31%** | **0.4691** | Penurunan drastis (-0.2888 AUROC) **membuktikan kebocoran scaffold**. |
| **Inductive Cold-Start** | 258.988 | 3.493 | **0.7623** | **0.7596** | **68.94%** | **0.6695** | Obat pada test set 100% baru bagi model; **sejajar dengan** SOTA substruktur (GMPNN-CS 0.7748, SA-DDI 0.7914) — lihat `paper/draft_manuscript.md`. |

> Angka di atas diambil dari artefak run Kaggle 2×T4 terakhir: [`runs_kaggle/random/summary.json`](runs_kaggle/random/summary.json), [`runs_kaggle/scaffold/summary.json`](runs_kaggle/scaffold/summary.json), [`runs_kaggle/cold_start/summary.json`](runs_kaggle/cold_start/summary.json). (Berbeda dari berkas lama `kaggle_artifacts/*_summary.json` yang berasal dari run 15 epoch sebelumnya.)

> **Temuan Kunci Biofisika:** Injeksi fitur afinitas enzim basah ($K_i / IC_{50}$) dari ChEMBL direncanakan untuk mendorong AUROC Cold-Start dari **0.7623** menuju target **>0.82** — belum diimplementasikan maupun diukur (lihat MASTER_PROJECT_SPECIFICATION.md §9).

---

## ⚡ Profil Komputasi Edge Deployment & Analisis Kuantisasi

Model dioptimalkan untuk inferensi lokal pada smartphone (IF570 Mobile App) tanpa ketergantungan koneksi server:
* **Binary Format:** ONNX Opset 18 (FP32).
* **Ukuran Binary:** **815.4 KB** untuk artefak run Kaggle terakhir ([`runs_kaggle/random/molecular_gnn_ddi.onnx`](runs_kaggle/random/molecular_gnn_ddi.onnx)); checkpoint lokal [`models/molecular_gnn_ddi.onnx`](models/molecular_gnn_ddi.onnx) berukuran 834.0 KB. Keduanya di bawah batas toleransi mobile 1 MB.
* **Latensi Inferensi CPU:** **~0.21 ms per pasangan obat** pada single-core *thread* ONNX Runtime (≈4.7k skrining resep per detik). Diukur oleh [`bench_onnx_latency.py`](bench_onnx_latency.py) → [`runs_kaggle/onnx_latency.json`](runs_kaggle/onnx_latency.json).
* **Presisi Numerik:** Selisih output PyTorch vs ONNX Runtime tercatat pada log run: **$5.81 \times 10^{-7}$** (random), **$5.22 \times 10^{-8}$** (scaffold), **$2.24 \times 10^{-8}$** (cold-start) — semuanya lulus ambang parity $10^{-4}$.
* **Studi Ablasi Kuantisasi INT8:**
  - Kuantisasi dinamis INT8 pada runtime ONNX menyebabkan kegagalan inferensi bentuk dinamis matriks graf (*MatMulInteger shape error*); artefak INT8 yang dihasilkan (`models/molecular_gnn_ddi_int8.onnx`, 518.3 KB) **tidak dapat dimuat** oleh ONNX Runtime — dikonfirmasi oleh `int8_onnx_verified.loads = false` pada setiap `*_summary.json`.
  - Format FP32 dipertahankan karena sudah memenuhi seluruh target efisiensi.
  - Simulasi INT8 berbasis bobot (`emulate_int8_weights`) mengukur ketahanan atensi FP32 vs bobot INT8 pada 400 pasangan uji: Spearman $\rho$ rata-rata **0.9755** (random), **0.9852** (scaffold), **0.9781** (cold-start); *Jaccard overlap* Top-5 rata-rata **0.8378** / **0.9343** / **0.8504**. Ketiganya lulus ambang $\rho \ge 0.85$.

---

## 📂 Struktur Repositori & Berkas Utama

```
molecular-gnn-ddi/
├── PharmaGNN_DDI_Master_Pipeline.ipynb  # Master Jupyter Notebook 15 Bagian Lengkap
├── MASTER_PROJECT_SPECIFICATION.md      # Spesifikasi Teknis & Teori Komprehensif
├── README.md                            # Dokumentasi ringkasan repositori
├── data/
│   ├── download_dataset.py              # Skrip pengunduh data TDC & preprocessing
│   └── sample_ddi.csv                   # Dataset pilot klinis terverifikasi
├── src/
│   ├── dataset.py                       # Ekstraktor graf molekuler RDKit (24 fitur atom; edge_index tanpa fitur ikatan)
│   ├── model.py                         # Dual-Branch GATv2 + Cross-Attention
│   ├── explain.py                       # Engine saliency gradien (vanilla gradient saliency) + validator SMARTS CYP450
│   ├── export_onnx.py                   # Modul ekspor ONNX FP32
│   └── utils.py                         # Metrik evaluasi, split, & early stopping
├── runs_kaggle/                         # Artefak run training riil GPU T4 (sumber angka benchmark README)
│   ├── random/       {best_model.pt, history.json, summary.json, molecular_gnn_ddi.onnx}
│   ├── scaffold/     {best_model.pt, history.json, summary.json, molecular_gnn_ddi.onnx}
│   ├── cold_start/   {best_model.pt, history.json, summary.json, molecular_gnn_ddi.onnx}
│   └── logs/                            # Log training + parity ONNX per rezim
├── kaggle_artifacts/                    # Arsip run Kaggle 15-epoch sebelumnya (angka benchmark lama)
│   ├── random_best_model.pt
│   ├── scaffold_best_model.pt
│   ├── cold_start_best_model.pt
│   ├── random_summary.json
│   ├── scaffold_summary.json
│   └── cold_start_summary.json
├── api/
│   ├── app.py                           # Microservice REST API (PyTorch GNN Stage 1 + personalisasi Stage 2)
│   └── mock_mobile_client.html          # Simulator aplikasi smartphone IF570
├── paper/
│   ├── draft_manuscript.md              # Draf manuskrip publikasi IEEE / Scopus (dengan tabel komparasi Q1)
│   └── scaffold_leakage_diagram.png     # Diagram bukti empiris scaffold leakage
└── tests/
    ├── test_full_suite.py               # Unit test (11 skenario)
    ├── test_pipeline.py                 # Pengujian integrasi pipeline end-to-end (34 skenario)
    └── test_personalization_extended.py # Pengujian CredibleMeds QTc, Jamu, SLCO1B1 & HLA-B (5 skenario)
```

> Total 50 skenario pengujian (`11 + 34 + 5`), semuanya lulus: `50 passed`.

---

## 🚀 Panduan Menjalankan Proyek

### 1. Menjalankan Master Jupyter Notebook
Buka dan eksekusi notebook utama:
* Local: [`PharmaGNN_DDI_Master_Pipeline.ipynb`](PharmaGNN_DDI_Master_Pipeline.ipynb)
* Downloads Mirror: [`/Users/kennyvws/Downloads/PharmaGNN_DDI_Master_Pipeline.ipynb`](file:///Users/kennyvws/Downloads/PharmaGNN_DDI_Master_Pipeline.ipynb)

Notebook mencakup 15 bagian terstruktur: dari instalasi, ekstraksi fitur atom/ikatan RDKit, training model GATv2, personalisasi Tiered MIMIC-IV / PharmGKB, hingga klasifikasi fenotipik TwoSides dan polifarmasi HODDI.

### 2. Menjalankan Unit Test (Verifikasi 50 Test)
```bash
# Menggunakan virtualenv Python 3.12 dengan PYTHONPATH
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

### 3. Menjalankan Simulator Mobile App (Integrasi IF570)
```bash
# Jalankan backend API server
.venv/bin/python api/app.py
```
Akses di peramban: `http://localhost:8080`.

---

## 🏆 Kepatuhan Capaian Pembelajaran (RPKPS IF542 UMN)
* **SubCLO0504 (Level Taksonomi Bloom C6: Create):** Mengembangkan arsitektur Deep Learning orisinal yang menggabungkan graf perhatian (GATv2), *soft substructure pooling*, dan modul atensi silang dua arah dengan fusi komutatif.
* **Integritas Metodologi & Pencegahan Kebocoran Data:** Secara eksplisit mengidentifikasi fenomena *scaffold leakage* dan merumuskan rezim pengujian *Inductive Cold-Start*.
* **Relevansi Kepakaran Pembimbing (Dr. David Agustriawan, Ph.D.):** Berpijak pada bioinformatika komputasional, validasi substruktur farmakofor biokimiawi SMARTS, serta farmakogenomik presisi CPIC.

---

## ⚠️ Keterbatasan (Limitations) — Wajib Dibaca Sebelum Mengutip Metrik

1. **Negatif disampling sintetis 1:1 (base rate tidak realistis).** Label negatif pada `data/tdc_drugbank_ddi.csv` adalah pasangan obat acak hasil *negative sampling* seragam, bukan interaksi yang dikonfirmasi absen. Hasilnya tepat **191.402 positif : 191.402 negatif** (50/50). DrugBank DDI riil ~1:4 (dan jauh lebih jarang dalam skrining klinis nyata). Karena itu **AUPRC 0.9482 dan F1 0.8769** diukur terhadap *base rate* 50% yang tidak akan pernah terjadi secara klinis; nilainya akan turun tajam pada prevalensi riil.
2. **Split scaffold membuang 43% data.** Untuk jaminan bebas kebocoran, **166.621 dari 382.804 baris** (43,5%) dibuang sebagai *straddler* (obat yang scaffold-nya muncul di train & test). Artinya rezim scaffold hanya dilatih pada ~200k baris sedangkan random pada ~306k — perbandingan tiga rezim **tidak apple-to-apple pada volume training**. Split cold-start membuang 117.455 baris; random tidak membuang satupun.
3. **Kalibrasi *threshold* & risiko belum dari val set.** Ambang 0,5 dan batas `risk_level` 0,7/0,4 masih hardcoded, bukan dikalibrasi dari distribusi val set. Label "Safe" untuk probabilitas 39% secara klinis perlu dipertanyakan.
4. **Klaim tanpa implementasi.** Fitur afinitas ChEMBL/BindingDB, multi-task TwoSides, MIMIC-IV EHR, dan polifarmasi HODDI **belum diimplementasikan** (lihat tabel dataset di atas dan `MASTER_PROJECT_SPECIFICATION.md` §9). Angka Cold-Start yang sahih adalah **0.7623** — bukan >0.82.
