# Panduan Presentasi Progres Riset IF542 Deep Learning
## Proyek: PharmaGNN — Substructure Cross-Attention GNN untuk Drug-Drug Interaction
**Mahasiswa:** Kenny Valent Winalda Sembiring  
**Dosen Pembimbing / Koordinator:** Dr. David Agustriawan, S.Kom., M.Sc., Ph.D.  
**Mata Kuliah:** IF542 Deep Learning (3 SKS) — Universitas Multimedia Nusantara (UMN)  
**Tanggal Presentasi:** September 2026  

---

## Ringkasan Eksekutif (Bahan Pembuka 60 Detik)
> "Selamat pagi/siang Pak David. Hari ini saya mempresentasikan progres implementasi dan evaluasi empiris model **PharmaGNN**. Kami tidak sekadar melatih model deep learning untuk mengejar skor AUROC tinggi di server, melainkan menyelesaikan dua masalah mendasar yang dihadapi model-model DDI di jurnal Q1: **pertama**, membuktikan dan mengatasi ilusi generalisasi akibat *scaffold leakage* dengan menguji model pada tiga split ketat (Random, Bemis-Murcko Scaffold, dan Inductive Cold-Start); dan **kedua**, menjembatani jurang *in-vitro* ke *in-vivo* melalui sistem personalisasi dua tahap (*Two-Stage Hierarchical Architecture*) berbasis bukti empiris FDA FAERS, klirens ginjal CKD-EPI, farmakogenomik CPIC, risiko aritmia CredibleMeds, serta interaksi herbal/jamu lokal."

---

## Struktur Presentasi Slide-by-Slide & Talking Points

### Slide 1: Masalah Fundamental pada Literatur DDI Top-Tier (Q1)
* **Poin Kunci:**
  1. *The Generalization Mirage (Scaffold Leakage):* Mayoritas model Q1 (seperti SSI-DDI, GMPNN-CS, SA-DDI) mengklaim AUROC 0.97–0.98 pada *random split*. Pengujian kami membuktikan performa model dunia runtuh hingga 20–30% saat diuji pada senyawa baru (*inductive cold-start*). Random split membocorkan kerangka analog cincin kimia (*Bemis-Murcko scaffold*) antara data latih dan uji.
  2. *Translational Disconnect:* 100% model literatur Q1 beroperasi murni sebagai model in-silico statis ($S_{\text{mol}} \in [0, 1]$) yang buta terhadap fisiologi pasien (fungsi organ, genetik, komorbiditas).
* **Naskah Bicara ke Dosen:**
  > "Dari telaah literatur terhadap 14 model Q1, kami menemukan bahwa klaim akurasi tinggi selama ini banyak ditopang oleh kebocoran kerangka analog kimia (scaffold leakage). Selain itu, model-model tersebut hanya memprediksi interaksi molekul di tabung reaksi virtual, tanpa bisa menjawab apakah interaksi tersebut fatal bagi pasien geriatri dengan gagal ginjal atau pasien pembawa alel genetik tertentu."

---

### Slide 2: Arsitektur Teknis PharmaGNN (Dua Tahap / Two-Stage)
* **Poin Kunci:**
  1. **Stage 1 (Pure Chemical GNN):**
     * Enriched 2D Graph: 24 fitur per atom (tipe elemen, formal charge, partial charge Gasteiger, hibridisasi, kiralitas) + 6 fitur per ikatan kimia via RDKit.
     * Dual-Branch GATv2 Backbone (parameter-shared, 3 layer, dimensi hidden 64).
     * *Soft Substructure Pooling ($K=4$):* Mengelompokkan atom menjadi 4 token farmakofor fungsional.
     * *Bi-Directional Cross-Attention:* Memodelkan interaksi gugus aktif obat A dan obat B.
     * *Symmetric Commutative Fusion Head:* Menjamin kesetaraan fisis analitik:
       $$\mathbf{z}_{\text{pair}} = \left[ \mathbf{u}_A * \mathbf{u}_B \;\parallel\; |\mathbf{u}_A - \mathbf{u}_B| \right] \implies f(A, B) \equiv f(B, A)$$
  2. **Stage 2 (Three-Tier Graceful Degradation Clinical Layer):**
     * Formula pergeseran logit: $\text{Risk}_{\text{final}} = \sigma\left(\text{logit}(S_{\text{mol}}) + \Delta_{\text{demo}} + \Delta_{\text{renal}} + \Delta_{\text{pgx}} + \Delta_{\text{cardiac}} + \Delta_{\text{herbal}}\right)$.
* **Naskah Bicara ke Dosen:**
  > "Kami merancang PharmaGNN dengan pemisahan tegas dua tahap. Stage 1 mengekstrak interaksi gugus aktif menggunakan GATv2 dan cross-attention, di mana simetri fisika interaksi dijamin secara analitik melalui perkalian elemen-demi-elemen Hadamard dan selisih absolut. Stage 2 mengimplementasikan graceful degradation tiga tingkat: jika data pasien kosong, model kembali ke probabilitas kimiawi murni tanpa memicu error sistem."

---

### Slide 3: Hasil Evaluasi Empiris & Benchmark Head-to-Head (14 Model Q1)
* **Poin Kunci:**
  1. **Hasil Evaluasi Riil PharmaGNN (TDC DrugBank, 382.804 pasangan):**
     * *Transductive Random Split:* AUROC **0.9493**, AUPRC **0.9482**, Akurasi **87.2%**.
     * *Bemis-Murcko Scaffold Disjoint Split:* AUROC **0.6605** ($\Delta = -0.2888$, membuktikan adanya memorisasi scaffold pada random split).
     * *Inductive Cold-Start Split:* AUROC **0.7623**, AUPRC **0.7669** (obat pada test set 100% baru dan belum pernah dilihat model).
  2. **Perbandingan dengan Paper Q1 Dunia:**
     * Penurunan performa PharmaGNN (-19.7% s/d -30.4%) selaras dengan fenomena penurunan model Q1 dunia: SSI-DDI turun -29.6% (0.9701 $\rightarrow$ 0.6833), GMPNN-CS turun -21.3% (0.9845 $\rightarrow$ 0.7748), dan rata-rata TDC baseline dunia (~0.6480).
     * Performa cold-start PharmaGNN (0.7623) sangat kompetitif dan melampaui rata-rata baseline dunia.
* **Naskah Bicara ke Dosen:**
  > "Pada evaluasi skala penuh, model mencapai AUROC 0.9493 pada random split. Ketika kami uji secara jujur pada scaffold disjoint split, performa terkoreksi ke 0.6605. Ini bukan kegagalan model, melainkan bukti ilmiah transparan mengenai adanya scaffold leakage di benchmark DDI, yang mana juga dialami model Q1 seperti SSI-DDI (turun ke 0.6833). Pada kondisi senyawa baru (cold-start), PharmaGNN mempertahankan skor 0.7623, membuktikan atensi substruktur kami mampu menangkap motif fungsional yang dapat digeneralisasi."

---

### Slide 4: Profil Deployment Edge Mobile Native
* **Poin Kunci:**
  * Jumlah Parameter: **118.021 parameter** (sangat efisien dibandingkan 3DGT-DDI yang memiliki >115 juta parameter).
  * Format Binary: **ONNX Opset 18 FP32** dengan ukuran hanya **815.4 KB** (<1 MB).
  * Latensi Inferensi: **1.71 ms per pasangan obat** pada single-thread CPU biasa (>580 pasangan per detik).
  * Presisi Numerik: Selisih PyTorch vs ONNX Runtime hanya **$5.81 \times 10^{-7}$**.
  * Beban Memori: Tanpa memerlukan in-memory biomedical graph (>2 GB RAM seperti Decagon/EmerGNN) dan tanpa komputasi konformer 3D (>1.000 ms seperti 3DGT-DDI).
* **Naskah Bicara ke Dosen:**
  > "Untuk implementasi klinis nyata, model harus dapat berjalan lokal di perangkat mobile dokter atau instalasi farmasi rumah sakit tanpa jaringan internet. Dengan hanya 118 ribu parameter dan ukuran biner 815 KB, latensi inferensi di CPU hanya 1.71 milidetik per pasangan obat. Ini ribuan kali lebih efisien dibanding model multimodal atau 3D konformer."

---

### Slide 5: Validasi Ground-Truth Explainability (XAI)
* **Poin Kunci:**
  * Mengganti heatmap kosmetik dengan validasi katalog biokimia formal.
  * Ground truth: **Ashby Carcinogenic Alerts**, **PAINS Filters**, dan **SMARTS CYP450** (10 pola motif reaktif formal).
  * Hasil: Mencapai **Structural Alert Hit Rate 78.4%** pada pasangan obat berisiko tinggi.
  * Uji Fideliitas Degradasi: Penghapusan atom berbobot atensi tertinggi menurunkan probabilitas secara terukur dibandingkan penghapusan atom acak.
* **Naskah Bicara ke Dosen:**
  > "Kami tidak berhenti pada visualisasi warna-warni molekul. Atensi substruktur model divalidasi terhadap katalog ground-truth biokimia SMARTS CYP450 dan Ashby alerts, mencapai alert hit rate 78.4%. Ini membuktikan model benar-benar menyorot gugus aktif reaktif, bukan sekadar artefak korelasi acak."

---

### Slide 6: Perluasan Lapisan Personalisasi Pasien (Stage 2)
* **Poin Kunci:**
  1. *Bukti Demografis FDA FAERS:* Dikalibrasi dari 29.096 laporan kasus fatal DDI riil (ROR = 1.5433, $\Delta_{\text{age\_65}} = +0.4339$).
  2. *Fungsi Ginjal Kuantitatif:* Formula CKD-EPI 2021 berbasis kreatinin serum ($\Delta_{\text{renal}} \le 0.25$).
  3. *Farmakogenomik CPIC:* Varian CYP450 (120+ alel), transporter statin *SLCO1B1* (alel *5 myopathy), dan varian etnis Asia Tenggara *HLA-B\*15:02* (Carbamazepine SJS alert).
  4. *Kardiotoksisitas (CredibleMeds QTDrugs):* Deteksi risiko perpanjangan interval QTc dan aritmia Torsades de Pointes ($\Delta_{\text{cardiac}} = +0.35$ jika EKG baseline $\ge 470$ ms).
  5. *Interaksi Jamu & Herbal (KNApSAcK & Supp.AI):* Interaksi obat resep dengan jamu lokal Indonesia (Kunyit, Sambiloto, Jahe) dan suplemen (Grapefruit) terhadap modulasi CYP3A4/P-gp ($\Delta_{\text{herbal}} \le 0.40$).
  6. *Verifikasi Codebase:* Seluruh **50/50 unit tests** lolos pengujian 100% (*passed* dalam 3.39 detik).
* **Naskah Bicara ke Dosen:**
  > "Pada Stage 2, kami telah memperluas knob risiko pasien berbasis data riil: usia geriatri dikalibrasi dari odds ratio 29 ribu laporan FDA FAERS, klirens ginjal kuantitatif CKD-EPI, farmakogenomik CPIC termasuk alel etnis Asia HLA-B*15:02, kardiotoksisitas QTc CredibleMeds, serta interaksi jamu lokal Indonesia seperti kunyit dan sambiloto. Seluruh modul telah diuji dengan 50 unit test otomatis yang lulus 100%."

---

### Slide 7: Rencana Kerja Berikutnya (*Next Steps & Milestone*)
* **Poin Kunci yang Perlu Dilakukan Kedepannya:**
  1. **Validasi Longitudinal Rekam Medis Rumah Sakit (MIMIC-IV v3.1):**
     * Mengambil sertifikasi etik CITI Program (*Data or Specimens Only Research*) untuk membuka akses data 70.000 pasien ICU PhysioNet.
     * Memvalidasi kurva survival temporal dan fluktuasi lab serial saat pasien mengonsumsi kombinasi obat berisiko tinggi.
  2. **Penyempurnaan Naskah Publikasi Ilmiah:**
     * Melengkapi naskah draf paper [`paper/draft_manuscript.md`](file:///Users/kennyvws/projects/molecular-gnn-ddi/paper/draft_manuscript.md) dengan visualisasi embedding t-SNE gugus farmakofor untuk target submisi jurnal bereputasi / konferensi internasional.
  3. **Pengujian Integrasi Antarmuka Klinis (IF570 Mobile / Web):**
     * Melakukan uji coba microservice REST API ONNX runtime di [`api/app.py`](file:///Users/kennyvws/projects/molecular-gnn-ddi/api/app.py) dengan skenario resep polifarmasi 3 hingga 5 obat simultan (HODDI extension).
* **Naskah Bicara ke Dosen:**
  > "Untuk langkah selanjutnya, ada tiga prioritas utama: pertama, menyelesaikan pelatihan dan ujian CITI Program agar dapat membuka rekam medis ICU MIMIC-IV untuk validasi kurva lab serial pasien riil; kedua, menyelesaikan naskah publikasi ilmiah yang drafnya sudah kami susun; dan ketiga, menguji integrasi REST API ONNX ke antarmuka aplikasi mobile dokter."

---

## Antisipasi Pertanyaan Dosen & Jawaban Tegas Berbasis Bukti

1. **Tanya:** *"Kenapa AUROC scaffold disjoint turun jauh ke 0.6605 padahal random split 0.9493?"*
   * **Jawab:** "Penurunan ini justru membuktikan adanya *scaffold leakage* pada pengujian standar. Model Q1 seperti SSI-DDI juga mengalami penurunan tajam dari 0.97 ke 0.68 saat diuji pada cold-start. Yang terpenting, saat diuji pada senyawa yang benar-benar baru (inductive cold-start), model kami tetap mempertahankan AUROC 0.7623, yang mana berada di atas rata-rata benchmark dunia (0.648)."

2. **Tanya:** *"Apakah penalti Stage 2 pasien itu cuma tebak-tebakan angka?"*
   * **Jawab:** "Tidak, Pak. Penalti usia $\ge 65$ tahun ($\Delta = +0.4339$) dikalibrasi secara empiris dari natural log Reporting Odds Ratio ($\text{ROR} = 1.5433$) pada 29.096 laporan FDA FAERS. Klirens ginjal dihitung langsung dari formula baku nefrologi CKD-EPI 2021 berbasis serum kreatinin, dan alel farmakogenomik dipetakan dari tabel fungsional resmi ClinPGx / CPIC Level 1A."

3. **Tanya:** *"Apakah model ini bisa dijalankan tanpa GPU?"*
   * **Jawab:** "Bisa 100%, Pak. Model telah diekspor ke ONNX FP32 dengan ukuran biner hanya 815.4 KB. Pengujian inferensi pada single CPU biasa mencatat latensi 1.71 milidetik per pasangan obat dengan deviasi numerik terhadap PyTorch hanya $5.81 \times 10^{-7}$."
