# Kandidat & Katalog Dataset Level Pasien (Stage 2 Personalization)

*Dokumentasi diperbarui: 2026-09-26.*  
Tujuan: Mengintegrasikan bukti klinis riil ke dalam formula personalisasi Stage 2 PharmaGNN:
$$\text{Risk}_{\text{final}} = \sigma\left(\text{logit}(S_{\text{mol}}) + \Delta_{\text{demo}} + \Delta_{\text{renal}} + \Delta_{\text{pgx}} + \Delta_{\text{cardiac}} + \Delta_{\text{herbal}}\right)$$

---

## 1. Dataset yang SUDAH Aktif Diintegrasikan ke Codebase

| Nama Dataset | Berkas / Sumber | Dimensi Klinis | Status Implementasi |
| :--- | :--- | :--- | :--- |
| **FDA FAERS** | [`data/delta_calibration.json`](file:///Users/kennyvws/projects/molecular-gnn-ddi/data/delta_calibration.json) | Demografi Geriatri & Komorbiditas | **Aktif**: Dikalibrasi dari 29.096 laporan kasus fatal DDI. $\Delta_{\text{age\_65}} = +0.4339$. Penalti statis: Hamil ($+0.10$), Hepar ($+0.12$). |
| **Formula 2021 CKD-EPI** | [`src/personalization.py`](file:///Users/kennyvws/projects/molecular-gnn-ddi/src/personalization.py) | Klirens Fungsi Ginjal (eGFR) | **Aktif**: Formula kuantitatif nefrologi berbasis kreatinin serum: $\Delta_{\text{renal}} \le 0.25$. |
| **ClinPGx / CPIC (CYP450)** | [`data/delta_calibration.json`](file:///Users/kennyvws/projects/molecular-gnn-ddi/data/delta_calibration.json) | Enzim Pemetabolisme Hati | **Aktif**: 120+ alel fungsional CYP2C9, CYP2C19, CYP2D6, CYP3A4. Penalti *Poor Metabolizer* ($+0.45$), *Intermediate* ($+0.225$). |
| **PharmGKB Extended (Transporter & HLA)** | [`data/delta_calibration.json`](file:///Users/kennyvws/projects/molecular-gnn-ddi/data/delta_calibration.json) | Toksisitas Statin & Sindrom Kulit SJS | **Aktif**: Gen transporter *SLCO1B1* (alel *5 myopathy) dan varian etnis Asia Tenggara *HLA-B\*15:02* (Carbamazepine SJS alert). |
| **CredibleMeds QTDrugs** | [`data/crediblemeds_qtc.json`](file:///Users/kennyvws/projects/molecular-gnn-ddi/data/crediblemeds_qtc.json) | Kardiotoksisitas & Aritmia TdP | **Aktif**: Pemetaan obat berisiko QTc (*Known Risk*, *Possible Risk*). EKG baseline $\ge 470$ ms $\rightarrow \Delta_{\text{cardiac}} = +0.35$. |
| **KNApSAcK Jamu & Supp.AI** | [`data/herbal_interactions.json`](file:///Users/kennyvws/projects/molecular-gnn-ddi/data/herbal_interactions.json) | Herbal-Drug Interactions (HDI) | **Aktif**: Interaksi Jamu Indonesia (Kunyit, Sambiloto, Jahe) & Suplemen (Grapefruit, St. John's Wort) terhadap CYP3A4/P-gp ($\Delta_{\text{herbal}} \le 0.40$). |

---

## 2. Dataset "Ultimate" Gold-Standard yang Wajib Ujian/Sertifikasi Etik

Bagi peneliti bioinformatika dan kedokteran komputasional, dataset klinis tingkat rumah sakit paling bergengsi (*gold-standard / ultimate*) tidak bisa diunduh secara bebas karena memuat data rekam medis pasien riil (*Protected Health Information / PHI*). Peneliti **diwajibkan lulus ujian sertifikasi etik penelitian manusia** terlebih dahulu.

### 🌟 1. MIMIC-IV v3.1 (Medical Information Mart for Intensive Care)
* **Penyedia:** MIT Laboratory for Computational Physiology & Beth Israel Deaconess Medical Center (Boston, USA).
* **Portal:** [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)
* **Skala Data:** >70.000 pasien ICU rawat inap riil dengan data longitudinal lengkap:
  * Resep obat aktual (`prescriptions` & `emar`): nama obat, dosis, rute, waktu injeksi per menit.
  * Hasil lab serial (`labevents`): kreatinin, eGFR serial, albumin, bilirubin, kalium, trombosit, INR.
  * Diagnosa penyakit ICD-9 / ICD-10 (`diagnoses_icd`): memetakan seluruh komorbiditas pasien.
  * Catatan klinis dokter & perawat (`mimic-iv-note`): narasi evaluasi efek samping obat.
* **Mengapa Disebut "Ultimate":** Ini adalah satu-satunya dataset dunia di mana Anda dapat mengamati **sebab-akibat temporal riil**: Pasien A meminum Obat 1 jam 08:00, Obat 2 jam 12:00, lalu pada jam 18:00 kadar kreatinin melonjak dan EKG menunjukkan aritmia.
* **Ujian / Tes yang Wajib Dilalui:**
  1. **CITI Program Course:** Pelatihan etik *Data or Specimens Only Research* di situs [citiprogram.org](https://about.citiprogram.org/).
  2. Terdiri dari 10–12 modul interaktif: sejarah etika kedokteran (Belmont Report), privasi pasien (HIPAA de-identification), dan tata kelola data sekunder.
  3. Di akhir setiap modul terdapat **Ujian Pilihan Ganda (Quiz)** dengan syarat kelulusan minimal skor **80%**.
  4. Sertifikat kelulusan (Completion Report PDF) diunggah ke profil PhysioNet.
  5. Menandatangani *Data Use Agreement (DUA)* bersama pembimbing akademik (Dr. David Agustriawan, Ph.D.).
  6. Proses verifikasi berkas memakan waktu 3–7 hari kerja (Gratis untuk mahasiswa/akademisi).

---

### 🌟 2. eICU Collaborative Research Database (eICU-CRD v2.0)
* **Penyedia:** Philips Healthcare & MIT PhysioNet.
* **Portal:** [https://physionet.org/content/eicu-crd/](https://physionet.org/content/eicu-crd/)
* **Skala Data:** >200.000 pasien ICU dari **ratusan rumah sakit berbeda di seluruh Amerika Serikat**.
* **Keunggulan:** Jika MIMIC-IV hanya berasal dari 1 rumah sakit di Boston, eICU membuktikan generalisasi model lintas rumah sakit, wilayah geografis, dan variasi protokol dokter.
* **Syarat Akses:** Identik dengan MIMIC-IV (Sertifikat CITI Program yang sama berlaku untuk eICU).

---

### 🌟 3. UK Biobank (UKB) — Tingkat Riset Genomik Tertinggi
* **Penyedia:** UK Biobank Consortium.
* **Portal:** [https://www.ukbiobank.ac.uk/](https://www.ukbiobank.ac.uk/)
* **Skala Data:** 500.000 partisipan dewasa di Inggris dengan data multi-omika raksasa:
  * Whole Genome Sequencing (WGS) & Whole Exome Sequencing (WES) 500k individu.
  * Rekam medis elektronik nasional (NHS primary care & prescription data).
  * Data gaya hidup: kebiasaan merokok, konsumsi alkohol, nutrisi harian, jam tidur, aktivitas fisik.
* **Syarat Akses:** Mengajukan proposal riset formal (*Research Application*) yang direview oleh komite ilmiah internasional UK Biobank, verifikasi institusi universitas (MOU), dan pembayaran biaya komputasi cloud (UKB Research Analysis Platform / DNAnexus).

---

## 3. Panduan Langkah Demi Langkah Mengikuti Ujian CITI Program (Untuk Membuka MIMIC-IV)

Jika Anda ingin membuka akses ke MIMIC-IV untuk tugas akhir atau publikasi jurnal Q1:

1. **Buat Akun PhysioNet:**
   * Kunjungi [https://physionet.org/](https://physionet.org/) dan registrasi menggunakan email institusi kampus (`@student.umn.ac.id` atau email akademik).
2. **Daftar Ujian di CITI Program:**
   * Kunjungi [https://www.citiprogram.org/](https://www.citiprogram.org/).
   * Buat akun dan pilih afiliasi organisasi (*Select Your Organization Affiliation*): cari universitas Anda, atau pilih **"Massachusetts Institute of Technology (Affiliate)"** / PhysioNet.
   * Pilih kursus: **"Data or Specimens Only Research"** (Human Research Curriculum).
3. **Mengerjakan Modul & Kuis:**
   * Kerjakan materi modul (bacaan teks ringkas tentang informed consent, HIPAA identifiers, dan keamanan data).
   * Jawab kuis di akhir setiap modul. Jika salah, kuis bisa diulang hingga mencapai skor $\ge 80\%$.
4. **Unggah Sertifikat ke PhysioNet:**
   * Unduh *Completion Report* (PDF) yang memuat nomor sertifikat dan tanggal kedaluwarsa.
   * Masuk ke halaman akun PhysioNet $\rightarrow$ *Training* $\rightarrow$ masukkan nomor ID sertifikat CITI Program Anda.
5. **Permohonan Akses MIMIC-IV:**
   * Buka halaman proyek [MIMIC-IV on PhysioNet](https://physionet.org/content/mimiciv/).
   * Klik tombol *Request Access*, tandatangani pakta integritas etika data (DUA), dan cantumkan nama dosen pembimbing riset (Dr. David Agustriawan, Ph.D.).
   * Setelah diverifikasi oleh tim MIT, seluruh data pasien ICU siap di-query via Google BigQuery atau diunduh langsung dalam format CSV/Parquet.
