# AlphaGroup_JC_DS_FT_BDG_07_FinalProject

# Proyek Analisis & Prediksi Customer Churn Jaya Telecom

Selamat datang di repositori proyek analisis *customer churn* Jaya Telecom. Proyek ini bukan sekadar membangun model *machine learning* biasa, tetapi sebuah studi mendalam untuk menemukan *akar masalah* dari *churn* dan memberikan rekomendasi bisnis yang spesifik dan dapat ditindaklanjuti.

**Dashboard Interaktif Final dari proyek ini dapat diakses di Tableau Public:**
[**https://public.tableau.com/app/profile/fadhilah.arfu/viz/TelcoChurnDashboard_17619271761100/SuperstoreDashboard?publish=yes**](https://public.tableau.com/app/profile/fadhilah.arfu/viz/TelcoChurnDashboard_17619271761100/SuperstoreDashboard?publish=yes)

Tujuan Utama: Beralih dari strategi retensi yang "reaktif" (menangani *customer* setelah mereka pergi) menjadi "proaktif" (mengidentifikasi *customer* berisiko sebelum mereka pergi).

---

## 1. Permasalahan Bisnis

Di industri telekomunikasi, biaya untuk mendapatkan *customer* baru jauh lebih mahal (diperkirakan 5x - 25x lipat) daripada mempertahankan *customer* yang sudah ada. Jaya Telecom menghadapi dua masalah utama:
1.  **Kehilangan Pendapatan:** Tingkat *churn* (perpindahan *customer*) yang tinggi mengikis profitabilitas secara langsung.
2.  **Anggaran Promosi Tidak Efisien:** Program retensi (seperti diskon) seringkali diberikan secara "merata", yang berarti anggaran terbuang untuk dua kelompok yang salah: *customer* yang sudah pasti loyal, atau *customer* yang sudah pasti pergi.

Model ini bertujuan untuk mengoptimalkan anggaran tersebut dengan memfokuskan intervensi **hanya** pada *customer* yang berisiko tinggi dan **bisa diselamatkan**.

## 2. Pendekatan & Solusi: "Membersihkan Data adalah Kunci"

Tantangan terbesar dalam proyek ini bukanlah pada model, melainkan pada **data**.

Analisis data awal (EDA) kami menemukan sebuah **bias krusial**: label `Churn = 'Yes'` ternyata mencampurkan dua tipe *customer* yang sangat berbeda:

1.  ***Voluntary Churn* (Churn Sukarela):** *Customer* yang secara sadar memutuskan berhenti karena tidak puas (layanan, harga, dll.). **Inilah *customer* yang ingin kita selamatkan.**
2.  ***Involuntary Churn* (Churn Paksa):** *Customer* yang diberhentikan oleh perusahaan, umumnya karena **gagal bayar**.

> **Insight Kunci:** Memberikan promosi retensi kepada *customer* yang gagal bayar adalah pemborosan anggaran 100%.

Oleh karena itu, kami melakukan investigasi mendalam dan menemukan bahwa *customer* yang *churn* dalam 1-2 bulan pertama mayoritas (83%) menggunakan metode pembayaran manual (`Electronic check` atau `Mailed check`). Kami mengklasifikasikan segmen ini sebagai *Involuntary Churn* dan **memfilternya dari data latih**.

Dengan data yang bersih ini, kami melatih model (XGBoost) yang 100% fokus mempelajari pola *customer* yang **tidak puas dan bisa diselamatkan**.

## 3. Temuan Utama: Faktor Pendorong Churn (The "Why")

Model yang telah dilatih menunjukkan tiga faktor utama yang paling kuat memprediksi seorang *customer* akan *churn* secara sukarela:

1.  **Tipe Kontrak (`Contract_Month-to-month`):** Ini adalah faktor risiko #1. *Customer* tanpa kontrak jangka panjang tidak memiliki "penghalang untuk keluar" (*barrier to exit*) dan bisa pergi kapan saja.
2.  **Layanan Internet (`InternetService_Fiber optic`):** *Customer* fiber optic membayar harga premium. Ini mengindikasikan sensitivitas harga yang tinggi atau ekspektasi layanan yang tidak terpenuhi.
3.  **Layanan Tambahan (`OnlineSecurity_No`):** *Customer* yang tidak memiliki layanan keamanan tambahan memiliki "keterikatan" (*stickiness*) yang lebih rendah terhadap ekosistem Jaya Telcom.

## 4. Rekomendasi Bisnis (The "So What?")

Model ini memungkinkan kita membuat strategi intervensi yang "cerdas", bukan sekadar memberi diskon acak.

> **Strategi Utama:** Jangan hanya memberi diskon. Gunakan promosi untuk **memperbaiki akar masalah** yang ditemukan model.

* **Untuk Masalah Kontrak Bulanan:**
    * **Rekomendasi:** Tawarkan **insentif untuk beralih kontrak**.
    * **Contoh Aksi:** "Ganti ke kontrak 1 Tahun dan dapatkan 2 bulan gratis." (Memindahkan mereka dari kelompok risiko tinggi ke risiko rendah).

* **Untuk Masalah Fiber Optic & Kurang Layanan Tambahan:**
    * **Rekomendasi:** Tingkatkan "keterikatan" (*stickiness*) layanan.
    * **Contoh Aksi:** "Sebagai apresiasi, dapatkan layanan Online Security gratis selama 6 bulan." (Meningkatkan nilai layanan di mata *customer*).

## 5. Estimasi Dampak Bisnis (ROI Riil)

Kami melakukan *tuning* model untuk memprioritaskan **Recall** (kemampuan "menangkap" *customer* berisiko) pada *threshold* 40%. Ini adalah keputusan bisnis untuk menerima lebih banyak *False Positive* (biaya promosi) demi menekan *False Negative* (biaya kehilangan *customer*).

Analisis ini didasarkan pada **penghematan riil** (fokus pada *Voluntary Churn*) berdasarkan 1.320 data uji:

| Skenario | Total Biaya / Kerugian |
| :--- | :--- |
| **Status Quo (Tanpa Model)** <br> *(Kehilangan 285 *customer*)* | Rp 285.000.000 |
| **Intervensi Tertarget (Dengan Model)** | **Rp 173.000.000** |
| **POTENSI PENGHEMATAN RIIL** | **Rp 112.000.000 (39,3%)** |

Dengan fokus hanya pada *customer* yang bisa diselamatkan, model ini dapat **mengurangi total kerugian akibat *churn* sekitar 39,3%**.

---

## 6. Struktur Repositori & Cara Menjalankan

* **`/` (Root):**
    * `machine_learning.ipynb`: *Notebook* utama yang berisi seluruh proses analisis, investigasi bias, *preprocessing*, *training* model, *tuning*, dan evaluasi.
    * `Cleaning_data.ipynb`: *Notebook* untuk proses pembersihan dan analisis data awal.
    * `model_churn_jaya_telcom.joblib`: File model (pipeline) final yang telah dilatih dan siap digunakan.
    * `app.py`: Aplikasi Streamlit untuk demonstrasi prediksi *live*.
    * `Telco-Customer-Churn.csv`: Data mentah yang digunakan (sebelum difilter).
    * `sample_telco_churn_100.csv`: Sampel data untuk pengujian aplikasi.
    * `README.md`: Dokumen ini.
    * `.gitignore`: File untuk mengabaikan data dan *cache*.

* **Cara Menjalankan Aplikasi Demo (Streamlit):**
    1.  Pastikan Anda memiliki semua *library* yang diperlukan (disarankan menggunakan *virtual environment*).
    2.  Jalankan perintah berikut di terminal Anda dari dalam folder proyek:
        ```bash
        streamlit run app.py
        ```
    3.  Buka *browser* Anda dan unggah file `sample_telco_churn_100.csv` untuk melihat hasilnya.

* **Link Dashboard Tableau:**
    * [https://public.tableau.com/app/profile/fadhilah.arfu/viz/TelcoChurnDashboard_17619271761100/SuperstoreDashboard?publish=yes](https://public.tableau.com/app/profile/fadhilah.arfu/viz/TelcoChurnDashboard_17619271761100/SuperstoreDashboard?publish=yes)
