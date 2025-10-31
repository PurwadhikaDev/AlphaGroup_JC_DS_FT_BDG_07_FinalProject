import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings

# --- 1. IMPORT PENTING UNTUK MEMPERBAIKI 'NameError' ---
from sklearn.base import BaseEstimator, TransformerMixin

# --- 2. DEFINISI KELAS 'FeatureEngineer' ---
# class ini harus sama persis dengan yang Anda gunakan saat training
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_new = X.copy()
        
        # 1. TotalAdditionalServices
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
        valid_service_cols = [col for col in service_cols if col in X_new.columns]
        X_new['TotalAdditionalServices'] = X_new[valid_service_cols].apply(
            lambda row: (row == 'Yes').sum(), axis=1
        )
        
        # 2. TenureGroup
        labels = ['New (0-12 bln)', 'Mid-Term (13-48 bln)', 'Loyal (49+ bln)']
        bins = [-1, 12, 48, 72]
        X_new['TenureGroup'] = pd.cut(X_new['tenure'], bins=bins, labels=labels, right=True)
        X_new['TenureGroup'] = X_new['TenureGroup'].cat.add_categories(
            'Very Loyal (72+)'
        ).fillna('Very Loyal (72+)')

        # 3. Had_No_Key_Support
        X_new['Had_No_Key_Support'] = (
            (X_new['OnlineSecurity'] == 'No') & 
            (X_new['TechSupport'] == 'No')
        ).astype(int)
        
        return X_new
# --- Batas akhir definisi kelas ---


# ---------------------------------
# Konfigurasi Halaman (Page Config)
# ---------------------------------
st.set_page_config(
    page_title="Prediksi Churn Jaya Telcom",
    page_icon="📡",
    layout="wide"
)

# ---------------------------------
# Fungsi Bantuan (Helper Functions)
# ---------------------------------

@st.cache_resource
def load_pipeline():
    """Memuat pipeline model yang sudah di-training."""
    try:
        # Pastikan nama file ini SAMA dengan file yang Anda simpan
        pipeline = joblib.load("churn_pipeline.joblib") 
        return pipeline
    except FileNotFoundError:
        st.error("File 'churn_pipeline.joblib' tidak ditemukan. Pastikan file tersebut ada di folder yang sama dengan app.py.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat pipeline: {e}")
        st.info("Pastikan Anda menyertakan class FeatureEngineer di app.py jika Anda menggunakannya di pipeline.")
        return None

def convert_df_to_csv(df):
    """Konversi DataFrame ke CSV untuk di-download."""
    return df.to_csv(index=False).encode('utf-8')

# --- MODIFIKASI 1: Ubah fungsi ini untuk threshold 0.40 ---
def categorize_risk(probability, threshold=0.40):
    """Mengubah probabilitas menjadi kategori risiko berdasarkan threshold."""
    if probability >= threshold:
        return "Berisiko (Target Promosi)"
    else:
        return "Risiko Rendah (Aman)"
# --- Batas Akhir Modifikasi 1 ---

# ---------------------------------
# Main App Interface (Tampilan Utama)
# ---------------------------------

# Judul Utama
st.title("📡 Aplikasi Prediksi Customer Churn - Jaya Telcom")
st.write("""
Aplikasi ini adalah **Proof of Concept (POC)** untuk mendemonstrasikan model Machine Learning
dalam memprediksi pelanggan yang berpotensi *churn*.

**Cara Penggunaan:**
1.  Siapkan data pelanggan baru dalam format file `.csv`.
2.  Pastikan file Anda memiliki kolom yang sama dengan data training (seperti `tenure`, `Contract`, `InternetService`, dll.).
3.  Upload file di bawah ini untuk melihat hasilnya.
""")

pipeline = load_pipeline()

if pipeline:
    uploaded_file = st.file_uploader("Unggah File CSV Data Pelanggan Baru", type=["csv"])

    if uploaded_file is not None:
        try:
            warnings.filterwarnings('ignore')
            
            new_data = pd.read_csv(uploaded_file)
            original_data_display = new_data.copy()

            customer_ids = None
            if 'customerID' in new_data.columns:
                customer_ids = new_data['customerID']
            
            data_to_predict = new_data.drop(['customerID', 'Churn'], axis=1, errors='ignore')
            
            st.info(f"Mendeteksi {len(data_to_predict)} data pelanggan. Memulai proses prediksi...")
            
            # 1. Dapatkan Probabilitas
            churn_probabilities = pipeline.predict_proba(data_to_predict)[:, 1]
            
            st.success("Prediksi Selesai!")

            # 2. Buat DataFrame hasil
            results_df = pd.DataFrame()
            if customer_ids is not None:
                results_df['customerID'] = customer_ids
            else:
                results_df['customerID'] = original_data_display.index.rename('customerID')
            
            results_df['Churn_Probability'] = churn_probabilities
            results_df['Churn_Probability_%'] = (results_df['Churn_Probability'] * 100).round(2)
            
            # --- MODIFIKASI 2: Tambahkan 'PaymentMethod' dan 'tenure' agar bisa difilter ---
            key_columns = ['tenure', 'Contract', 'InternetService', 'MonthlyCharges', 'PaymentMethod']
            for col in key_columns:
                if col in original_data_display.columns:
                    results_df[col] = original_data_display[col].values
            
            # --- MODIFIKASI 3: Terapkan threshold 0.40 ---
            results_df['Risk_Category'] = results_df['Churn_Probability'].apply(categorize_risk)

            # --- MODIFIKASI 4: Terapkan FILTER Involuntary Churn ---
            manual_payment_methods = ['Electronic check', 'Mailed check']
            
            # Tentukan kondisi involuntary churn
            involuntary_mask = (
                (results_df['tenure'] <= 2) &
                (results_df['PaymentMethod'].isin(manual_payment_methods))
            )
            
            # Timpa Risk_Category untuk customer ini, APAPUN SKORNYA
            results_df.loc[involuntary_mask, 'Risk_Category'] = "Involuntary (Tdk Promosi)"
            # --- Batas Akhir Modifikasi 4 ---
            
            results_df_sorted = results_df.sort_values(by='Churn_Probability', ascending=False)
            
            # --- MODIFIKASI 5: Perbarui Ringkasan Eksekutif ---
            st.subheader("📊 Ringkasan Eksekutif (Executive Summary)")

            total_customers = len(results_df_sorted)
            
            # Hitung yang BERISIKO (yang sudah difilter TDK termasuk)
            high_risk_count = results_df_sorted[results_df_sorted['Risk_Category'] == 'Berisiko (Target Promosi)'].shape[0]
            
            # Hitung yang di-filter (Involuntary)
            filtered_count = results_df_sorted[results_df_sorted['Risk_Category'] == 'Involuntary (Tdk Promosi)'].shape[0]

            high_risk_percent = (high_risk_count / total_customers) * 100 if total_customers > 0 else 0

            # Tampilkan metrik dalam kolom
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Pelanggan Dianalisis", f"{total_customers}")
            col2.metric("Pelanggan Berisiko (Target Promosi)", f"{high_risk_count}", f"{high_risk_percent:.1f}% dari total")
            col3.metric("Involuntary Churn (Difilter)", f"{filtered_count}", "Tdk diberi promosi")
            # --- Batas Akhir Modifikasi 5 ---
            
            
            # --- MODIFIKASI 6: Perbarui Insight Cepat ---
            if high_risk_count > 0:
                st.markdown("---")
                st.subheader("💡 Insight Cepat: Profil Pelanggan 'Berisiko'")
                
                # Gunakan kategori baru
                high_risk_df = results_df_sorted[results_df_sorted['Risk_Category'] == 'Berisiko (Target Promosi)']
                
                col1_insight, col2_insight = st.columns(2)
                
                try:
                    # Rata-rata tenure
                    avg_tenure_high_risk = high_risk_df['tenure'].mean()
                    col1_insight.metric("Rata-rata Tenure (High Risk)", f"{avg_tenure_high_risk:.1f} bulan", "Cenderung pelanggan baru")

                    # Tipe kontrak paling umum
                    common_contract_high_risk = high_risk_df['Contract'].mode()[0]
                    col2_insight.metric("Tipe Kontrak Paling Umum (High Risk)", f"{common_contract_high_risk}")
                except Exception as e:
                    st.warning(f"Tidak dapat menghitung insight tambahan: {e}")
            # --- Batas Akhir Modifikasi 6 ---

            
            st.markdown("---")
            
            # 4. Tampilkan Hasil Detail
            st.subheader("📋 Hasil Prediksi dan Skor Risiko (Detail)")
            
            display_columns = ['customerID', 'Risk_Category', 'Churn_Probability_%', 'Contract', 'tenure', 'InternetService', 'MonthlyCharges', 'PaymentMethod']
            display_columns = [col for col in display_columns if col in results_df_sorted.columns]

            st.dataframe(results_df_sorted[display_columns].reset_index(drop=True))

            # --- MODIFIKASI 7: Perbarui Teks Rekomendasi ---
            st.write("""
            **Rekomendasi:**
            - **Berisiko (Target Promosi):** Pelanggan ini harus segera dihubungi oleh tim retensi dengan penawaran yang relevan (misal: ganti kontrak, bonus layanan).
            - **Risiko Rendah (Aman):** Tidak memerlukan tindakan segera.
            - **Involuntary (Tdk Promosi):** Pelanggan ini diprediksi *churn* karena gagal bayar. **JANGAN** beri promosi.
            """)
            # --- Batas Akhir Modifikasi 7 ---
            
            
            # 5. Tombol Download
            st.subheader("💾 Download Hasil Lengkap")
            
            if customer_ids is None:
                 original_data_display = original_data_display.reset_index(names='customerID')

            download_data = original_data_display.merge(
                results_df[['customerID', 'Churn_Probability_%', 'Risk_Category']], 
                on='customerID', 
                how='left'
            )
            
            csv_data = convert_df_to_csv(download_data)
            
            st.download_button(
                label="Download sebagai CSV",
                data=csv_data,
                file_name=f"hasil_prediksi_churn.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            st.warning("Pastikan kolom di file CSV Anda sudah sesuai dengan data training.")