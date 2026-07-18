import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Drop-Out Siswa", page_icon="🎓", layout="wide")

# ==========================================
# SIDEBAR (MENU SAMPING) - UNTUK UPLOAD CSV
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Sistem AI")
st.sidebar.write("*(Fitur ini disematkan untuk memenuhi syarat wajib praktikum)*")

uploaded_file = st.sidebar.file_uploader("1. Upload Dataset (.csv)", type=["csv"])

# Variabel untuk mengecek apakah model sudah siap
model_ready = False 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    target_column = st.sidebar.selectbox("2. Pilih Target Label:", df.columns)
    
    if st.sidebar.button("Latih Model Sekarang"):
        with st.spinner('Sistem sedang belajar...'):
            df = df.dropna()
            X = df.drop(columns=[target_column])
            y = df[target_column]
            X_encoded = pd.get_dummies(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
            
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # Simpan ke memori Streamlit
            st.session_state['model'] = model
            st.session_state['X_columns'] = X_encoded.columns
            st.session_state['X_mean'] = X_encoded.mean()
            st.session_state['classes'] = model.classes_
            
            st.sidebar.success("✅ Model Siap Digunakan!")
            model_ready = True
elif 'model' in st.session_state:
    model_ready = True

# ==========================================
# HALAMAN UTAMA - KALKULATOR PROBABILITAS
# ==========================================
st.title("🎓 Kalkulator Probabilitas Drop-Out Siswa")
st.markdown("---")

if not model_ready:
    # Tampilan jika belum ada data yang diupload
    st.info("👈 **Sistem Belum Aktif.** Silakan buka menu di sebelah kiri, unggah file CSV dataset, dan klik 'Latih Model' untuk mengaktifkan kalkulator.")
else:
    # Tampilan jika model sudah siap (Halaman utama bersih dari upload file)
    st.write("Sesuaikan nilai metrik akademik di bawah ini untuk melihat persentase potensi siswa putus studi secara langsung.")
    
    model = st.session_state['model']
    X_columns = st.session_state['X_columns']
    
    # Mengambil 4 fitur terpenting untuk dijadikan slider
    importances = pd.Series(model.feature_importances_, index=X_columns)
    top_features = importances.nlargest(4).index.tolist()
    
    # Layout untuk Slider
    user_inputs = {}
    st.markdown("### 📊 Parameter Analisis Siswa")
    cols = st.columns(2)
    
    for i, feature in enumerate(top_features):
        with cols[i % 2]:
            min_val = float(st.session_state['X_mean'][feature] * 0) # Estimasi batas bawah
            max_val = float(st.session_state['X_mean'][feature] * 2) # Estimasi batas atas
            mean_val = float(st.session_state['X_mean'][feature])
            step_val = 1.0 if mean_val > 10 else 0.1
            
            user_inputs[feature] = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val, step=step_val)
            
    st.markdown("---")
    
    # MENGHITUNG PROBABILITAS OTOMATIS
    input_data = pd.DataFrame([st.session_state['X_mean']], columns=X_columns)
    for feature in top_features:
        input_data.at[0, feature] = user_inputs[feature]
        
    probabilities = model.predict_proba(input_data)[0]
    
    st.markdown("### 🎯 Hasil Prediksi Sistem:")
    
    # Menampilkan bar probabilitas untuk masing-masing kemungkinan
    for i, class_name in enumerate(st.session_state['classes']):
        prob_percent = probabilities[i] * 100
        
        # Mengatur warna tulisan berdasarkan probabilitas
        if prob_percent > 50:
            st.error(f"**Probabilitas {class_name}: {prob_percent:.1f}%**")
        elif prob_percent > 20:
            st.warning(f"**Probabilitas {class_name}: {prob_percent:.1f}%**")
        else:
            st.success(f"**Probabilitas {class_name}: {prob_percent:.1f}%**")
            
        st.progress(float(probabilities[i]))