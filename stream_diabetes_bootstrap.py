# stream_diabetes_bootstrap.py
"""
Streamlit app with Bootstrap styling for Diabetes Prediction
Author : Yusuf Ghozali (NIM 2221400110)
"""

import pickle
import streamlit as st

# ----------------------------
# Load the trained model (.sav)
# ----------------------------
MODEL_PATH = "diabetes_svm_nosmote.sav"

def load_model(path: str):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        st.error(f"❌ Model tidak ditemukan. Pastikan file '{path}' tersedia.")
        st.stop()

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

diabetes_model = get_model()

# ----------------------------
# Page Config & Custom CSS
# ----------------------------
PAGE_TITLE = "Prediksi Diabetes dengan ML"
PAGE_ICON  = "🩺"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body       { font-family: 'Poppins', sans-serif; background:#f8f9fa; }
        .card-custom     { border-radius:1rem; box-shadow:0 0.5rem 1rem rgba(0,0,0,.1);
                           background:#fff; padding:2rem; margin-top:1rem; }
        .stButton>button { width:100%; background:#0d6efd; color:#fff; font-weight:600;
                           border-radius:.5rem; padding:.6rem; transition:.3s; }
        .stButton>button:hover { background:#0b5ed7; }
        footer           { background:#343a40; color:#fff; padding:1rem 0; text-align:center;
                           margin-top:3rem; border-top-left-radius:.5rem; border-top-right-radius:.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Main Card (titel + form)
# ----------------------------
with st.container():
    st.markdown("<div class='card-custom'>", unsafe_allow_html=True)

    # --- Judul halaman di dalam kartu putih ---
    st.markdown(f"<h1 class='text-center mb-3'>{PAGE_ICON} {PAGE_TITLE}</h1>", unsafe_allow_html=True)

    # --- Sub‑judul form ---
    st.markdown("<h4 class='mb-4 text-center'>🔢 Masukkan Data Pasien</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pregnancies    = st.number_input("Kehamilan (Pregnancies)",        min_value=0,   value=0, step=1)
        blood_pressure = st.number_input("Tekanan Darah",                  min_value=0,   value=0)
        insulin        = st.number_input("Insulin",                        min_value=0,   value=0)
        dpf            = st.number_input("Diabetes Pedigree Function",     min_value=0.0, value=0.0, step=0.01, format="%.2f")

    with col2:
        glucose        = st.number_input("Glukosa",                        min_value=0,   value=0)
        skin_thickness = st.number_input("Ketebalan Kulit",                min_value=0,   value=0)
        bmi            = st.number_input("BMI (Indeks Massa Tubuh)",       min_value=0.0, value=0.0, step=0.1, format="%.1f")
        age            = st.number_input("Usia",                           min_value=0,   value=0)

    if st.button("🔍 Cek Prediksi"):
        if all(v == 0 for v in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]):
            st.warning("⚠️ Isi minimal satu kolom dengan nilai > 0 untuk prediksi.")
        else:
            X = [[float(pregnancies), float(glucose), float(blood_pressure),
                  float(skin_thickness), float(insulin), float(bmi), float(dpf), float(age)]]
            y_pred = diabetes_model.predict(X)[0]

            if y_pred == 1:
                st.markdown(
                    "<div class='alert alert-danger mt-4' role='alert'>"
                    "⚠️ <strong>Hasil:</strong> Pasien kemungkinan <strong>terkena diabetes</strong>."
                    "</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='alert alert-success mt-4' role='alert'>"
                    "✅ <strong>Hasil:</strong> Pasien <strong>tidak terkena diabetes</strong>."
                    "</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
    <footer>
      <div class="container">
        Dibuat oleh <strong>Yusuf Ghozali</strong> | NIM 2221400110
      </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
