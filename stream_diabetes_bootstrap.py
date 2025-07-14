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
# Page Config & Custom CSS
# ----------------------------
PAGE_TITLE = "Prediksi Diabetes dengan ML"
PAGE_ICON = "🩺"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# Inject Bootstrap & custom styling
st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        html, body {
            font-family: 'Poppins', sans-serif;
            background-color: #f8f9fa;
        }
        .navbar-brand {
            font-size: 1.75rem;
            font-weight: 700;
        }
        .card-custom {
            border-radius: 1rem;
            box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.1);
            background-color: white;
            padding: 2rem;
            margin-top: 1rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #0d6efd;
            color: white;
            font-weight: 600;
            border-radius: 0.5rem;
            padding: 0.6rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0b5ed7;
        }
        footer {
            background: #343a40;
            color: #fff;
            padding: 1rem 0;
            text-align: center;
            margin-top: 3rem;
            border-top-left-radius: .5rem;
            border-top-right-radius: .5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Navbar (Header)
# ----------------------------
navbar = f"""
<nav class="navbar navbar-expand-lg navbar-dark bg-primary mb-4">
  <div class="container-fluid justify-content-center">
    <a class="navbar-brand" href="#">{PAGE_ICON} {PAGE_TITLE}</a>
  </div>
</nav>
"""
st.markdown(navbar, unsafe_allow_html=True)

# ----------------------------
# Input Form
# ----------------------------
with st.container():
    st.markdown("<div class='card-custom'>", unsafe_allow_html=True)
    st.markdown("<h3 class='mb-4'>🔢 Masukkan Data Pasien</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Kehamilan (Pregnancies)", min_value=0, value=0, step=1)
        blood_pressure = st.number_input("Tekanan Darah", min_value=0, value=0)
        insulin = st.number_input("Insulin", min_value=0, value=0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.0, step=0.01, format="%.2f")

    with col2:
        glucose = st.number_input("Glukosa", min_value=0, value=0)
        skin_thickness = st.number_input("Ketebalan Kulit", min_value=0, value=0)
        bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
        age = st.number_input("Usia", min_value=0, value=0)

    predict_btn = st.button("🔍 Cek Prediksi")

    if predict_btn:
        if all(val == 0 for val in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]):
            st.warning("⚠️ Isi minimal satu kolom dengan nilai > 0 untuk prediksi.")
        else:
            input_data = [
                float(pregnancies), float(glucose), float(blood_pressure),
                float(skin_thickness), float(insulin), float(bmi),
                float(dpf), float(age)
            ]
            prediction = diabetes_model.predict([input_data])

            if prediction[0] == 1:
                st.markdown(
                    """
                    <div class="alert alert-danger mt-4" role="alert">
                        ⚠️ <strong>Hasil:</strong> Pasien kemungkinan <strong>terkena diabetes</strong>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="alert alert-success mt-4" role="alert">
                        ✅ <strong>Hasil:</strong> Pasien <strong>tidak terkena diabetes</strong>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
footer = """
<footer>
  <div class="container">
      Dibuat oleh <strong>Yusuf Ghozali</strong> | NIM 2221400110
  </div>
</footer>
"""

st.markdown(footer, unsafe_allow_html=True)
