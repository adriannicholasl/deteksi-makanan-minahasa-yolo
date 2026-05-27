import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Konfigurasi Halaman (Harus di baris paling atas)
st.set_page_config(
    page_title="Deteksi Makanan Minahasa",
    page_icon="🍲",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Injeksi CSS Khusus untuk Tampilan Mobile/Smartphone yang Responsif
st.markdown("""
    <style>
    /* Mengatur lebar maksimum gambar agar tidak melebihi layar HP */
    img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
    }
    /* Memperbesar ukuran tombol agar mudah ditekan di layar sentuh */
    .stButton>button {
        width: 100%;
        padding: 0.75rem;
        font-weight: bold;
        border-radius: 8px;
    }
    /* Mengatur jarak dan perataan teks */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: gray;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Judul Aplikasi
st.title("🍲 Deteksi Makanan Khas Minahasa")
st.markdown("**Sistem Deteksi Objek Cerdas (YOLOv12)**")
st.write("Unggah foto makanan atau gunakan kamera *smartphone* Anda untuk mendeteksi: Lalampa, Panada, Ayam Rica, Ikan Woku, Cakalang Fufu, Nasi Jaha, dan Tinutuan.")

st.markdown("---")

# Load Model YOLO (Gunakan cache agar tidak dimuat ulang setiap kali layar disentuh)
@st.cache_resource
def load_model():
    # Pastikan file best.pt berada di folder yang sama dengan app.py
    return YOLO("best.pt")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error("⚠️ File model 'best.pt' belum ditemukan. Pastikan sudah diunduh dari Colab dan diletakkan di folder yang sama.")
    model_loaded = False

# Tombol Upload (Otomatis mendukung akses kamera jika dibuka dari HP)
uploaded_file = st.file_uploader("Ambil foto atau unggah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    # Membaca gambar yang diunggah
    image = Image.open(uploaded_file)
    
    # Menampilkan gambar asli yang diunggah pengguna
    st.subheader("📷 Gambar Inputsadasdsa")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    
    # Tombol untuk memicu proses deteksi
    if st.button("🔍 Deteksi Makanan"):
        with st.spinner("Sedang memproses gambar..."):
            # Proses inferensi menggunakan model YOLO
            results = model.predict(image, conf=0.5) # conf=0.5 artinya hanya menampilkan deteksi dengan keyakinan di atas 50%
            
            # Mengambil hasil gambar yang sudah ada kotak (bounding box)
            res_image = results[0].plot()
            
            # Mengubah array numpy (hasil YOLO) kembali menjadi gambar (PIL)
            # Karena plot() mengembalikan format BGR (OpenCV), kita ubah ke RGB
            detected_image = Image.fromarray(res_image[..., ::-1])
            
            st.markdown("---")
            st.subheader("✨ Hasil Deteksi")
            st.image(detected_image, caption="Hasil Analisis YOLO", use_container_width=True)
            
            # Menampilkan detail kelas apa saja yang terdeteksi dalam bentuk list
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"Berhasil mendeteksi {len(boxes)} objek!")
                st.markdown("**Detail Objek Terdeteksi:**")
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0]) * 100
                    st.write(f"- **{class_name}** (Akurasi: {confidence:.2f}%)")
            else:
                st.warning("Tidak ada makanan yang dikenali pada gambar ini.")
                

# Footer Khusus Skripsi
st.markdown("---")
st.markdown('<div class="footer">Dikembangkan oleh Adrian Nicholas Lumowa<br>Teknik Informatika - Universitas Prisma</div>', unsafe_allow_html=True)