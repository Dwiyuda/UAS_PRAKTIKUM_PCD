import streamlit as st
import cv2
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image

# --- KONFIGURASI HALAMAN & CSS ---
st.set_page_config(page_title="Object Detection - Kelompok 4", layout="wide")

# Custom CSS untuk Warna User
# #b21f1f (Merah Marun), #5555b7 (Ungu Biru), #03102f (Navy Gelap)
st.markdown("""
<style>
    /* =========================================
       BAGIAN 1: SIDEBAR (MENU KIRI)
       ========================================= */
    [data-testid="stSidebar"] {
        background-color: #03102f;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
        font-size: 16px !important;
    }

    /* =========================================
       BAGIAN 2: HALAMAN UTAMA (KANAN)
       ========================================= */
    .stApp {
        background-color: #FFFFFF; /* Background Putih Bersih */
        color: #000000;            /* Tulisan Hitam Pekat */
        font-size: 20px;
    }
    
    /* Judul & Subjudul */
    h1 { color: #03102f !important; font-size: 40px !important; }
    h2, h3 { color: #b21f1f !important; }

    /* =========================================
       BAGIAN 3: KOMPONEN UI LAIN (WIDGETS)
       ========================================= */
    /* Slider Color */
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
        background-color: #b21f1f;
    }

    /* --- TAMBAHAN BARU: MEMAKSA LABEL WIDGET JADI HITAM --- */
    
    /* 1. Judul di atas Slider & Radio (Misal: "Atur Tingkat Blur") */
    label[data-testid="stWidgetLabel"] p {
        color: #000000 !important;
        font-size: 20 !important;
    }

    /* 2. Teks Pilihan Radio Button (Misal: "Otsu", "Manual") */
    div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
        font-size: 20 !important;
    }
    
    /* 3. Angka indikator Slider (Angka kecil di kanan slider) */
    div[data-testid="stSliderTickBar"] {
        color: #000000 !important;
    }

    /* =========================================
       BAGIAN 4: FILE UPLOADER (PERBAIKAN WARNA)
       ========================================= */
    [data-testid="stFileUploader"] section {
        background-color: #000000;  /* Ubah ke Abu Terang agar kontras dengan tulisan hitam */
        border: 2px dashed #b21f1f; 
        color: #000000 !important; 
    }
    [data-testid="stFileUploader"] section small {
        color: #555555 !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #03102f !important;
        color: white !important;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI NAVIGASI SIDEBAR ---
with st.sidebar:
    # Tampilkan Logo di Sidebar (Jika ada di folder assets)
    try:
        # Ganti nama file sesuai icon anda
        logo = Image.open("assets/lego.png") 
        st.image(logo, use_container_width=True)
    except:
        st.write("📂 (Icon tidak ditemukan di folder assets)")

    st.markdown("<h3 style='text-align: center; color: #5555b7;'>Navigasi Implementasi Modul 📚</h3>", unsafe_allow_html=True)
    
    # Menu Navigasi Keren
    selected = option_menu(
        menu_title=None,
        options=["Input Gambar", "Grayscale", "Noise Reduction", "Segmentasi", "Morfologi", "Analisis Akhir"],
        icons=["cloud-upload", "camera", "magic", "layers", "gem", "check-circle"], 
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#03102f"},
            "icon": {"color": "#5555b7", "font-size": "30px"}, 
            "nav-link": {"font-size": "12px", "text-align": "left", "margin":"5px", "--hover-color": "#5555b7"},
            "nav-link-selected": {"background-color": "#b21f1f"}, # Warna Merah Marun saat dipilih
        }
    )

# --- FUNGSI UTAMA: SESSION STATE ---
# Kita simpan gambar di memori agar bisa dipakai antar halaman
if 'image_ori' not in st.session_state:
    st.session_state['image_ori'] = None

# =================================================================================
# HALAMAN 1: INPUT GAMBAR (HOME)
# =================================================================================
if selected == "Input Gambar":
    st.title("📷 Tahap 1: Input Citra Asli")
    st.write("Silakan upload citra yang akan diproses.")
    
    uploaded_file = st.file_uploader("Pilih Gambar (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Simpan ke Session State (Memori)
        st.session_state['image_ori'] = image
        
        st.success("Gambar berhasil diupload! Silakan lanjut ke menu 'Grayscale'.")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Citra Asli", width=500)
    else:
        st.info("Belum ada gambar yang diupload.")

# =================================================================================
# LOGIKA PROTEKSI: CEK APAKAH SUDAH UPLOAD?
# =================================================================================
elif st.session_state['image_ori'] is None:
    st.error("⚠️ Halaman Ini Masih Terkunci, Selesaikan Tahap 1 Terlebih Dahulu!")
    st.stop() # Hentikan kode di sini agar tidak error

# =================================================================================
# JIKA SUDAH UPLOAD, LANJUTKAN PROSES SESUAI HALAMAN
# =================================================================================
else:
    # Ambil gambar dari memori
    img = st.session_state['image_ori']
    
    # --- PROSES SEKUENSIAL (PIPELINE) ---
    # Kita jalankan proses background agar siap ditampilkan di tahap manapun
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HALAMAN 2: GRAYSCALE
    if selected == "Grayscale":
        st.title("👀 Tahap 2: GrayScale")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input: Citra Asli", use_container_width=True)
        with col2:
            st.image(gray, caption="Output: Grayscale", use_container_width=True)
        
        st.info("Citra diubah menjadi derajat keabuan (1 channel) untuk memudahkan pemrosesan.")

    # HALAMAN 3: NOISE REDUCTION
    elif selected == "Noise Reduction":
        st.title("✨ Tahap 3: Filtering")
        st.markdown("---")
        
        # Slider Kontrol ada di Halaman Utama
        k_size = st.slider("Atur Tingkat Blur (Kernel Ganjil)", 1, 15, 5, step=2)
        
        # Proses Median Blur
        img_blur = cv2.medianBlur(gray, k_size)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray, caption="Input: Grayscale", use_container_width=True)
        with col2:
            st.image(img_blur, caption=f"Output: Median Blur (k={k_size})", use_container_width=True)
            
        st.info("Median Filter digunakan untuk menghilangkan noise bintik tanpa merusak tepi objek.")

    # HALAMAN 4: SEGMENTASI (THRESHOLD)
    elif selected == "Segmentasi":
        st.title("🙂‍↕️ Tahap 4: Thresholding ")
        st.markdown("---")
        
        # Perlu Blur dulu (Default kernel 5 jika user tidak setting, atau kita fix kan)
        img_blur = cv2.medianBlur(gray, 5) 
        
        method = st.radio("Metode Threshold", ["Otsu (Otomatis)", "Manual"])
        
        if method == "Manual":
            thresh_val = st.slider("Nilai Ambang", 0, 255, 127)
            ret, img_thresh = cv2.threshold(img_blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
        else:
            ret, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            st.success(f"Nilai Otsu ditemukan: {ret}")

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_blur, caption="Input: Image Blur", use_container_width=True)
        with col2:
            st.image(img_thresh, caption="Output: Citra Biner", use_container_width=True)

    # HALAMAN 5: MORFOLOGI
    elif selected == "Morfologi":
        st.title("🔧 Tahap 5: Morfologi")
        st.markdown("---")
        
        # Pipeline sebelumnya (Fixed parameter agar konsisten)
        img_blur = cv2.medianBlur(gray, 5)
        ret, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Slider Morfologi
        iterasi = st.slider("Jumlah Iterasi (Pemisahan Objek)", 1, 10, 2)
        kernel = np.ones((3,3), np.uint8)
        
        # Opening: Erosi lalu Dilasi (Memisahkan objek nempel)
        img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=iterasi)
        # Closing: Menutup lubang
        img_morph = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_thresh, caption="Input: Threshold Kasar", use_container_width=True)
        with col2:
            st.image(img_morph, caption="Output: Morfologi (Rapih)", use_container_width=True)
            
    # HALAMAN 6: ANALISIS
    elif selected == "Analisis Akhir":
        st.title("✅ Tahap 6: Counting")
        st.markdown("---")
        
        # --- RE-RUN PIPELINE LENGKAP ---
        # Kita jalankan semua proses di belakang layar dengan parameter default/terbaik
        # Agar hasil akhir langsung muncul
        
        # 1. Blur
        img_blur = cv2.medianBlur(gray, 5)
        # 2. Threshold
        ret, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 3. Morfologi
        kernel = np.ones((3,3), np.uint8)
        img_morph = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 4. Filter Area (Slider untuk User Tuning hasil akhir)
        min_area = st.slider("Filter Luas Area Minimum (Pixel)", 10, 5000, 100)

        # 5. Find Contours
        contours, _ = cv2.findContours(img_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_img = img.copy()
        count = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(final_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                count += 1
                cv2.putText(final_img, str(count), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_morph, caption="Input: Citra Bersih", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), caption=f"Output Akhir: {count} Objek", use_container_width=True)
            
        st.success(f"Analisis Selesai. Total Objek Terdeteksi: {count}")