"""
Ứng dụng Web Demo – Chương 9: Các phương pháp Học máy Cổ điển trong Xử lý Ảnh
(Classical Machine Learning Methods in Image Processing)

Yêu cầu cài đặt:
    pip install streamlit numpy scikit-learn scikit-image "opencv-python-headless<5.0" pillow matplotlib

Chạy ứng dụng:
    streamlit run app.py
"""

import os
import io
import warnings

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from skimage import img_as_float, data as sk_data

# ── scikit-learn imports ──────────────────────────────────────────────────────
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import fetch_lfw_people, load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Cấu hình chung ───────────────────────────────────────────────────────────
matplotlib.use("Agg")  # backend không có GUI
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Demo – ML trong Xử lý Ảnh",
    page_icon="🖼️",
    layout="wide",
)

st.title("🖼️ Chương 9 – Các phương pháp Học máy Cổ điển trong Xử lý Ảnh")
st.caption("Classical Machine Learning Methods in Image Processing")

# ── Thư mục ảnh mẫu & Haar cascade ──────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER & CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_data(show_spinner="Đang tải tập dữ liệu LFW Faces …")
def load_lfw_dataset(min_faces: int = 60, resize: float = 0.4):
    """Tải tập Labeled Faces in the Wild (LFW)."""
    faces_data = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize)
    return faces_data.images, faces_data.target, faces_data.target_names


@st.cache_data(show_spinner="Đang tải tập dữ liệu Digits …")
def load_digits_dataset():
    """Tải tập chữ số viết tay 8×8 (sklearn digits)."""
    digits_data = load_digits()
    return digits_data.images, digits_data.data, digits_data.target, digits_data.target_names


def get_face_cascade():
    """Khởi tạo bộ phân loại Haar Cascade an toàn với xử lý ngoại lệ."""
    if not hasattr(cv2, "CascadeClassifier"):
        return None, (
            "Thư viện OpenCV hiện tại không chứa module `cv2.CascadeClassifier`.\n\n"
            "Cách khắc phục: Vui lòng chạy lệnh sau trong terminal:\n"
            "```bash\npip install --force-reinstall \"opencv-python-headless<5.0\"\n```"
        )

    # Danh sách các vị trí kiểm tra file xml
    search_paths = [
        HAAR_CASCADE_PATH,
        os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml"),
    ]
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        search_paths.append(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

    for p in search_paths:
        if p and os.path.isfile(p):
            try:
                cascade = cv2.CascadeClassifier(p)
                if not cascade.empty():
                    return cascade, None
            except Exception as e:
                continue

    return None, (
        f"Không tìm thấy file `haarcascade_frontalface_default.xml` hợp lệ.\n\n"
        f"Đã tìm kiếm tại: `{HAAR_CASCADE_PATH}`.\n"
        "Vui lòng đảm bảo file XML đã được tải về cùng thư mục với `app.py`."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎨 1. Gom cụm điểm ảnh",
        "🧮 2. PCA & Eigenfaces",
        "✍️ 3. Phân loại ảnh",
        "👤 4. Phát hiện khuôn mặt",
    ]
)

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 – Gom cụm điểm ảnh (K-Means & Spectral Clustering)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

with tab1:
    st.header("Gom cụm điểm ảnh (Unsupervised Learning)")
    st.markdown(
        "Sử dụng **K-Means** để lượng hóa màu (color quantization) "
        "và **Spectral Clustering** để phân đoạn tiền cảnh / hậu cảnh."
    )

    t1_source = st.radio(
        "Chọn nguồn ảnh:",
        ["📁 Tải ảnh từ máy tính", "🖼️ Sử dụng ảnh mẫu có sẵn"],
        horizontal=True,
        key="tab1_source_radio",
    )

    pil_img = None

    if t1_source == "📁 Tải ảnh từ máy tính":
        uploaded_file = st.file_uploader(
            "Tải ảnh lên",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="tab1_upload",
        )
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file).convert("RGB")
            except Exception as exc:
                st.error(f"❌ Không thể đọc ảnh tải lên: {exc}")
    else:
        sample_choice = st.selectbox(
            "Chọn ảnh mẫu:",
            ["Tách cà phê (Nhiều dải màu)", "Nhà du hành vũ trụ", "Thợ chụp ảnh"],
            key="tab1_sample_select",
        )
        try:
            if "cà phê" in sample_choice.lower():
                pil_img = Image.fromarray(sk_data.coffee())
            elif "du hành" in sample_choice.lower():
                pil_img = Image.fromarray(sk_data.astronaut())
            else:
                pil_img = Image.fromarray(sk_data.camera()).convert("RGB")
        except Exception as exc:
            st.error(f"❌ Không thể nạp ảnh mẫu: {exc}")

    if pil_img is not None:
        img_np = np.array(pil_img)

        # Giới hạn kích thước để xử lý mượt mà trên trình duyệt
        MAX_DIM = 512
        h0, w0 = img_np.shape[:2]
        if max(h0, w0) > MAX_DIM:
            scale = MAX_DIM / max(h0, w0)
            new_size = (int(w0 * scale), int(h0 * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            img_np = np.array(pil_img)
            st.info(f"ℹ️ Ảnh đã được tối ưu kích thước về {img_np.shape[1]}×{img_np.shape[0]} px để tăng tốc tính toán.")

        st.image(pil_img, caption="Ảnh đầu vào", use_container_width=False, width=380)

        # ── K-Means Color Quantization ────────────────────────────────────────
        st.subheader("🔹 1. Lượng hóa màu bằng K-Means")
        k_colors = st.slider(
            "Chọn số cụm màu k (2 đến 64)",
            min_value=2,
            max_value=64,
            value=8,
            help="Số màu đại diện sau khi nén. Giá trị càng nhỏ thì ảnh càng ít màu (hiệu ứng poster hóa).",
        )

        if st.button("▶ Chạy K-Means Quantization", key="btn_kmeans"):
            with st.spinner("Đang thực hiện phân cụm K-Means …"):
                try:
                    img_float = img_as_float(img_np)
                    h, w, d = img_float.shape
                    pixels = img_float.reshape(-1, d)

                    kmeans = KMeans(n_clusters=k_colors, n_init=4, random_state=42)
                    kmeans.fit(pixels)

                    quantized = kmeans.cluster_centers_[kmeans.labels_]
                    quantized_img = np.clip(quantized.reshape(h, w, d), 0, 1)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(pil_img, caption="Ảnh gốc", use_container_width=True)
                    with col_b:
                        st.image(
                            quantized_img,
                            caption=f"Ảnh sau khi nén còn {k_colors} màu",
                            use_container_width=True,
                        )

                    # Bảng màu đại diện
                    fig_palette, ax_palette = plt.subplots(1, 1, figsize=(8, 1))
                    ax_palette.imshow([kmeans.cluster_centers_], aspect="auto")
                    ax_palette.set_xticks(range(k_colors))
                    ax_palette.set_yticks([])
                    ax_palette.set_title(f"Bảng {k_colors} màu trọng tâm (Cluster Centers)")
                    st.pyplot(fig_palette)
                    plt.close(fig_palette)
                except Exception as exc:
                    st.error(f"❌ Lỗi khi chạy K-Means: {exc}")

        # ── Spectral Clustering ───────────────────────────────────────────────
        st.subheader("🔹 2. Phân đoạn ảnh bằng Spectral Clustering")
        st.markdown(
            "Spectral Clustering sử dụng đồ thị tương đồng (similarity graph) để phân đoạn đối tượng "
            "tiền cảnh và hậu cảnh. Để đảm bảo tốc độ đáp ứng thời gian thực, ảnh được điều chỉnh về kích thước phù hợp."
        )

        spectral_k = st.slider(
            "Số vùng phân đoạn (k)",
            min_value=2,
            max_value=6,
            value=2,
            key="spectral_k",
        )

        if st.button("▶ Chạy Spectral Clustering", key="btn_spectral"):
            with st.spinner("Đang tính toán ma trận tương đồng và véc-tơ riêng (10-20s) …"):
                try:
                    SPEC_DIM = 75
                    scale_s = SPEC_DIM / max(img_np.shape[:2])
                    small_size = (
                        max(int(img_np.shape[1] * scale_s), 1),
                        max(int(img_np.shape[0] * scale_s), 1),
                    )
                    small_img = np.array(
                        Image.fromarray(img_np).resize(small_size, Image.LANCZOS)
                    )
                    img_float_s = img_as_float(small_img)
                    hs, ws, ds = img_float_s.shape
                    pixels_s = img_float_s.reshape(-1, ds)

                    sc = SpectralClustering(
                        n_clusters=spectral_k,
                        affinity="nearest_neighbors",
                        n_neighbors=8,
                        assign_labels="kmeans",
                        random_state=42,
                    )
                    labels_s = sc.fit_predict(pixels_s)
                    seg_map = labels_s.reshape(hs, ws)

                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        fig_in, ax_in = plt.subplots(figsize=(4, 4))
                        ax_in.imshow(small_img)
                        ax_in.set_title(f"Ảnh đầu vào ({ws}×{hs})")
                        ax_in.axis("off")
                        st.pyplot(fig_in)
                        plt.close(fig_in)
                    with col_s2:
                        fig_sc, ax_sc = plt.subplots(figsize=(4, 4))
                        cax = ax_sc.imshow(seg_map, cmap="viridis")
                        ax_sc.set_title(f"Spectral Clustering ({spectral_k} cụm)")
                        ax_sc.axis("off")
                        fig_sc.colorbar(cax, ax=ax_sc, fraction=0.046, pad=0.04)
                        st.pyplot(fig_sc)
                        plt.close(fig_sc)
                except Exception as exc:
                    st.error(f"❌ Lỗi trong quá trình Spectral Clustering: {exc}")
    else:
        st.info("👆 Vui lòng tải lên ảnh hoặc chọn ảnh mẫu phía trên để bắt đầu thử nghiệm.")

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 – PCA & Eigenfaces                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

with tab2:
    st.header("Giảm chiều dữ liệu & Eigenfaces (PCA – Unsupervised Learning)")
    st.markdown(
        "Thuật toán **PCA (Principal Component Analysis)** tìm các trục tọa độ trực giao sao cho "
        "phương sai của dữ liệu lớn nhất. Trong xử lý ảnh khuôn mặt, các thành phần chính này được gọi là **Eigenfaces**."
    )

    with st.spinner("Đang chuẩn bị tập dữ liệu LFW Faces …"):
        try:
            lfw_images, lfw_target, lfw_names = load_lfw_dataset()
            n_samples, h_face, w_face = lfw_images.shape
            st.success(
                f"✅ Đã tải thành công **{n_samples}** ảnh khuôn mặt ({h_face}×{w_face} px) "
                f"thuộc **{len(lfw_names)}** nhân vật nổi tiếng."
            )
        except Exception as exc:
            st.error(f"❌ Không thể nạp tập dữ liệu LFW: {exc}")
            st.stop()

    # Hiển thị một số khuôn mặt mẫu
    st.subheader("📷 Một số khuôn mặt mẫu từ tập dữ liệu")
    sample_cols = st.columns(8)
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n_samples, size=8, replace=False)
    for i, idx in enumerate(sample_indices):
        with sample_cols[i]:
            fig_s, ax_s = plt.subplots(figsize=(1.5, 2))
            ax_s.imshow(lfw_images[idx], cmap="gray")
            ax_s.set_title(lfw_names[lfw_target[idx]].split()[-1], fontsize=8)
            ax_s.axis("off")
            st.pyplot(fig_s)
            plt.close(fig_s)

    st.subheader("🔹 Khảo sát số lượng thành phần chính (Eigenfaces)")
    n_components = st.slider(
        "Số lượng thành phần chính k (PCs)",
        min_value=5,
        max_value=120,
        value=50,
        step=5,
        help="Số lượng Eigenfaces dùng để đại diện và tái tạo lại khuôn mặt.",
    )

    if st.button("▶ Phân tích PCA & Trích xuất Eigenfaces", key="btn_pca"):
        with st.spinner("Đang áp dụng PCA trên ma trận khuôn mặt …"):
            try:
                X_faces = lfw_images.reshape(n_samples, -1).astype(np.float64)

                pca = PCA(n_components=n_components, whiten=True, random_state=42)
                X_pca = pca.fit_transform(X_faces)

                # ── Trực quan hóa Eigenfaces ──────────────────────────────────
                st.subheader(f"👤 Trực quan hóa {min(n_components, 16)} Eigenfaces đầu tiên")
                n_show = min(n_components, 16)
                n_cols_ef = 8
                n_rows_ef = int(np.ceil(n_show / n_cols_ef))
                fig_ef, axes_ef = plt.subplots(
                    n_rows_ef, n_cols_ef, figsize=(n_cols_ef * 1.6, n_rows_ef * 2.2)
                )
                axes_ef = np.atleast_2d(axes_ef)
                for i in range(n_rows_ef * n_cols_ef):
                    ax = axes_ef.flat[i]
                    if i < n_show:
                        eigenface = pca.components_[i].reshape(h_face, w_face)
                        ax.imshow(eigenface, cmap="RdBu_r")
                        ax.set_title(f"PC {i + 1}", fontsize=8)
                    ax.axis("off")
                fig_ef.suptitle("Các đặc trưng trực giao (Eigenfaces)", fontsize=11)
                fig_ef.tight_layout()
                st.pyplot(fig_ef)
                plt.close(fig_ef)

                # ── Phương sai giải thích ──────────────────────────────────────
                st.subheader("📊 Tỉ lệ phương sai giải thích tích lũy (Cumulative Explained Variance)")
                cumulative_var = np.cumsum(pca.explained_variance_ratio_) * 100
                fig_var, ax_var = plt.subplots(figsize=(8, 3))
                ax_var.plot(range(1, n_components + 1), cumulative_var, "o-", markersize=3, color="#1f77b4")
                ax_var.set_xlabel("Số thành phần chính")
                ax_var.set_ylabel("Phương sai tích lũy (%)")
                ax_var.grid(True, alpha=0.3)
                ax_var.axhline(y=90, color="orange", linestyle="--", label="Ngưỡng 90%")
                ax_var.axhline(y=95, color="red", linestyle="--", label="Ngưỡng 95%")
                ax_var.legend()
                fig_var.tight_layout()
                st.pyplot(fig_var)
                plt.close(fig_var)

                st.info(
                    f"💡 Với **{n_components}** chiều thay vì không gian gốc {h_face*w_face} chiều, "
                    f"mô hình đã lưu giữ được **{cumulative_var[-1]:.2f}%** lượng thông tin của tập dữ liệu."
                )

                # ── Tái tạo khuôn mặt từ k thành phần ─────────────────────────
                st.subheader("🔄 Minh họa khả năng nén & tái tạo khuôn mặt")
                test_idx = rng.choice(n_samples)
                orig_face = X_faces[test_idx]
                recon_face = pca.inverse_transform(X_pca[test_idx])

                col_o, col_r = st.columns(2)
                with col_o:
                    fig_orig, ax_orig = plt.subplots(figsize=(3, 3.8))
                    ax_orig.imshow(orig_face.reshape(h_face, w_face), cmap="gray")
                    ax_orig.set_title("Ảnh gốc (Đầy đủ chiều)")
                    ax_orig.axis("off")
                    st.pyplot(fig_orig)
                    plt.close(fig_orig)
                with col_r:
                    fig_rec, ax_rec = plt.subplots(figsize=(3, 3.8))
                    ax_rec.imshow(recon_face.reshape(h_face, w_face), cmap="gray")
                    ax_rec.set_title(f"Tái tạo từ {n_components} Eigenfaces")
                    ax_rec.axis("off")
                    st.pyplot(fig_rec)
                    plt.close(fig_rec)
            except Exception as exc:
                st.error(f"❌ Lỗi khi phân tích PCA: {exc}")

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 – Phân loại ảnh (kNN, Gaussian Bayes, SVM)                           ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

with tab3:
    st.header("Phân loại chữ số viết tay (Supervised Learning - Classification)")
    st.markdown(
        "So sánh 3 mô hình học máy kinh điển: **k-Nearest Neighbors (kNN)**, "
        "**Gaussian Naïve Bayes** và **Support Vector Machine (SVM)** trên tập chữ số viết tay."
    )

    with st.spinner("Đang nạp tập dữ liệu chữ số MNIST / Digits …"):
        try:
            dig_images, dig_data, dig_target, dig_names = load_digits_dataset()
            n_dig = len(dig_target)
            st.success(f"✅ Đã nạp thành công **{n_dig}** ảnh chữ số (kích thước 8×8 pixels, 10 lớp từ 0 đến 9).")
        except Exception as exc:
            st.error(f"❌ Không thể tải tập dữ liệu Digits: {exc}")
            st.stop()

    st.subheader("📷 Một số mẫu chữ số viết tay tiêu biểu")
    sample_cols3 = st.columns(10)
    for digit_val in range(10):
        with sample_cols3[digit_val]:
            fig_d, ax_d = plt.subplots(figsize=(1.2, 1.2))
            sample_pos = np.where(dig_target == digit_val)[0][0]
            ax_d.imshow(dig_images[sample_pos], cmap="gray_r", interpolation="nearest")
            ax_d.set_title(str(digit_val), fontsize=10)
            ax_d.axis("off")
            st.pyplot(fig_d)
            plt.close(fig_d)

    # ── Điều chỉnh tham số ───────────────────────────────────────────────────
    st.subheader("⚙️ Thiết lập tham số huấn luyện")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        test_ratio = st.slider("Tỉ lệ tập Test (%)", 10, 50, 25, step=5)
    with col_p2:
        knn_neighbors = st.slider("Số láng giềng k (kNN)", 1, 15, 5, step=1)
    with col_p3:
        svm_kernel_choice = st.selectbox("Kernel cho SVM", ["rbf", "linear", "poly"], index=0)

    if st.button("▶ Huấn luyện & Đánh giá 3 Mô hình", key="btn_train_classify"):
        with st.spinner("Đang chia dữ liệu và huấn luyện các mô hình …"):
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    dig_data,
                    dig_target,
                    test_size=test_ratio / 100.0,
                    random_state=42,
                    stratify=dig_target,
                )

                st.write(f"Số lượng mẫu huấn luyện: **{len(X_tr)}** | Số lượng mẫu kiểm tra: **{len(X_te)}**")

                models = {
                    f"k-Nearest Neighbors (k={knn_neighbors})": KNeighborsClassifier(n_neighbors=knn_neighbors),
                    "Gaussian Naïve Bayes": GaussianNB(),
                    f"SVM ({svm_kernel_choice} kernel)": make_pipeline(
                        StandardScaler(), SVC(kernel=svm_kernel_choice, gamma="scale")
                    ),
                }

                eval_results = {}
                progress_bar = st.progress(0, text="Bắt đầu huấn luyện …")

                for idx_m, (name_m, model_obj) in enumerate(models.items()):
                    progress_bar.progress((idx_m + 1) / len(models), text=f"Đang huấn luyện {name_m} …")
                    model_obj.fit(X_tr, y_tr)
                    y_pred = model_obj.predict(X_te)
                    acc = accuracy_score(y_te, y_pred)
                    eval_results[name_m] = {
                        "accuracy": acc,
                        "y_pred": y_pred,
                        "report": classification_report(
                            y_te, y_pred, target_names=[str(i) for i in range(10)]
                        ),
                    }
                progress_bar.empty()

                # Bảng so sánh độ chính xác
                st.subheader("📊 Bảng so sánh độ chính xác (Accuracy)")
                summary_data = {
                    "Thuật toán": list(eval_results.keys()),
                    "Độ chính xác (Accuracy)": [
                        f"{r['accuracy'] * 100:.2f} %" for r in eval_results.values()
                    ],
                }
                st.table(summary_data)

                # Ma trận nhầm lẫn (Confusion Matrix)
                st.subheader("🗂️ Ma trận nhầm lẫn (Confusion Matrix)")
                cm_cols = st.columns(len(models))
                for idx_m, (name_m, res) in enumerate(eval_results.items()):
                    with cm_cols[idx_m]:
                        cm = confusion_matrix(y_te, res["y_pred"])
                        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.6))
                        cax = ax_cm.matshow(cm, cmap="Blues")
                        fig_cm.colorbar(cax, ax=ax_cm, fraction=0.046)
                        ax_cm.set_xlabel("Dự đoán (Predicted)")
                        ax_cm.set_ylabel("Thực tế (True)")
                        ax_cm.set_title(f"{name_m}\nAcc: {res['accuracy'] * 100:.1f}%", fontsize=9)
                        for (r_i, c_j), val in np.ndenumerate(cm):
                            ax_cm.text(c_j, r_i, str(val), ha="center", va="center", fontsize=6)
                        fig_cm.tight_layout()
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)

                # Trực quan hóa các mẫu nhận dạng sai
                st.subheader("❌ Phân tích các trường hợp nhận dạng sai (Mô hình SVM)")
                svm_name = list(eval_results.keys())[-1]
                svm_pred = eval_results[svm_name]["y_pred"]
                wrong_cases = np.where(y_te != svm_pred)[0]

                if len(wrong_cases) > 0:
                    show_count = min(8, len(wrong_cases))
                    err_cols = st.columns(show_count)
                    for j in range(show_count):
                        with err_cols[j]:
                            err_idx = wrong_cases[j]
                            fig_err, ax_err = plt.subplots(figsize=(1.2, 1.2))
                            ax_err.imshow(X_te[err_idx].reshape(8, 8), cmap="gray_r")
                            ax_err.set_title(f"Thực: {y_te[err_idx]}\nĐoán: {svm_pred[err_idx]}", fontsize=8, color="red")
                            ax_err.axis("off")
                            st.pyplot(fig_err)
                            plt.close(fig_err)
                else:
                    st.success("🎉 Mô hình dự đoán chính xác tuyệt đối trên tập kiểm tra!")

                with st.expander("📝 Xem Báo cáo phân loại chi tiết (Precision, Recall, F1-score)"):
                    for name_m, res in eval_results.items():
                        st.markdown(f"**{name_m}**")
                        st.code(res["report"], language="text")

            except Exception as exc:
                st.error(f"❌ Lỗi trong quá trình phân loại: {exc}")

# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 – Phát hiện đối tượng (Viola-Jones / Haar Cascade)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

with tab4:
    st.header("Phát hiện đối tượng (Supervised Learning - Object Detection)")
    st.markdown(
        "Thuật toán **Viola-Jones** (2001) kết hợp 4 kỹ thuật cốt lõi:\n"
        "1. **Đặc trưng dạng Haar (Haar-like features)** để nắm bắt cấu trúc khuôn mặt (mắt, sống mũi, miệng).\n"
        "2. **Ảnh tích hợp (Integral Image)** giúp tính toán tổng mức xám vùng chữ nhật cực nhanh với độ phức tạp $O(1)$.\n"
        "3. **Thuật toán AdaBoost** chọn lọc các đặc trưng phân loại mạnh nhất.\n"
        "4. **Cấu trúc thác (Cascading Classifier)** loại bỏ nhanh các vùng nền không phải mặt."
    )

    # Nạp Haar Cascade an toàn
    face_cascade, cascade_err = get_face_cascade()
    if cascade_err is not None:
        st.error(f"❌ {cascade_err}")
        st.stop()

    t4_source = st.radio(
        "Chọn nguồn ảnh khuôn mặt:",
        ["🖼️ Sử dụng ảnh mẫu có sẵn (Eileen Collins - NASA)", "📁 Tải ảnh từ máy tính"],
        horizontal=True,
        key="tab4_source_radio",
    )

    pil_face = None

    if t4_source == "📁 Tải ảnh từ máy tính":
        uploaded_face = st.file_uploader(
            "Tải ảnh chân dung hoặc nhóm người",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="tab4_upload",
        )
        if uploaded_face is not None:
            try:
                pil_face = Image.open(uploaded_face).convert("RGB")
            except Exception as exc:
                st.error(f"❌ Không thể đọc ảnh khuôn mặt: {exc}")
    else:
        try:
            # sk_data.astronaut() là ảnh phi hành gia Eileen Collins
            pil_face = Image.fromarray(sk_data.astronaut())
        except Exception as exc:
            st.error(f"❌ Không thể nạp ảnh mẫu: {exc}")

    # ── Tinh chỉnh tham số Viola-Jones ───────────────────────────────────────
    st.subheader("⚙️ Tinh chỉnh tham số bộ dò Viola-Jones")
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        scale_factor = st.slider(
            "scaleFactor (Tỉ lệ co giãn kim tự tháp ảnh)",
            min_value=1.02,
            max_value=1.40,
            value=1.10,
            step=0.01,
            help="Hệ số giảm kích thước ảnh ở mỗi mức kim tự tháp (Scale Pyramid). Giá trị nhỏ (1.05 - 1.15) sẽ quét kỹ hơn nhưng tốn thời gian hơn.",
        )
    with col_v2:
        min_neighbors = st.slider(
            "minNeighbors (Số hàng xóm tối thiểu)",
            min_value=1,
            max_value=10,
            value=4,
            help="Số lượng ô bao lân cận xác nhận cùng một khuôn mặt. Giá trị cao giúp loại trừ nhiễu (False Positives).",
        )
    with col_v3:
        min_face_size = st.slider(
            "minSize (Kích thước mặt nhỏ nhất px)",
            min_value=20,
            max_value=150,
            value=30,
            step=5,
            help="Kích thước tối thiểu của cửa sổ khuôn mặt cần phát hiện.",
        )

    if pil_face is not None:
        face_np = np.array(pil_face)

        if st.button("▶ Chạy phát hiện khuôn mặt (Viola-Jones)", key="btn_detect_face"):
            with st.spinner("Đang áp dụng bộ lọc Haar Cascade qua cửa sổ trượt …"):
                try:
                    gray_img = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
                    detected_faces = face_cascade.detectMultiScale(
                        gray_img,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(min_face_size, min_face_size),
                    )

                    annotated_img = face_np.copy()
                    for f_idx, (x, y, w_f, h_f) in enumerate(detected_faces):
                        cv2.rectangle(
                            annotated_img,
                            (x, y),
                            (x + w_f, y + h_f),
                            (0, 255, 0),
                            3,
                        )
                        cv2.putText(
                            annotated_img,
                            f"Face #{f_idx + 1}",
                            (x, max(y - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.image(pil_face, caption="Ảnh gốc", use_container_width=True)
                    with col_f2:
                        st.image(
                            annotated_img,
                            caption=f"Kết quả nhận diện: {len(detected_faces)} khuôn mặt",
                            use_container_width=True,
                        )

                    if len(detected_faces) > 0:
                        st.success(f"🎯 Phát hiện thành công **{len(detected_faces)}** khuôn mặt!")

                        st.subheader("✂️ Cắt trích xuất các khuôn mặt phát hiện được")
                        crop_cols = st.columns(min(len(detected_faces), 8))
                        for idx_crop, (x, y, w_f, h_f) in enumerate(detected_faces):
                            if idx_crop >= 8:
                                break
                            with crop_cols[idx_crop]:
                                cropped_face = face_np[y : y + h_f, x : x + w_f]
                                st.image(cropped_face, caption=f"Mặt #{idx_crop + 1}", use_container_width=True)
                    else:
                        st.warning(
                            "⚠️ Không phát hiện thấy khuôn mặt nào với bộ tham số hiện tại. "
                            "Hãy thử giảm `minNeighbors` hoặc giảm `scaleFactor` xuống 1.05 - 1.10."
                        )

                except Exception as exc:
                    st.error(f"❌ Lỗi trong quá trình phát hiện khuôn mặt: {exc}")
    else:
        st.info("👆 Vui lòng chọn ảnh mẫu hoặc tải ảnh chân dung lên để thử nghiệm.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🎓 **Đồ án môn học / Báo cáo thuyết trình**: Chương 9 – Các phương pháp Học máy Cổ điển trong Xử lý Ảnh\n\n"
    "Công nghệ sử dụng: Streamlit • Scikit-Learn • OpenCV (Haar-Cascade) • Scikit-Image • Matplotlib"
)