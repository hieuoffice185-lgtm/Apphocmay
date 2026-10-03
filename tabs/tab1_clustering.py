"""
tabs/tab1_clustering.py - Gom cụm điểm ảnh (Unsupervised Learning)
Bao gồm:
  1. Lượng hóa màu (Color Quantization) bằng thuật toán K-Means.
  2. Phân đoạn ảnh (Image Segmentation) bằng thuật toán Spectral Clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from skimage import img_as_float
from sklearn.cluster import KMeans, SpectralClustering

from utils import get_sample_image


def render_tab1():
    """Hiển thị toàn bộ nội dung và logic của Tab 1."""
    st.header("Gom cụm điểm ảnh (Unsupervised Learning)")
    st.markdown(
        "Sử dụng **K-Means** để lượng hóa màu (color quantization) "
        "và **Spectral Clustering** để phân đoạn tiền cảnh / hậu cảnh."
    )

    t1_source = st.radio(
        "Chọn nguồn ảnh:",
        [" Tải ảnh từ máy tính", " Sử dụng ảnh mẫu có sẵn"],
        horizontal=True,
        key="tab1_source_radio",
    )

    pil_img = None

    if t1_source == " Tải ảnh từ máy tính":
        uploaded_file = st.file_uploader(
            "Tải ảnh lên",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="tab1_upload",
        )
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file).convert("RGB")
            except Exception as exc:
                st.error(f" Không thể đọc ảnh tải lên: {exc}")
    else:
        sample_choice = st.selectbox(
            "Chọn ảnh mẫu:",
            ["Tách cà phê (Nhiều dải màu)", "Nhà du hành vũ trụ", "Thợ chụp ảnh"],
            key="tab1_sample_select",
        )
        try:
            pil_img = get_sample_image(sample_choice)
        except Exception as exc:
            st.error(f" Không thể nạp ảnh mẫu: {exc}")

    if pil_img is not None:
        img_np = np.array(pil_img)

        # Giới hạn kích thước ảnh tối đa để tính toán nhanh mượt
        MAX_DIM = 512
        h0, w0 = img_np.shape[:2]
        if max(h0, w0) > MAX_DIM:
            scale = MAX_DIM / max(h0, w0)
            new_size = (int(w0 * scale), int(h0 * scale))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            img_np = np.array(pil_img)
            st.info(
                f" Ảnh đã được tối ưu kích thước về {img_np.shape[1]}×{img_np.shape[0]} px để tăng tốc tính toán."
            )

        st.image(pil_img, caption="Ảnh đầu vào", use_container_width=False, width=380)

        #  1. K-Means Color Quantization 
        st.subheader(" 1. Lượng hóa màu bằng K-Means")
        st.caption("Thuật toán gom cụm các pixel trong không gian màu 3D (R, G, B) thành k màu đại diện.")

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

                    # Bảng màu đại diện của các cluster centers
                    fig_palette, ax_palette = plt.subplots(1, 1, figsize=(8, 1))
                    ax_palette.imshow([kmeans.cluster_centers_], aspect="auto")
                    ax_palette.set_xticks(range(k_colors))
                    ax_palette.set_yticks([])
                    ax_palette.set_title(f"Bảng {k_colors} màu trọng tâm (Cluster Centers)")
                    st.pyplot(fig_palette)
                    plt.close(fig_palette)
                except Exception as exc:
                    st.error(f"Lỗi khi chạy K-Means: {exc}")

        # 2. Spectral Clustering 
        st.subheader(" 2. Phân đoạn ảnh bằng Spectral Clustering")
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
                    st.error(f"Lỗi trong quá trình Spectral Clustering: {exc}")
    else:
        st.info(" Vui lòng tải lên ảnh hoặc chọn ảnh mẫu phía trên để bắt đầu thử nghiệm.")
