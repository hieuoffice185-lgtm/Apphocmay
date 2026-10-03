"""
tabs/tab2_pca.py - Giảm chiều dữ liệu & Eigenfaces (PCA – Unsupervised Learning)
Bao gồm:
  1. Trích xuất các thành phần chính (Eigenfaces) trên tập khuôn mặt LFW.
  2. Phân tích phương sai giải thích tích lũy (Cumulative Explained Variance).
  3. Minh họa khả năng nén và tái tạo ảnh khuôn mặt từ k thành phần chính.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import PCA

from utils import load_lfw_dataset


def render_tab2():
    """Hiển thị toàn bộ nội dung và logic của Tab 2."""
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
                f" Đã nạp thành công **{n_samples}** ảnh khuôn mặt ({h_face}×{w_face} px) "
                f"thuộc **{len(lfw_names)}** nhân vật nổi tiếng."
            )
        except Exception as exc:
            st.error(f" Không thể nạp tập dữ liệu LFW: {exc}")
            return

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

                #  1. Trực quan hóa Eigenfaces 
                st.subheader(f" Trực quan hóa {min(n_components, 16)} Eigenfaces đầu tiên")
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

                #  2. Phương sai giải thích 
                st.subheader("Tỉ lệ phương sai giải thích tích lũy (Cumulative Explained Variance)")
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

                #  3. Tái tạo khuôn mặt từ k thành phần 
                st.subheader(" Minh họa khả năng nén & tái tạo khuôn mặt")
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
                st.error(f" Lỗi khi phân tích PCA: {exc}")
