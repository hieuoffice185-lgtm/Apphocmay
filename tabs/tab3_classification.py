"""
tabs/tab3_classification.py - Phân loại ảnh chữ số (Supervised Learning)
Bao gồm:
  1. Nạp và trực quan hóa tập dữ liệu chữ số viết tay Digits / MNIST.
  2. Huấn luyện và so sánh 3 mô hình: k-Nearest Neighbors, Gaussian Naive Bayes, SVM.
  3. Bảng so sánh Accuracy, ma trận nhầm lẫn (Confusion Matrix) và phân tích lỗi.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils import load_digits_dataset


def render_tab3():
    """Hiển thị toàn bộ nội dung và logic của Tab 3."""
    st.header("Phân loại chữ số viết tay (Supervised Learning - Classification)")
    st.markdown(
        "So sánh 3 mô hình học máy kinh điển: **k-Nearest Neighbors (kNN)**, "
        "**Gaussian Naïve Bayes** và **Support Vector Machine (SVM)** trên tập chữ số viết tay."
    )

    with st.spinner("Đang nạp tập dữ liệu chữ số MNIST / Digits …"):
        try:
            dig_images, dig_data, dig_target, dig_names = load_digits_dataset()
            n_dig = len(dig_target)
            st.success(
                f" Đã nạp thành công **{n_dig}** ảnh chữ số (kích thước 8×8 pixels, 10 lớp từ 0 đến 9)."
            )
        except Exception as exc:
            st.error(f" Không thể tải tập dữ liệu Digits: {exc}")
            return

    # Hiển thị các chữ số mẫu
    st.subheader("Một số mẫu chữ số viết tay tiêu biểu")
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

    #  Thiết lập tham số 
    st.subheader(" Thiết lập tham số huấn luyện")
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

                #  1. Bảng so sánh độ chính xác 
                st.subheader(" Bảng so sánh độ chính xác (Accuracy)")
                summary_data = {
                    "Thuật toán": list(eval_results.keys()),
                    "Độ chính xác (Accuracy)": [
                        f"{r['accuracy'] * 100:.2f} %" for r in eval_results.values()
                    ],
                }
                st.table(summary_data)

                #  2. Ma trận nhầm lẫn (Confusion Matrix) 
                st.subheader(" Ma trận nhầm lẫn (Confusion Matrix)")
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

                #  3. Phân tích các trường hợp nhận dạng sai 
                st.subheader(" Phân tích các trường hợp nhận dạng sai (Mô hình SVM)")
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
                            ax_err.set_title(
                                f"Thực: {y_te[err_idx]}\nĐoán: {svm_pred[err_idx]}",
                                fontsize=8,
                                color="red",
                            )
                            ax_err.axis("off")
                            st.pyplot(fig_err)
                            plt.close(fig_err)
                else:
                    st.success(" Mô hình dự đoán chính xác tuyệt đối trên tập kiểm tra!")

                with st.expander(" Xem Báo cáo phân loại chi tiết (Precision, Recall, F1-score)"):
                    for name_m, res in eval_results.items():
                        st.markdown(f"**{name_m}**")
                        st.code(res["report"], language="text")

            except Exception as exc:
                st.error(f" Lỗi trong quá trình phân loại: {exc}")
