"""
tabs/tab4_detection.py - Phát hiện đối tượng (Supervised Learning - Object Detection)
Bao gồm:
  1. Giới thiệu kiến trúc bộ phân loại Viola-Jones (Haar-like Features, Integral Image, AdaBoost, Cascade).
  2. Nạp bộ phân loại Haar Cascade an toàn và cho phép tinh chỉnh tham số (scaleFactor, minNeighbors, minSize).
  3. Khoanh vùng và cắt trích xuất các khuôn mặt phát hiện được trên ảnh.
"""

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from utils import get_face_cascade, get_sample_image


def render_tab4():
    """Hiển thị toàn bộ nội dung và logic của Tab 4."""
    st.header("Phát hiện đối tượng (Supervised Learning - Object Detection)")
    st.markdown(
        "Thuật toán **Viola-Jones** (2001) kết hợp 4 kỹ thuật cốt lõi:\n"
        "1. **Đặc trưng dạng Haar (Haar-like features)** để nắm bắt cấu trúc khuôn mặt (mắt, sống mũi, miệng).\n"
        "2. **Ảnh tích hợp (Integral Image)** giúp tính toán tổng mức xám vùng chữ nhật cực nhanh với độ phức tạp $O(1)$.\n"
        "3. **Thuật toán AdaBoost** chọn lọc các đặc trưng phân loại mạnh nhất.\n"
        "4. **Cấu trúc thác (Cascading Classifier)** loại bỏ nhanh các vùng nền không phải mặt."
    )

    # Nạp Haar Cascade an toàn từ utils
    face_cascade, cascade_err = get_face_cascade()
    if cascade_err is not None:
        st.error(f" {cascade_err}")
        return

    t4_source = st.radio(
        "Chọn nguồn ảnh khuôn mặt:",
        [" Sử dụng ảnh mẫu có sẵn (Eileen Collins - NASA)", " Tải ảnh từ máy tính"],
        horizontal=True,
        key="tab4_source_radio",
    )

    pil_face = None

    if t4_source == " Tải ảnh từ máy tính":
        uploaded_face = st.file_uploader(
            "Tải ảnh chân dung hoặc nhóm người",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="tab4_upload",
        )
        if uploaded_face is not None:
            try:
                pil_face = Image.open(uploaded_face).convert("RGB")
            except Exception as exc:
                st.error(f" Không thể đọc ảnh khuôn mặt: {exc}")
    else:
        try:
            pil_face = get_sample_image("astronaut")
        except Exception as exc:
            st.error(f" Không thể nạp ảnh mẫu: {exc}")

    # Tinh chỉnh tham số Viola-Jones 
    st.subheader(" Tinh chỉnh tham số bộ dò Viola-Jones")
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

        if st.button("Chạy phát hiện khuôn mặt (Viola-Jones)", key="btn_detect_face"):
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
                        st.success(f" Phát hiện thành công **{len(detected_faces)}** khuôn mặt!")

                        st.subheader(" Cắt trích xuất các khuôn mặt phát hiện được")
                        crop_cols = st.columns(min(len(detected_faces), 8))
                        for idx_crop, (x, y, w_f, h_f) in enumerate(detected_faces):
                            if idx_crop >= 8:
                                break
                            with crop_cols[idx_crop]:
                                cropped_face = face_np[y : y + h_f, x : x + w_f]
                                st.image(cropped_face, caption=f"Mặt #{idx_crop + 1}", use_container_width=True)
                    else:
                        st.warning(
                            " Không phát hiện thấy khuôn mặt nào với bộ tham số hiện tại. "
                            "Hãy thử giảm `minNeighbors` hoặc giảm `scaleFactor` xuống 1.05 - 1.10."
                        )

                except Exception as exc:
                    st.error(f" Lỗi trong quá trình phát hiện khuôn mặt: {exc}")
    else:
        st.info(" Vui lòng chọn ảnh mẫu hoặc tải ảnh chân dung lên để thử nghiệm.")
