"""
utils.py - Các hàm tiện ích, cấu hình và nạp dữ liệu dùng chung
cho ứng dụng Xử lý Ảnh bằng Machine Learning Cổ điển.
"""

import os
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from skimage import data as sk_data
from sklearn.datasets import fetch_lfw_people, load_digits

#  Hằng số đường dẫn 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAAR_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)


#  Nạp dữ liệu với cơ chế Cache 
@st.cache_data(show_spinner="Đang tải tập dữ liệu LFW Faces …")
def load_lfw_dataset(min_faces: int = 60, resize: float = 0.4):
    """
    Tải tập dữ liệu khuôn mặt Labeled Faces in the Wild (LFW).
    Được cache lại để không cần tải lại giữa các lần re-run.
    """
    faces_data = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize)
    return faces_data.images, faces_data.target, faces_data.target_names


@st.cache_data(show_spinner="Đang tải tập dữ liệu chữ số MNIST / Digits …")
def load_digits_dataset():
    """
    Tải tập dữ liệu chữ số viết tay 8×8 (Scikit-Learn Digits - biến thể chuẩn của MNIST).
    """
    digits_data = load_digits()
    return digits_data.images, digits_data.data, digits_data.target, digits_data.target_names


#  Nạp bộ phân loại Viola-Jones (Haar Cascade) 
def get_face_cascade():
    """
    Khởi tạo bộ phân loại Haar Cascade an toàn với xử lý ngoại lệ và cơ chế Fallback.
    Trả về: (cascade_object, error_message)
    """
    if not hasattr(cv2, "CascadeClassifier"):
        return None, (
            "Thư viện OpenCV hiện tại không chứa module `cv2.CascadeClassifier`.\n\n"
            " Vui lòng cài đặt bản tương thích: `pip install --force-reinstall \"opencv-python-headless<5.0\"`"
        )

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
            except Exception:
                continue

    return None, (
        f"Không tìm thấy file `haarcascade_frontalface_default.xml` hợp lệ.\n\n"
        f"Đã tìm kiếm tại: `{HAAR_CASCADE_PATH}`.\n"
        "Vui lòng đảm bảo file XML nằm trong thư mục gốc của ứng dụng."
    )


# Helper nạp ảnh mẫu 
def get_sample_image(name: str) -> Image.Image:
    """
    Trả về đối tượng PIL Image từ thư viện scikit-image chuẩn.
    """
    name_lower = name.lower()
    if "cà phê" in name_lower or "coffee" in name_lower:
        return Image.fromarray(sk_data.coffee())
    elif "du hành" in name_lower or "astronaut" in name_lower:
        return Image.fromarray(sk_data.astronaut())
    elif "máy ảnh" in name_lower or "camera" in name_lower:
        return Image.fromarray(sk_data.camera()).convert("RGB")
    else:
        return Image.fromarray(sk_data.astronaut())


def fig_to_image(fig: plt.Figure) -> Image.Image:
    """Chuyển đổi một Figure của Matplotlib thành ảnh PIL."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img
