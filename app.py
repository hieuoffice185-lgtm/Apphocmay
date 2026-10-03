"""
app.py - Điểm khởi chạy chính của ứng dụng Web Demo
Chương 9: Các phương pháp Học máy Cổ điển trong Xử lý Ảnh
(Classical Machine Learning Methods in Image Processing)

Chạy ứng dụng:
    streamlit run app.py
"""

import warnings
import matplotlib
import streamlit as st

# Nạp các module Tab từ package tabs
from tabs import render_tab1, render_tab2, render_tab3, render_tab4

# Cấu hình matplotlib backend không GUI
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Demo – ML trong Xử lý Ảnh",
    page_icon="",
    layout="wide",
)

# Tiêu đề và giới thiệu
st.title(" Chương 9 – Các phương pháp Học máy Cổ điển trong Xử lý Ảnh")
st.caption("Classical Machine Learning Methods in Image Processing • Streamlit Web Demo")

# Khởi tạo 4 Tab chức năng
tab1, tab2, tab3, tab4 = st.tabs(
    [
        " 1. Gom cụm điểm ảnh",
        " 2. PCA & Eigenfaces",
        " 3. Phân loại ảnh",
        " 4. Phát hiện khuôn mặt",
    ]
)

with tab1:
    render_tab1()

with tab2:
    render_tab2()

with tab3:
    render_tab3()

with tab4:
    render_tab4()

# Phần chân trang (Footer)
st.divider()
st.caption(
    "🎓 **Báo cáo thuyết trình**: Chương 9 – Các phương pháp Học máy Cổ điển trong Xử lý Ảnh\n\n"
    "Công nghệ sử dụng: Streamlit • Scikit-Learn • OpenCV (Haar-Cascade) • Scikit-Image • Matplotlib"
)