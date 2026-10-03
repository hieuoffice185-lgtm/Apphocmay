Ứng dụng được thiết kế theo cấu trúc module hóa, chia thành 4 Tab tính năng bám sát nội dung cốt lõi của Chương 9:

Tab 1: Gom cụm điểm ảnh (Unsupervised Learning): Cho phép tải ảnh lên, sử dụng thanh trượt để điều chỉnh số cụm màu (k = 2 đến 64) nhằm thực hiện lượng hóa màu bằng K-Means, và cung cấp nút bấm để chạy thuật toán Spectral Clustering tách biệt tiền cảnh/hậu cảnh

Tab 2: Giảm chiều dữ liệu & Eigenfaces (Unsupervised Learning): Trực quan hóa ảnh khuôn mặt trung bình cùng các "khuôn mặt ma" (Eigenfaces) từ tập dữ liệu, đồng thời sử dụng thanh trượt chọn số lượng thành phần chính (PCs) để quan sát quá trình tái cấu trúc độ sắc nét của khuôn mặt

Tab 3: Phân loại chữ số viết tay (Supervised Learning): Lấy mẫu ngẫu nhiên từ tập dữ liệu MNIST và so sánh kết quả dự đoán đa mô hình giữa k-Nearest Neighbors (kNN), Gaussian Bayes và Support Vector Machine (SVM)

Tab 4: Phát hiện đối tượng (Supervised Learning): Nhận diện khuôn mặt tập thể bằng thuật toán Viola-Jones và phát hiện người đi bộ bằng HOG-SVM, kết hợp với công tắc Bật/Tắt kỹ thuật gộp khung Non-Maximum Suppression (NMS) để quan sát sự tối ưu của thuật toán
