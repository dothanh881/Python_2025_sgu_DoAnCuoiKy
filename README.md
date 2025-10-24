# Image Captioning with PyTorch

**Reference:** [A PyTorch Tutorial to Image Captioning (sgrvinod)](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)  
**Paper:** [Show, Attend and Tell (Xu et al., 2015)](https://arxiv.org/abs/1502.03044)

---
##  Giảng Viên Hướng Dẫn
**ThS. Phùng Thái Thiên Trang*  

*Khoa Công nghệ Thông tin – Trường Đại học Sài Gòn*

##  Nhóm Thực Hiện

| STT | Họ và Tên          | Mã Số Sinh Viên |
|:---:|--------------------|:---------------:|
| 1 | Đỗ Phú Thành       | 3122411189 |
| 2 | Huỳnh Duy Khang | 3122411088 |
| 3 | Võ Thành Danh   | 3122411024 |

---

##  Objective
Xây dựng mô hình **tự động sinh chú thích (caption)** cho ảnh theo kiến trúc **Encoder–Decoder với Attention**.  
Dự án **tham khảo source gốc** và sẽ **mở rộng thêm mô hình Transformer** để so sánh hiệu quả giữa LSTM-Attention và Self-Attention.

---

##  Model Architecture

### 1) Encoder
- **ResNet-101** pretrained (ImageNet) — *Transfer Learning*.  


### 2) Attention Mechanism
- **Soft Attention**: tính trọng số cho từng vùng ảnh dựa trên **hidden state** hiện tại của Decoder.  

### 3) Decoder
- **LSTM Decoder** sinh caption theo từng bước.  

### 4) Beam Search (Inference)
- Dùng **Beam Search** để chọn **chuỗi có xác suất tổng cao nhất**, tránh greedy kém tối ưu.

---

##  Workflow Overview

**Dataset**
- **Flickr8k** dataset (thay cho MSCOCO 2014 trong source gốc).  
- **Caption splits**: theo định nghĩa của **Andrej Karpathy** (train / val / test).  
- Mỗi ảnh có 5 caption mô tả khác nhau.

**Preprocessing**
- Chuẩn hoá ảnh theo **ImageNet mean/std**.  
- Tokenize, map từ → chỉ số; thêm **`<start>`, `<end>`, `<pad>`**.  
- Lưu ảnh vào **HDF5**, caption & độ dài vào **JSON**.

**Training**
- **Loss**: CrossEntropy **+** *Doubly Stochastic Regularization* (khuyến khích chú ý đủ các vùng).  
- **Optimizer**: Adam.  
- **Fine-tune Encoder** sau khi Decoder ổn định.

**Evaluation**
- **Metric**: BLEU-4 (NLTK).  
- **Inference** không dùng Teacher Forcing; hỗ trợ Beam Search.

---

##  Knowledge & Techniques Used

| Kiến thức | Vai trò |
|---|---|
| CNN (ResNet-101) | Encoder trích đặc trưng ảnh |
| LSTM | Decoder sinh chuỗi caption |
| Attention Mechanism | Tập trung vào vùng ảnh quan trọng |
| Transfer Learning | Khởi tạo từ mô hình pretrained |
| Beam Search | Tối ưu quá trình giải mã chuỗi |
| BLEU Score | Đánh giá chất lượng caption |
| HDF5 / JSON | Quản lý dữ liệu ảnh & chú thích |
| Teacher Forcing | Hỗ trợ huấn luyện nhanh hơn |

---

##  Future Work: Transformer-based Captioning
- Thay LSTM bằng **Transformer Decoder** (Self-Attention).  
- So sánh **BLEU**, tốc độ huấn luyện, khả năng tổng quát giữa:
  - **ResNet + LSTM-Attention** (baseline hiện tại).  
  - **ResNet + Transformer Decoder** (mở rộng).  


---

##  References
- Xu et al. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*. [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)  
- Source: [sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
