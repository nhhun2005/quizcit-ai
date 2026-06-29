# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class AttemptData(BaseModel):
    responses: list  # [1,0,1,1,...]
    difficulties: list  # [0.2, -0.5, ...]
    weights: list        # [0.8, 0.3, 1.0, ...]

def rasch_theta(responses, difficulties, weights, max_iter=20,
                theta_min=-4.0, theta_max=4.0):
    # Khởi tạo giá trị
    theta = 0.0
    # sigma điều khiển độ mạnh của prior N(0, sigma^2)
    # sigma nhỏ → prior mạnh → θ bị kéo về 0 nhiều hơn (ít tin dữ liệu)
    # sigma lớn → prior yếu → θ linh hoạt hơn (tin dữ liệu nhiều hơn)
    sigma = 1
    # Công thức toán học xác suất thống kê Bayes. sigma càng lớn => niềm tin rằng câu hỏi này đánh giá đúng càng cao, sigma càng thấp => niềm tin rằng câu hỏi này đánh giá đúng càng thấp 
    #sigma = 1, nghĩa là theta kì vọng nằm khoảng (0 - sigma^2) => tin rằng đa số học viên có năng lực từ khoảng 0 - 1

    # Đổi kiểu dữ liệu sang numpy, cho việc thực hiện tính vector hóa nhanh hơn
    responses = np.array(responses, dtype=float)
    difficulties = np.array(difficulties, dtype=float)
    weights = np.array(weights, dtype=float)

    # Lặp tối đa 20 lần đề tìm theta tốt nhất, chặn trường hợp bị giao động mạnh và gây ra lặp vô tận
    for _ in range(max_iter):
        
        # Công thức IRT tính xác suất làm đúng câu hỏi
        # Càng cao thì kì vọng sinh viên trả lời đúng câu đó càng cao
        p = 1 / (1 + np.exp(-(theta - difficulties)))
        
        # gradient (chiều và độ lớn), thg này lớn thì sv đang làm tốt hơn những gì theta đang thể hiện => tăng theta và ngược lại
        gradient = np.sum(weights * (responses - p)) - theta / (sigma**2)
        
        # hessian (độ tin tưởng vào gradient)
        hessian = -np.sum(weights * p * (1 - p)) - 1 / (sigma**2)
        # Newton-Raphson update
        new_theta = theta- gradient / hessian
        new_theta = np.clip(new_theta, theta_min, theta_max)  # clamp đưa giá trị theta về khoảng [-4:4]

        # Convergencce, thực hiện kết thúc vòng lặp sớm nếu như đã hội tụ
        if abs(new_theta - theta) < 1e-4:
            break
        theta = new_theta
        
    
    return {
        "theta_raw":    round(float(theta), 4),        # logit gốc, dùng cho tính toán
    }


# BẢNG GIÁ TRỊ ĐỂ ĐỔI THETA_RAW SANG ĐIỂM CHỮ - ĐIỂM 4.
#         (-4.0, -2.0, "F",  0.0),
#         (-2.0, -1.0, "D",  1.0),
#         (-1.0, -0.5, "D+", 1.5),
#         (-0.5,  0.5, "C",  2.0),
#         ( 0.5,  1.0, "C+", 2.5),
#         ( 1.0,  2.0, "B",  3.0),
#         ( 2.0,  3.0, "B+", 3.5),
#         ( 3.0,  4.0, "A",  4.0),


# Phần này hiện chưa xài, chỉ xài để cập nhật độ khó câu hỏi nếu cuối kỳ cảm thấy cần
# def update_difficulty(
#     b_old: float,
#     theta_raws: list[float],   # theta_raw lấy từ ability_history.theta
#     responses: list[int],      # 0/1 từ responses.score
#     decay: float = 0.6
# ) -> float:
#     """
#     Cập nhật difficulty_b cuối kỳ.
#     Bắt buộc dùng theta_raw — KHÔNG dùng theta_scaled.
#     Cần ít nhất 10 lượt làm mới cập nhật.
#     """
#     if len(theta_raws) < 10:
#         return round(b_old, 4)

#     theta_arr    = np.array(theta_raws, dtype=float)
#     response_arr = np.array(responses,  dtype=float)

#     theta_mean  = float(np.mean(theta_arr))
#     p_observed  = float(np.clip(np.mean(response_arr), 0.01, 0.99))

#     b_new   = theta_mean - np.log(p_observed / (1 - p_observed))
#     b_new   = float(np.clip(b_new, THETA_MIN, THETA_MAX))
#     b_final = b_old * decay + b_new * (1 - decay)

#     return round(b_final, 4)

@app.post("/estimate")
def estimate_theta(data: AttemptData):
    return rasch_theta(data.responses, data.difficulties, data.weights)