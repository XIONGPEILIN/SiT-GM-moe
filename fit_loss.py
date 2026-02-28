import re
import numpy as np
import scipy.optimize
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

log_file = "results_a100/a6000/002-SiT-XL-2-Linear-velocity-None/log.txt"

steps = []
losses = []
flow_losses = []
jump_losses = []

with open(log_file, 'r') as f:
    for line in f:
        match = re.search(r'\(step=(\d+)\) Train Loss: ([\d\.]+) \(Flow: ([\d\.]+), Jump: ([\d\.]+)\)', line)
        if match:
            s_val = int(match.group(1))
            steps.append(s_val)
            losses.append(float(match.group(2)))
            flow_losses.append(float(match.group(3)))
            jump_losses.append(float(match.group(4)))

steps = np.array(steps)
losses = np.array(losses)
flow_losses = np.array(flow_losses)
jump_losses = np.array(jump_losses)

# 1. 深度学习常用学习曲线拟合 (Power Law 幂律法则): L(s) = a * s^(-b) + c
def power_law(s, a, b, c):
    # s 加上一个小常数避免s=0时除以0或负数次幂问题
    return a * np.power(s + 1e-5, -b) + c

# 2. 深度学习模型预测 (MLP 神经网络去拟合曲线步数 -> Loss)
class CurveNet(nn.Module):
    def __init__(self):
        super(CurveNet, self).__init__()
        # 使用 Sigmoid 而不是 ReLU，因为 ReLU 在外推 (extrapolation) 时呈线性发散，会导致负数
        # Sigmoid 在输入变大时会自然饱和 (saturate)，相当于趋于一个极限常数，这完全符合 Loss 收敛的先验假设
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Sigmoid(),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, 1),
            nn.Softplus() # Softplus 强行保证最终输出大于 0，绝对不会出现负数 Loss
        )
    def forward(self, x):
        return self.net(x)

def dl_predict_limit(x, y, target_step=2000000):
    # 对于 x 依然进行标准化
    x_mean, x_std = x.mean(), x.std()
    x_t = torch.tensor((x - x_mean) / x_std, dtype=torch.float32).unsqueeze(1)
    
    # 因为加了 Softplus，y 直接用原始值，不用标准化，因为 loss 本来就在 0~10 之间
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    model = CurveNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 多训练一些 epoch，让网络拟合得更好
    for epoch in range(3000):
        optimizer.zero_grad()
        pred = model(x_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        
    x_target = torch.tensor([[(target_step - x_mean) / x_std]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        final_pred = model(x_target).item()
    
    return final_pred

def fit_and_predict(name, x, y):
    print(f"========== {name} ==========")
    print(f"Max step recorded: {x[-1]}, Current Final Value: {y[-1]:.6f}")

    # ===== 多项式预测 (Polynomial Fit) =====
    # 二次多项式: y = a*x^2 + b*x + c
    # 对于开口向上的抛物线，极值在 x = -b / (2a)
    try:
        p2 = np.polyfit(x, y, 2)
        print(f"Polynominal (Degree 2): {p2[0]:.4e}*s^2 + {p2[1]:.4e}*s + {p2[2]:.6f}")
        if p2[0] > 0:
            extreme_x = -p2[1] / (2 * p2[0])
            extreme_y = p2[0]*extreme_x**2 + p2[1]*extreme_x + p2[2]
            print(f"  -> 二次多项式极值点在 step={extreme_x:.0f}, 极值={extreme_y:.6f}")
        else:
            print("  -> 二次多项式开口向下，无极小值下限")
    except Exception as e:
        print(f"Polynomial 2 failed: {e}")
        
    try:
        p3 = np.polyfit(x, y, 3)
        print(f"Polynominal (Degree 3): {p3[0]:.4e}*s^3 + {p3[1]:.4e}*s^2 + {p3[2]:.4e}*s + {p3[3]:.6f}")
        # 极值点导数 3a*s^2 + 2b*s + c = 0
        a, b, c = 3*p3[0], 2*p3[1], p3[2]
        delta = b**2 - 4*a*c
        if delta >= 0 and a != 0:
            root1 = (-b + np.sqrt(delta))/(2*a)
            root2 = (-b - np.sqrt(delta))/(2*a)
            ex_x = root1 if p3[0]*root1 > 0 else root2 # 极大极小判断
            ex_y = p3[0]*ex_x**3 + p3[1]*ex_x**2 + p3[2]*ex_x + p3[3]
            print(f"  -> 三次多项式局部极小值可能在 step={ex_x:.0f}, 极值={ex_y:.6f}")
    except Exception as e:
        pass

    # ===== 深度学习 Scaling Law 参数拟合 (Power Law) =====
    # 常常用于深度学习模型 Loss 预测
    try:
        # a, b, c 初始值猜测
        p0 = [y[0]-min(y), 0.5, min(y)]
        popt_pow, _ = scipy.optimize.curve_fit(power_law, x, y, p0=p0, maxfev=10000)
        print(f"Scaling Law 拟合 (a*s^(-b) + limit):")
        print(f"  -> limit 参数 (c): {popt_pow[2]:.6f}, a={popt_pow[0]:.4f}, b={popt_pow[1]:.4f}")
    except Exception as e:
        print(f"Scaling Law fit failed: {e}")

    # ===== 真正的深度学习神经网络预测 =====
    # 预测到达某个大数字例如 200,000 步的 loss
    try:
        nn_pred = dl_predict_limit(x, y, target_step=2000000)
        print(f"DL MLP 神经网络预测 (step=2,000,000): {nn_pred:.6f}")
    except Exception as e:
        print(f"DL MLP failed: {e}")
        
    print()

if len(steps) > 0:
    fit_and_predict('Total Loss', steps, losses)
    fit_and_predict('Flow Loss', steps, flow_losses)
    fit_and_predict('Jump Loss', steps, jump_losses)
else:
    print("No matched lines found. Please check log format or file path.")
