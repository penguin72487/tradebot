import numpy as np

# ====== 參數 ======
T = 10.0           # 總時間 (1 day, 隨便)
dt = 1/(60*60*24)       # 每分鐘更新一次
N  = int(T/dt)

sigma = 0.02      # 價格 volatility (隨便假設)
gamma = 0.1       # 風險趨避
A = 1.5           # Poisson 基礎強度
k = 1.5           # 深度彈性

S0 = 1000.0        # 初始 mid
q0 = 1            # 初始庫存
X0 = 1000          # 初始現金

# ====== 狀態 ======
S = S0
q = q0
X = X0

def optimal_quotes(S, q, t):
    tau = T - t
    r = S - q * gamma * (sigma**2) * tau
    delta = 0.5 * gamma * (sigma**2) * tau + (1.0 / gamma) * np.log(1.0 + gamma / k)
    pb = r - delta
    pa = r + delta
    return pb, pa

def intensity(delta):
    return A * np.exp(-k * delta)

# ====== 模擬 loop ======
for i in range(N):
    t = i * dt

    # 1) 更新 mid-price (Brownian)
    dW = np.sqrt(dt) * np.random.randn()
    S = S + sigma * S0 * dW  # 你也可以用純 additive: S += sigma * dW

    # 2) 算出最佳掛價
    pb, pa = optimal_quotes(S, q, t)
    delta_b = S - pb
    delta_a = pa - S

    # 3) 計算成交強度
    lam_b = intensity(delta_b)
    lam_a = intensity(delta_a)

    # 4) 用 Poisson(λ dt) 決定這個 dt 是否成交
    #    這邊簡單用 Bernoulli(λ dt) 近似
    if np.random.rand() < lam_b * dt:
        # 對手打你的 bid，你買到一單
        q += 1
        X -= pb  # 無手續費

    if np.random.rand() < lam_a * dt:
        # 對手打你的 ask，你賣出一單
        q -= 1
        X += pa  # 無手續費

# 最後總財富
W_T = X + q * S
print("Final wealth:", W_T, "Inventory:", q, "Price:", S)
