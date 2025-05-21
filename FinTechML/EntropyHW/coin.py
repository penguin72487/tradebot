import numpy as np
import matplotlib.pyplot as plt

# 定義二元熵函數 H2(p)
def H2(p):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            (p == 0) | (p == 1),
            0,
            -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        )

# 建立從 0.001 到 0.999 的 p 值範圍（避免 log(0)）
p_values = np.linspace(0.001, 0.999, 1000)
h_values = H2(p_values)

# 畫圖
plt.figure(figsize=(10, 6))
plt.plot(p_values, h_values, label=r'$H_2(p) = -p \log_2 p - (1-p) \log_2 (1-p)$', linewidth=2, color='blue')
plt.title('Binary Entropy Function H2(p)', fontsize=16)
plt.xlabel('Probability p (of Head)', fontsize=14)
plt.ylabel('Entropy H2(p)', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()

# 儲存圖檔
output_path = "binary_entropy_function.png"
plt.savefig(output_path, dpi=300)
print(f"✅ 熵函數圖已儲存為：{output_path}")
