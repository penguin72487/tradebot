import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def entropy_table(probs, label):
    chars = ['a', 'b', 'c', 'd']
    i_si = [-np.log2(p) if p > 0 else 0 for p in probs]
    h_si = [p * i for p, i in zip(probs, i_si)]
    total_entropy = sum(h_si)
    df = pd.DataFrame({
        'S': [label] * len(chars),
        'Si': chars,
        'p(si)': probs,
        'i(si) = log2(1/p(si))': i_si,
        'H(si) = p(si) * i(si)': h_si,
        'H(S)': [f"{total_entropy:.4f}"] * len(chars)
    })
    return df

# 三組機率
S1 = [0.25, 0.25, 0.25, 0.25]
S2 = [0.625, 0.0625, 0.0625, 0.25]
S3 = [0.8125, 0.0625, 0.0625, 0.0625]

# 建表
df1 = entropy_table(S1, 'S1')
df2 = entropy_table(S2, 'S2')
df3 = entropy_table(S3, 'S3')

# 合併所有資料
df_all = pd.concat([df1, df2, df3], ignore_index=True)

# ===== 畫圖模擬合併儲存格 =====
fig, ax = plt.subplots(figsize=(15, 7))
ax.axis('off')

# 計算合併儲存格用的rowspan
row_spans = [4, 4, 4]
merge_rows = []
start_row = 1  # 從1開始算，因為0是標題列
for span in row_spans:
    merge_rows.append((start_row, span))
    start_row += span

# 建立表格
table = ax.table(cellText=df_all.values,
                 colLabels=df_all.columns,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# 美化與合併視覺模擬
for (row_start, span) in merge_rows:
    for col in [0, 5]:  # 合併 S 和 H(S) 欄
        for i in range(1, span):
            table[(row_start + i, col)].visible_edges = ''
            table[(row_start + i, col)].get_text().set_text('')

# 標題列樣式
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e')
    else:
        cell.set_facecolor('#f1f1f2' if key[0] % 2 == 0 else '#ffffff')

# 儲存圖片
output_path = "entropy_table.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"✅ 表格已儲存為：{output_path}")


# 感謝 GPT-4o 的幫助，右邊是對話紀錄 https://chatgpt.com/share/682dbae8-e514-8005-8dea-a2fbd2c7ea74