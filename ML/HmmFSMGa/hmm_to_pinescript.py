import pandas as pd
import os

def convert_csv_to_pinescript_array(input_csv_path, output_txt_path):
    df = pd.read_csv(input_csv_path)
    lines = []

    for _, row in df.iterrows():
        state = int(row['state'])
        mean_str = row['hmm_means']
        covar_str = row['hmm_covars']

        mu_line = f"mu{state} = array.from_list([{mean_str}])"
        var_line = f"var{state} = array.from_list([{covar_str}])"

        lines.append(mu_line)
        lines.append(var_line)
        lines.append("")  # 空行間隔

    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ 轉換完成！已儲存到：{output_txt_path}")

# ======== 使用範例 ========
if __name__ == "__main__":
    input_csv = "hmm_parameters_n15.csv"  # 你的輸入檔
    output_txt = "pine_hmm_arrays.txt"    # 輸出的 Pine Script 陣列
    convert_csv_to_pinescript_array(input_csv, output_txt)
