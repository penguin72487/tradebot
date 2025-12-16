import qlib
from qlib.data import D
import os
from pathlib import Path
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(base_dir, "clean_csv_for_qlib", "BTCUSDT.csv")
OUTPUT_DIR = os.path.join(base_dir, "qlib_data")
FREQ = "15min"

qlib.init(provider_uri=OUTPUT_DIR, region="cn")  # 路徑記得改成你實際的 OUTPUT_DIR

# 先嘗試用 qlib 的 API 取得 calendar；失敗時退回到我們建立的 calendar.txt
try:
    print("calendar sample:", D.calendar(freq=FREQ)[:5])
except Exception as e:
    print("D.calendar failed:", e)
    cal_path = Path(OUTPUT_DIR) / "cal" / FREQ / "calendar.txt"
    if cal_path.exists():
        with open(cal_path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh.readlines()][:5]
        print("fallback calendar sample:", lines)
    else:
        print("no fallback calendar found at:", cal_path)

# 試著用 qlib 的 features 讀取；如果失敗，就顯示我們的後備 feature pickles 的樣本
try:
    df = D.features(
        instruments=["BTCUSDT"],
        fields=["close", "RSI Line"],  # 這裡欄名要跟你 CSV / feature 一樣
        start_time="2024-01-01",
        end_time="2024-01-02",
        freq=FREQ,
    )
    print(df.head())
except Exception as e:
    print("D.features failed:", e)
    feat_dir = Path(OUTPUT_DIR) / "feature" / "BTCUSDT"
    if feat_dir.exists():
        print("Listing fallback feature files and small samples:")
        for p in sorted(feat_dir.glob("*.pkl"))[:10]:
            try:
                s = pd.read_pickle(p)
                print(p.name, s.head(3).to_dict())
            except Exception as e2:
                print("failed to read", p, e2)
    else:
        print("no fallback feature directory found at:", feat_dir)
