import time
import datetime
import subprocess
import psutil
import os
import logging

# ==================== 📝 日誌設定 ====================

# 🏡 程式根目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "auto_scheduler.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8'),
        logging.StreamHandler()  # 同時顯示在 console
    ]
)
log = logging.getLogger()

# 🧠 任務設定
# GA_SCRIPT = "GaFeatureStrategyMutiModel.py"
GA_SCRIPT = "GaFeatureStrategyMutiModelEX.py"  # 遺傳演算法策略腳本
FETCH_SCRIPTS = ["fetch_Muti_yahoo_financials.py", "merge_and_compute_features.py"]

# 💤 休息時間區段（不執行任何任務）
SLEEP_WINDOWS = [
    ("01:00", "02:00"),
    ("05:00", "06:00"),
    ("09:00", "10:00"),
    ("17:00", "18:00"),
]

# 📊 抓資料與合併任務時間
FETCH_WINDOW = ("18:00", "19:00")

# 🔄 當前任務狀態
current_process = None
current_task = None
last_fetch_date = None  # 記錄最近一次抓資料的日期


# 🔍 判斷時間是否在區間內
def time_in_range(start_str, end_str, now=None):
    now = now or datetime.datetime.now().time()
    start = datetime.datetime.strptime(start_str, "%H:%M").time()
    end = datetime.datetime.strptime(end_str, "%H:%M").time()
    if start < end:
        return start <= now < end
    else:
        return now >= start or now < end  # 跨午夜區段

# 🛑 結束目前任務
def kill_current_process():
    global current_process, current_task
    if current_process and psutil.pid_exists(current_process.pid):
        log.info(f"🛑 結束目前任務：{current_task}")
        current_process.terminate()
        try:
            current_process.wait(timeout=10)
        except:
            current_process.kill()
    current_process = None
    current_task = None

# 🚀 啟動指定任務
# 🚀 啟動指定任務
def launch(script_name):
    global current_process, current_task
    script_path = os.path.join(BASE_DIR, script_name)
    log.info(f"🚀 啟動任務：{script_path}")
    current_task = script_name

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{os.path.splitext(script_name)[0]}_{timestamp}.log"
    log_path = os.path.join(LOG_DIR, log_filename)

    log_file = open(log_path, 'a', encoding='utf-8')
    current_process = subprocess.Popen(
        ["python", script_path],
        stdout=log_file,
        stderr=log_file,
    )
    log.info(f"✅ 任務 {script_name} 啟動成功，輸出記錄至 {log_filename}")



# 🕓 主排程迴圈
log.info("⏱️ 自動排程監控開始喵～")

while True:
    now = datetime.datetime.now().time()

    # 😴 休息時間
    if any(time_in_range(start, end, now) for start, end in SLEEP_WINDOWS):
        if current_task:
            kill_current_process()
        log.info(f"😴 {datetime.datetime.now()} 現在是休息時間，什麼都不執行喵～")

    # 📥 抓資料時段
    elif time_in_range(*FETCH_WINDOW, now) and (last_fetch_date and last_fetch_date != datetime.datetime.now().strftime("%Y-%m-%d")):
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        if last_fetch_date != today_str:
            kill_current_process()

            log.info(f"📥 {datetime.datetime.now()} 現在是抓資料時段，開始執行抓資料任務...")
            # 1. 抓資料
            launch(FETCH_SCRIPTS[0])
            current_process.wait()

            log.info(f"📥 {datetime.datetime.now()} 資料抓取完成，開始合併特徵...")
            # 2. 合併特徵
            launch(FETCH_SCRIPTS[1])
            current_process.wait()

            log.info(f"📥 {datetime.datetime.now()} 特徵合併完成，開始遺傳演算法...")
            # 3. 遺傳演算法
            # 記錄今天已經抓過資料
            last_fetch_date = today_str


    # 📈 GA 策略時段
    else:
        if current_task != GA_SCRIPT or (current_process and current_process.poll() is not None):
            log.info("💥 偵測到 GA 任務未執行或已崩潰，重新啟動...")
            kill_current_process()
            launch(GA_SCRIPT)


    time.sleep(60)  # 每分鐘檢查一次
