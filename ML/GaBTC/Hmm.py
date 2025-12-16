# -*- coding: utf-8 -*-
# HMM + L0 Gate Benchmark（先 GPU 後 CPU）
import os, math, time, copy
import torch
from torch import nn

# ========= 你直接在這裡改參數 =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_PARQUET = os.path.join(BASE_DIR, "features_240_4112.parquet")
B = 2**8; T = 1024; D = 2**10; N = 2**3   # 批次×時間×特徵×狀態
STEPS = 1000; WARMUP = 5
LR = 5e-3; LAM = 5e-3
CPU_THREADS = os.cpu_count()
USE_COMPILE = True
USE_AMP = True   # 只影響 GPU

def reade_parquet(path):
    """
    Try to read a parquet file and reshape it to (B, T, D) if possible;
    on failure, return a random tensor of shape (B, T, D).
    This ensures the original call `reade_parquet(...)` is defined.
    """
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        arr = df.values
        # Ensure at least 1-D
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # If array has enough elements, reshape into (B, T, D)
        if arr.size >= B * T * D:
            arr = arr.ravel()[:B * T * D].reshape(B, T, D)
            return torch.tensor(arr, dtype=torch.float32)
        # If already 3D and matches shape, accept it
        if arr.ndim == 3 and arr.shape == (B, T, D):
            return torch.tensor(arr, dtype=torch.float32)
        # Otherwise fall back
        raise ValueError("parquet data cannot be reshaped to (B,T,D)")
    except Exception:
        print(f"Warning: failed to read parquet '{path}', generating random data.")
        return torch.randn(B, T, D, dtype=torch.float32)

# ====== << 只在程式一開始設定一次 CPU 執行緒 >> ======
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(max(1, CPU_THREADS // 2))
torch.backends.mkldnn.enabled = True

# ====== 其他效能設定 ======
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# ========= 模型（同前，略） =========
class HardConcreteGate(nn.Module):
    def __init__(self, D, temperature=2./3., gamma=-0.1, zeta=1.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(D))
        self.temperature, self.gamma, self.zeta = temperature, gamma, zeta
    def regularizer(self):
        s = torch.sigmoid(self.log_alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0, 1).sum()
    def forward(self, X, deterministic=True):
        s = torch.sigmoid(self.log_alpha)
        if deterministic:
            z = s * (self.zeta - self.gamma) + self.gamma
        else:
            u = torch.rand_like(s)
            l = torch.log(u) - torch.log1p(-u)
            s_tilde = torch.sigmoid((l + self.log_alpha) / self.temperature)
            z = s_tilde * (self.zeta - self.gamma) + self.gamma
        return X * z.clamp(0, 1)

class GaussianHMM(nn.Module):
    def __init__(self, N, D, init_sticky=0.9):
        super().__init__()
        self.N, self.D = N, D
        self.log_pi = nn.Parameter(torch.full((N,), -math.log(N)))
        A = torch.eye(N) * init_sticky + (1 - init_sticky) / N
        self.log_A = nn.Parameter(A.log())
        self.mu = nn.Parameter(torch.zeros(N, D))
        self.log_var = nn.Parameter(torch.zeros(N, D))
        self.gate = HardConcreteGate(D)

    # -------- GPU 高吞吐：一次把 [B,T,N] 的 log_B 算完（GEMM）--------
    def emission_log_prob_matmul(self, X):  # X: [B,T,D]
        B, T, D = X.shape
        Xg = self.gate(X)                                  # [B,T,D]
        Xg2 = torch.square(Xg)                             # [B,T,D]
        X2 = Xg2.reshape(B*T, D).contiguous()              # [BT,D]
        X1 = Xg.reshape(B*T, D).contiguous()               # [BT,D]

        inv_var = torch.exp(-self.log_var)                 # [N,D]
        W1 = inv_var.t().contiguous()                      # [D,N]
        W2 = (self.mu * inv_var).t().contiguous()          # [D,N]
        const = (self.mu.square() * inv_var).sum(dim=1)    # [N]

        # 兩個 GEMM → [BT,N]
        x2_term = X2 @ W1
        xm_term = X1 @ W2
        quad = x2_term - 2.0 * xm_term + const.view(1, -1) # [BT,N]

        logdet = self.log_var.sum(dim=1)                   # [N]
        lp = -0.5 * (quad + logdet.view(1, -1) + self.D * math.log(2*math.pi))
        return lp.view(B, T, self.N)                       # [B,T,N]

    # -------- 前向：吃預先算好的 log_B（避免重算與 kernel storm）--------
    def forward_loglik_from_logB(self, log_B, mask=None):  # log_B: [B,T,N]
        B, T, N = log_B.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=log_B.device)
        alpha = self.log_pi.view(1, N) + log_B[:, 0, :]    # [B,N]
        alpha = torch.where(mask[:, 0, None], alpha, torch.full_like(alpha, float("-inf")))
        for t in range(1, T):                               # O(T·N^2)（N 通常很小）
            m = alpha.unsqueeze(2) + self.log_A.unsqueeze(0)  # [B,N,N]
            alpha_t = torch.logsumexp(m, dim=1) + log_B[:, t, :]
            alpha = torch.where(mask[:, t, None], alpha_t, alpha)
        return torch.logsumexp(alpha, dim=1)                # [B]

    # -------- CPU 省記憶體版本（保留給 CPU）--------
    def emission_log_prob_stream(self, X):  # 舊版三項式 tensordot（可留作 CPU）
        Xg = self.gate(X)
        inv_var = torch.exp(-self.log_var); mu = self.mu
        x2 = torch.tensordot(torch.square(Xg), inv_var.t(), dims=([2],[0]))
        xm = torch.tensordot(Xg, (mu * inv_var).t(), dims=([2],[0]))
        const = (mu.square() * inv_var).sum(dim=1).view(1,1,-1)
        logdet = self.log_var.sum(dim=1).view(1,1,-1)
        quad = x2 - 2.0 * xm + const
        return -0.5 * (quad + logdet + self.D * math.log(2*math.pi))

    def forward_loglik_stream(self, X, mask=None):  # 原本 streaming 省記憶體
        B, T, _ = X.shape
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=X.device)
        log_pi, log_A = self.log_pi, self.log_A
        log_B0 = self.emission_log_prob_stream(X[:, :1, :])[:, 0, :]
        alpha = log_pi.view(1, -1) + log_B0
        alpha = torch.where(mask[:, 0, None], alpha, torch.full_like(alpha, float("-inf")))
        for t in range(1, T):
            log_Bt = self.emission_log_prob_stream(X[:, t:t+1, :])[:, 0, :]
            m = alpha.unsqueeze(2) + log_A.unsqueeze(0)
            alpha_t = torch.logsumexp(m, dim=1) + log_Bt
            alpha = torch.where(mask[:, t, None], alpha_t, alpha)
        return torch.logsumexp(alpha, dim=1)

    # -------- 統一的 loss，依裝置自動選最快路徑 --------
    def loss(self, X, mask=None, lam=1e-3):
        if X.is_cuda:
            log_B = self.emission_log_prob_matmul(X)              # 一次算完
            loglik = self.forward_loglik_from_logB(log_B, mask)
        else:
            loglik = self.forward_loglik_stream(X, mask)          # CPU 走省記憶體
        nll = -loglik.mean()
        reg = self.gate.regularizer()
        return nll + lam * reg, {'nll': float(nll.detach()), 'l0': float(reg.detach())}

# ========= 單裝置基準測試（不再改 threads！） =========
def bench_one(device, init_sd, X_cpu, mask_cpu, steps, lr, lam, use_compile=True, use_amp=True):
    amp_enabled = (use_amp and device.startswith("cuda"))
    model = GaussianHMM(N=init_sd["_meta_N"], D=init_sd["_meta_D"]).to(device)
    state = {k: v for k, v in init_sd.items() if not k.startswith("_meta_")}
    model.load_state_dict(state)

    compiled = False
    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, dynamic=False, mode="max-autotune")  # 固定形狀較易優化
            compiled = True
        except Exception as e:
            print(f"[{device}] torch.compile 失敗，退回原版：{e}")

    # 非同步拷貝 +（若可）釘選記憶體
    X = X_cpu.pin_memory().to(device, non_blocking=True)
    mask = mask_cpu.pin_memory().to(device, non_blocking=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []; last_loss = None; last_info = None
    t0 = time.perf_counter()
    for s in range(WARMUP + steps):
        if device.startswith("cuda"): torch.cuda.synchronize()
        t_s0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, info = model.loss(X, mask, lam=lam)   # 走 GPU 專用路徑
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss, info = model.loss(X, mask, lam=lam)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if device.startswith("cuda"): torch.cuda.synchronize()
        t_s1 = time.perf_counter()
        if s >= WARMUP: times.append(t_s1 - t_s0)
        last_loss, last_info = float(loss.detach()), info
        print(f"[{device}] step {s+1:03d}/{WARMUP+steps:03d} | loss={last_loss:.4f} (nll={info['nll']:.4f}, L0={info['l0']:.2f}) | step_sec={(t_s1 - t_s0):.4f}")

        # ⚠️ 不要在這裡 print（會同步 GPU、拖慢）
    total = time.perf_counter() - t0
    avg = sum(times) / max(1, len(times))
    res = {
        "device": device, "compiled": compiled, "amp": amp_enabled,
        "avg_step_sec": avg, "total_sec": total, "steps": steps, "warmup": WARMUP,
        "final_loss": last_loss, "final_nll": last_info["nll"], "final_L0": last_info["l0"],
    }
    if device.startswith("cuda"):
        res["peak_gpu_gib"] = torch.cuda.max_memory_allocated()/(1024**3)
    return res




# ========= 主程式 =========
def main():
    torch.manual_seed(0)
    # 資料（CPU 建一次）
    X_cpu = reade_parquet(FEATURE_PARQUET)
    lengths = torch.randint(low=T//2, high=T, size=(B,), device="cpu")
    mask_cpu = (torch.arange(T, device="cpu").view(1, T) < lengths.view(B, 1))

    # 基準初始化
    base = GaussianHMM(N, D).to("cpu")
    init_sd = copy.deepcopy(base.state_dict()); init_sd["_meta_N"] = N; init_sd["_meta_D"] = D

    results = []
    # 先 GPU
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} | AMP={USE_AMP} | compile={USE_COMPILE}")
        gpu_res = bench_one("cuda", init_sd, X_cpu, mask_cpu, STEPS, LR, LAM, USE_COMPILE, USE_AMP)
        results.append(gpu_res)
        print(f"[GPU] avg_step={gpu_res['avg_step_sec']:.4f}s | total={gpu_res['total_sec']:.2f}s | peak_mem={gpu_res.get('peak_gpu_gib',0):.2f} GiB")
    else:
        print("[GPU] CUDA 不可用，略過。")

    # 再 CPU（此處只印出目前 threads，不再重設）
    print(f"[CPU] threads: intra={torch.get_num_threads()}, inter={torch.get_num_interop_threads()} | compile={USE_COMPILE}")
    cpu_res = bench_one("cpu", init_sd, X_cpu, mask_cpu, STEPS, LR, LAM, USE_COMPILE, False)
    results.append(cpu_res)
    print(f"[CPU] avg_step={cpu_res['avg_step_sec']:.4f}s | total={cpu_res['total_sec']:.2f}s")

    # 總結
    print("\n========== Summary ==========")
    for r in results:
        tag = "GPU" if r["device"].startswith("cuda") else "CPU"
        extra = f", peak_mem={r.get('peak_gpu_gib',0):.2f}GiB" if tag=="GPU" else ""
        print(f"[{tag}] avg_step={r['avg_step_sec']:.4f}s, total={r['total_sec']:.2f}s, steps={r['steps']}, warmup={r['warmup']}, compile={r['compiled']}{extra}")
    if len(results) == 2 and results[0]["device"].startswith("cuda"):
        speedup = results[1]["avg_step_sec"] / max(1e-9, results[0]["avg_step_sec"])
        print(f"\nSpeedup (CPU avg_step / GPU avg_step) = {speedup:.2f}×  （>1 代表 GPU 較快）")

if __name__ == "__main__":
    main()
