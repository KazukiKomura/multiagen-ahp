#!/usr/bin/env bash
# RBCS experiment runner (train -> select -> eval -> summary)
# usage: ./run_experiments.sh
set -Eeuo pipefail

### ==== 設定（必要に応じて変更） ====
PROJ_DIR="/Users/kazukikomura/Developer/research/multiagentahp"
PYTHON="python"                             # 別のPython使うならパス指定
SCRIPT="${PROJ_DIR}/rbcs_full_enhanced.py"  # 実行対象

# 学習ループ
SEEDS=("0" "1" "2" "3" "4")
EPISODES_TRAIN=80
N_AGENTS=5
N_ALTS=5
N_CRITS=5
TMAX=200
MODE_TRAIN="full"

# LinUCB / 探索パラメタ
UCB_ALPHA=0.6
SOFTMAX_TEMP=0.7

# 安全柵
SAFETY_CI_MAX=0.03
SAFETY_TAU_MAX=0.05
SAFETY_GINI_MAX=0.03

# OPE（推定）
OPE_W_CLIP=5.0
OPE_RIDGE_LAM=5.0
OPE_MIN_PER_ACTION=20

# MLflow（trueなら --mlflow を付与）
USE_MLFLOW=true
# export MLFLOW_TRACKING_URI="file:${PROJ_DIR}/mlruns"  # 必要なら有効化

# 選抜（SNIPSブートストラップ）
RUNS_GLOB="${PROJ_DIR}/runs/2025*"   # 過去も含めベスト選抜。直近に絞るなら適宜変更
N_BOOT=1000

# 評価
EVAL_EPISODES=50
EVAL_MODES=("full" "explain" "baseline")

### ==== ここから自動 ====
cd "${PROJ_DIR}"

# venv があれば有効化
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

# スクリプト存在確認
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: not found -> ${SCRIPT}" >&2
  exit 1
fi

# MLflowフラグ
MLFLOW_FLAG=()
if [[ "${USE_MLFLOW}" == "true" ]]; then
  MLFLOW_FLAG=(--mlflow)
fi

echo "=== TRAIN (seeds: ${SEEDS[*]}) ==="
for s in "${SEEDS[@]}"; do
  echo "--- seed ${s} ---"
  # Hydraを使わずargparseで実行（環境変数で明示オフ）
  RBCS_NO_HYDRA=1 "${PYTHON}" "${SCRIPT}" \
    --task train \
    --mode "${MODE_TRAIN}" \
    --episodes "${EPISODES_TRAIN}" \
    --n_agents "${N_AGENTS}" \
    --n_alternatives "${N_ALTS}" \
    --n_criteria "${N_CRITS}" \
    --Tmax "${TMAX}" \
    --seed "${s}" \
    --ucb_alpha "${UCB_ALPHA}" \
    --ucb_softmax_temp "${SOFTMAX_TEMP}" \
    --safety_ci_increase_max "${SAFETY_CI_MAX}" \
    --safety_tau_drop_max "${SAFETY_TAU_MAX}" \
    --safety_gini_increase_max "${SAFETY_GINI_MAX}" \
    --ope_w_clip "${OPE_W_CLIP}" \
    --ope_ridge_lam "${OPE_RIDGE_LAM}" \
    --n_boot "${N_BOOT}" \
    "${MLFLOW_FLAG[@]}" | tee -a "runs/train_seed${s}.log"
done

echo "=== SELECT (boot=${N_BOOT}, glob=${RUNS_GLOB}) ==="
RBCS_NO_HYDRA=1 "${PYTHON}" "${SCRIPT}" \
  --task select \
  --runs_glob "${RUNS_GLOB}" \
  --n_boot "${N_BOOT}"

FROZEN="runs/frozen_best/frozen_policy.json"
if [[ ! -f "${FROZEN}" ]]; then
  echo "ERROR: frozen policy not found at ${FROZEN}" >&2
  exit 1
fi

echo "=== EVAL (episodes=${EVAL_EPISODES}) ==="
for m in "${EVAL_MODES[@]}"; do
  echo "--- mode ${m} ---"
  RBCS_NO_HYDRA=1 "${PYTHON}" "${SCRIPT}" \
    --task eval \
    --mode "${m}" \
    --episodes "${EVAL_EPISODES}" \
    --policy_path "${FROZEN}"
done

echo "=== SUMMARY (CSV出力) ==="
# runs/*/events.jsonl を集約して runs/summary.csv を作成
"${PYTHON}" - <<'PY'
import json, glob, os, csv

rows = []
for path in sorted(glob.glob("runs/*/events.jsonl")):
    run_dir = os.path.dirname(path)
    phase = "train"
    sum_R, sum_steps = 0.0, 0
    tau, ci_s, gini = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "step":
                tau.append(ev.get("tau"))
                ci_s.append(ev.get("ci_s", ev.get("mean_ci_saaty")))
                gini.append(ev.get("gini"))
                if ev.get("phase") == "eval":
                    phase = "eval"
            if ev.get("type") in ("episode_end","episode_end_eval"):
                sum_R += float(ev.get("return", 0.0))
                sum_steps += int(ev.get("steps", 0))
    if tau:
        rows.append([
            run_dir, phase,
            f"{sum_R:.6f}", sum_steps,
            f"{(sum(tau)/len(tau)):.6f}",
            f"{(sum(ci_s)/len(ci_s)):.6f}",
            f"{(sum(gini)/len(gini)):.6f}",
        ])

os.makedirs("runs", exist_ok=True)
out = "runs/summary.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["run","phase","sum_return","sum_steps","mean_tau","mean_ci_s","mean_gini"])
    w.writerows(rows)
print(f"[summary] wrote {out}  (rows={len(rows)})")
PY

echo "=== DONE ==="
echo "・凍結ポリシー: ${FROZEN}"
echo "・集計CSV: runs/summary.csv"
