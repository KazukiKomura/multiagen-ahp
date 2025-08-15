# MSC Simulation with "No Hard Reject": Applied / Clipped / Calibrate / Blocked
# Includes Strict mode, consent-gated median sync, and PRE-SYNC consensus check.
# Outputs:
#  - ablation_trials_calibrate.csv
#  - ablation_summary_calibrate.csv
#  - 5 bar charts (Rounds, Rank Reversal, CI Area, Final Gap, Calibrate Resolved Rate)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os, math, random

rng = np.random.default_rng(7)
np.set_printoptions(precision=4, suppress=True)

ALTS = ["X","Y","Z"]
CRITS = ["cost","quality","delivery"]
A2I = {a:i for i,a in enumerate(ALTS)}
C2I = {c:i for i,c in enumerate(CRITS)}

def project_simplex_with_floor(v, lam=0.05):
    v = np.maximum(v, lam)
    v = v / v.sum()
    v = np.maximum(v, lam)
    v = v / v.sum()
    return v

def normalize_col_with_floor(col, tau=0.05):
    col = np.maximum(col, tau)
    col = col / col.sum()
    return col

def order_from_scores(scores):
    return list(np.argsort(-scores))

def kendall_tau_distance(order1, order2):
    pairs = [(0,1),(0,2),(1,2)]
    r1 = {order1[i]: i for i in range(len(order1))}
    r2 = {order2[i]: i for i in range(len(order2))}
    disc=0
    for i,j in pairs:
        s1 = np.sign(r1[i]-r1[j]); s2 = np.sign(r2[i]-r2[j])
        if s1 != s2: disc += 1
    return disc/3.0

def compute_agent_CI(w,S):
    U = S @ w
    oU = order_from_scores(U)
    res = 0.0
    for c in range(S.shape[1]):
        oc = order_from_scores(S[:,c])
        res += kendall_tau_distance(oU, oc)
    return res / S.shape[1]

def consensus_gap_W(W_list):
    n=len(W_list)
    if n<2: return 0.0
    tot=0.0; cnt=0
    for i in range(n):
        for j in range(i+1,n):
            tot += np.abs(W_list[i]-W_list[j]).sum(); cnt+=1
    return tot/cnt

def top1_alt(w,S):
    U = S @ w
    return int(np.argmax(U))

def make_cards_pool():
    pool = [
        (C2I["quality"],  A2I["X"], A2I["Y"], +0.06),
        (C2I["cost"],     A2I["Z"], A2I["Y"], +0.06),
        (C2I["delivery"], A2I["Y"], A2I["X"], +0.05),
        (C2I["quality"],  A2I["X"], A2I["Z"], +0.05),
        (C2I["cost"],     A2I["X"], A2I["Z"], +0.06),
        (C2I["delivery"], A2I["Z"], A2I["Y"], +0.05),
        (C2I["quality"],  A2I["Y"], A2I["Z"], -0.05),
    ]
    return pool

def consent_rule(S, w, card, noise=0.05):
    c,a,b,delta = card
    local_ok = (S[a,c] >= S[b,c]) if delta>0 else (S[a,c] <= S[b,c])
    U = S @ w
    overall_ok = (U[a] >= U[b]) if delta>0 else (U[a] <= U[b])
    base_prob = 0.85 if (local_ok or overall_ok) else 0.15
    p = base_prob*(1-noise) + noise*0.5
    return rng.random() < p

def try_apply_card(w, S, card, eta, w_step, tau=0.05, lam=0.05):
    c,a,b,delta = card
    S2 = S.copy()
    S2[a,c] += delta*eta
    S2[b,c] -= delta*eta
    S2[:,c] = normalize_col_with_floor(S2[:,c], tau=tau)
    w2 = w.copy()
    wd = np.zeros_like(w2); wd[c] = np.sign(delta)
    w2 = project_simplex_with_floor(w2 + w_step*eta*wd, lam=lam)
    return w2, S2

def calibrate_small_fix(w, S, card, eta_try, w_step, fix_step=0.02):
    c,a,b,delta = card
    CI0 = compute_agent_CI(w,S)
    U = S @ w
    oU = order_from_scores(U)
    worst_c = None; worst_d = -1.0
    for cc in range(S.shape[1]):
        oc = order_from_scores(S[:,cc])
        d = kendall_tau_distance(oU, oc)
        if d > worst_d:
            worst_d = d; worst_c = cc
    S_fix = S.copy()
    top = oU[0]; bot = oU[-1]
    S_fix[top,worst_c] += fix_step
    S_fix[bot,worst_c] -= fix_step
    S_fix[:,worst_c] = normalize_col_with_floor(S_fix[:,worst_c], tau=0.05)
    w2, S2 = try_apply_card(w, S_fix, card, eta_try, w_step)
    CI1 = compute_agent_CI(w2, S2)
    if CI1 - CI0 <= 1e-12:
        return w2, S2, True
    return w, S, False

def apply_with_CI_pathways(w, S, card, eta0=1.0, beta=0.5, eta_min=0.05, w_step=0.03):
    CI0 = compute_agent_CI(w,S)
    eta = eta0
    while eta >= eta_min:
        w2, S2 = try_apply_card(w, S, card, eta, w_step)
        CI1 = compute_agent_CI(w2, S2)
        if CI1 - CI0 <= 1e-12:
            status = "Applied" if abs(eta-eta0) < 1e-12 else "Clipped"
            return w2, S2, status, {"eta":eta, "calibrated":False}
        eta *= beta
    w2, S2, ok = calibrate_small_fix(w, S, card, eta_try=eta_min, w_step=w_step, fix_step=0.02)
    if ok:
        return w2, S2, "Calibrate", {"eta":eta_min, "calibrated":True}
    return w, S, "Blocked", {"eta":0.0, "calibrated":False}

def median_sync(W_list, coords=None, alpha=0.5):
    if coords is None:
        coords = list(range(len(W_list[0])))
    W = np.vstack(W_list)
    m = np.median(W, axis=0)
    out = []
    for v in W_list:
        u = v.copy()
        for k in coords:
            u[k] = (1-alpha)*v[k] + alpha*m[k]
        u = project_simplex_with_floor(u, lam=0.05)
        out.append(u)
    return out

def median_sync_strict(W_list, consent_mask, coords=None, alpha=0.5):
    if coords is None:
        coords = list(range(len(W_list[0])))
    W = np.vstack(W_list)
    m = np.median(W, axis=0)
    out = []
    for v,ok in zip(W_list, consent_mask):
        u = v.copy()
        if ok:
            for k in coords:
                u[k] = (1-alpha)*v[k] + alpha*m[k]
        u = project_simplex_with_floor(u, lam=0.05)
        out.append(u)
    return out

def consensus_reached(W_list, S_list, eps=0.15):
    gap = consensus_gap_W(W_list)
    if gap > eps: 
        return False, gap, None
    tops = [top1_alt(w,S) for w,S in zip(W_list, S_list)]
    if len(set(tops)) == 1:
        return True, gap, tops[0]
    return False, gap, None

def run_trial(condition, rounds_max=18, seed=None):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)
    N=5
    W=[]; S=[]
    for _ in range(N):
        w = rng.dirichlet(np.ones(3))
        w = project_simplex_with_floor(w, lam=0.05)
        W.append(w)
        cols=[]
        for _c in range(3):
            col = rng.dirichlet(np.ones(3))
            col = normalize_col_with_floor(col, tau=0.05)
            cols.append(col)
        Smat = np.column_stack(cols)
        S.append(Smat)

    cards = make_cards_pool()
    seq = [cards[i % len(cards)] for i in range(rounds_max)]

    use_ci_lock = condition in ["M2","MSC","MSC_Strict","NoGossip","Wdirect_MSC"]
    use_median  = condition in ["M1","MSC","MSC_Strict"]
    use_mean    = condition in ["Base","M2"]
    use_sync    = use_median or use_mean
    strict      = condition == "MSC_Strict"
    w_direct    = condition == "Wdirect_MSC"

    alpha=0.5
    rounds=0
    prev_group_top=None
    rank_reversals=0
    ci_area=0.0
    final_gap=0.0
    not_converged=0
    status_counts = Counter()

    for r in range(rounds_max):
        rounds = r+1
        c,a,b,delta = seq[r]
        pre_CI = [compute_agent_CI(W[i],S[i]) for i in range(N)]

        accepted_any=False
        touched=set()
        consent_mask=[False]*N

        for i in range(N):
            agree = consent_rule(S[i], W[i], (c,a,b,delta), noise=0.05)
            if not agree: 
                continue
            consent_mask[i]=True
            accepted_any=True
            touched.add(c)

            if use_ci_lock:
                wstep = 0.06 if w_direct else 0.03
                Wi,Si,status,meta = apply_with_CI_pathways(W[i], S[i], (c,a,b,delta),
                                                           eta0=1.0, beta=0.5, eta_min=0.05, w_step=wstep)
                W[i], S[i] = Wi, Si
                status_counts[status]+=1
            else:
                Si = S[i].copy()
                Si[a,c]+=delta; Si[b,c]-=delta
                Si[:,c]=normalize_col_with_floor(Si[:,c], tau=0.05)
                Wi = W[i].copy()
                wd = np.zeros_like(Wi); wd[c]=np.sign(delta)
                wstep = 0.06 if w_direct else 0.02
                Wi = project_simplex_with_floor(Wi + wstep*wd, lam=0.05)
                W[i],S[i]=Wi,Si
                status_counts["Applied"]+=1

        post_CI = [compute_agent_CI(W[i],S[i]) for i in range(N)]
        for i in range(N):
            d = post_CI[i]-pre_CI[i]
            if d>0: ci_area += d

        pre_gap = consensus_gap_W(W)
        group_top = Counter([top1_alt(W[i],S[i]) for i in range(N)]).most_common(1)[0][0]
        if prev_group_top is not None and group_top != prev_group_top:
            rank_reversals += 1
        prev_group_top = group_top

        reached, gap_check, top1 = consensus_reached(W,S, eps=0.15)
        if reached:
            final_gap = gap_check
            break

        if condition!="NoGossip" and accepted_any and len(touched)>0 and use_sync:
            coords = list(sorted(list(touched)))
            if use_median:
                if strict:
                    W = median_sync_strict(W, consent_mask, coords=coords, alpha=alpha)
                else:
                    W = median_sync(W, coords=coords, alpha=alpha)
            elif use_mean:
                Wmat = np.vstack(W)
                mean_vec = np.mean(Wmat, axis=0)
                W_new=[]
                for v in W:
                    u=v.copy()
                    for k in coords:
                        u[k]=(1-alpha)*v[k]+alpha*mean_vec[k]
                    u=project_simplex_with_floor(u, lam=0.05)
                    W_new.append(u)
                W=W_new

    final_gap = consensus_gap_W(W)
    reached, gap_check, top1 = consensus_reached(W,S, eps=0.15)
    if not reached:
        not_converged=1

    total_status = sum(status_counts.values()) if sum(status_counts.values())>0 else 1
    res = {
        "condition": condition,
        "rounds": rounds,
        "rank_reversal": rank_reversals,
        "ci_area": ci_area,
        "final_gap": final_gap,
        "not_converged": not_converged,
        "applied_rate": status_counts["Applied"]/total_status,
        "clipped_rate": status_counts["Clipped"]/total_status,
        "calibrate_rate": status_counts["Calibrate"]/total_status,
        "blocked_rate": status_counts["Blocked"]/total_status,
    }
    return res

def run_all(trials=24, rounds_max=18, seed0=20250816):
    conds=["Base","M1","M2","MSC","MSC_Strict","NoGossip","Wdirect_MSC"]
    rows=[]
    for ci,cond in enumerate(conds):
        for t in range(trials):
            res = run_trial(cond, rounds_max=rounds_max, seed=seed0+100*ci+t)
            res["trial"]=t+1
            rows.append(res)
    return pd.DataFrame(rows)

TRIALS=24
ROUNDS_MAX=18

df_trials = run_all(trials=TRIALS, rounds_max=ROUNDS_MAX, seed0=20250816)
df_summary = df_trials.groupby("condition").agg(
    mean_rounds=("rounds","mean"),
    mean_rank_reversal=("rank_reversal","mean"),
    mean_ci_area=("ci_area","mean"),
    mean_final_gap=("final_gap","mean"),
    not_converged_rate=("not_converged","mean"),
    mean_applied_rate=("applied_rate","mean"),
    mean_clipped_rate=("clipped_rate","mean"),
    mean_calibrate_rate=("calibrate_rate","mean"),
    mean_blocked_rate=("blocked_rate","mean"),
    n=("trial","count"),
).reset_index()

os.makedirs("/mnt/data", exist_ok=True)
path_trials = "/mnt/data/ablation_trials_calibrate.csv"
path_summary = "/mnt/data/ablation_summary_calibrate.csv"
df_trials.to_csv(path_trials, index=False)
df_summary.to_csv(path_summary, index=False)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Trials (No-Hard-Reject)", df_trials)
caas_jupyter_tools.display_dataframe_to_user("Summary (No-Hard-Reject)", df_summary)

plt.figure()
dfp = df_summary.sort_values("mean_rounds")
plt.bar(dfp["condition"], dfp["mean_rounds"])
plt.title("Rounds to Consensus (mean)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure()
dfp = df_summary.sort_values("mean_rank_reversal")
plt.bar(dfp["condition"], dfp["mean_rank_reversal"])
plt.title("Rank Reversal (mean)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure()
dfp = df_summary.sort_values("mean_ci_area")
plt.bar(dfp["condition"], dfp["mean_ci_area"])
plt.title("CI Area (mean)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure()
dfp = df_summary.sort_values("mean_final_gap")
plt.bar(dfp["condition"], dfp["mean_final_gap"])
plt.title("Final Consensus Gap (mean)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure()
dfp = df_summary.sort_values("mean_calibrate_rate")
plt.bar(dfp["condition"], dfp["mean_calibrate_rate"])
plt.title("Calibrate Rate (mean)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("CSV saved:")
print(" - Trials:", path_trials)
print(" - Summary:", path_summary)
