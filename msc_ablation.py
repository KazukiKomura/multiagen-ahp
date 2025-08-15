# Ablation study for MSC (CI-lock + Median) with S-first simulator.
# We'll add two ablations:
#  A) No-Gossip (no synchronization step) with CI-lock ON
#  B) W-Direct cards (users act on abstract weights), with CI-lock ON and Median gossip
# We will also extend S-first to record CI-lock rejections.
#
# Then, we will run 24 trials per configuration and compare:
#  - rounds_to_consensus, not_converged rate
#  - rank_reversal_count
#  - CI_area
#  - final_consensus_gap
#  - w_update_reject_rate (for those with CI-lock)

import numpy as np
import pandas as pd
import math
# import matplotlib.pyplot as plt  # Comment out for CLI execution
from pathlib import Path
# --------- Core utility functions (moved from msc_phaseA_sim_S_first.py) ----------

def project_simplex_with_floor(v, lam):
    m = v.shape[0]
    s = 1.0 - m * lam
    u = v - lam
    u_sorted = np.sort(u)[::-1]
    cssv = np.cumsum(u_sorted)
    rho_idx = np.nonzero(u_sorted - (cssv - s) / (np.arange(m) + 1) > 0)[0]
    theta = (cssv[rho_idx[-1]] - s) / (rho_idx[-1] + 1.0) if len(rho_idx) else 0.0
    w = np.maximum(u - theta, 0.0) + lam
    w = np.maximum(w, lam); w = w / w.sum(); return w

def project_matrix_columns_with_floor(S, tau):
    A, C = S.shape
    out = np.zeros_like(S)
    for c in range(C):
        out[:, c] = project_simplex_with_floor(S[:, c], tau)
    return out

def pairwise_log_from_weights(w):
    lw = np.log(w + 1e-12)
    L = lw[None, :] - lw[:, None]
    np.fill_diagonal(L, 0.0); L = 0.5*(L - L.T); return L

def exp_pairwise_from_log(L):
    P = np.exp(L); P = (P + 1.0/np.maximum(P.T, 1e-12))*0.5
    np.fill_diagonal(P, 1.0); return P

def consistency_index_from_pairwise(P):
    n = P.shape[0]
    eigvals = np.linalg.eigvals(P)
    lam_max = np.max(eigvals.real)
    ci = float((lam_max - n) / (n - 1)) if n > 1 else 0.0
    return max(ci, 0.0)

def init_agent(K=3, A=3, lam=0.05, tau=0.05, ci_noise=0.2, rng=None):
    r = np.random.default_rng() if rng is None else rng
    w = r.dirichlet(np.ones(K)); w = project_simplex_with_floor(w, lam)
    S = np.zeros((A, K))
    for c in range(K):
        S[:, c] = project_simplex_with_floor(r.dirichlet(np.ones(A)), tau)
    L = pairwise_log_from_weights(w)
    for i in range(K):
        for j in range(i+1, K):
            eps = r.normal(0.0, ci_noise)
            L[i, j] += eps; L[j, i] -= eps
    P = exp_pairwise_from_log(L); CI = consistency_index_from_pairwise(P)
    attn = np.log(w + 1e-12).copy()
    return {"w": w, "S": S, "L": L, "P": P, "CI": CI, "attn": attn, "top1": None}

def alt_scores(agent): return agent["S"] @ agent["w"]
def top1_alt(agent): return int(np.argmax(alt_scores(agent)))

def apply_S_pairwise_card(S, c, x, y, eta, tau):
    S2 = S.copy()
    S2[x, c] *= math.exp(+eta); S2[y, c] *= math.exp(-eta)
    return project_matrix_columns_with_floor(S2, tau)

def gossip_step_WS(agents, mode, alpha, lam, tau):
    W = np.array([ag["w"] for ag in agents])
    S_stack = np.stack([ag["S"] for ag in agents], axis=0)
    if mode == "mean":
        agg_w = W.mean(axis=0); agg_S = S_stack.mean(axis=0)
    elif mode == "median":
        agg_w = np.median(W, axis=0); agg_S = np.median(S_stack, axis=0)
    else:
        raise ValueError("mode")
    agg_w = project_simplex_with_floor(agg_w, lam)
    agg_S = project_matrix_columns_with_floor(agg_S, tau)
    for ag in agents:
        ag["w"] = project_simplex_with_floor((1-alpha)*ag["w"] + alpha*agg_w, lam)
        ag["S"] = project_matrix_columns_with_floor((1-alpha)*ag["S"] + alpha*agg_S, tau)

class UCB1:
    def __init__(self, n_arms): self.n=np.zeros(n_arms,int); self.mu=np.zeros(n_arms,float); self.t=0
    def select(self):
        self.t+=1
        for a in range(len(self.n)):
            if self.n[a]==0: return a
        ucb=self.mu+np.sqrt(2.0*np.log(self.t)/(self.n+1e-12))
        return int(np.argmax(ucb))
    def update(self,a,r): self.n[a]+=1; self.mu[a]+= (r-self.mu[a])/self.n[a]

# --------- Extended S-first simulator that returns CI-lock rejection stats ----------
def simulate_trial_Sfirst_v2(condition: str,
                             N_agents=5, K=3, A=3,
                             lam=0.05, tau=0.05,
                             ci_noise=0.2,
                             kappa=0.6,          # attention increment scale
                             alpha_w=0.4,        # w step size (with backtracking)
                             beta=0.5,           # backtrack factor
                             gamma=0.3,          # L blend ratio when CI-lock
                             step_ci_noise=0.05, # L noise when no CI-lock
                             gossip_alpha=0.5,
                             epsilon=0.15,
                             max_rounds=50,
                             seed=None):
    r = np.random.default_rng(seed)
    agents = [init_agent(K, A, lam, tau, ci_noise, rng=r) for _ in range(N_agents)]
    for ag in agents:
        ag["top1"] = top1_alt(ag)

    # Arms: (c, x, y, m) with x != y, m in {0.15, 0.30}
    mags = [0.15, 0.30]
    arms = []
    for c in range(K):
        for x in range(A):
            for y in range(A):
                if x == y: continue
                for m in mags:
                    arms.append((c, x, y, m))
    bandit = UCB1(len(arms))

    def avg_CI():
        return float(np.mean([ag["CI"] for ag in agents]))

    def consensus_gap():
        W = np.array([ag["w"] for ag in agents])
        n = W.shape[0]; total = 0.0; cnt = 0
        for i in range(n):
            for j in range(i+1, n):
                total += np.abs(W[i]-W[j]).sum(); cnt += 1
        return float(total / max(cnt, 1))

    base_CI = avg_CI()
    ci_area = 0.0
    prev_gap = consensus_gap()
    prev_tops = [ag["top1"] for ag in agents]
    rank_reversals = 0

    use_median = condition in ("M1", "MSC")
    use_ci_lock = condition in ("M2", "MSC")
    use_gossip = True
    if condition == "NoGossip":
        use_gossip = False
        use_median = True  # irrelevant, but keep "median" semantics

    converged = False
    rounds_used = max_rounds

    w_update_attempts = 0
    w_update_rejects = 0

    for t in range(1, max_rounds+1):
        # 1) RL chooses an S-card
        arm = bandit.select()
        c, x, y, m = arms[arm]
        eta = m

        # 2) Apply S-card to all agents (human-judgable change)
        for ag in agents:
            ag["S"] = apply_S_pairwise_card(ag["S"], c, x, y, eta, tau)

        # 3) Acceptance-aware attention bump
        for ag in agents:
            accept = 1 if (ag["w"][c] >= 1.0/K or ag["S"][x, c] >= ag["S"][y, c]) else 0
            ag["attn"][c] += kappa * m * accept

        # 4) Propose w update from attention with/without CI-lock
        w_update_attempts += 1
        alpha_try = alpha_w
        applied_w = False
        while True:
            new_states = []
            for ag in agents:
                attn = ag["attn"]
                w_target = np.exp(attn - np.max(attn)); w_target = w_target / w_target.sum()
                w_target = project_simplex_with_floor(w_target, lam)
                w_prop = project_simplex_with_floor((1-alpha_try)*ag["w"] + alpha_try*w_target, lam)

                if use_ci_lock:
                    L_new = (1.0 - gamma)*ag["L"] + gamma*pairwise_log_from_weights(w_prop)
                    L_new = 0.5 * (L_new - L_new.T)
                else:
                    L_new = ag["L"].copy()
                    for i in range(K):
                        for j in range(i+1, K):
                            eps = r.normal(0.0, step_ci_noise)
                            L_new[i, j] += eps; L_new[j, i] -= eps
                    L_new = 0.5 * (L_new - L_new.T)

                P_new = exp_pairwise_from_log(L_new)
                CI_new = consistency_index_from_pairwise(P_new)
                new_states.append((w_prop, L_new, P_new, CI_new))

            new_avg_CI = float(np.mean([ns[3] for ns in new_states]))
            if use_ci_lock:
                if new_avg_CI <= avg_CI() + 1e-12:
                    applied_w = True
                    break
                else:
                    alpha_try *= beta
                    if alpha_try < 1e-3:
                        applied_w = False
                        break
            else:
                applied_w = True
                break

        if not applied_w and use_ci_lock:
            w_update_rejects += 1

        if applied_w:
            for ag, (w_prop, L_new, P_new, CI_new) in zip(agents, new_states):
                ag["w"] = w_prop; ag["L"] = L_new; ag["P"] = P_new; ag["CI"] = CI_new

        # 5) Gossip
        if use_gossip:
            gossip_mode = "median" if use_median else "mean"
            gossip_step_WS(agents, gossip_mode, gossip_alpha, lam, tau)

        # 6) Metrics & bandit reward
        current_tops = []
        flips = 0
        for idx, ag in enumerate(agents):
            t1 = top1_alt(ag)
            current_tops.append(t1)
            if t1 != prev_tops[idx]:
                flips += 1
            ag["top1"] = t1
        rank_reversals += flips

        aci = avg_CI()
        ci_area += max(aci - base_CI, 0.0)

        gap = consensus_gap()
        delta_consensus = prev_gap - gap
        reward = delta_consensus - (1.0 if flips > 0 else 0.0) - (0.5 if (use_ci_lock and not applied_w) else 0.0)
        bandit.update(arm, reward)

        all_same_top1 = len(set(current_tops)) == 1
        if (gap <= epsilon) and all_same_top1:
            converged = True
            rounds_used = t
            break

        prev_gap = gap
        prev_tops = current_tops

    rej_rate = (w_update_rejects / max(w_update_attempts,1)) if use_ci_lock else 0.0
    return {
        "condition": condition,
        "rounds_to_consensus": rounds_used,
        "not_converged": int(not converged),
        "rank_reversal_count": rank_reversals,
        "CI_area": ci_area,
        "final_consensus_gap": prev_gap,
        "w_update_reject_rate": rej_rate
    }

# --------- W-direct simulator (cards act on w; abstract; CI-lock ON) ----------
def apply_w_card(w: np.ndarray, cidx: int, sign: int, eta: float, lam: float) -> np.ndarray:
    w_new = w.copy()
    w_new[cidx] *= math.exp(sign * eta)
    w_new = w_new / w_new.sum()
    w_new = project_simplex_with_floor(w_new, lam)
    return w_new

def simulate_trial_Wdirect_MSC(N_agents=5, K=3, A=3,
                               lam=0.05, tau=0.05,
                               ci_noise=0.2,
                               eta0=0.3, beta=0.5,
                               gamma=0.3,
                               gossip_alpha=0.5,
                               epsilon=0.15, max_rounds=50, seed=None):
    r = np.random.default_rng(seed)
    agents = [init_agent(K, A, lam, tau, ci_noise, rng=r) for _ in range(N_agents)]
    for ag in agents:
        ag["top1"] = top1_alt(ag)

    # arms: w_c up/down with small/medium magnitudes
    mags = [0.15, 0.30]
    arms = []
    for c in range(K):
        for s in (+1, -1):
            for m in mags:
                arms.append((c, s, m))
    bandit = UCB1(len(arms))

    def avg_CI():
        return float(np.mean([ag["CI"] for ag in agents]))

    def consensus_gap():
        W = np.array([ag["w"] for ag in agents])
        n = W.shape[0]; total = 0.0; cnt = 0
        for i in range(n):
            for j in range(i+1, n):
                total += np.abs(W[i]-W[j]).sum(); cnt += 1
        return float(total / max(cnt, 1))

    base_CI = avg_CI()
    ci_area = 0.0
    prev_gap = consensus_gap()
    prev_tops = [ag["top1"] for ag in agents]
    rank_reversals = 0
    w_update_attempts = 0
    w_update_rejects = 0

    for t in range(1, max_rounds+1):
        arm = bandit.select()
        cidx, sign, m = arms[arm]

        # CI-lock backtracking on w directly
        eta = m
        w_update_attempts += 1
        applied = False
        while True:
            candidates = []
            for ag in agents:
                w_prop = apply_w_card(ag["w"], cidx, sign, eta, lam)
                L_new = (1.0 - gamma)*ag["L"] + gamma*pairwise_log_from_weights(w_prop)
                L_new = 0.5*(L_new - L_new.T)
                P_new = exp_pairwise_from_log(L_new)
                CI_new = consistency_index_from_pairwise(P_new)
                candidates.append((w_prop, L_new, P_new, CI_new))
            new_avg_CI = float(np.mean([c[3] for c in candidates]))
            if new_avg_CI <= avg_CI() + 1e-12:
                applied = True
                break
            else:
                eta *= beta
                if eta < 1e-3:
                    applied = False
                    break
        if not applied:
            w_update_rejects += 1
        if applied:
            for ag, (w_prop, L_new, P_new, CI_new) in zip(agents, candidates):
                ag["w"] = w_prop; ag["L"] = L_new; ag["P"] = P_new; ag["CI"] = CI_new

        # Gossip median on w & S
        gossip_step_WS(agents, "median", gossip_alpha, lam, tau)

        # Metrics & reward
        current_tops=[]; flips=0
        for idx, ag in enumerate(agents):
            t1 = top1_alt(ag)
            current_tops.append(t1)
            if t1 != prev_tops[idx]:
                flips += 1
            ag["top1"] = t1
        rank_reversals += flips

        aci = avg_CI()
        ci_area += max(aci - base_CI, 0.0)

        gap = consensus_gap()
        delta_consensus = prev_gap - gap
        reward = delta_consensus - (1.0 if flips > 0 else 0.0) - (0.5 if not applied else 0.0)
        bandit.update(arm, reward)

        all_same_top1 = len(set(current_tops)) == 1
        if (gap <= epsilon) and all_same_top1:
            rounds_used = t
            return {
                "condition":"Wdirect_MSC",
                "rounds_to_consensus": rounds_used,
                "not_converged": 0,
                "rank_reversal_count": rank_reversals,
                "CI_area": ci_area,
                "final_consensus_gap": gap,
                "w_update_reject_rate": w_update_rejects / max(w_update_attempts,1)
            }

        prev_gap = gap
        prev_tops = current_tops

    # not converged
    return {
        "condition":"Wdirect_MSC",
        "rounds_to_consensus": max_rounds,
        "not_converged": 1,
        "rank_reversal_count": rank_reversals,
        "CI_area": ci_area,
        "final_consensus_gap": prev_gap,
        "w_update_reject_rate": w_update_rejects / max(w_update_attempts,1)
    }

# --------- Runner for all configs ----------
def run_ablation(n_trials=24, seed=777):
    ss = np.random.SeedSequence(seed)
    child = ss.spawn(6 * n_trials)  # enough entropy
    results = []

    configs = ["Base", "M1", "M2", "MSC", "NoGossip", "Wdirect_MSC"]
    k = 0
    for cfg in configs:
        for _ in range(n_trials):
            rs = child[k].generate_state(1)[0]; k += 1
            if cfg in ["Base","M1","M2","MSC","NoGossip"]:
                out = simulate_trial_Sfirst_v2(cfg, seed=rs)
            else:
                out = simulate_trial_Wdirect_MSC(seed=rs)
            results.append(out)

    df = pd.DataFrame(results)
    summary = df.groupby("condition").agg(
        trials=("condition","count"),
        mean_rounds=("rounds_to_consensus","mean"),
        median_rounds=("rounds_to_consensus","median"),
        not_conv_rate=("not_converged","mean"),
        mean_rank_reversals=("rank_reversal_count","mean"),
        mean_CI_area=("CI_area","mean"),
        mean_final_gap=("final_consensus_gap","mean"),
        mean_w_reject_rate=("w_update_reject_rate","mean")
    ).reset_index()
    return df, summary

# ---- Execute ablation ----
df_ab, sm_ab = run_ablation(n_trials=24, seed=777)

# Show summary table
print("\n=== MSC Ablation Study Results ===")
print(sm_ab.to_string(index=False))

# Summary metrics (plotting disabled for CLI execution)
print("\n=== Additional Metrics Summary ===")
conditions = ["Base","M1","M2","MSC","NoGossip","Wdirect_MSC"]
print(f"{'Condition':<12} {'Rounds(avg)':<12} {'RankRev(avg)':<12} {'CI_Area(avg)':<12} {'Gap(avg)':<12} {'Reject(avg)':<12}")
print("-" * 80)
for c in conditions:
    subset = df_ab[df_ab.condition==c]
    rounds_avg = subset["rounds_to_consensus"].mean()
    rr_avg = subset["rank_reversal_count"].mean()
    ci_avg = subset["CI_area"].mean()
    gap_avg = subset["final_consensus_gap"].mean()
    rej_avg = subset["w_update_reject_rate"].mean()
    print(f"{c:<12} {rounds_avg:<12.2f} {rr_avg:<12.2f} {ci_avg:<12.3f} {gap_avg:<12.3f} {rej_avg:<12.3f}")

# Save outputs
out_dir = Path("./results")
out_dir.mkdir(parents=True, exist_ok=True)
df_ab.to_csv(out_dir/"ablation_trials.csv", index=False)
sm_ab.to_csv(out_dir/"ablation_summary.csv", index=False)
print(f"\nSaved results to: {out_dir.absolute()}")
print(f"- {out_dir}/ablation_trials.csv")
print(f"- {out_dir}/ablation_summary.csv")
