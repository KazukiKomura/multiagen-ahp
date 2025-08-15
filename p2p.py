
# p2p.py
# 完全実装版：AHP(ペア比較+CI)を各個体に持たせ、VAF+AGAU+Gossipを回しつつ
# PPOで「論点選択」方策を学習。Veto/フロア/CIアウェアで単一基準崩壊を抑制。
# deps: numpy, networkx, torch, (pandas/matplotlibは可視化スクリプト側)

import os, math, random, numpy as np, networkx as nx
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import argparse, json, time

# ===================== Utility =====================

rng = np.random.default_rng(42)
torch.manual_seed(42)

def normalize_simplex(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 1e-12, None)
    return v / v.sum()

def project_simplex_with_floor(v: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """floor_i を下回らず、かつ総和=1 となるように射影する。余剰質量を (v-floor)^+ に比例配分。"""
    v = np.asarray(v, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    assert v.shape == floor.shape, "floor shape mismatch"
    assert np.all(floor >= 0.0), "floor must be non-negative"
    s_floor = float(floor.sum())
    # 余剰質量 r
    r = 1.0 - s_floor
    if r < 0.0:
        # floor合計が1を超える場合 -> 等分でOK (理論上起こらない)
        return normalize_simplex(floor)
    surplus = np.maximum(v - floor, 0.0)
    tot = float(surplus.sum())
    if tot < 1e-12:
        # v がすべて floor 以下
        n = len(v)
        return floor + r * (np.ones(n, dtype=np.float64) / n)
    out = floor + (r * surplus / tot)
    # 数値安定化
    out = np.clip(out, floor, 1.0)
    return normalize_simplex(out)

def pearson_scaled(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    sx, sy = x.std(), y.std()
    if not np.isfinite(sx) or not np.isfinite(sy) or sx < 1e-9 or sy < 1e-9:
        return 0.5
    c = np.corrcoef(x, y)
    r = c[0,1]
    if not np.isfinite(r): return 0.5
    return float((r + 1.0) * 0.5)

def weight_entropy(w: np.ndarray) -> float:
    w = np.clip(w, 1e-12, 1.0)
    H = -np.sum(w * np.log(w))
    return H / np.log(len(w))  # [0,1] 正規化

# === Preflight / invariant check toggle via env or CLI ===
CHECK_INVARIANTS_DEFAULT = bool(int(os.getenv("CHECK_INVARIANTS", "1")))
# ============== Invariant Checks (Preflight) ==============

def assert_state(node_id: int, st: "NodeState"):
    # w: simplex + floor
    assert np.isfinite(st.w).all(), f"w nan/inf at node {node_id}"
    assert abs(st.w.sum() - 1.0) < 1e-6, f"w not simplex at node {node_id}"
    assert np.all(st.w >= st.w_floor - 1e-12), f"w below floor at node {node_id}"
    # S: per-criterion simplex + tau floor
    A, C = st.S.shape
    for c in range(C):
        col = st.S[:, c]
        assert np.isfinite(col).all(), f"S nan/inf at node {node_id}, c={c}"
        assert abs(col.sum() - 1.0) < 1e-6, f"S col not simplex at node {node_id}, c={c}"
        assert np.all(col >= st.tau[:, c] - 1e-12), f"S below tau at node {node_id}, c={c}"
    # CI finite
    assert np.isfinite(st.CIcrit), f"CI not finite at node {node_id}"

def apply_floor_and_normalize_columns(S: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """列ごとに project_simplex_with_floor を適用。"""
    A, C = S.shape
    S2 = S.copy()
    for c in range(C):
        S2[:, c] = project_simplex_with_floor(S2[:, c], tau[:, c])
    return S2

# ============== AHP: Pairwise & CI =================

def clamp_saaty(x: float, lo=1/9, hi=9) -> float:
    return float(np.clip(x, lo, hi))

def random_pairwise(n: int, intensity: float = 0.6) -> np.ndarray:
    """ランダムな（相反・正の）ペア比較行列。intensity で鋭さを調整。"""
    # 基準の“真の強さ”ベクトルを作り、そこから比を作る
    base = rng.lognormal(mean=0.0, sigma=intensity, size=n)
    base = base / base.mean()
    P = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            r = base[i] / base[j]
            # ノイズ
            noise = rng.lognormal(mean=0.0, sigma=0.25)
            val = clamp_saaty(r * noise)
            P[i,j] = val
            P[j,i] = 1.0/val
    np.fill_diagonal(P, 1.0)
    return P

def ahp_eigvec_ci(P: np.ndarray, iters: int = 200, tol: float = 1e-10) -> Tuple[np.ndarray, float, float]:
    """固有ベクトル近似（べき乗法）とCI。"""
    n = P.shape[0]
    v = np.ones(n, dtype=np.float64) / n
    for _ in range(iters):
        v_new = P @ v
        v_new = v_new / v_new.sum()
        if np.linalg.norm(v_new - v, 1) < tol:
            v = v_new; break
        v = v_new
    # 近似固有値
    Pv = P @ v
    lam = float((v @ Pv) / (v @ v))
    ci = float((lam - n) / (n - 1)) if n > 2 else 0.0
    v = normalize_simplex(v)
    return v, ci, lam

def consistent_pairwise_from_w(w: np.ndarray) -> np.ndarray:
    n = len(w); P = np.ones((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i==j: P[i,j]=1.0
            else: P[i,j] = clamp_saaty(w[i]/w[j])
    return P

def blend_pairwise_to_w(P: np.ndarray, w: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """log空間で P を一貫行列 w_i/w_j にブレンド。"""
    n = P.shape[0]
    logP = np.log(np.clip(P, 1e-9, 1e9))
    W = consistent_pairwise_from_w(w)
    logW = np.log(W)
    logP_new = (1 - lam) * logP + lam * logW
    Pn = np.exp(logP_new)
    # reciprocity & clamp
    for i in range(n):
        for j in range(i+1, n):
            val = clamp_saaty(Pn[i,j])
            Pn[i,j] = val; Pn[j,i] = 1.0/val
    np.fill_diagonal(Pn, 1.0)
    return Pn

def update_Palt_prefer(P: np.ndarray, a_idx: int, delta: float, cap: float = 2.0) -> np.ndarray:
    """代替案 a の“相対優位”を delta だけ log空間で押し上げ（/下げ）、
       P[a,k] のみ変更（k!=a）。 reciprocity を保ち、クランプ。"""
    n = P.shape[0]
    d = float(np.clip(delta, -cap, cap))
    for k in range(n):
        if k == a_idx: continue
        logval = np.log(np.clip(P[a_idx, k], 1e-9, 1e9))
        logval_new = logval + d
        val = clamp_saaty(np.exp(logval_new))
        P[a_idx, k] = val
        P[k, a_idx] = 1.0/val
    np.fill_diagonal(P, 1.0)
    return P

# ============== Argumentation (VAF personal) ==============

@dataclass
class Arg:
    owner: int
    kind: str            # 'w' or 'S'
    crit: int            # which criterion c
    target: Tuple[int,int]  # (a,c) for S, (-1,c) for w
    sign: int            # +1 / -1
    conf: float          # [0,1]
    strength: float      # {1,2,3}

def conflicts(a: Arg, b: Arg) -> bool:
    if a.kind != b.kind: return False
    return (a.target == b.target) and (a.sign != b.sign)
def effective_attack_personal(att: Arg, tgt: Arg,
                              w: np.ndarray,
                              veto_crit: set,
                              w_floor: np.ndarray,
                              S: np.ndarray,
                              tau: np.ndarray,
                              stats: Dict[str,int] = None) -> bool:
    """個人VAFの攻撃可否判定。ブロック理由を stats にカウントする。"""
    # Veto: その人にとって“下げ”は無効
    if tgt.kind == 'w' and att.sign == -1 and (tgt.target[1] in veto_crit):
        if stats is not None: stats["veto"] = stats.get("veto",0) + 1
        return False
    # フロア（重み）
    if tgt.kind == 'w' and att.sign == -1 and w[tgt.target[1]] <= w_floor[tgt.target[1]] + 1e-12:
        if stats is not None: stats["floor"] = stats.get("floor",0) + 1
        return False
    # Sの下限（代替案スコア）
    if tgt.kind == 'S' and att.sign == -1:
        a, c = tgt.target
        if S[a, c] <= tau[a, c] + 1e-12:
            if stats is not None: stats["tau"] = stats.get("tau",0) + 1
            return False
    # 価値依存
    return conflicts(att, tgt) and (w[att.crit] + 1e-12 >= w[tgt.crit])

def grounded_vaf_personal(args: List[Arg], w: np.ndarray,
                          veto_crit: set, w_floor: np.ndarray,
                          S: np.ndarray, tau: np.ndarray,
                          stats: Dict[str,int] = None) -> List[int]:
    n = len(args)
    attackers = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if effective_attack_personal(args[i], args[j], w, veto_crit, w_floor, S, tau, stats=stats):
                attackers[j].add(i)

    undec = set(range(n)); acc, rej = set(), set()
    changed = True
    while changed:
        changed = False
        newly_acc = {v for v in list(undec) if all(u in rej for u in attackers[v])}
        if newly_acc:
            acc |= newly_acc; undec -= newly_acc; changed = True
        newly_rej = {v for v in list(undec) if any(u in acc for u in attackers[v])}
        if newly_rej:
            rej |= newly_rej; undec -= newly_rej; changed = True
    return sorted(list(acc))


# ============== Node (Agent) state =================

@dataclass
class NodeState:
    w: np.ndarray            # (C,)
    S: np.ndarray            # (A,C)
    Pcrit: np.ndarray        # (C,C) pairwise criteria
    Palt: List[np.ndarray]   # list of (A,A) per criterion
    CIcrit: float
    inbox: List[Arg]
    last_si: float
    last_cons: float
    w_prior: np.ndarray
    veto_crit: set
    w_floor: np.ndarray
    tau: np.ndarray          # (A,C) Sの下限

def init_states(N: int, A: int, C: int) -> Dict[int, NodeState]:
    st = {}
    for i in range(N):
        # criteria pairwise → w
        Pcrit = random_pairwise(C, intensity=0.6)
        w_i, ci_i, _ = ahp_eigvec_ci(Pcrit)

        # 下限（先に用意）
        w_floor = np.full(C, 0.05, dtype=np.float64)    # 5% floor（従来通り）
        tau = np.full((A, C), 0.05, dtype=np.float64)   # tauは0.05に固定（平均・更新での再適用で担保）

        # w に floor を適用して正規化（下限付き射影）
        w_i = project_simplex_with_floor(w_i, w_floor)

        # per-criterion alternative pairwise → S
        Palt = [random_pairwise(A, intensity=0.6) for _ in range(C)]
        S = np.zeros((A, C), dtype=np.float64)
        for c in range(C):
            s_c, _, _ = ahp_eigvec_ci(Palt[c])
            S[:, c] = s_c

        # S に floor を適用して再正規化
        S = apply_floor_and_normalize_columns(S, tau)

        veto = set([int(rng.integers(0, C))])

        st[i] = NodeState(
            w=w_i.copy(), S=S, Pcrit=Pcrit, Palt=Palt, CIcrit=ci_i,
            inbox=[], last_si=0.5, last_cons=1.0,
            w_prior=w_i.copy(), veto_crit=veto, w_floor=w_floor, tau=tau
        )
    return st

# ============== Gossip & Utilities =================

def local_consensus(i: int, states: Dict[int,NodeState], G: nx.Graph) -> float:
    nbrs = list(G.neighbors(i))
    if not nbrs: return 0.0
    di = 0.0
    for j in nbrs:
        di += np.linalg.norm(states[i].w - states[j].w, ord=1)
    return di / len(nbrs)

def group_utility(states: Dict[int,NodeState]) -> np.ndarray:
    S_mean = np.stack([s.S for s in states.values()], axis=0).mean(axis=0) # (A,C)
    w_mean = np.stack([s.w for s in states.values()], axis=0).mean(axis=0) # (C,)
    util = S_mean @ w_mean
    return util

def gossip_step_w(i: int, w_all: Dict[int,np.ndarray], G: nx.Graph) -> np.ndarray:
    nbrs = list(G.neighbors(i)) + [i]
    W = np.stack([w_all[j] for j in nbrs], axis=0)
    return normalize_simplex(W.mean(axis=0))

# ============== AGAU (safe, AHP-consistent) ==============

def adjust_eta_by_ci_entropy(w: np.ndarray, CI: float, eta: float,
                             H_min: float = 0.85, CI_hi: float = 0.15) -> float:
    """尖り/不整合が強いときは慎重（eta↓）。拒否ではなく“弱く適用”に留める。"""
    Hn = weight_entropy(w)  # [0,1]
    eta_eff = eta
    if Hn < H_min: eta_eff *= 0.5
    if CI > CI_hi: eta_eff *= 0.5
    return eta_eff

def agau_update_w_safe(w: np.ndarray, args: List[Arg], accepted: List[int], eta: float,
                       w_floor: np.ndarray, w_prior: np.ndarray, lam_prior: float = 0.05) -> np.ndarray:
    s = np.zeros_like(w)
    for k in accepted:
        a = args[k]
        if a.kind == 'w':
            c = a.target[1]
            s[c] += a.sign * a.conf * a.strength
    # 指数更新
    w1 = normalize_simplex(w * np.exp(eta * s))
    # 事前に引き戻し（非拘束・慎重さ）
    w2 = normalize_simplex((1 - lam_prior) * w1 + lam_prior * w_prior)
    # フロア付き射影で下限と合計=1を同時に満たす
    return project_simplex_with_floor(w2, w_floor)

def agau_update_S_via_Palt(S: np.ndarray, Palt: List[np.ndarray], args: List[Arg], accepted: List[int],
                           eta: float, tau: np.ndarray, s_min: float = 1e-3, s_max: float = 1e3) -> Tuple[np.ndarray, List[np.ndarray]]:
    """受理S論証を Palt に反映→固有ベクトルで S[:,c] を再計算。"""
    for k in accepted:
        a = args[k]
        if a.kind != 'S': continue
        r, c = a.target
        delta = float(eta * a.sign * a.conf * a.strength)
        Palt[c] = update_Palt_prefer(Palt[c], a_idx=r, delta=delta, cap=2.0)
    # 再計算
    A, C = S.shape
    for c in range(C):
        s_c, _, _ = ahp_eigvec_ci(Palt[c])
        S[:, c] = s_c
    # tauフロア適用 → 列ごと再正規化 → クリップ（主に上限保護）
    S = apply_floor_and_normalize_columns(S, tau)
    # 上限保護のみ（下限は tau に任せる）
    S = np.minimum(S, s_max)
    return S, Palt

# ============== Action space =================

@dataclass
class ActionSpec:
    kind: str          # 'w' or 'S'
    crit: int
    sign: int          # +/-1
    mag: float         # {1.0, 2.0, 3.0}
    alt: int           # 0..A-1 (for S). if kind=='w', alt is -1.

def build_action_space(C: int, A: int) -> List[ActionSpec]:
    actions = []
    for kind in ('w','S'):
        for c in range(C):
            for sign in (-1, +1):
                for mag in (1.0, 2.0, 3.0):
                    if kind == 'w':
                        actions.append(ActionSpec(kind, c, sign, mag, -1))
                    else:
                        for a in range(A):
                            actions.append(ActionSpec(kind, c, sign, mag, a))
    return actions

# ============== Policy / Value =================

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
    def forward(self, x): return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

# ============== PPO Buffer & GAE ==============

class RolloutBuffer:
    def __init__(self): self.reset()
    def reset(self):
        self.obs = []; self.act = []; self.logp = []; self.rew = []; self.val = []; self.done = []
    def add(self, o,a,lp,r,v,d):
        self.obs.append(o); self.act.append(a); self.logp.append(lp); self.rew.append(r); self.val.append(v); self.done.append(d)
    def get(self, device):
        to_t = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32, device=device)
        return (to_t(self.obs),
                torch.as_tensor(np.array(self.act), dtype=torch.long, device=device),
                to_t(self.logp), to_t(self.rew), to_t(self.val), to_t(self.done))

def compute_gae(rew, val, done, gamma=0.95, lam=0.95):
    T = len(rew); adv = np.zeros(T, dtype=np.float32); last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        next_v = val[t+1] if t+1<T else 0.0
        delta = rew[t] + gamma * next_v * nonterminal - val[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret

# ============== Observation =================

def build_observation(i: int, states: Dict[int,NodeState], G: nx.Graph) -> np.ndarray:
    si = states[i]; C = si.w.shape[0]
    nbrs = list(G.neighbors(i))
    if nbrs:
        w_nei = np.stack([states[j].w for j in nbrs], axis=0)
        w_mean = w_nei.mean(axis=0); w_std = w_nei.std(axis=0)
    else:
        w_mean = np.zeros_like(si.w); w_std = np.zeros_like(si.w)
    cons = local_consensus(i, states, G)
    # 観測: 自分w, 近傍平均w, 乖離, std, last_si, last_cons, CIcrit, H(w)
    obs = np.concatenate([
        si.w, w_mean, np.abs(si.w - w_mean), w_std,
        np.array([si.last_si, si.last_cons, si.CIcrit, weight_entropy(si.w)], dtype=np.float32)
    ])
    return np.nan_to_num(obs.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)

# ============== Environment Step =================

def step_env(actions_idx: Dict[int,int], action_space: List[ActionSpec],
             states: Dict[int,NodeState], G: nx.Graph, A: int, C: int,
             eta: float = 0.06,
             lam_prior: float = 0.05,
             lam_incons_w: float = 0.10) -> Dict[str, Any]:
    """全ノード同時に行動→VAF受理→AGAU（w/SとPcrit/Palt）→gossip→報酬+ログ"""
    # Preflight invariant checks (optional)
    if getattr(step_env, "check_invariants", False):
        for _i in G.nodes:
            assert_state(_i, states[_i])
    # 1) materialize arguments, broadcast to neighbors+self
    outbox: Dict[int, List[Arg]] = defaultdict(list)
    own_arg_per_node: Dict[int, Arg] = {}
    own_key: Dict[int, Tuple[str,int,int,float]] = {}
    for i in G.nodes:
        spec = action_space[actions_idx[i]]
        if spec.kind == 'w':
            arg = Arg(i, 'w', spec.crit, (-1, spec.crit), spec.sign,
                      conf=float(rng.uniform(0.6, 1.0)), strength=spec.mag)
        else:
            arg = Arg(i, 'S', spec.crit, (spec.alt, spec.crit), spec.sign,
                      conf=float(rng.uniform(0.6, 1.0)), strength=spec.mag)
        own_arg_per_node[i] = arg
        own_key[i] = (spec.kind, spec.crit, spec.sign, spec.mag)
        for j in list(G.neighbors(i)) + [i]:
            outbox[j].append(arg)

    # 2) deliver
    for i in G.nodes:
        states[i].inbox.extend(outbox[i])
    # 3) VAF受理→AGAU更新（個人ごと）
    accept_map: Dict[int,List[int]] = {}
    safety_stats_all: Dict[int, Dict[str,int]] = {}  # <= NEW
    for i in G.nodes:
        st = states[i]; args = st.inbox
        if args:
            local_stats = {"veto":0,"floor":0,"tau":0}  # <= NEW
            acc = grounded_vaf_personal(args, st.w, st.veto_crit, st.w_floor, st.S, st.tau, stats=local_stats)  # <= MOD
            safety_stats_all[i] = local_stats  # <= NEW
            accept_map[i] = acc
            # CI/エントロピーに応じてeta調整
            eta_eff = adjust_eta_by_ci_entropy(st.w, st.CIcrit, eta, H_min=0.85, CI_hi=0.15)
            # w更新
            st.w = agau_update_w_safe(st.w, args, acc, eta_eff, st.w_floor, st.w_prior, lam_prior)
            # Pcrit を w に合わせてブレンド
            st.Pcrit = blend_pairwise_to_w(st.Pcrit, st.w, lam=lam_incons_w)
            _, st.CIcrit, _ = ahp_eigvec_ci(st.Pcrit)
            # S更新（tauフロア適用を内包）
            st.S, st.Palt = agau_update_S_via_Palt(st.S, st.Palt, args, acc, eta_eff, st.tau)
            # 念のため、ここでも列ごとに tau フロアと正規化を強制
            st.S = apply_floor_and_normalize_columns(st.S, st.tau)
        else:
            accept_map[i] = []
            safety_stats_all[i] = {"veto":0,"floor":0,"tau":0}  # <= NEW


    # 4) Gossip（wのみ）。その後、Pcritをwに微ブレンドして整合を保つ。
    w_all = {i: states[i].w for i in G.nodes}
    for i in G.nodes:
        states[i].w = gossip_step_w(i, w_all, G)
        states[i].w = project_simplex_with_floor(states[i].w, states[i].w_floor)
        states[i].Pcrit = blend_pairwise_to_w(states[i].Pcrit, states[i].w, lam=0.05)
        _, states[i].CIcrit, _ = ahp_eigvec_ci(states[i].Pcrit)

    # 5) 報酬（ΔSI + Δ合意度 + 自論受理）
    util_g = group_utility(states)
    rewards: Dict[int,float] = {}
    own_accepted_exact: Dict[int,int] = {}
    d_si_map: Dict[int,float] = {}
    d_cons_map: Dict[int,float] = {}
    # 重み乖離（基準ごと）
    W = np.stack([states[i].w for i in G.nodes], axis=0)
    wbar = W.mean(axis=0)
    div_vec = np.mean(np.abs(W - wbar[None,:]), axis=0)

    for i in G.nodes:
        st = states[i]
        util_i = st.S @ st.w
        si_now = pearson_scaled(util_i, util_g); d_si = si_now - st.last_si
        cons_now = local_consensus(i, states, G); d_cons = st.last_cons - cons_now

        # 自論受理（kindとcrit, signで判定）
        acc_ids = accept_map[i]
        accepted = 0
        if st.inbox and acc_ids:
            for k in acc_ids:
                a = st.inbox[k]
                if a.owner == i and a.kind == own_arg_per_node[i].kind and a.crit == own_arg_per_node[i].crit and a.sign == own_arg_per_node[i].sign:
                    accepted = 1; break

        rewards[i] = 1.0 * d_si + 0.3 * d_cons + 0.2 * accepted
        own_accepted_exact[i] = accepted
        d_si_map[i] = d_si; d_cons_map[i] = d_cons

        st.last_si = si_now; st.last_cons = cons_now
        st.inbox = st.inbox[-80:]  # keep bounded
    safety_totals = {"veto":0,"floor":0,"tau":0}
    for i in safety_stats_all:
        for k in safety_totals:
            safety_totals[k] += safety_stats_all[i][k]

    return {
        "rewards": rewards,
        "own_key": own_key,
        "own_accepted_exact": own_accepted_exact,
        "d_si": d_si_map,
        "d_cons": d_cons_map,
        "div_vec": div_vec,
        "safety_events": safety_totals  # <= NEW
    }

# ============== Training (PPO) =================

def train_ppo(
    N=4, A=3, C=3, steps_per_update=64, updates=60,
    eta=0.06, gamma=0.95, gae_lam=0.95, clip_eps=0.2,
    pi_lr=3e-4, vf_lr=5e-4, pi_epochs=4, minibatch_frac=0.5, seed=123,
    eval_every=10, val_steps=64, save_dir="artifacts", check_invariants=CHECK_INVARIANTS_DEFAULT,
    export_templates=False
):
    os.makedirs(save_dir, exist_ok=True)
    # enable invariant checks inside step_env
    step_env.check_invariants = bool(check_invariants)

    def compute_D_from_W(Wsnap: np.ndarray) -> float:
        wbar = Wsnap.mean(axis=0)
        return float(np.max(wbar))

    def eval_once(val_steps=64):
        # 学習OFFで policy を用い、短いロールアウトを回す
        G_eval = nx.watts_strogatz_graph(N, k=4, p=0.2, seed=seed+999)
        states_eval = init_states(N, A, C)
        dsi_list, dcons_list, ownacc_list = [], [], []
        Wsnaps = []
        for _ in range(val_steps):
            obs_all = [build_observation(i, states_eval, G_eval) for i in G_eval.nodes]
            obs_np = np.stack(obs_all, axis=0)
            with torch.no_grad():
                logits = policy(torch.as_tensor(obs_np, dtype=torch.float32))
                probs = torch.distributions.Categorical(logits=torch.nan_to_num(logits))
                acts = probs.sample().cpu().numpy()
            actions_idx = {i: int(a) for i, a in zip(G_eval.nodes, acts)}
            out = step_env(actions_idx, action_space, states_eval, G_eval, A, C, eta=eta)
            dsi_list.append(np.mean([out["d_si"][i] for i in G_eval.nodes]))
            dcons_list.append(np.mean([out["d_cons"][i] for i in G_eval.nodes]))
            ownacc_list.append(np.mean([out["own_accepted_exact"][i] for i in G_eval.nodes]))
            Wsnaps.append(np.stack([states_eval[i].w for i in G_eval.nodes], axis=0))
        D_final = compute_D_from_W(Wsnaps[-1])
        J = float(np.mean(dsi_list) + 0.3*np.mean(dcons_list) + 0.2*np.mean(ownacc_list) - 0.5*D_final)
        return J, D_final
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs("logs", exist_ok=True)

    # graph & states
    G = nx.watts_strogatz_graph(N, k=4, p=0.2, seed=seed)
    states = init_states(N, A, C)
    action_space = build_action_space(C, A)
    act_dim = len(action_space)

    # obs dim: w(C)+mean(C)+|diff|(C)+std(C)+[last_si,last_cons,CI,H]=4C+4
    obs_dim = 4*C + 4

    device = torch.device("cpu")
    policy = PolicyNet(obs_dim, act_dim).to(device)
    valuef = ValueNet(obs_dim).to(device)
    opt_pi = optim.Adam(policy.parameters(), lr=pi_lr)
    opt_vf = optim.Adam(valuef.parameters(), lr=vf_lr)

    buf = RolloutBuffer()

    # ---- instrumentation ----
    action_freq: Counter = Counter()
    accept_stats: Dict[Tuple[str,int,int], List[int]] = defaultdict(lambda: [0,0])
    traj_dsi, traj_dcons, traj_hit = [], [], []
    weight_div_series: List[np.ndarray] = []

    # best model tracking
    best_J = -1e9
    best_paths: Dict[str, str] = {}

    def policy_step(obs_batch: np.ndarray):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            logits = policy(obs_t)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
            probs = torch.distributions.Categorical(logits=logits)
            a = probs.sample()
            logp = probs.log_prob(a)
            v = valuef(obs_t).squeeze(-1)
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    for upd in range(1, updates+1):
        buf.reset()
        # rollout
        for t in range(steps_per_update):
            obs_all = []; node_index = []
            for i in G.nodes:
                obs = build_observation(i, states, G)
                obs_all.append(obs); node_index.append(i)
            obs_np = np.stack(obs_all, axis=0)

            acts, logps, vals = policy_step(obs_np)
            actions_idx = {node_index[k]: int(acts[k]) for k in range(N)}

            out = step_env(actions_idx, action_space, states, G, A, C, eta=eta)
            rews = np.array([out["rewards"][i] for i in node_index], dtype=np.float32)
            done_flags = np.zeros(N, dtype=np.float32)

            # instrumentation
            for i in node_index:
                kind, crit, sign, mag = out["own_key"][i]
                action_freq[(kind, crit, sign, mag)] += 1
                akey = (kind, crit, sign)
                accept_stats[akey][1] += 1
                if out["own_accepted_exact"][i] == 1:
                    accept_stats[akey][0] += 1
            traj_dsi.append(float(np.mean([out["d_si"][i] for i in node_index])))
            traj_dcons.append(float(np.mean([out["d_cons"][i] for i in node_index])))
            traj_hit.append(float(np.mean([out["own_accepted_exact"][i] for i in node_index])))
            weight_div_series.append(out["div_vec"])

            for k in range(N):
                buf.add(obs_np[k], acts[k], logps[k], rews[k], vals[k], done_flags[k])

        # PPO update
        obs_t, act_t, logp_t, rew_t, val_t, done_t = buf.get(device)
        adv_np, ret_np = compute_gae(rew_t.cpu().numpy(), val_t.cpu().numpy(), done_t.cpu().numpy(),
                                     gamma=gamma, lam=gae_lam)
        adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=device)
        ret_t = torch.as_tensor(ret_np, dtype=torch.float32, device=device)

        B = obs_t.shape[0]; idxs = np.arange(B)
        mb_size = max(1, int(B * minibatch_frac))
        for _ in range(pi_epochs):
            np.random.shuffle(idxs)
            for start in range(0, B, mb_size):
                mb = idxs[start:start+mb_size]
                logits = policy(obs_t[mb]); logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
                probs = torch.distributions.Categorical(logits=logits)
                logp_new = probs.log_prob(act_t[mb])
                ratio = torch.exp(logp_new - logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1.0-clip_eps, 1.0+clip_eps) * adv_t[mb]
                loss_pi = -torch.mean(torch.min(surr1, surr2))

                v_pred = valuef(obs_t[mb]).squeeze(-1)
                loss_v = torch.mean((v_pred - ret_t[mb])**2)

                opt_pi.zero_grad(); loss_pi.backward(); opt_pi.step()
                opt_vf.zero_grad(); loss_v.backward(); opt_vf.step()

        with torch.no_grad():
            ent = torch.distributions.Categorical(logits=torch.nan_to_num(policy(obs_t))).entropy().mean().item()
            loss_v_eval = torch.mean((valuef(obs_t).squeeze(-1) - ret_t)**2).item()

        # quick stats
        W = np.stack([states[i].w for i in G.nodes], axis=0)
        cons_global = np.max([local_consensus(i, states, G) for i in G.nodes])
        util_g = group_utility(states)
        si_mean = np.mean([pearson_scaled(states[i].S @ states[i].w, util_g) for i in G.nodes])

        print(f"[upd {upd:03d}] entropy={ent:.2f} vf_loss={loss_v_eval:.3f} "
              f"cons(max L1 nbr)={cons_global:.3f} SI~{si_mean:.3f}")

        # === eval & save best ===
        if (upd % eval_every) == 0:
            J, Dfin = eval_once(val_steps=val_steps)
            print(f"[eval @upd {upd}] J={J:.4f}  D_final={Dfin:.3f}")
            if J > best_J:
                best_J = J
                pt_path = os.path.join(save_dir, f"policy_v1_seed{seed}_best.pt")
                ts_path = os.path.join(save_dir, f"policy_v1_seed{seed}_best.ts")
                scripted = torch.jit.script(policy)
                torch.save(policy.state_dict(), pt_path)
                torch.jit.save(scripted, ts_path)
                best_paths = {"pt": pt_path, "ts": ts_path}
                if export_templates:
                    export_action_templates(action_space,
                                            os.path.join(save_dir, "action_space.json"),
                                            os.path.join(save_dir, "templates.json"))
                print(f"  >> saved BEST: J={best_J:.4f} to {best_paths}")

    # dump logs
    import csv
    with open("logs/action_freq.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kind","crit","sign","magnitude","count"])
        for (kind,c,sg,mg),cnt in sorted(action_freq.items(), key=lambda kv: -kv[1]):
            w.writerow([kind,c,sg,mg,cnt])

    with open("logs/accept_rate.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["kind","crit","sign","accepted","total","rate"])
        for (kind,c,sg),(acc,tot) in sorted(accept_stats.items()):
            rate = acc/tot if tot>0 else 0.0
            w.writerow([kind,c,sg,acc,tot,rate])

    with open("logs/reward_decomp.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step","dSI","dConsensus","ownAccepted"])
        for t,(a,b,c3) in enumerate(zip(traj_dsi, traj_dcons, traj_hit)):
            w.writerow([t,a,b,c3])

    wd = np.stack(weight_div_series, axis=0) if len(weight_div_series)>0 else np.zeros((0,C))
    np.save("logs/weight_divergence.npy", wd)
    with open("logs/weight_divergence.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step"] + [f"crit_{k}" for k in range(C)])
        for t in range(wd.shape[0]):
            w.writerow([t] + list(map(float, wd[t])))

    # final snapshot
    w_mean = W.mean(axis=0); util = util_g; ranking = list(map(int, np.argsort(-util)))
    print("\n==== FINAL ====")
    print("Group weights (mean):", np.round(w_mean, 3))
    print("Final group utility per alternative:", np.round(util, 3))
    print("Ranking (best->worst):", ranking)
    print("Saved logs in ./logs (compatible with viz_logs.py)")

# ============== Export helpers (action space & NL templates) ==============

def export_action_templates(action_space: List["ActionSpec"], path_space: str, path_templates: str):
    # action_space.json
    items = []
    for idx, spec in enumerate(action_space):
        items.append({
            "index": idx,
            "kind": spec.kind,
            "crit": spec.crit,
            "sign": spec.sign,
            "mag": spec.mag,
            "alt": spec.alt
        })
    with open(path_space, "w") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    # templates.json（固定テンプレ：後でUI側で細かく調整可）
    templates = {
        "w": {
            "+1": {
                "1.0": "基準{c}の重みを少し上げる",
                "2.0": "基準{c}の重みを上げる",
                "3.0": "基準{c}の重みを強く上げる"
            },
            "-1": {
                "1.0": "基準{c}の重みを少し下げる（Veto/フロアに注意）",
                "2.0": "基準{c}の重みを下げる（Veto/フロアに注意）",
                "3.0": "基準{c}の重みを強く下げる（Veto/フロアに注意）"
            }
        },
        "S": {
            "+1": "代替案{a}×基準{c}の評価を上げる（強度{mag}）",
            "-1": "代替案{a}×基準{c}の評価を下げる（強度{mag}）"
        }
    }
    with open(path_templates, "w") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--A", type=int, default=3)
    p.add_argument("--C", type=int, default=4)
    p.add_argument("--steps_per_update", type=int, default=64)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--eta", type=float, default=0.06)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--gae_lam", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--pi_lr", type=float, default=3e-4)
    p.add_argument("--vf_lr", type=float, default=5e-4)
    p.add_argument("--pi_epochs", type=int, default=4)
    p.add_argument("--minibatch_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--val_steps", type=int, default=64)
    p.add_argument("--save_dir", type=str, default="artifacts")
    p.add_argument("--check_invariants", type=int, default=int(CHECK_INVARIANTS_DEFAULT))
    p.add_argument("--export_templates", action="store_true")
    args = p.parse_args()

    train_ppo(
        N=args.N, A=args.A, C=args.C,
        steps_per_update=args.steps_per_update, updates=args.updates,
        eta=args.eta, gamma=args.gamma, gae_lam=args.gae_lam,
        clip_eps=args.clip_eps, pi_lr=args.pi_lr, vf_lr=args.vf_lr,
        pi_epochs=args.pi_epochs, minibatch_frac=args.minibatch_frac, seed=args.seed,
        eval_every=args.eval_every, val_steps=args.val_steps, save_dir=args.save_dir,
        check_invariants=bool(args.check_invariants),
        export_templates=args.export_templates
    )