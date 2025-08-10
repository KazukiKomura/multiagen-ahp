"""
Reason-Based Consensus System — Minimal Full-Version Prototype
VAF (value-based AF) × AGAU × AHP × Mediator Policy (LinUCB-like) × Multi-agent Simulator

目的: 事前学習をすぐ回せる“単一ファイルのミニ実装”。
- 依存: numpy, networkx, scipy(optional), typing, dataclasses
- 想定: 小規模(agents<=7, alternatives<=6, criteria<=6)で数千ステップをCPUで回せる
- 注意: 研究コード。最適化/厳密CIは簡略。安全柵とログを最小限実装。

今後の分割先:
- core/semantics.py, core/agau.py, core/measures.py, agents/policy.py, agents/argument_gen.py,
  environment/simulator.py, experiments/train.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import networkx as nx
import math
import random

# ============================
# ユーティリティ
# ============================

def normalize_simplex(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= eps:
        # 退避: 一様化
        return np.ones_like(w) / len(w)
    return w / s


def softclip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================
# データ構造
# ============================

TOPIC = ("Self", "Criteria", "Alt")

@dataclass
class Argument:
    id: str
    topic: str  # "Self" | "Criteria" | "Alt"
    target: Tuple  # (component, index...) e.g. ("weight", k) or ("score", a, k) or ("pairwise", k, i, j)
    sign: int    # +1/-1
    magnitude: float  # >=0
    confidence: float # [0,1]
    sender: Optional[int] = None
    recipient: Optional[int] = None
    value: Optional[int] = None  # which criterion this argument appeals to (index)


@dataclass
class AHPState:
    w: np.ndarray                 # (m,)
    S: np.ndarray                 # (A, m)
    pairwise: Dict[int, np.ndarray]  # per criterion: (n_alt, n_alt) with reciprocity
    CI: float                     # aggregate CI (simplified)


@dataclass
class Participant:
    w: np.ndarray
    S: np.ndarray
    pairwise: Dict[int, np.ndarray]
    CI: float
    theta: Tuple[float, float, float] = (1.2, 0.8, 0.6)  # (value align, evidence, repetition fatigue)
    accept_ema: float = 0.5

    def accept_prob(self, arg: Argument, w_bar: np.ndarray, recent_same: int = 0) -> float:
        # 価値整合: 自身重みとarg.valueの一致度
        align = 0.0
        if arg.value is not None:
            k = int(arg.value)
            align = float(self.w[k])
        evidence = float(arg.confidence) * float(arg.magnitude)
        fatigue = float(recent_same)
        z = self.theta[0] * align + self.theta[1] * evidence - self.theta[2] * fatigue
        return 1.0 / (1.0 + math.exp(-z))


# ============================
# VAF (grounded, 価値順序依存)
# ============================

def value_order_from_weights(w: np.ndarray) -> List[int]:
    # 高重み→高優先の並び (indices)
    return list(np.argsort(-w))


def vaf_effective_attack(u_val: int, v_val: int, order: List[int]) -> bool:
    # order での優先順位: 位置が前ほど優先
    pos = {k: i for i, k in enumerate(order)}
    return pos[u_val] <= pos[v_val]


def grounded_vaf(G: nx.DiGraph, w_values: np.ndarray) -> List[str]:
    """ grounded semantics with value-based attack filtering.
    ノード属性: value(int), id(str)
    エッジ属性: attacks=True (支持辺はGに入れない/別扱い)
    """
    order = value_order_from_weights(w_values)

    # 有効攻撃のみを残したサブグラフ
    H = nx.DiGraph()
    for n, data in G.nodes(data=True):
        H.add_node(n, **data)
    for u, v, data in G.edges(data=True):
        if not data.get("attacks", False):
            continue
        u_val, v_val = G.nodes[u].get("value"), G.nodes[v].get("value")
        if u_val is None or v_val is None:
            continue
        if vaf_effective_attack(int(u_val), int(v_val), order):
            H.add_edge(u, v)

    undec = set(H.nodes())
    acc: set = set()
    rej: set = set()
    changed = True
    while changed:
        changed = False
        for v in list(undec):
            # 攻撃者がすべて拒否なら受理
            attackers = [u for u in H.predecessors(v)]
            if all(u in rej for u in attackers):
                acc.add(v); undec.remove(v); changed = True
        for v in list(undec):
            # 攻撃者に受理が一つでもあれば拒否
            attackers = [u for u in H.predecessors(v)]
            if any(u in acc for u in attackers):
                rej.add(v); undec.remove(v); changed = True
    return list(acc)


# ============================
# AGAU 更新子
# ============================

def agau_weight_update(w: np.ndarray, s: np.ndarray, eta_w: float, cap: float = 0.15) -> np.ndarray:
    eta = softclip(eta_w, 0.0, cap)
    w_new = w * np.exp(eta * s)
    return normalize_simplex(w_new)


def agau_score_update(S: np.ndarray, M: np.ndarray, eta_S: float, cap: float = 0.15) -> np.ndarray:
    eta = softclip(eta_S, 0.0, cap)
    return S * np.exp(eta * M)


def nearest_consistent_matrix(P: np.ndarray) -> np.ndarray:
    # 近似: 一貫行列 u_i/u_j へ射影（最大固有ベクトル）
    # P は正の相反行列を仮定
    vals, vecs = np.linalg.eig(P)
    u = np.real(vecs[:, np.argmax(np.real(vals))])
    u = np.clip(u, 1e-9, None)
    U = np.outer(u, 1.0 / u)
    return U


def agau_pairwise_blend(P: np.ndarray, lam: float, lam_max: float = 0.1) -> np.ndarray:
    lam = softclip(lam, 0.0, lam_max)
    U = nearest_consistent_matrix(P)
    # log-space blend
    logP = np.log(P + 1e-12)
    logU = np.log(U + 1e-12)
    logP_new = (1 - lam) * logP + lam * logU
    P_new = np.exp(logP_new)
    # reciprocity enforce
    P_new = (P_new + 1.0 / P_new.T) / 2.0
    np.fill_diagonal(P_new, 1.0)
    return P_new


def ci_harmonic(P: np.ndarray) -> float:
    # 簡略HCI: Stein & Mizzi に準拠した近似（正規化は省略）
    n = P.shape[0]
    # 相反性前提
    h = np.sum(1.0 / (P + 1e-12)) - n
    h_max = n * (n - 1)
    return float(h / h_max)


# ============================
# 指標
# ============================

def kendall_tau_from_scores(S: np.ndarray, w: np.ndarray) -> float:
    # ここでは擬似: 現順位の安定度を0.5〜1.0の範囲で近似
    scores = S @ w
    rank = np.argsort(-scores)
    # 疑似安定度: 上位差のマージンで代替
    diffs = np.diff(np.sort(scores))
    margin = float(np.mean(np.abs(diffs))) if len(diffs) else 0.0
    return float(0.5 + min(0.5, margin))


def gini_coeff(x: np.ndarray) -> float:
    x = np.array(x, dtype=float).flatten()
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(np.clip(x, 0, None))
    n = len(x)
    cum = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(max(0.0, min(1.0, g)))


# ============================
# 進行役方策 (簡易LinUCB風)
# ============================

@dataclass
class LinUCBPolicy:
    d: int
    alpha: float = 0.8
    topics: Tuple[str, ...] = ("Self", "Criteria", "Alt")
    A: Dict[str, np.ndarray] = field(default_factory=dict)
    b: Dict[str, np.ndarray] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        for a in self.topics:
            self.A[a] = np.eye(self.d)
            self.b[a] = np.zeros((self.d,))

    def features(self, obs: Dict) -> np.ndarray:
        feats = np.array([
            obs.get("mean_ci", 0.2),
            obs.get("max_ci", 0.3),
            obs.get("w_var", 0.05),
            obs.get("top_split", 0.1),
            obs.get("ema_accept", 0.5),
            obs.get("step_left", 1.0),
        ], dtype=float)
        # 固定次元に揃える
        if len(feats) < self.d:
            feats = np.pad(feats, (0, self.d - len(feats)))
        elif len(feats) > self.d:
            feats = feats[: self.d]
        return feats

    def select(self, obs: Dict) -> Tuple[str, float]:
        x = self.features(obs)
        best_a, best_ucb = None, -1e9
        for a in self.topics:
            A = self.A[a]
            b = self.b[a]
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            mean = float(theta @ x)
            conf = float(self.alpha * math.sqrt(x @ A_inv @ x))
            ucb = mean + conf
            if ucb > best_ucb:
                best_ucb, best_a = ucb, a
        propensity = 1.0 / len(self.topics)  # ログ用（簡略）
        return best_a, propensity

    def update(self, obs: Dict, action: str, reward: float):
        x = self.features(obs)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x


# ============================
# 環境（Mediator）
# ============================

@dataclass
class MediatorEnv:
    n_agents: int
    n_alt: int
    n_crit: int
    Tmax: int = 200
    eta_w: float = 0.05
    eta_S: float = 0.03
    lam_max: float = 0.1
    diffuse_to_recipient: float = 1.0
    diffuse_to_others: float = 0.2

    rng: random.Random = field(default_factory=random.Random)

    # 状態
    participants: List[Participant] = field(default_factory=list)
    t: int = 0
    ema_accept: float = 0.5

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            self.rng.seed(seed)
        self.participants = []
        for _ in range(self.n_agents):
            w = normalize_simplex(np.random.dirichlet(np.ones(self.n_crit)))
            S = np.abs(np.random.randn(self.n_alt, self.n_crit))
            pairwise = {}
            for k in range(self.n_crit):
                P = np.abs(np.random.lognormal(mean=0.0, sigma=0.5, size=(self.n_alt, self.n_alt)))
                P = (P + 1.0 / P.T) / 2.0
                np.fill_diagonal(P, 1.0)
                pairwise[k] = P
            CI = float(np.mean([ci_harmonic(P) for P in pairwise.values()]))
            self.participants.append(Participant(w=w, S=S, pairwise=pairwise, CI=CI))
        self.t = 0
        self.ema_accept = 0.5
        return self.observe()

    # 受け手の選択（ヒューリスティック）
    def pick_recipient(self) -> int:
        # CIが最大の人を優先
        cis = [p.CI for p in self.participants]
        return int(np.argmax(cis))

    # 対象の選択（ヒューリスティック）
    def pick_target(self, topic: str) -> Tuple:
        if topic == "Self":
            j = self.pick_recipient()
            # CIが高い基準を選ぶ（近似）
            k = self.rng.randrange(self.n_crit)
            i = self.rng.randrange(self.n_alt)
            q = ("pairwise", k, i, (i + 1) % self.n_alt)
            return q
        if topic == "Criteria":
            k = self.rng.randrange(self.n_crit)
            return ("weight", k)
        if topic == "Alt":
            a = self.rng.randrange(self.n_alt)
            k = self.rng.randrange(self.n_crit)
            return ("score", a, k)
        raise ValueError("unknown topic")

    def w_bar(self) -> np.ndarray:
        return normalize_simplex(np.mean([p.w for p in self.participants], axis=0))

    def current_ci_mean(self) -> float:
        return float(np.mean([p.CI for p in self.participants]))

    def argument_from(self, topic: str, recipient: Optional[int] = None) -> Argument:
        tgt = self.pick_target(topic)
        k_val = None
        if tgt[0] == "weight":
            k_val = int(tgt[1])
        elif tgt[0] == "score":
            k_val = int(tgt[2])
        elif tgt[0] == "pairwise":
            k_val = int(tgt[1])
        arg = Argument(
            id=f"t{self.t}-{topic}", topic=topic, target=tgt,
            sign=+1, magnitude=1.0, confidence=0.8,
            sender=None, recipient=recipient, value=k_val,
        )
        return arg

    def vaf_judge(self, arg: Argument) -> bool:
        # 1枚カードの判定を簡略: 有効攻撃を仮定し、受け手の確率受理→採否
        j = self.pick_recipient() if arg.recipient is None else arg.recipient
        p = self.participants[j]
        order = value_order_from_weights(self.w_bar())
        # 有効攻撃性チェック: value順位でOKなら継続
        if arg.value is not None:
            # ここでは自己関係比較は常に有効とみなす
            pass
        pr = p.accept_prob(arg, self.w_bar())
        accepted = (self.rng.random() < pr)
        # EMA 更新
        self.ema_accept = 0.7 * self.ema_accept + 0.3 * (1.0 if accepted else 0.0)
        return accepted

    def agau_apply(self, arg: Argument, accepted: bool):
        if not accepted:
            return
        # 支持ベクトル/行列を極小に構成
        if arg.topic == "Criteria" and arg.target[0] == "weight":
            k = int(arg.target[1])
            s = np.zeros(self.n_crit)
            s[k] = arg.sign * arg.magnitude * arg.confidence
            for idx, p in enumerate(self.participants):
                eta = self.eta_w * (self.diffuse_to_recipient if idx == arg.recipient else self.diffuse_to_others)
                p.w = agau_weight_update(p.w, s, eta)
        elif arg.topic == "Alt" and arg.target[0] == "score":
            a, k = int(arg.target[1]), int(arg.target[2])
            M = np.zeros((self.n_alt, self.n_crit))
            M[a, k] = arg.sign * arg.magnitude * arg.confidence
            for idx, p in enumerate(self.participants):
                eta = self.eta_S * (self.diffuse_to_recipient if idx == arg.recipient else self.diffuse_to_others)
                p.S = agau_score_update(p.S, M, eta)
        elif arg.topic == "Self" and arg.target[0] == "pairwise":
            k, i, j = int(arg.target[1]), int(arg.target[2]), int(arg.target[3])
            for p in self.participants:
                lam = min(self.lam_max, 0.05 + 0.05 * arg.confidence)
                P = p.pairwise[k]
                p.pairwise[k] = agau_pairwise_blend(P, lam, self.lam_max)
                # CI再計算（粗）
                p.CI = float(np.mean([ci_harmonic(Pk) for Pk in p.pairwise.values()]))
        # 受理でなくてもCIの自然減衰など追加可能

    def observe(self) -> Dict:
        wbar = self.w_bar()
        mean_ci = self.current_ci_mean()
        # グループ順位安定度（粗近似）
        taus = [kendall_tau_from_scores(p.S, p.w) for p in self.participants]
        tau_mean = float(np.mean(taus))
        w_var = float(np.var(np.vstack([p.w for p in self.participants]), axis=0).mean())
        # 上位案の割れ度（粗）
        top_idx = [int(np.argmax(p.S @ p.w)) for p in self.participants]
        split = 1.0 - (np.bincount(top_idx, minlength=self.n_alt).max() / max(1, len(top_idx)))
        return {
            "w_bar": wbar,
            "mean_ci": mean_ci,
            "max_ci": float(max([p.CI for p in self.participants])),
            "tau": tau_mean,
            "w_var": w_var,
            "top_split": float(split),
            "ema_accept": float(self.ema_accept),
            "step_left": float(max(0, self.Tmax - self.t) / self.Tmax),
        }

    def should_stop(self) -> bool:
        obs = self.observe()
        top_stable = obs["tau"] >= 0.85
        accept_stall = self.ema_accept < 0.1
        timeup = self.t >= self.Tmax
        return bool(top_stable or accept_stall or timeup)

    def step(self, topic: str) -> Tuple[Dict, float, bool, Dict]:
        self.t += 1
        recipient = self.pick_recipient()
        arg = self.argument_from(topic, recipient)
        accepted = self.vaf_judge(arg)
        ci_before = self.current_ci_mean()
        self.agau_apply(arg, accepted)
        obs = self.observe()
        # 報酬: 進捗(τ↑) + 一貫性(CI↓) - 手数
        reward = (obs["tau"] - 0.5) - (obs["mean_ci"] - ci_before) - 0.01
        info = {"arg": arg, "accepted": accepted, "obs": obs}
        done = self.should_stop()
        return obs, float(reward), done, info


# ============================
# 事前学習（最小）
# ============================

def pretrain(seed: int = 0, episodes: int = 50):
    env = MediatorEnv(n_agents=5, n_alt=5, n_crit=5)
    policy = LinUCBPolicy(d=6, alpha=0.8)
    logs = []
    for ep in range(episodes):
        obs = env.reset(seed + ep)
        total_r = 0.0
        steps = 0
        while True:
            topic, prop = policy.select(obs)
            obs_next, r, done, info = env.step(topic)
            policy.update(obs, topic, r)
            total_r += r
            steps += 1
            obs = obs_next
            if done:
                break
        logs.append({"episode": ep, "return": total_r, "steps": steps, **env.observe()})
        print(f"[ep {ep}] return={total_r:.3f} steps={steps} tau={obs['tau']:.3f} CI={obs['mean_ci']:.3f}")
    return logs


if __name__ == "__main__":
    # クイックドライラン（小規模）
    pretrain(seed=42, episodes=5)
