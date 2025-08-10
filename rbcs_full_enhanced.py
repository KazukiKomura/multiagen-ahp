"""
RBCS Full Enhanced — single-file prototype
- 指標と安全柵の強化（CI/HCI、ロールバック、クリップ、悪化検知）
- 受理モデルの高度化（価値整合×証拠×反復疲労＋個人差）
- Hydra設定＋MLflow連携（未導入環境では自動フォールバック）

実行例:
  # 依存が無い最小実行（Hydra/MLflow無しで動く）
  python rbcs_full_enhanced.py --episodes 10 --n_agents 5

  # Hydraが使える場合（推奨）
  python rbcs_full_enhanced.py hydra=on experiment.name=pretrain_v2 env.n_agents=5 episodes=200

ログ:
  - JSONL: ./runs/<timestamp>/events.jsonl
  - MLflow: 利用可能なら自動で記録
"""
from __future__ import annotations
import os, json, math, random, time, argparse
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any, Mapping
from collections import Counter

import numpy as np
import networkx as nx

# -------- Optional scipy for true Kendall-τ --------
try:
    from scipy.stats import kendalltau  # type: ignore
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# -------- Optional deps (Hydra/MLflow) --------
try:
    if os.environ.get("RBCS_NO_HYDRA") == "1":
        raise ImportError("force argparse fallback")
    import hydra  # type: ignore
    from omegaconf import OmegaConf, DictConfig  # type: ignore
    HYDRA_OK = True
except Exception:
    HYDRA_OK = False
    DictConfig = dict  # fallback type alias


try:
    import mlflow  # type: ignore
    MLFLOW_OK = True
except Exception:
    MLFLOW_OK = False

# ============================
# Utils & Logging
# ============================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_jsonable(x):
    import numpy as _np
    if isinstance(x, _np.ndarray):
        return x.tolist()
    if isinstance(x, (_np.floating, _np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x

class JsonlLogger:
    def __init__(self, run_dir: str):
        ensure_dir(run_dir)
        self.f = open(os.path.join(run_dir, "events.jsonl"), "a", encoding="utf-8")
        self.start = time.time()
        self.run_dir = run_dir
    def write(self, obj: dict):
        obj = {"ts": time.time(), **obj}
        self.f.write(json.dumps(_to_jsonable(obj), ensure_ascii=False) + "\n")
        self.f.flush()
    def close(self):
        try:
            self.f.close()
        except:  # noqa: E722
            pass


def normalize_simplex(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= eps:
        return np.ones_like(w) / len(w)
    return w / s


def softclip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sufficient_samples(dataset, min_n=20):
    cnt = Counter([it["action"] for it in dataset])
    return bool(cnt) and all(c >= min_n for c in cnt.values())

# ============================
# Consistency indices & metrics
# ============================

# Saaty CI for a single pairwise matrix P

def saaty_ci(P: np.ndarray) -> float:
    # largest eigenvalue λ_max via power iteration (stable)
    n = P.shape[0]
    v = np.ones((n,)) / n
    for _ in range(50):
        v = P @ v
        v = v / (np.linalg.norm(v) + 1e-12)
    lam = float((v @ (P @ v)) / (v @ v))
    ci = (lam - n) / max(1, (n - 1))
    return float(max(0.0, ci))

# Harmonic Consistency Index (Stein & Mizzi, simplified practical form)

def harmonic_ci(P: np.ndarray) -> float:
    # reciprocal assumed; use row-wise harmonic means
    n = P.shape[0]
    H = np.zeros(n)
    for i in range(n):
        row = P[i, :]
        H[i] = n / np.sum(1.0 / (row + 1e-12))
    # ideal consistent row ratios proportional to weights; deviation proxy
    h_norm = H / (np.mean(H) + 1e-12)
    dev = float(np.mean(np.abs(h_norm - 1.0)))
    return float(min(1.0, max(0.0, dev)))


def gini_coeff(x: np.ndarray) -> float:
    x = np.array(x, dtype=float).flatten()
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(np.clip(x, 0, None))
    n = len(x)
    cum = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(max(0.0, min(1.0, g)))


def compute_group_ranking(participants: List) -> List[int]:
    """Compute group ranking using Borda count aggregation.
    This is more sensitive to individual disagreements than simple averaging.
    Returns list of alternative IDs in descending order of preference.
    """
    if not participants:
        return []
    
    n_alt = participants[0].S.shape[0]
    borda_scores = np.zeros(n_alt)
    
    for p in participants:
        utilities = p.S @ p.w  # utility scores for this participant
        # Convert to individual ranking
        individual_ranking = list(np.argsort(-utilities))  # desc order
        
        # Assign Borda count scores (n-1 for 1st, n-2 for 2nd, etc.)
        for rank_pos, alt_id in enumerate(individual_ranking):
            borda_scores[alt_id] += (n_alt - 1 - rank_pos)
    
    # Convert Borda scores to ranking (higher score = better rank)
    ranking = list(np.argsort(-borda_scores))  # descending order
    return ranking


def ranking_to_rankvec(order: List[int]) -> List[int]:
    """Convert ranking order (alt_ids in desc score) to rank vector.
    
    Args:
        order: List of alternative IDs in descending score order
        
    Returns:
        rank: List where rank[alt_id] = position (0=best, 1=second, ...)
    """
    n = len(order)
    rank = [0] * n
    for pos, alt_id in enumerate(order):
        rank[alt_id] = pos
    return rank

def compute_kendall_tau(r_prev: Optional[List[int]], r_current: List[int]) -> float:
    """Compute true Kendall-τ between two rankings using rank vectors.
    Returns τ ∈ [-1, 1] where 1 = perfect agreement, 0 = no correlation.
    """
    if r_prev is None or len(r_prev) != len(r_current) or len(r_current) <= 1:
        return 0.0  # No previous ranking or invalid input
    
    # Convert to rank vectors for proper comparison
    try:
        rank_prev = ranking_to_rankvec(r_prev)
        rank_curr = ranking_to_rankvec(r_current)
        
        if SCIPY_OK:
            # Use scipy for true Kendall-τ on rank vectors
            tau, _ = kendalltau(rank_prev, rank_curr)
            return float(tau) if not np.isnan(tau) else 0.0
        else:
            # Fallback: simple correlation-like measure
            # Count concordant vs discordant pairs
            n = len(rank_curr)
            concordant = 0
            discordant = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    # Check if pair (i,j) has same relative order in both rankings
                    prev_order = (rank_prev[i] < rank_prev[j])
                    curr_order = (rank_curr[i] < rank_curr[j])
                    
                    if prev_order == curr_order:
                        concordant += 1
                    else:
                        discordant += 1
            
            total_pairs = concordant + discordant
            if total_pairs == 0:
                return 0.0
            
            tau = (concordant - discordant) / total_pairs
            return float(tau)
    except Exception:
        return 0.0


def kendall_tau_like(S: np.ndarray, w: np.ndarray) -> float:
    # proxy stability score 0.5..1.0 based on margin of top scores
    scores = S @ w
    diffs = np.diff(np.sort(scores))
    margin = float(np.mean(np.abs(diffs))) if diffs.size else 0.0
    return float(0.5 + min(0.5, margin))

# ============================
# VAF (grounded, value-ordered attacks) - Full Implementation
# ============================

def value_order_from_weights(w: np.ndarray) -> List[int]:
    return list(np.argsort(-w))


def vaf_effective_attack(u_val: int, v_val: int, order: List[int]) -> bool:
    pos = {k: i for i, k in enumerate(order)}
    return pos[u_val] <= pos[v_val]


@dataclass  
class ArgumentNode:
    """Single argument in the VAF graph"""
    arg_id: str
    topic: str  # Self / Criteria / Alt
    target: Tuple  # (component, idx...)
    sign: int
    magnitude: float
    confidence: float
    value: Optional[int]  # Associated criterion value
    
    
class VAFGraph:
    """Value-based Argumentation Framework Graph using NetworkX"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.arguments: Dict[str, ArgumentNode] = {}
        
    def add_argument(self, arg_node: ArgumentNode) -> None:
        """Add argument node to the graph"""
        self.graph.add_node(arg_node.arg_id)
        self.arguments[arg_node.arg_id] = arg_node
        
    def add_attack(self, attacker_id: str, target_id: str) -> None:
        """Add attack relation between arguments"""
        self.graph.add_edge(attacker_id, target_id, relation='attack')
        
    def add_support(self, supporter_id: str, target_id: str) -> None:
        """Add support relation between arguments"""
        self.graph.add_edge(supporter_id, target_id, relation='support')
        
    def compute_attacks_by_value_order(self, value_order: List[int]) -> None:
        """Compute attack relations based on value ordering"""
        # Clear existing attack/support edges
        self.graph.clear_edges()
        
        # Generate attacks based on value preferences
        for arg1_id, arg1 in self.arguments.items():
            for arg2_id, arg2 in self.arguments.items():
                if arg1_id == arg2_id:
                    continue
                    
                # If both arguments have values, check attack relation
                if arg1.value is not None and arg2.value is not None:
                    if vaf_effective_attack(arg1.value, arg2.value, value_order):
                        # arg1 can attack arg2 if arg1's value is preferred
                        if arg1.topic == arg2.topic and arg1.sign != arg2.sign:
                            self.add_attack(arg1_id, arg2_id)
                        elif arg1.confidence > arg2.confidence:  # Confidence-based attack
                            self.add_attack(arg1_id, arg2_id)
                            
    def compute_grounded_extension(self) -> set:
        """Compute grounded extension using iterative algorithm"""
        all_args = set(self.graph.nodes())
        in_ext = set()  # Accepted arguments
        out_ext = set()  # Rejected arguments
        
        changed = True
        while changed:
            changed = False
            
            # Find arguments with no undefeated attackers
            for arg in list(all_args - in_ext - out_ext):
                attackers = {pred for pred in self.graph.predecessors(arg) 
                           if self.graph[pred][arg].get('relation') == 'attack'}
                
                if not attackers or attackers.issubset(out_ext):
                    # No attackers or all attackers defeated -> accept
                    in_ext.add(arg)
                    changed = True
                    
            # Find arguments attacked by accepted arguments
            for arg in list(all_args - in_ext - out_ext):
                attackers = {pred for pred in self.graph.predecessors(arg)
                           if self.graph[pred][arg].get('relation') == 'attack'}
                
                if attackers.intersection(in_ext):
                    # Attacked by accepted argument -> reject
                    out_ext.add(arg)
                    changed = True
                    
        return in_ext
        
    def get_accepted_arguments(self, value_order: List[int]) -> set:
        """Get accepted arguments based on value ordering"""
        self.compute_attacks_by_value_order(value_order)
        return self.compute_grounded_extension()


def grounded_vaf_one(arg_value: Optional[int], w_values: np.ndarray) -> bool:
    """Legacy simple VAF acceptance (for backward compatibility)"""
    if arg_value is None:
        return True
    order = value_order_from_weights(w_values)
    # 攻撃先の価値はここでは仮定しない→単に上位価値の主張は通りやすい
    return order.index(arg_value) <= max(0, len(order)//2)

# ============================
# AGAU updates
# ============================

def agau_weight_update(w: np.ndarray, s: np.ndarray, eta_w: float, cap: float) -> np.ndarray:
    eta = softclip(eta_w, 0.0, cap)
    w_new = w * np.exp(eta * s)
    return normalize_simplex(w_new)


def agau_score_update(S: np.ndarray, M: np.ndarray, eta_S: float, cap: float) -> np.ndarray:
    eta = softclip(eta_S, 0.0, cap)
    return S * np.exp(eta * M)


def nearest_consistent_matrix(P: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(P)
    u = np.real(vecs[:, np.argmax(np.real(vals))])
    u = np.clip(u, 1e-9, None)
    U = np.outer(u, 1.0 / u)
    return U


def agau_pairwise_blend(P: np.ndarray, lam: float, lam_max: float) -> np.ndarray:
    lam = softclip(lam, 0.0, lam_max)
    U = nearest_consistent_matrix(P)
    logP = np.log(P + 1e-12)
    logU = np.log(U + 1e-12)
    logP_new = (1 - lam) * logP + lam * logU
    P_new = np.exp(logP_new)
    P_new = (P_new + 1.0 / P_new.T) / 2.0
    np.fill_diagonal(P_new, 1.0)
    return P_new


def geom_mean_pairwise(mats: List[np.ndarray]) -> np.ndarray:
    """Element-wise geometric mean for a set of reciprocal pairwise matrices.
    Enforces reciprocity and unit diagonals.
    """
    if not mats:
        raise ValueError("geom_mean_pairwise: empty mats")
    stack = np.stack([np.log(np.clip(M, 1e-12, None)) for M in mats], axis=0)
    log_mean = np.mean(stack, axis=0)
    P = np.exp(log_mean)
    P = (P + 1.0 / P.T) / 2.0
    np.fill_diagonal(P, 1.0)
    return P

# ============================
# Data structures
# ============================

TOPIC = ("Self", "Criteria", "Alt")

@dataclass
class Argument:
    topic: str  # Self / Criteria / Alt
    target: Tuple  # (component, idx...)
    sign: int
    magnitude: float
    confidence: float
    recipient: Optional[int]
    value: Optional[int]

@dataclass
class Participant:
    w: np.ndarray
    S: np.ndarray
    pairwise: Dict[int, np.ndarray]
    CI_saaty: float
    CI_harm: float
    # 個人差パラメタ
    theta_value: float = 1.2
    theta_evidence: float = 0.8
    theta_fatigue: float = 0.6
    bias: float = 0.0
    noise: float = 0.0
    # 履歴（疲労用）
    recent_topics: List[str] = field(default_factory=list)

    def accept_prob(self, arg: Argument, w_bar: np.ndarray) -> float:
        # 価値整合
        align = 0.0
        if arg.value is not None:
            align = float(self.w[int(arg.value)])
        # 証拠
        evidence = float(arg.confidence) * float(arg.magnitude)
        # 反復疲労（直近K=5の同topic回数）
        K = 5
        fatigue = float(self.recent_topics[-K:].count(arg.topic))
        z = (self.bias
             + self.theta_value * align
             + self.theta_evidence * evidence
             - self.theta_fatigue * fatigue)
        if self.noise > 0:
            z += np.random.normal(0, self.noise)
        return float(1.0 / (1.0 + math.exp(-z)))

    def push_topic(self, topic: str):
        self.recent_topics.append(topic)
        if len(self.recent_topics) > 32:
            self.recent_topics = self.recent_topics[-32:]

# ============================
# Policy (LinUCB-like for topics)
# ============================

@dataclass
class LinUCBPolicy:
    d: int
    alpha: float = 0.8
    topics: Tuple[str, ...] = ("Self", "Criteria", "Alt")
    A: Dict[str, np.ndarray] = field(default_factory=dict)
    b: Dict[str, np.ndarray] = field(default_factory=dict)
    softmax_temp: float = 0.5  # 行動確率化の温度

    def __post_init__(self):
        for a in self.topics:
            self.A[a] = np.eye(self.d)
            self.b[a] = np.zeros((self.d,))

    def features(self, obs: Dict) -> np.ndarray:
        feats = np.array([
            obs.get("mean_ci_saaty", 0.2),
            obs.get("mean_ci_harm", 0.2),
            obs.get("max_ci_saaty", 0.3),
            obs.get("w_var", 0.05),
            obs.get("top_split", 0.1),
            obs.get("ema_accept", 0.5),
            obs.get("step_left", 1.0),
        ], dtype=float)
        if feats.size < self.d:
            feats = np.pad(feats, (0, self.d - feats.size))
        elif feats.size > self.d:
            feats = feats[: self.d]
        return feats

    def _scores(self, x: np.ndarray) -> Dict[str, float]:
        scores = {}
        for a in self.topics:
            A = self.A[a]
            b = self.b[a]
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            mean = float(theta @ x)
            conf = float(self.alpha * math.sqrt(max(1e-12, x @ A_inv @ x)))
            scores[a] = mean + conf
        return scores

    def _softmax(self, scores: Dict[str, float], temp: Optional[float] = None) -> Dict[str, float]:
        if temp is None or temp <= 0:
            temp = 1e-3
        arr = np.array([scores[a] for a in self.topics], dtype=float) / float(temp)
        arr = arr - np.max(arr)
        prob = np.exp(arr)
        prob = prob / (np.sum(prob) + 1e-12)
        return {a: float(prob[i]) for i, a in enumerate(self.topics)}

    def select(self, obs: Dict) -> Tuple[str, float, np.ndarray, Dict[str, float], Dict[str, float]]:
        x = self.features(obs)
        scores = self._scores(x)
        probs = self._softmax(scores, self.softmax_temp)
        actions = list(self.topics)
        pvec = np.array([probs[a] for a in actions])
        idx = int(np.random.choice(len(actions), p=pvec))
        a = actions[idx]
        return a, float(probs[a]), x, probs, scores

    def update(self, obs: Dict, action: str, reward: float):
        x = self.features(obs)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

    def update(self, obs: Dict, action: str, reward: float):
        x = self.features(obs)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x

# ============================
# Environment (Mediator)
# ============================

@dataclass
class EnvConfig:
    n_agents: int = 5
    n_alt: int = 5
    n_crit: int = 5
    Tmax: int = 200
    eta_w: float = 0.05
    eta_S: float = 0.03
    lam_max: float = 0.1
    diffuse_recipient: float = 1.0
    diffuse_others: float = 0.2
    # convergence threshold for true Kendall-τ
    tau_star: float = 0.85  # τ* threshold for convergence
    # safety rails
    clip_eta_w: float = 0.15
    clip_eta_S: float = 0.15
    safety_ci_increase_max: float = 0.02
    safety_tau_drop_max: float = 0.03
    safety_gini_increase_max: float = 0.03
    # individual rail for CI increase (applied per-person)
    safety_ci_increase_max_ind: float = 0.03
    # experiment mode: 'full' | 'explain' | 'baseline'
    mode: str = "full"
    # testing: create intentional disagreement to test tau computation
    create_disagreement: bool = False

@dataclass
class MediatorEnv:
    cfg: EnvConfig
    rng: random.Random = field(default_factory=random.Random)
    participants: List[Participant] = field(default_factory=list)
    t: int = 0
    ema_accept: float = 0.5
    prev_ranking: Optional[List[int]] = None  # Previous ranking r(t-1) for Kendall-τ
    vaf_graph: VAFGraph = field(default_factory=VAFGraph)  # VAF argument graph

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            self.rng.seed(seed)
        self.participants = []
        
        # Check if we should create intentional disagreement for testing
        create_disagreement = getattr(self.cfg, 'create_disagreement', False)
        
        for i in range(self.cfg.n_agents):
            # 個人差サンプル
            theta_v = np.random.normal(1.2, 0.2)
            theta_e = np.random.normal(0.8, 0.2)
            theta_f = np.random.normal(0.6, 0.2)
            bias = np.random.normal(0.0, 0.3)
            noise = abs(np.random.normal(0.0, 0.1))

            if create_disagreement and self.cfg.n_agents >= 2 and self.cfg.n_alt >= 2:
                # Create intentional disagreement: alternate ranking preferences
                if i % 2 == 0:
                    # Preference for alternatives 0, 1, 2, ... in that order
                    w = normalize_simplex(np.random.dirichlet([2.0] + [0.5] * (self.cfg.n_crit - 1)))
                    S = np.zeros((self.cfg.n_alt, self.cfg.n_crit))
                    for a in range(self.cfg.n_alt):
                        for k in range(self.cfg.n_crit):
                            # Decreasing scores for alternatives: 0=best, 1=second, etc.
                            base_score = 3.0 - 0.5 * a + np.random.normal(0, 0.2)
                            S[a, k] = max(0.1, base_score)
                else:
                    # Opposite preference: alternatives in reverse order
                    w = normalize_simplex(np.random.dirichlet([0.5] * (self.cfg.n_crit - 1) + [2.0]))
                    S = np.zeros((self.cfg.n_alt, self.cfg.n_crit))
                    for a in range(self.cfg.n_alt):
                        for k in range(self.cfg.n_crit):
                            # Increasing scores for alternatives: reverse order
                            base_score = 2.0 + 0.5 * a + np.random.normal(0, 0.2)
                            S[a, k] = max(0.1, base_score)
            else:
                # Original random initialization
                w = normalize_simplex(np.random.dirichlet(np.ones(self.cfg.n_crit)))
                S = np.abs(np.random.randn(self.cfg.n_alt, self.cfg.n_crit))
            
            pairwise = {}
            CI_s_list, CI_h_list = [], []
            for k in range(self.cfg.n_crit):
                P = np.abs(np.random.lognormal(mean=0.0, sigma=0.5, size=(self.cfg.n_alt, self.cfg.n_alt)))
                P = (P + 1.0 / P.T) / 2.0
                np.fill_diagonal(P, 1.0)
                pairwise[k] = P
                CI_s_list.append(saaty_ci(P))
                CI_h_list.append(harmonic_ci(P))
            p = Participant(w=w, S=S, pairwise=pairwise,
                            CI_saaty=float(np.mean(CI_s_list)), CI_harm=float(np.mean(CI_h_list)),
                            theta_value=theta_v, theta_evidence=theta_e,
                            theta_fatigue=theta_f, bias=bias, noise=noise)
            self.participants.append(p)
        self.t = 0
        self.ema_accept = 0.5
        self.prev_ranking = None
        self.vaf_graph = VAFGraph()  # Reset VAF graph
        return self.observe()

    # heuristics
    def pick_recipient(self) -> int:
        cis = [p.CI_saaty for p in self.participants]
        return int(np.argmax(cis))

    def pick_target(self, topic: str) -> Tuple:
        """Enhanced target selection based on CI/influence/divergence."""
        if topic == "Self":
            # Select criterion and recipient with highest CI
            j = self.pick_recipient()  # Already selects highest CI participant
            
            # Select criterion with highest CI for this participant
            if j < len(self.participants):
                p = self.participants[j]
                crit_cis = [saaty_ci(p.pairwise[k]) for k in range(self.cfg.n_crit)]
                if crit_cis:
                    # Weighted selection favoring high CI
                    weights = np.array(crit_cis) + 0.01  # Add small constant to avoid zeros
                    weights = weights / weights.sum()
                    k = int(np.random.choice(self.cfg.n_crit, p=weights))
                else:
                    k = self.rng.randrange(self.cfg.n_crit)
            else:
                k = self.rng.randrange(self.cfg.n_crit)
                
            # Select alternatives with largest inconsistency
            i = self.rng.randrange(self.cfg.n_alt)
            i2 = (i + 1) % self.cfg.n_alt
            return ("pairwise", k, i, i2)
            
        elif topic == "Criteria":
            # Select criterion with highest variance across participants
            if self.participants:
                weights_matrix = np.vstack([p.w for p in self.participants])
                variances = np.var(weights_matrix, axis=0)
                if variances.size > 0:
                    # Weighted selection favoring high variance criteria
                    weights = variances + 0.01  # Add small constant
                    weights = weights / weights.sum()
                    k = int(np.random.choice(self.cfg.n_crit, p=weights))
                else:
                    k = self.rng.randrange(self.cfg.n_crit)
            else:
                k = self.rng.randrange(self.cfg.n_crit)
            return ("weight", k)
            
        elif topic == "Alt":
            # Select alternative-criterion pair with highest disagreement
            if self.participants and len(self.participants) > 1:
                max_var = -1
                best_a, best_k = 0, 0
                
                for a in range(self.cfg.n_alt):
                    for k in range(self.cfg.n_crit):
                        # Calculate variance of scores for this (alt, crit) pair
                        scores = [p.S[a, k] for p in self.participants]
                        var = float(np.var(scores))
                        if var > max_var:
                            max_var = var
                            best_a, best_k = a, k
                            
                # With probability 0.7, select the highest variance pair
                if self.rng.random() < 0.7:
                    return ("score", best_a, best_k)
                    
            # Fallback to random selection
            a = self.rng.randrange(self.cfg.n_alt)
            k = self.rng.randrange(self.cfg.n_crit)
            return ("score", a, k)
            
        raise ValueError("unknown topic")

    def w_bar(self) -> np.ndarray:
        return normalize_simplex(np.mean([p.w for p in self.participants], axis=0))

    def group_gini(self) -> float:
        utils = [float((p.S @ p.w).max()) for p in self.participants]
        return gini_coeff(np.array(utils))

    def group_pairwise_AIJ(self) -> Dict[int, np.ndarray]:
        """Aggregate individual pairwise matrices per criterion by AIJ (geometric mean)."""
        if not self.participants:
            return {}
        n_crit = self.cfg.n_crit
        P_grp: Dict[int, np.ndarray] = {}
        for k in range(n_crit):
            mats = [p.pairwise[k] for p in self.participants]
            P_grp[k] = geom_mean_pairwise(mats)
        return P_grp

    def group_ci_from_AIJ(self, P_grp: Optional[Dict[int, np.ndarray]] = None) -> Tuple[float, float]:
        if P_grp is None:
            P_grp = self.group_pairwise_AIJ()
        if not P_grp:
            return 0.0, 0.0
        ci_s = float(np.mean([saaty_ci(P) for P in P_grp.values()]))
        ci_h = float(np.mean([harmonic_ci(P) for P in P_grp.values()]))
        return ci_s, ci_h

    def observe(self, commit_prev_ranking: bool = False) -> Dict:
        """Observe current state without side effects (unless commit_prev_ranking=True).
        
        Args:
            commit_prev_ranking: If True, update prev_ranking (only call once per step)
        """
        wbar = self.w_bar()
        
        # Compute current ranking r(t)
        current_ranking = compute_group_ranking(self.participants)
        
        # Compute true Kendall-τ with previous ranking (without side effects)
        tau = compute_kendall_tau(self.prev_ranking, current_ranking)
        
        # For backward compatibility, also compute tau_like
        tau_like = float(np.mean([kendall_tau_like(p.S, p.w) for p in self.participants]))
        
        # Only update previous ranking if explicitly requested
        if commit_prev_ranking:
            self.prev_ranking = current_ranking.copy() if current_ranking else None
        
        ci_s = float(np.mean([p.CI_saaty for p in self.participants]))
        ci_h = float(np.mean([p.CI_harm for p in self.participants]))
        w_var = float(np.var(np.vstack([p.w for p in self.participants]), axis=0).mean())
        top_idx = [int(np.argmax(p.S @ p.w)) for p in self.participants]
        split = 1.0 - (np.bincount(top_idx, minlength=self.cfg.n_alt).max() / max(1, len(top_idx)))
        
        return {
            "w_bar": wbar,
            "tau": tau,  # True Kendall-τ between r(t-1) and r(t)
            "tau_like": tau_like,  # Legacy proxy metric
            "ranking": current_ranking,  # Current ranking r(t)
            "mean_ci_saaty": ci_s,
            "mean_ci_harm": ci_h,
            "max_ci_saaty": float(max([p.CI_saaty for p in self.participants])),
            "w_var": w_var,
            "top_split": float(split),
            "gini": self.group_gini(),
            "ema_accept": float(self.ema_accept),
            "step_left": float(max(0, self.cfg.Tmax - self.t) / self.cfg.Tmax),
        }

    def argument_from(self, topic: str, recipient: Optional[int] = None) -> Argument:
        tgt = self.pick_target(topic)
        if recipient is None:
            recipient = self.pick_recipient()
        if tgt[0] == "weight":
            v = int(tgt[1])
        elif tgt[0] == "score":
            v = int(tgt[2])
        else:
            v = int(tgt[1])
        return Argument(topic=topic, target=tgt, sign=+1, magnitude=1.0,
                        confidence=0.8, recipient=recipient, value=v)

    def vaf_and_accept(self, arg: Argument) -> Tuple[bool, List[bool], float, set]:
        """VAF allowed check using full argumentation graph, then per-individual acceptance.
        Returns (allowed, accept_mask, accept_rate, accepted_arg_set).
        """
        # Add current argument to VAF graph
        arg_id = f"arg_{self.t}_{arg.topic}_{arg.target}"
        arg_node = ArgumentNode(
            arg_id=arg_id,
            topic=arg.topic,
            target=arg.target,
            sign=arg.sign,
            magnitude=arg.magnitude,
            confidence=arg.confidence,
            value=arg.value
        )
        self.vaf_graph.add_argument(arg_node)
        
        # Compute accepted arguments using VAF grounded semantics
        value_order = value_order_from_weights(self.w_bar())
        accepted_arg_ids = self.vaf_graph.get_accepted_arguments(value_order)
        
        # Check if current argument is in accepted set
        allowed = arg_id in accepted_arg_ids
        n = len(self.participants)
        
        if not allowed:
            self.ema_accept = 0.7 * self.ema_accept + 0.3 * 0.0
            return False, [False] * n, 0.0, accepted_arg_ids
            
        # per-individual acceptance (for individual differences)
        probs = [p.accept_prob(arg, self.w_bar()) for p in self.participants]
        accept_mask = [(random.random() < pr) for pr in probs]
        
        # update fatigue history for everyone exposed to the reason
        for p in self.participants:
            p.push_topic(arg.topic)
            
        acc_rate = float(np.mean(accept_mask)) if accept_mask else 0.0
        self.ema_accept = 0.7 * self.ema_accept + 0.3 * acc_rate
        return True, accept_mask, acc_rate, accepted_arg_ids

    def build_agau_parameters_from_accepted_set(self, accepted_arg_ids: set) -> Tuple[np.ndarray, np.ndarray, float]:
        """Build AGAU parameters (s, M, λ) from accepted argument set.
        Returns (support_vector, support_matrix, consistency_lambda).
        """
        # Initialize parameters
        s = np.zeros(self.cfg.n_crit)  # Criteria weight support vector
        M = np.zeros((self.cfg.n_alt, self.cfg.n_crit))  # Score support matrix
        lam = 0.0  # Consistency correction strength
        
        # Accumulate support from accepted arguments
        for arg_id in accepted_arg_ids:
            if arg_id not in self.vaf_graph.arguments:
                continue
                
            arg_node = self.vaf_graph.arguments[arg_id]
            
            if arg_node.topic == "Criteria" and len(arg_node.target) >= 2:
                if arg_node.target[0] == "weight":
                    k = int(arg_node.target[1])
                    if 0 <= k < self.cfg.n_crit:
                        s[k] += arg_node.sign * arg_node.magnitude * arg_node.confidence
                        
            elif arg_node.topic == "Alt" and len(arg_node.target) >= 3:
                if arg_node.target[0] == "score":
                    a, k = int(arg_node.target[1]), int(arg_node.target[2])
                    if 0 <= a < self.cfg.n_alt and 0 <= k < self.cfg.n_crit:
                        M[a, k] += arg_node.sign * arg_node.magnitude * arg_node.confidence
                        
            elif arg_node.topic == "Self" and len(arg_node.target) >= 2:
                if arg_node.target[0] == "pairwise":
                    # Accumulate consistency correction strength
                    lam += 0.08 * arg_node.confidence  # Base strength * confidence
                    
        # Normalize λ to reasonable range
        lam = min(lam, self.cfg.lam_max)
        
        return s, M, lam

    def agau_apply_collective(self, accepted_arg_set: set, accept_mask: List[bool]) -> Dict:
        """Apply AGAU updates based on collective accepted argument set.
        This is the new method that aligns with the concept flow.
        """
        if not any(accept_mask) or not accepted_arg_set:
            return {"applied": False, "reason": "not_accepted"}
            
        # Build AGAU parameters from accepted argument set
        s, M, lam = self.build_agau_parameters_from_accepted_set(accepted_arg_set)
        
        # Full snapshot for potential rollback
        snap = [(p.w.copy(), p.S.copy(), {k: v.copy() for k, v in p.pairwise.items()}, p.CI_saaty, p.CI_harm)
                for p in self.participants]
        obs_before = self.observe()
        P_grp_prev = self.group_pairwise_AIJ()
        grp_ci_s_prev, grp_ci_h_prev = self.group_ci_from_AIJ(P_grp_prev)
        
        # Apply weight updates if s has non-zero components
        if np.any(np.abs(s) > 1e-6):
            for idx, p in enumerate(self.participants):
                if accept_mask[idx]:
                    p.w = agau_weight_update(p.w, s, self.cfg.eta_w, self.cfg.clip_eta_w)
                    
        # Apply score updates if M has non-zero components
        if np.any(np.abs(M) > 1e-6):
            for idx, p in enumerate(self.participants):
                if accept_mask[idx]:
                    p.S = agau_score_update(p.S, M, self.cfg.eta_S, self.cfg.clip_eta_S)
                    
        # Apply pairwise consistency updates if λ > 0
        if lam > 1e-6:
            # Individual snapshots for per-person rollback
            indiv_snaps = [(
                p.w.copy(), p.S.copy(), {kk: vv.copy() for kk, vv in p.pairwise.items()}, p.CI_saaty, p.CI_harm
            ) for p in self.participants]
            
            for idx, p in enumerate(self.participants):
                if accept_mask[idx]:
                    for k in range(self.cfg.n_crit):
                        P = p.pairwise[k]
                        p.pairwise[k] = agau_pairwise_blend(P, lam, self.cfg.lam_max)
                        
            # Recompute CI and apply individual rail
            for idx, p in enumerate(self.participants):
                p.CI_saaty = float(np.mean([saaty_ci(Pk) for Pk in p.pairwise.values()]))
                p.CI_harm = float(np.mean([harmonic_ci(Pk) for Pk in p.pairwise.values()]))
                w0, S0, P0, ci_s0, ci_h0 = indiv_snaps[idx]
                if (p.CI_saaty - ci_s0) > self.cfg.safety_ci_increase_max_ind:
                    # Rollback this individual only
                    p.w = w0; p.S = S0; p.pairwise = P0; p.CI_saaty = ci_s0; p.CI_harm = ci_h0
        
        # Group AIJ rail check
        obs_after = self.observe()
        P_grp_new = self.group_pairwise_AIJ()
        grp_ci_s_new, grp_ci_h_new = self.group_ci_from_AIJ(P_grp_new)
        ci_up = grp_ci_s_new - grp_ci_s_prev
        tau_drop = obs_before["tau"] - obs_after["tau"]
        gini_up = obs_after["gini"] - obs_before["gini"]
        
        violated = (ci_up > self.cfg.safety_ci_increase_max or
                    tau_drop > self.cfg.safety_tau_drop_max or
                    gini_up > self.cfg.safety_gini_increase_max)
                    
        if violated:
            for p, (w, S, Pdict, ci_s, ci_h) in zip(self.participants, snap):
                p.w = w; p.S = S; p.pairwise = Pdict; p.CI_saaty = ci_s; p.CI_harm = ci_h
            return {"applied": False, "reason": "safety_rollback",
                    "delta": {"ci_up": ci_up, "tau_drop": tau_drop, "gini_up": gini_up}}
                    
        return {"applied": True,
                "delta": {"ci_up": ci_up, "tau_drop": tau_drop, "gini_up": gini_up},
                "agau_params": {"s_norm": float(np.linalg.norm(s)), "M_norm": float(np.linalg.norm(M)), "lambda": float(lam)}}

    def agau_apply_safe(self, arg: Argument, accepted: Any) -> Dict:
        """Apply AGAU only to accepted individuals; enforce individual CI rails and group AIJ rails.
        accepted can be bool or List[bool] (per-individual)."""
        # unify accepted mask
        if isinstance(accepted, list):
            accept_mask = [bool(x) for x in accepted]
            accepted_any = any(accept_mask)
        else:
            accept_mask = [bool(accepted)] * len(self.participants)
            accepted_any = bool(accepted)
        if not accepted_any:
            return {"applied": False, "reason": "not_accepted"}
        # full snapshot for potential group rollback
        snap = [(p.w.copy(), p.S.copy(), {k: v.copy() for k, v in p.pairwise.items()}, p.CI_saaty, p.CI_harm)
                for p in self.participants]
        obs_before = self.observe()
        P_grp_prev = self.group_pairwise_AIJ()
        grp_ci_s_prev, grp_ci_h_prev = self.group_ci_from_AIJ(P_grp_prev)
        if arg.topic == "Criteria" and arg.target[0] == "weight":
            k = int(arg.target[1])
            s = np.zeros(self.cfg.n_crit)
            s[k] = arg.sign * arg.magnitude * arg.confidence
            for idx, p in enumerate(self.participants):
                if accept_mask[idx]:
                    eta = self.cfg.eta_w
                    p.w = agau_weight_update(p.w, s, eta, self.cfg.clip_eta_w)
        elif arg.topic == "Alt" and arg.target[0] == "score":
            a, k = int(arg.target[1]), int(arg.target[2])
            M = np.zeros((self.cfg.n_alt, self.cfg.n_crit))
            M[a, k] = arg.sign * arg.magnitude * arg.confidence
            for idx, p in enumerate(self.participants):
                if accept_mask[idx]:
                    eta = self.cfg.eta_S
                    p.S = agau_score_update(p.S, M, eta, self.cfg.clip_eta_S)
        elif arg.topic == "Self" and arg.target[0] == "pairwise":
            k = int(arg.target[1])
            # individual snapshots for per-person rollback
            indiv_snaps = [(
                p.w.copy(), p.S.copy(), {kk: vv.copy() for kk, vv in p.pairwise.items()}, p.CI_saaty, p.CI_harm
            ) for p in self.participants]
            for idx, p in enumerate(self.participants):
                if accept_mask[idx]:
                    P = p.pairwise[k]
                    p.pairwise[k] = agau_pairwise_blend(P, lam=0.08, lam_max=self.cfg.lam_max)
            # recompute CI and apply individual rail
            violated_any_ind = False
            for idx, p in enumerate(self.participants):
                # recompute CI for this individual
                p.CI_saaty = float(np.mean([saaty_ci(Pk) for Pk in p.pairwise.values()]))
                p.CI_harm  = float(np.mean([harmonic_ci(Pk) for Pk in p.pairwise.values()]))
                w0, S0, P0, ci_s0, ci_h0 = indiv_snaps[idx]
                if (p.CI_saaty - ci_s0) > self.cfg.safety_ci_increase_max_ind:
                    # rollback this individual only
                    violated_any_ind = True
                    p.w = w0; p.S = S0; p.pairwise = P0; p.CI_saaty = ci_s0; p.CI_harm = ci_h0
        else:
            return {"applied": False, "reason": "unknown_target"}
        # group AIJ rail
        obs_after = self.observe()
        P_grp_new = self.group_pairwise_AIJ()
        grp_ci_s_new, grp_ci_h_new = self.group_ci_from_AIJ(P_grp_new)
        ci_up = grp_ci_s_new - grp_ci_s_prev
        tau_drop = obs_before["tau"] - obs_after["tau"]
        gini_up = obs_after["gini"] - obs_before["gini"]
        violated = (ci_up > self.cfg.safety_ci_increase_max or
                    tau_drop > self.cfg.safety_tau_drop_max or
                    gini_up > self.cfg.safety_gini_increase_max)
        if violated:
            for p, (w, S, Pdict, ci_s, ci_h) in zip(self.participants, snap):
                p.w = w; p.S = S; p.pairwise = Pdict; p.CI_saaty = ci_s; p.CI_harm = ci_h
            return {"applied": False, "reason": "safety_rollback",
                    "delta": {"ci_up": ci_up, "tau_drop": tau_drop, "gini_up": gini_up}}
        return {"applied": True,
                "delta": {"ci_up": ci_up, "tau_drop": tau_drop, "gini_up": gini_up}}

    def should_stop(self) -> bool:
        """Check convergence without side effects using cached tau value."""
        # Use cached tau from the last step computation (no additional observe() call)
        tau_converged = getattr(self, 'tau_last', 0.0) >= self.cfg.tau_star
        
        # Safety conditions: timeout and acceptance stall
        timeup = self.t >= self.cfg.Tmax
        accept_stall = self.ema_accept < 0.1  # Prevent infinite loops
        
        if self.cfg.mode == "explain":
            # Explain mode: use time limit (no actual updates)
            return bool(tau_converged or timeup or self.t >= min(120, self.cfg.Tmax))
        elif self.cfg.mode == "baseline":
            # Baseline mode: traditional AHP convergence with CI focus
            # Get current state without side effects for CI check
            current_ci = float(np.mean([p.CI_saaty for p in self.participants]))
            ci_ok = current_ci <= 0.01
            tau_stagnant = getattr(self, 'tau_last', 0.0) <= 0.72
            early = ci_ok and tau_stagnant and self.t >= 60
            return bool(tau_converged or timeup or early)
        else:  # full mode
            # Full RBCS: primary τ≥τ* with safety nets
            return bool(tau_converged or accept_stall or timeup)

    def step(self, topic: str) -> Tuple[Dict, float, bool, Dict]:
        self.t += 1
        
        # Get previous ranking before any updates (no side effects)
        r_prev = self.prev_ranking or compute_group_ranking(self.participants)
        
        # Get observation before intervention (no side effects)
        obs_before = self.observe(commit_prev_ranking=False)
        
        arg = self.argument_from(topic)
        
        # --- mode branches ---
        if self.cfg.mode == "explain":
            allowed, accept_mask, acc_rate, accepted_arg_set = self.vaf_and_accept(arg)
            res = {"applied": False, "reason": "explain_mode"}
        elif self.cfg.mode == "baseline":
            # AHP-only: Self(topic)のCI是正のみ適用、VAFは参照しない
            accepted = True if arg.topic == "Self" else False
            accept_mask = [accepted] * len(self.participants)
            acc_rate = float(np.mean(accept_mask)) if accept_mask else 0.0
            accepted_arg_set = set()  # No VAF in baseline
            if arg.topic == "Self" and arg.target[0] == "pairwise":
                res = self.agau_apply_safe(arg, True)
            else:
                res = {"applied": False, "reason": "baseline_no_update"}
        else:  # full
            allowed, accept_mask, acc_rate, accepted_arg_set = self.vaf_and_accept(arg)
            # Apply AGAU based on accepted argument set
            res = self.agau_apply_collective(accepted_arg_set, accept_mask if allowed else [False] * len(self.participants))
        
        # Compute current ranking after updates
        r_now = compute_group_ranking(self.participants)
        
        # Compute true tau between r_prev and r_now
        tau_true = compute_kendall_tau(r_prev, r_now)
        
        # Cache tau for should_stop() to use
        self.tau_last = tau_true
        
        # Update prev_ranking only once at the end
        self.prev_ranking = r_now.copy() if r_now else None
        
        # Get final observation (no side effects, prev_ranking already updated)
        obs_after = self.observe(commit_prev_ranking=False)
        # Override tau with the true computed value
        obs_after["tau"] = tau_true
        
        # reward
        r = (obs_after["tau"] - obs_before["tau"]) \
            - (obs_after["mean_ci_saaty"] - obs_before["mean_ci_saaty"]) \
            - 0.5 * (obs_after["gini"] - obs_before["gini"]) \
            - 0.01
        
        done = self.should_stop()
        accepted_any = bool(allowed and any(accept_mask)) if 'allowed' in locals() else bool(accepted)
        
        info = {
            "arg": asdict(arg),
            "accepted": accepted_any,
            "accept_rate_step": float(acc_rate if 'acc_rate' in locals() else (1.0 if accepted_any else 0.0)),
            "apply": res,
            "obs_before": obs_before,
            "obs_after": obs_after,
            "r_prev": r_prev,  # For debugging
            "r_now": r_now,   # For debugging
            "tau_computed": tau_true  # For debugging
        }
        return obs_after, float(r), done, info

# ============================
# OPE (IPS / DR) utilities
# ============================

class FrozenPolicy(LinUCBPolicy):
    def update(self, *args, **kwargs):
        return  # no-op
    def prob_given_x(self, x: np.ndarray, action: str) -> float:
        scores = self._scores(x)
        probs = self._softmax(scores, temp=1e-3)  # near-greedy
        return float(probs[action])


def save_policy(policy: LinUCBPolicy, path: str):
    data = {
        "d": policy.d,
        "alpha": policy.alpha,
        "topics": list(policy.topics),
        "softmax_temp": policy.softmax_temp,
        "A": {a: policy.A[a].tolist() for a in policy.topics},
        "b": {a: policy.b[a].tolist() for a in policy.topics},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_policy(path: str) -> FrozenPolicy:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pol = FrozenPolicy(d=int(data["d"]), alpha=float(data["alpha"]))
    pol.softmax_temp = float(data.get("softmax_temp", 0.5))
    for a in data["topics"]:
        pol.A[a] = np.array(data["A"][a], dtype=float)
        pol.b[a] = np.array(data["b"][a], dtype=float)
    return pol


def estimate_ips(dataset: List[dict], target: FrozenPolicy, w_clip: float = 10.0, self_norm: bool = True) -> float:
    ws, rs = [], []
    for it in dataset:
        x = np.array(it["x"])  # features
        a = it["action"]
        r = float(it["reward"])
        pb = float(it["b_prob"])  # behavior prob
        pt = float(target.prob_given_x(x, a))
        pb = max(pb, 1e-3)
        pt = max(pt, 1e-6)
        w = min(pt / pb, w_clip)
        ws.append(w); rs.append(r)
    if self_norm:
        return float(np.sum(np.array(ws) * np.array(rs)) / (np.sum(ws) + 1e-12))
    return float(np.mean(np.array(ws) * np.array(rs)))


def fit_ridge_per_action(dataset: List[dict], lam: float = 1.0) -> Dict[str, Tuple[np.ndarray, float]]:
    # returns {action: (theta, bias)} minimizing ||X theta + b - y||^2 + lam||theta||^2
    lam = max(lam, 5.0)
    by_act: Dict[str, List[Tuple[np.ndarray, float]]] = {}
    for it in dataset:
        by_act.setdefault(it["action"], []).append((np.array(it["x"]), float(it["reward"])) )
    models: Dict[str, Tuple[np.ndarray, float]] = {}
    for a, pairs in by_act.items():
        X = np.stack([x for x, _ in pairs], axis=0)
        y = np.array([y for _, y in pairs], dtype=float)
        n, d = X.shape
        X1 = np.c_[X, np.ones((n, 1))]
        I = np.eye(d + 1); I[-1, -1] = 0.0  # bias not regularized
        theta_b = np.linalg.pinv(X1.T @ X1 + lam * I) @ (X1.T @ y)
        theta, b = theta_b[:-1], float(theta_b[-1])
        models[a] = (theta, b)
    return models


def predict_reward(models: Dict[str, Tuple[np.ndarray, float]], x: np.ndarray, a: str) -> float:
    if a not in models:
        return 0.0
    theta, b = models[a]
    return float(x @ theta + b)


def estimate_dr(dataset: List[dict], target: FrozenPolicy, models: Dict[str, Tuple[np.ndarray, float]], w_clip: float = 10.0) -> float:
    vals = []
    for it in dataset:
        x = np.array(it["x"])
        a = it["action"]
        r = float(it["reward"]) 
        pb = float(it["b_prob"]) 
        pt = float(target.prob_given_x(x, a))
        pb = max(pb, 1e-3)
        pt = max(pt, 1e-6)
        w = min(pt / pb, w_clip)
        # Direct term: sum_a pi(a|x) * Q_hat(x,a)
        direct = 0.0
        # approximate over existing actions
        for aa in target.topics:
            pt_aa = float(target.prob_given_x(x, aa))
            direct += pt_aa * predict_reward(models, x, aa)
        # control variate
        q_hat = predict_reward(models, x, a)
        vals.append(direct + w * (r - q_hat))
    return float(np.mean(vals))

# ============================
# Training loop
# ============================

def run_training(cfg: Mapping[str, Any]):
    # run dir
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", stamp)
    logger = JsonlLogger(run_dir)

    # env / policy
    env = MediatorEnv(EnvConfig(
        n_agents=cfg.get("n_agents", 5),
        n_alt=cfg.get("n_alternatives", 5),
        n_crit=cfg.get("n_criteria", 5),
        Tmax=cfg.get("Tmax", 200),
        tau_star=cfg.get("tau_star", 0.85),  # Convergence threshold
        eta_w=cfg.get("eta_w", 0.05),
        eta_S=cfg.get("eta_S", 0.03),
        lam_max=cfg.get("lam_max", 0.1),
        diffuse_recipient=cfg.get("diffuse_recipient", 1.0),
        diffuse_others=cfg.get("diffuse_others", 0.2),
        clip_eta_w=cfg.get("clip_eta_w", 0.15),
        clip_eta_S=cfg.get("clip_eta_S", 0.15),
        safety_ci_increase_max=cfg.get("safety_ci_increase_max", 0.02),
        safety_tau_drop_max=cfg.get("safety_tau_drop_max", 0.03),
        safety_gini_increase_max=cfg.get("safety_gini_increase_max", 0.03),
        mode=cfg.get("mode", "full"),
    ))
    policy = LinUCBPolicy(d=7, alpha=float(cfg.get("ucb_alpha", 0.8)), softmax_temp=float(cfg.get("ucb_softmax_temp", 0.5)))

    episodes = int(cfg.get("episodes", 50))
    seed = int(cfg.get("seed", 42))

    if MLFLOW_OK and cfg.get("mlflow", True):
        mlflow.set_experiment(cfg.get("experiment_name", "rbcs_pretrain"))
        mlflow.start_run(run_name=f"run_{stamp}")
        mlflow.log_params({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))})
        mlflow.log_param("run_dir", run_dir)

    dataset_steps: List[dict] = []

    try:
        for ep in range(episodes):
            obs = env.reset(seed + ep)
            total_r = 0.0
            steps = 0
            while True:
                action, b_prob, x, prob_dict, score_dict = policy.select(obs)
                obs_next, r, done, info = env.step(action)
                policy.update(obs, action, r)
                total_r += r; steps += 1
                # collect dataset for OPE
                dataset_steps.append({
                    "x": x.tolist(),
                    "action": action,
                    "b_prob": float(b_prob),
                    "reward": float(r),
                })
                # logging per step (compact)
                logger.write({
                    "type": "step",
                    "ep": ep, "t": env.t,
                    "action": {"topic": action, "propensity": b_prob},
                    "b_prob": float(b_prob),
                    "reward": float(r),
                    "accepted": info.get("accepted", False),
                    "apply": info.get("apply", {}),
                    "tau": obs_next["tau"],  # True Kendall-τ
                    "tau_like": obs_next.get("tau_like", obs_next["tau"]),  # Legacy proxy
                    "ranking": obs_next.get("ranking", []),  # Current ranking r(t)
                    "ci_s": obs_next["mean_ci_saaty"],
                    "ci_h": obs_next["mean_ci_harm"],
                    "gini": obs_next["gini"],
                    "accept_rate": float(env.ema_accept),
                    "rollback": 1.0 if (info.get("apply", {}).get("reason") == "safety_rollback") else 0.0,
                    "x": x.tolist(),
                })
                obs = obs_next
                if done:
                    break
            summary = {"episode": ep, "return": total_r, "steps": steps, **env.observe()}
            logger.write({"type": "episode_end", **summary})
            print(f"[ep {ep}] R={total_r:.3f} steps={steps} tau={summary['tau']:.3f} CI={summary['mean_ci_saaty']:.3f} gini={summary['gini']:.3f}")
            if MLFLOW_OK and cfg.get("mlflow", True):
                mlflow.log_metrics({"return": total_r, "steps": steps,
                                   "tau": summary['tau'], "ci_s": summary['mean_ci_saaty'],
                                   "ci_h": summary['mean_ci_harm'], "gini": summary['gini']}, step=ep)
        # ---- freeze policy and OPE ----
        policy_path = os.path.join(run_dir, "policy.json")
        save_policy(policy, policy_path)
        frozen = load_policy(policy_path)
        ips = estimate_ips(dataset_steps, frozen, w_clip=float(cfg.get("ope_w_clip", 5.0)), self_norm=True)
        if sufficient_samples(dataset_steps, min_n=int(cfg.get("ope_min_per_action", 20))):
            models = fit_ridge_per_action(dataset_steps, lam=float(cfg.get("ope_ridge_lam", 5.0)))
            dr = estimate_dr(dataset_steps, frozen, models, w_clip=float(cfg.get("ope_w_clip", 5.0)))
        else:
            dr = float("nan")
        logger.write({"type": "ope", "ips": ips, "dr": dr, "policy_path": policy_path})
        print(f"OPE => IPS: {ips:.4f}  DR: {dr:.4f}  saved: {policy_path}")
        if MLFLOW_OK and cfg.get("mlflow", True):
            mlflow.log_metric("ips", ips)
            if math.isfinite(dr):
                mlflow.log_metric("dr", dr)
            mlflow.log_artifact(policy_path)
    finally:
        logger.close()
        if MLFLOW_OK and cfg.get("mlflow", True):
            mlflow.end_run()

# ============================
# Entrypoint (Hydra fallback)
# ============================

def run_selection(cfg: Mapping[str, Any]):
    """
    すでに存在する runs/* を走査し、OPE(IPS)をブートストラップで評価。
    下側95%CIが最大の run を「最良」として policy を凍結コピー。
    """
    import glob, shutil

    runs_glob = cfg.get("runs_glob", "runs/2025*")
    n_boot = int(cfg.get("n_boot", 1000))
    w_clip = float(cfg.get("ope_w_clip", 5.0))

    def load_steps(run_dir: str):
        ds = []
        path = os.path.join(run_dir, "events.jsonl")
        if not os.path.exists(path):
            return ds
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "step" and all(k in ev for k in ("x", "action", "gini")):
                    r = ev.get("reward", None)
                    if r is None:
                        continue
                    # accept both top-level b_prob or nested under action
                    if "b_prob" in ev:
                        bprob = float(ev["b_prob"])
                    else:
                        bprob = float(ev.get("action", {}).get("propensity", 0.0))
                    a = ev["action"]["topic"] if isinstance(ev.get("action"), dict) else ev.get("action")
                    if a is None:
                        continue
                    ds.append({
                        "x": ev["x"],
                        "action": a,
                        "b_prob": bprob,
                        "reward": float(r),
                    })
        return ds

    def snips(policy: FrozenPolicy, dataset, w_clip=5.0):
        # Self-normalized IPS
        num = 0.0; den = 0.0
        for it in dataset:
            x = np.array(it["x"], dtype=float)
            a = it["action"]
            r = float(it["reward"]); pb = float(it["b_prob"])
            pt = float(policy.prob_given_x(x, a))
            pb = max(pb, 1e-3); pt = max(pt, 1e-6)
            w = min(pt / pb, w_clip)
            num += w * r; den += w
        return float(num / (den + 1e-12))

    def boot_ci(values, alpha=0.05):
        vals = np.array(values, dtype=float)
        return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))

    candidates = []
    for run_dir in sorted(glob.glob(runs_glob)):
        pol_path = os.path.join(run_dir, "policy.json")
        if not os.path.exists(pol_path):
            continue
        try:
            pol = load_policy(pol_path)
        except Exception:
            continue
        ds = load_steps(run_dir)
        if not ds:
            continue

        rng = np.random.default_rng(0)
        boots = []
        n = len(ds)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = [ds[i] for i in idx]
            boots.append(snips(pol, sample, w_clip=w_clip))
        lb, ub = boot_ci(boots, alpha=0.05)
        mid = float(np.mean(boots))
        candidates.append({"run_dir": run_dir, "policy": pol_path, "ips_mid": mid, "ips_lb": lb, "ips_ub": ub})

    if not candidates:
        print("[select] No candidates found. Set runs_glob=... correctly.")
        return

    candidates.sort(key=lambda x: (x["ips_lb"], x["ips_mid"], x["run_dir"]), reverse=True)
    best = candidates[0]
    print(f"[select] best={best['run_dir']}  IPS_mid={best['ips_mid']:.4f}  CI95=[{best['ips_lb']:.4f},{best['ips_ub']:.4f}]")

    frozen_dir = os.path.join("runs", "frozen_best")
    os.makedirs(frozen_dir, exist_ok=True)
    dst = os.path.join(frozen_dir, "frozen_policy.json")
    shutil.copy2(best["policy"], dst)
    with open(os.path.join(frozen_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(f"[select] saved frozen policy -> {dst}")


def run_eval(cfg: Mapping[str, Any]):
    """
    凍結ポリシーで学習せずに評価（policy.update しない）。
    mode は full/explain/baseline をそのまま使用（Full-frozen = policy凍結の意）。
    """
    policy_path = cfg.get("policy_path", "runs/frozen_best/frozen_policy.json")
    episodes = int(cfg.get("episodes", 50))
    seed = int(cfg.get("seed", 2025))

    frozen = load_policy(policy_path)
    env = MediatorEnv(EnvConfig(
        n_agents=cfg.get("n_agents", 5),
        n_alt=cfg.get("n_alternatives", 5),
        n_crit=cfg.get("n_criteria", 5),
        Tmax=cfg.get("Tmax", 200),
        tau_star=cfg.get("tau_star", 0.85),
        mode=cfg.get("mode", "full"),
        eta_w=cfg.get("eta_w", 0.05),
        eta_S=cfg.get("eta_S", 0.03),
        lam_max=cfg.get("lam_max", 0.1),
        diffuse_recipient=cfg.get("diffuse_recipient", 1.0),
        diffuse_others=cfg.get("diffuse_others", 0.2),
        clip_eta_w=cfg.get("clip_eta_w", 0.15),
        clip_eta_S=cfg.get("clip_eta_S", 0.15),
        safety_ci_increase_max=cfg.get("safety_ci_increase_max", 0.03),
        safety_tau_drop_max=cfg.get("safety_tau_drop_max", 0.05),
        safety_gini_increase_max=cfg.get("safety_gini_increase_max", 0.03),
    ))

    stamp = time.strftime("eval_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", stamp)
    logger = JsonlLogger(run_dir)

    try:
        for ep in range(episodes):
            obs = env.reset(seed + ep)
            total_r, steps = 0.0, 0
            while True:
                x = FrozenPolicy(d=7).features(obs)
                a, _, _, probs, _ = frozen.select(obs)
                b_prob = float(probs[a])
                obs_next, r, done, info = env.step(a)
                total_r += r; steps += 1

                logger.write({
                    "type": "step", "phase": "eval", "ep": ep, "t": env.t,
                    "action": {"topic": a, "propensity": b_prob},
                    "b_prob": b_prob,
                    "reward": float(r),
                    "tau": obs_next["tau"],
                    "ci_s": obs_next["mean_ci_saaty"],
                    "ci_h": obs_next["mean_ci_harm"],
                    "gini": obs_next["gini"],
                    "accept_rate": float(env.ema_accept),
                    "rollback": 1.0 if (info.get("apply", {}).get("reason") == "safety_rollback") else 0.0,
                    "x": x.tolist(),
                })
                obs = obs_next
                if done:
                    break

            summ = {"episode": ep, "return": total_r, "steps": steps, **env.observe()}
            logger.write({"type": "episode_end_eval", **summ})
            print(f"[eval ep {ep}] R={total_r:.3f} steps={steps} tau={summ['tau']:.3f} CI={summ['mean_ci_saaty']:.3f} gini={summ['gini']:.3f}")
    finally:
        logger.close()


def dispatch(cfg: Mapping[str, Any]):
    task = cfg.get("task", "train")
    if task == "train":
        return run_training(cfg)
    elif task == "select":
        return run_selection(cfg)
    elif task == "eval":
        return run_eval(cfg)
    else:
        print(f"[main] unknown task={task}, use one of train/select/eval")

def parse_args_fallback():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="train")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_agents", type=int, default=5)
    ap.add_argument("--n_alternatives", type=int, default=5)
    ap.add_argument("--n_criteria", type=int, default=5)
    ap.add_argument("--Tmax", type=int, default=200)
    ap.add_argument("--tau_star", type=float, default=0.85)
    ap.add_argument("--mode", type=str, default="full")
    ap.add_argument("--eta_w", type=float, default=0.05)
    ap.add_argument("--eta_S", type=float, default=0.03)
    ap.add_argument("--lam_max", type=float, default=0.1)
    ap.add_argument("--diffuse_recipient", type=float, default=1.0)
    ap.add_argument("--diffuse_others", type=float, default=0.2)
    ap.add_argument("--clip_eta_w", type=float, default=0.15)
    ap.add_argument("--clip_eta_S", type=float, default=0.15)
    ap.add_argument("--ucb_alpha", type=float, default=0.8)
    ap.add_argument("--ucb_softmax_temp", type=float, default=0.5)
    # safety rails thresholds
    ap.add_argument("--safety_ci_increase_max", type=float, default=0.02)
    ap.add_argument("--safety_tau_drop_max", type=float, default=0.03)
    ap.add_argument("--safety_gini_increase_max", type=float, default=0.03)
    # OPE / selection / eval
    ap.add_argument("--ope_w_clip", type=float, default=5.0)
    ap.add_argument("--ope_ridge_lam", type=float, default=5.0)
    ap.add_argument("--runs_glob", type=str, default="runs/2025*")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--policy_path", type=str, default="runs/frozen_best/frozen_policy.json")
    ap.add_argument("--mlflow", action="store_true")
    return vars(ap.parse_args())

if __name__ == "__main__":
    if HYDRA_OK:
        @hydra.main(version_base=None, config_path=None)
        def _main(cfg: Any):
            # allow simple CLI like hydra=on episodes=100
            cfg = OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(cfg, dict):
                cfg = dict(cfg)
            cfg.setdefault("experiment_name", "rbcs_pretrain")
            dispatch(cfg)
        _main()
    else:
        cfg = parse_args_fallback()
        cfg.setdefault("experiment_name", "rbcs_pretrain")
        dispatch(cfg)
