#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RBCS Simple — reason-based consensus (norollback edition)
---------------------------------------------------------
- Three argument topics: Criteria(重み)/Alt(スコア)/Self(CI是正)
- VAF(grounded) → 受理集合 → AGAUでw/S/ペア比較を微更新（指数/対数ブレンド）
- Safety railsは**検知のみ**（デフォルトは rollback 無効）。違反はログしつつ適用継続。
- 収束判定を強化：min_steps・min_applies・stable_needed(τ連続達成)
- 初期化を接戦寄せ：小さな更新で順位が動きやすい

Usage:
  python rbcs_simple.py --episodes 10 --n_agents 5
"""
from __future__ import annotations
import os, json, time, math, random, argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Mapping

import numpy as np
import networkx as nx

# ---------- Optional SciPy for true Kendall-τ ----------
try:
    from scipy.stats import kendalltau  # type: ignore
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ============== Utils & Logging ==============

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def _to_jsonable(x):
    import numpy as _np
    if isinstance(x, _np.ndarray): return x.tolist()
    if isinstance(x, (_np.floating, _np.integer)): return x.item()
    if isinstance(x, dict): return {k:_to_jsonable(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [_to_jsonable(v) for v in x]
    return x

class JsonlLogger:
    def __init__(self, run_dir: str):
        ensure_dir(run_dir)
        self.f = open(os.path.join(run_dir, "events.jsonl"), "a", encoding="utf-8")
        self.run_dir = run_dir
    def write(self, obj: dict):
        obj = {"ts": time.time(), **obj}
        self.f.write(json.dumps(_to_jsonable(obj), ensure_ascii=False) + "\n")
        self.f.flush()
    def close(self):
        try: self.f.close()
        except: pass

def normalize_simplex(w: np.ndarray, eps: float=1e-12) -> np.ndarray:
    w = np.clip(np.asarray(w, float), 0.0, None)
    s = float(w.sum())
    return (w / s) if s > eps else np.ones_like(w)/len(w)

def softclip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ============== Metrics (CI/HCI/Gini/Tau) ==============

def saaty_ci(P: np.ndarray) -> float:
    n = P.shape[0]; v = np.ones(n)/n
    for _ in range(50):
        v = P @ v; v = v / (np.linalg.norm(v) + 1e-12)
    lam = float((v @ (P @ v)) / (v @ v))
    return float(max(0.0, (lam - n) / max(1, n-1)))

def harmonic_ci(P: np.ndarray) -> float:
    n = P.shape[0]; H = np.zeros(n)
    for i in range(n):
        row = np.clip(P[i,:], 1e-12, None)
        H[i] = n / np.sum(1.0/row)
    h_norm = H / (np.mean(H) + 1e-12)
    dev = float(np.mean(np.abs(h_norm - 1.0)))
    return float(min(1.0, max(0.0, dev)))

def gini_coeff(x: np.ndarray) -> float:
    x = np.array(x, float).flatten()
    if np.allclose(x, 0): return 0.0
    x = np.sort(np.clip(x, 0, None)); n = len(x); cum = np.cumsum(x)
    g = (n + 1 - 2*np.sum(cum)/cum[-1]) / n
    return float(max(0.0, min(1.0, g)))

def compute_group_ranking(participants: List["Participant"]) -> List[int]:
    if not participants: return []
    n_alt = participants[0].S.shape[0]
    borda = np.zeros(n_alt)
    for p in participants:
        util = p.S @ p.w
        order = list(np.argsort(-util))
        for pos, a in enumerate(order):
            borda[a] += (n_alt - 1 - pos)
    return list(np.argsort(-borda))

def ranking_to_rankvec(order: List[int]) -> List[int]:
    n = len(order); v = [0]*n
    for pos, a in enumerate(order): v[a] = pos
    return v

def tau_between(r_prev: Optional[List[int]], r_curr: List[int]) -> float:
    if r_prev is None or len(r_prev)!=len(r_curr) or len(r_curr)<=1: return 0.0
    rp, rc = ranking_to_rankvec(r_prev), ranking_to_rankvec(r_curr)
    if SCIPY_OK:
        t,_ = kendalltau(rp, rc); return float(t) if not np.isnan(t) else 0.0
    n = len(rc); conc = disc = 0
    for i in range(n):
        for j in range(i+1, n):
            po, co = (rp[i] < rp[j]), (rc[i] < rc[j])
            if po == co: conc += 1
            else: disc += 1
    tot = conc + disc
    return float((conc - disc)/tot) if tot>0 else 0.0

def tau_like(S: np.ndarray, w: np.ndarray) -> float:
    scores = S @ w
    diffs = np.diff(np.sort(scores))
    margin = float(np.mean(np.abs(diffs))) if diffs.size else 0.0
    return float(0.5 + min(0.5, margin))

# ============== VAF (grounded, value-ordered) ==============

def value_order_from_weights(w: np.ndarray) -> List[int]:
    return list(np.argsort(-w))

def vaf_better(val_u: int, val_v: int, order: List[int]) -> bool:
    pos = {k:i for i,k in enumerate(order)}
    return pos[val_u] <= pos[val_v]

@dataclass
class ArgumentNode:
    arg_id: str
    topic: str                # "Self" | "Criteria" | "Alt"
    target: Tuple             # e.g., ("weight", k) | ("score", a, k) | ("pairwise", k, i, j)
    sign: int
    magnitude: float
    confidence: float
    value: Optional[int]      # associated criterion id

class VAFGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.nodes: Dict[str, ArgumentNode] = {}
    def add_argument(self, n: ArgumentNode):
        self.G.add_node(n.arg_id); self.nodes[n.arg_id] = n
    def add_attack(self, u: str, v: str): self.G.add_edge(u, v, relation='attack')
    def compute_attacks(self, value_order: List[int]):
        self.G.remove_edges_from(list(self.G.edges()))
        ids = list(self.nodes.keys())
        for i,u in enumerate(ids):
            a = self.nodes[u]
            for j,v in enumerate(ids):
                if i==j: continue
                b = self.nodes[v]
                if a.value is None or b.value is None: continue
                if a.topic==b.topic and a.sign!=b.sign and vaf_better(a.value, b.value, value_order):
                    self.add_attack(u, v)
                elif a.confidence > b.confidence and vaf_better(a.value, b.value, value_order):
                    self.add_attack(u, v)
    def grounded(self) -> set:
        alln = set(self.G.nodes()); acc=set(); rej=set(); changed=True
        while changed:
            changed=False
            for x in list(alln-acc-rej):
                attackers = {p for p in self.G.predecessors(x)
                             if self.G[p][x].get('relation')=='attack'}
                if not attackers or attackers.issubset(rej):
                    acc.add(x); changed=True
            for x in list(alln-acc-rej):
                attackers = {p for p in self.G.predecessors(x)
                             if self.G[p][x].get('relation')=='attack'}
                if attackers.intersection(acc):
                    rej.add(x); changed=True
        return acc
    def accepted(self, value_order: List[int]) -> set:
        self.compute_attacks(value_order); return self.grounded()

# ============== AGAU (updates) ==============

def agau_weight_update(w: np.ndarray, s: np.ndarray, eta: float, cap: float) -> np.ndarray:
    eta = softclip(eta, 0.0, cap); return normalize_simplex(w * np.exp(eta * s))

def agau_score_update(S: np.ndarray, M: np.ndarray, eta: float, cap: float) -> np.ndarray:
    eta = softclip(eta, 0.0, cap); return S * np.exp(eta * M)

def nearest_consistent(P: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(P)
    u = np.real(vecs[:, np.argmax(np.real(vals))]); u = np.clip(u, 1e-9, None)
    U = np.outer(u, 1.0/u); return U

def pairwise_blend(P: np.ndarray, lam: float, lam_max: float) -> np.ndarray:
    lam = softclip(lam, 0.0, lam_max)
    U = nearest_consistent(P)
    logP = np.log(P + 1e-12); logU = np.log(U + 1e-12)
    logP_new = (1-lam)*logP + lam*logU
    Pn = np.exp(logP_new)
    Pn = (Pn + 1.0/Pn.T)/2.0; np.fill_diagonal(Pn, 1.0)
    return Pn

def geom_mean_pairwise(mats: List[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.log(np.clip(M, 1e-12, None)) for M in mats], axis=0)
    P = np.exp(np.mean(stack, axis=0))
    P = (P + 1.0/P.T)/2.0; np.fill_diagonal(P, 1.0)
    return P

# ============== Data structures ==============

TOPICS = ("Self", "Criteria", "Alt")

@dataclass
class Argument:
    topic: str
    target: Tuple
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
    theta_value: float = 1.2
    theta_evidence: float = 0.8
    theta_fatigue: float = 0.6
    bias: float = 0.0
    noise: float = 0.0
    recent_topics: List[str] = field(default_factory=list)

    def accept_prob(self, arg: Argument, w_bar: np.ndarray) -> float:
        align = float(self.w[int(arg.value)]) if arg.value is not None else 0.0
        evidence = float(arg.confidence) * float(arg.magnitude)
        fatigue = float(self.recent_topics[-5:].count(arg.topic))
        z = (self.bias + self.theta_value*align + self.theta_evidence*evidence - self.theta_fatigue*fatigue)
        if self.noise>0: z += np.random.normal(0, self.noise)
        return float(1.0/(1.0+math.exp(-z)))
    def push_topic(self, t: str):
        self.recent_topics.append(t)
        if len(self.recent_topics)>32: self.recent_topics=self.recent_topics[-32:]

# ============== LinUCB Policy (topics) ==============

@dataclass
class LinUCB:
    d: int = 7
    alpha: float = 0.8
    topics: Tuple[str,...] = TOPICS
    softmax_temp: float = 0.5
    A: Dict[str, np.ndarray] = field(default_factory=dict)
    b: Dict[str, np.ndarray] = field(default_factory=dict)

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
        if feats.size < self.d: feats = np.pad(feats, (0, self.d-feats.size))
        elif feats.size > self.d: feats = feats[:self.d]
        return feats

    def _scores(self, x: np.ndarray) -> Dict[str, float]:
        scores={}
        for a in self.topics:
            A, b = self.A[a], self.b[a]
            Ainv = np.linalg.inv(A); theta = Ainv @ b
            mean = float(theta @ x)
            conf = float(self.alpha * math.sqrt(max(1e-12, x @ Ainv @ x)))
            scores[a] = mean + conf
        return scores

    def _softmax(self, scores: Dict[str,float], temp: float) -> Dict[str,float]:
        t = max(1e-3, temp)
        arr = np.array([scores[a] for a in self.topics])/t
        arr = arr - np.max(arr)
        p = np.exp(arr); p = p/(np.sum(p)+1e-12)
        return {a: float(p[i]) for i,a in enumerate(self.topics)}

    def select(self, obs: Dict) -> Tuple[str,float,np.ndarray,Dict[str,float],Dict[str,float]]:
        x = self.features(obs); sc = self._scores(x)
        probs = self._softmax(sc, self.softmax_temp)
        actions = list(self.topics); pvec = np.array([probs[a] for a in actions])
        idx = int(np.random.choice(len(actions), p=pvec)); a = actions[idx]
        return a, float(probs[a]), x, probs, sc

    def update(self, obs: Dict, action: str, reward: float):
        x = self.features(obs); self.A[action] += np.outer(x,x); self.b[action] += reward*x

# Frozen variant for OPE
class FrozenPolicy(LinUCB):
    def update(self, *args, **kwargs): return
    def prob_given_x(self, x: np.ndarray, action: str) -> float:
        sc = self._scores(x); probs = self._softmax(sc, temp=1e-3)
        return float(probs[action])

def save_policy(pol: LinUCB, path: str):
    data = {
        "d": pol.d, "alpha": pol.alpha, "topics": list(pol.topics),
        "softmax_temp": pol.softmax_temp,
        "A": {a: pol.A[a].tolist() for a in pol.topics},
        "b": {a: pol.b[a].tolist() for a in pol.topics},
    }
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f)

def load_policy(path: str) -> FrozenPolicy:
    with open(path, "r", encoding="utf-8") as f: data=json.load(f)
    pol = FrozenPolicy(d=int(data["d"]), alpha=float(data["alpha"]))
    pol.softmax_temp = float(data.get("softmax_temp", 0.5))
    for a in data["topics"]:
        pol.A[a] = np.array(data["A"][a], float)
        pol.b[a] = np.array(data["b"][a], float)
    return pol

# ============== Environment ==============

@dataclass
class EnvCfg:
    n_agents: int = 5
    n_alt: int = 5
    n_crit: int = 5
    Tmax: int = 200
    eta_w: float = 0.05
    eta_S: float = 0.03
    lam_max: float = 0.1
    tau_star: float = 0.85
    clip_eta_w: float = 0.15
    clip_eta_S: float = 0.15
    safety_ci_increase_max: float = 0.02
    safety_tau_drop_max: float = 0.03
    safety_gini_increase_max: float = 0.03
    safety_ci_increase_max_ind: float = 0.03
    mode: str = "full"   # "full" | "explain" | "baseline"
    # NEW: convergence gates & rollback switch
    min_steps: int = 20
    stable_needed: int = 3
    min_applies: int = 5
    rollback_enabled: bool = False

@dataclass
class Mediator:
    cfg: EnvCfg
    rng: random.Random = field(default_factory=random.Random)
    participants: List[Participant] = field(default_factory=list)
    t: int = 0
    ema_accept: float = 0.5
    prev_ranking: Optional[List[int]] = None
    vaf: VAFGraph = field(default_factory=VAFGraph)
    tau_last: float = 0.0
    stable_streak: int = 0
    applies: int = 0

    def reset(self, seed: Optional[int]=None) -> Dict:
        if seed is not None:
            np.random.seed(seed); self.rng.seed(seed)
        self.participants=[]
        for _ in range(self.cfg.n_agents):
            # 接戦初期化：均等近傍の重み + 小さなスコア差
            w = normalize_simplex(np.random.dirichlet(np.ones(self.cfg.n_crit)*5.0))
            S = 1.0 + 0.05 * np.random.randn(self.cfg.n_alt, self.cfg.n_crit)
            S = np.clip(S, 0.1, None)
            pairwise = {}
            ci_s_list, ci_h_list = [], []
            for k in range(self.cfg.n_crit):
                P = np.abs(np.random.lognormal(mean=0.0, sigma=0.5, size=(self.cfg.n_alt, self.cfg.n_alt)))
                P = (P + 1.0/P.T)/2.0; np.fill_diagonal(P, 1.0)
                pairwise[k]=P; ci_s_list.append(saaty_ci(P)); ci_h_list.append(harmonic_ci(P))
            p = Participant(w=w, S=S, pairwise=pairwise,
                            CI_saaty=float(np.mean(ci_s_list)), CI_harm=float(np.mean(ci_h_list)),
                            theta_value=np.random.normal(1.2,0.2),
                            theta_evidence=np.random.normal(0.8,0.2),
                            theta_fatigue=np.random.normal(0.6,0.2),
                            bias=np.random.normal(0.0,0.3),
                            noise=abs(np.random.normal(0.0,0.1)))
            self.participants.append(p)
        self.t=0; self.ema_accept=0.5; self.prev_ranking=None; self.vaf=VAFGraph(); self.tau_last=0.0
        self.stable_streak = 0; self.applies = 0
        return self.observe(commit_prev=False)

    def w_bar(self) -> np.ndarray:
        return normalize_simplex(np.mean([p.w for p in self.participants], axis=0))

    def group_gini(self) -> float:
        utils = [float((p.S @ p.w).max()) for p in self.participants]
        return gini_coeff(np.array(utils))

    def group_pairwise_AIJ(self) -> Dict[int, np.ndarray]:
        Pgrp: Dict[int,np.ndarray]={}
        for k in range(self.cfg.n_crit):
            Pgrp[k] = geom_mean_pairwise([p.pairwise[k] for p in self.participants])
        return Pgrp

    def group_ci(self, Pgrp: Optional[Dict[int,np.ndarray]]=None) -> Tuple[float,float]:
        if Pgrp is None: Pgrp = self.group_pairwise_AIJ()
        ci_s = float(np.mean([saaty_ci(P) for P in Pgrp.values()]))
        ci_h = float(np.mean([harmonic_ci(P) for P in Pgrp.values()]))
        return ci_s, ci_h

    def observe(self, commit_prev: bool=False) -> Dict:
        wbar = self.w_bar()
        cur_rank = compute_group_ranking(self.participants)
        tau_true = tau_between(self.prev_ranking, cur_rank)
        tau_like_mean = float(np.mean([tau_like(p.S, p.w) for p in self.participants]))
        if commit_prev: self.prev_ranking = cur_rank.copy() if cur_rank else None
        ci_s = float(np.mean([p.CI_saaty for p in self.participants]))
        ci_h = float(np.mean([p.CI_harm for p in self.participants]))
        w_var = float(np.var(np.vstack([p.w for p in self.participants]), axis=0).mean())
        top_idx = [int(np.argmax(p.S @ p.w)) for p in self.participants]
        split = 1.0 - (np.bincount(top_idx, minlength=self.cfg.n_alt).max()/max(1,len(top_idx)))
        return {
            "w_bar": wbar,
            "tau": tau_true, "tau_like": tau_like_mean,
            "ranking": cur_rank,
            "mean_ci_saaty": ci_s, "mean_ci_harm": ci_h, "max_ci_saaty": float(max([p.CI_saaty for p in self.participants])),
            "w_var": w_var, "top_split": float(split),
            "gini": self.group_gini(),
            "ema_accept": float(self.ema_accept),
            "step_left": float(max(0, self.cfg.Tmax - self.t)/self.cfg.Tmax),
        }

    def pick_recipient(self) -> int:
        cis = [p.CI_saaty for p in self.participants]; return int(np.argmax(cis))

    def pick_target(self, topic: str) -> Tuple:
        if topic=="Self":
            j = self.pick_recipient(); k = int(np.argmax([saaty_ci(self.participants[j].pairwise[kk]) for kk in range(self.cfg.n_crit)]))
            i = self.rng.randrange(self.cfg.n_alt); i2 = (i+1)%self.cfg.n_alt
            return ("pairwise", k, i, i2)
        elif topic=="Criteria":
            W = np.vstack([p.w for p in self.participants]); var = np.var(W, axis=0); var = var+1e-6; var=var/var.sum()
            k = int(np.random.choice(self.cfg.n_crit, p=var)); return ("weight", k)
        elif topic=="Alt":
            maxv=-1; best=(0,0)
            for a in range(self.cfg.n_alt):
                for k in range(self.cfg.n_crit):
                    v = float(np.var([p.S[a,k] for p in self.participants]))
                    if v>maxv: maxv=v; best=(a,k)
            if self.rng.random()<0.7: return ("score", best[0], best[1])
            return ("score", self.rng.randrange(self.cfg.n_alt), self.rng.randrange(self.cfg.n_crit))
        raise ValueError("unknown topic")

    def argument_from(self, topic: str, recipient: Optional[int]=None) -> Argument:
        tgt = self.pick_target(topic)
        if recipient is None: recipient=self.pick_recipient()
        if tgt[0]=="weight": v=int(tgt[1])
        elif tgt[0]=="score": v=int(tgt[2])
        else: v=int(tgt[1])
        return Argument(topic=topic, target=tgt, sign=+1, magnitude=1.0, confidence=0.8, recipient=recipient, value=v)

    def vaf_and_accept(self, arg: Argument) -> Tuple[bool, List[bool], float, set]:
        arg_id = f"arg_{self.t}_{arg.topic}_{arg.target}"
        self.vaf.add_argument(ArgumentNode(arg_id, arg.topic, arg.target, arg.sign, arg.magnitude, arg.confidence, arg.value))
        accepted_ids = self.vaf.accepted(value_order_from_weights(self.w_bar()))
        allowed = arg_id in accepted_ids
        n = len(self.participants)
        if not allowed:
            self.ema_accept = 0.7*self.ema_accept + 0.3*0.0
            return False, [False]*n, 0.0, accepted_ids
        probs = [p.accept_prob(arg, self.w_bar()) for p in self.participants]
        acc_mask = [(random.random() < pr) for pr in probs]
        for p in self.participants: p.push_topic(arg.topic)
        rate = float(np.mean(acc_mask)) if acc_mask else 0.0
        self.ema_accept = 0.7*self.ema_accept + 0.3*rate
        return True, acc_mask, rate, accepted_ids

    def build_agau_params(self, acc_set: set) -> Tuple[np.ndarray, np.ndarray, float]:
        s = np.zeros(self.cfg.n_crit); M = np.zeros((self.cfg.n_alt, self.cfg.n_crit)); lam=0.0
        for aid in acc_set:
            n = self.vaf.nodes.get(aid); 
            if n is None: continue
            if n.topic=="Criteria" and n.target[0]=="weight":
                k=int(n.target[1]); s[k]+= n.sign*n.magnitude*n.confidence
            elif n.topic=="Alt" and n.target[0]=="score":
                a,k=int(n.target[1]), int(n.target[2]); M[a,k]+= n.sign*n.magnitude*n.confidence
            elif n.topic=="Self" and n.target[0]=="pairwise":
                lam += 0.08*n.confidence
        lam = min(lam, self.cfg.lam_max)
        return s, M, lam

    def agau_apply_collective(self, acc_set: set, acc_mask: List[bool]) -> Dict:
        if not any(acc_mask) or not acc_set: return {"applied": False, "reason": "not_accepted"}
        s, M, lam = self.build_agau_params(acc_set)
        # snapshot (only used if rollback_enabled)
        if self.cfg.rollback_enabled:
            snap = [(p.w.copy(), p.S.copy(), {k:v.copy() for k,v in p.pairwise.items()}, p.CI_saaty, p.CI_harm) for p in self.participants]
        obs_b = self.observe(commit_prev=False); P_prev = self.group_pairwise_AIJ(); ci_s_prev, _ = self.group_ci(P_prev)
        # weights
        if np.any(np.abs(s)>1e-9):
            for i,p in enumerate(self.participants):
                if acc_mask[i]: p.w = agau_weight_update(p.w, s, self.cfg.eta_w, self.cfg.clip_eta_w)
        # scores
        if np.any(np.abs(M)>1e-9):
            for i,p in enumerate(self.participants):
                if acc_mask[i]: p.S = agau_score_update(p.S, M, self.cfg.eta_S, self.cfg.clip_eta_S)
        # pairwise
        if lam>1e-9:
            if self.cfg.rollback_enabled:
                indiv_snap = [(p.w.copy(), p.S.copy(), {kk:vv.copy() for kk,vv in p.pairwise.items()}, p.CI_saaty, p.CI_harm) for p in self.participants]
            for i,p in enumerate(self.participants):
                if acc_mask[i]:
                    for k in range(self.cfg.n_crit):
                        p.pairwise[k] = pairwise_blend(p.pairwise[k], lam, self.cfg.lam_max)
            # update CI + (optional) individual rail
            for i,p in enumerate(self.participants):
                prev_ci_s = p.CI_saaty
                p.CI_saaty = float(np.mean([saaty_ci(Pk) for Pk in p.pairwise.values()]))
                p.CI_harm  = float(np.mean([harmonic_ci(Pk) for Pk in p.pairwise.values()]))
                if self.cfg.rollback_enabled:
                    w0,S0,P0,ci0,_ = indiv_snap[i]
                    if (p.CI_saaty - ci0) > self.cfg.safety_ci_increase_max_ind:
                        p.w, p.S, p.pairwise, p.CI_saaty, p.CI_harm = w0,S0,P0,ci0, p.CI_harm
        # group rail (detect only by default)
        obs_a = self.observe(commit_prev=False); P_new = self.group_pairwise_AIJ(); ci_s_new,_ = self.group_ci(P_new)
        ci_up = ci_s_new - ci_s_prev
        tau_drop = obs_b["tau"] - obs_a["tau"]
        gini_up = obs_a["gini"] - obs_b["gini"]
        violated = (ci_up>self.cfg.safety_ci_increase_max or tau_drop>self.cfg.safety_tau_drop_max or gini_up>self.cfg.safety_gini_increase_max)
        if violated and self.cfg.rollback_enabled:
            for p,(w,S,Pd,ci_s,ci_h) in zip(self.participants, snap):
                p.w=w; p.S=S; p.pairwise=Pd; p.CI_saaty=ci_s; p.CI_harm=ci_h
            return {"applied": False, "reason": "safety_rollback",
                    "delta": {"ci_up": ci_up, "tau_drop": tau_drop, "gini_up": gini_up}}
        # default: no rollback, just report violation flag
        return {"applied": True,
                "delta": {"ci_up": ci_up, "tau_drop": tau_drop, "gini_up": gini_up},
                "violated": bool(violated),
                "agau_params": {"s_norm": float(np.linalg.norm(s)), "M_norm": float(np.linalg.norm(M)), "lambda": float(lam)}}

    def should_stop(self) -> bool:
        # Gates
        if self.t < self.cfg.min_steps or self.applies < self.cfg.min_applies:
            return False
        tau_ok = self.tau_last >= self.cfg.tau_star
        timeup = self.t >= self.cfg.Tmax
        stall = self.ema_accept < 0.1
        if self.cfg.mode=="explain":
            return bool((self.stable_streak >= self.cfg.stable_needed) or timeup or self.t>=min(120,self.cfg.Tmax))
        if self.cfg.mode=="baseline":
            ci_now = float(np.mean([p.CI_saaty for p in self.participants]))
            ci_ok = ci_now <= 0.01; tau_stagnant = self.tau_last <= 0.72
            early = ci_ok and tau_stagnant and self.t>=60
            return bool((self.stable_streak >= self.cfg.stable_needed) or timeup or early)
        return bool((self.stable_streak >= self.cfg.stable_needed) or stall or timeup)

    def step(self, topic: str) -> Tuple[Dict, float, bool, Dict]:
        self.t += 1
        # FIX: 初回は prev_ranking=None のまま（tau_between(None, r_now)=0.0）
        r_prev = self.prev_ranking
        obs_b = self.observe(commit_prev=False)
        arg = self.argument_from(topic)
        if self.cfg.mode=="explain":
            allowed, acc_mask, acc_rate, acc_set = self.vaf_and_accept(arg); res={"applied": False, "reason": "explain_mode"}
        elif self.cfg.mode=="baseline":
            accepted = True if arg.topic=="Self" and arg.target[0]=="pairwise" else False
            acc_mask = [accepted]*len(self.participants); acc_rate = float(np.mean(acc_mask)) if acc_mask else 0.0; acc_set=set()
            res = self.agau_apply_collective(acc_set, acc_mask) if accepted else {"applied": False, "reason": "baseline_no_update"}
        else:
            allowed, acc_mask, acc_rate, acc_set = self.vaf_and_accept(arg)
            res = self.agau_apply_collective(acc_set, acc_mask if allowed else [False]*len(self.participants))

        r_now = compute_group_ranking(self.participants)
        tau_true = tau_between(r_prev, r_now)
        self.tau_last = tau_true
        self.prev_ranking = r_now.copy() if r_now else None
        # 連続安定カウント
        self.stable_streak = self.stable_streak + 1 if tau_true >= self.cfg.tau_star else 0
        # 反映回数カウント
        if res.get("applied"): self.applies += 1

        obs_a = self.observe(commit_prev=False); obs_a["tau"]=tau_true
        reward = (obs_a["tau"] - obs_b["tau"]) - (obs_a["mean_ci_saaty"] - obs_b["mean_ci_saaty"]) - 0.5*(obs_a["gini"] - obs_b["gini"]) - 0.01
        done = self.should_stop()
        info = {
            "arg": asdict(arg), "accepted": bool(any(acc_mask)),
            "accept_rate_step": float(acc_rate),
            "apply": res, "obs_before": obs_b, "obs_after": obs_a,
            "r_prev": r_prev, "r_now": r_now, "tau_computed": tau_true
        }
        return obs_a, float(reward), bool(done), info

# ============== OPE utils ==============

def estimate_ips(dataset: List[dict], target: FrozenPolicy, w_clip: float=10.0, self_norm: bool=True) -> float:
    ws, rs = [], []
    for it in dataset:
        x = np.array(it["x"]); a = it["action"]; r = float(it["reward"]); pb = float(it["b_prob"])
        pt = float(target.prob_given_x(x, a)); pb=max(pb,1e-3); pt=max(pt,1e-6)
        w = min(pt/pb, w_clip); ws.append(w); rs.append(r)
    ws=np.array(ws); rs=np.array(rs)
    return float(np.sum(ws*rs)/(np.sum(ws)+1e-12)) if self_norm else float(np.mean(ws*rs))

def fit_ridge_per_action(dataset: List[dict], lam: float=5.0) -> Dict[str, Tuple[np.ndarray, float]]:
    by: Dict[str, List[Tuple[np.ndarray, float]]] = {}
    for it in dataset: by.setdefault(it["action"], []).append((np.array(it["x"]), float(it["reward"])))
    models={}
    for a,pairs in by.items():
        X = np.stack([x for x,_ in pairs], axis=0); y = np.array([y for _,y in pairs], float)
        n,d = X.shape; X1 = np.c_[X, np.ones((n,1))]; I = np.eye(d+1); I[-1,-1]=0.0
        theta_b = np.linalg.pinv(X1.T@X1 + lam*I) @ (X1.T@y)
        models[a] = (theta_b[:-1], float(theta_b[-1]))
    return models

def predict_reward(models: Dict[str, Tuple[np.ndarray, float]], x: np.ndarray, a: str) -> float:
    if a not in models: return 0.0
    theta,b = models[a]; return float(x@theta + b)

def estimate_dr(dataset: List[dict], target: FrozenPolicy, models: Dict[str, Tuple[np.ndarray, float]], w_clip: float=10.0) -> float:
    vals=[]
    for it in dataset:
        x = np.array(it["x"]); a = it["action"]; r = float(it["reward"]); pb=float(it["b_prob"])
        pt = float(target.prob_given_x(x, a)); pb=max(pb,1e-3); pt=max(pt,1e-6)
        w = min(pt/pb, w_clip)
        direct=0.0
        for aa in target.topics:
            pt_aa = float(target.prob_given_x(x, aa))
            direct += pt_aa * predict_reward(models, x, aa)
        qhat = predict_reward(models, x, a)
        vals.append(direct + w*(r - qhat))
    return float(np.mean(vals))

# ============== Training / Eval ==============

def run_training(cfg: Mapping[str, Any]):
    stamp = time.strftime("%Y%m%d_%H%M%S"); run_dir = os.path.join("runs", stamp); logger = JsonlLogger(run_dir)
    env = Mediator(EnvCfg(
        n_agents=int(cfg.get("n_agents",5)), n_alt=int(cfg.get("n_alternatives",5)), n_crit=int(cfg.get("n_criteria",5)),
        Tmax=int(cfg.get("Tmax",200)), tau_star=float(cfg.get("tau_star",0.85)),
        eta_w=float(cfg.get("eta_w",0.05)), eta_S=float(cfg.get("eta_S",0.03)), lam_max=float(cfg.get("lam_max",0.1)),
        clip_eta_w=float(cfg.get("clip_eta_w",0.15)), clip_eta_S=float(cfg.get("clip_eta_S",0.15)),
        safety_ci_increase_max=float(cfg.get("safety_ci_increase_max",0.02)), safety_tau_drop_max=float(cfg.get("safety_tau_drop_max",0.03)),
        safety_gini_increase_max=float(cfg.get("safety_gini_increase_max",0.03)),
        safety_ci_increase_max_ind=float(cfg.get("safety_ci_increase_max_ind",0.03)),
        mode=str(cfg.get("mode","full")),
        min_steps=int(cfg.get("min_steps",20)),
        stable_needed=int(cfg.get("stable_needed",3)),
        min_applies=int(cfg.get("min_applies",5)),
        rollback_enabled=bool(cfg.get("rollback_enabled", False)),
    ))
    policy = LinUCB(d=7, alpha=float(cfg.get("ucb_alpha",0.8)), softmax_temp=float(cfg.get("ucb_softmax_temp",0.5)))
    episodes=int(cfg.get("episodes",20)); seed=int(cfg.get("seed",42))
    dataset_steps=[]
    try:
        for ep in range(episodes):
            obs = env.reset(seed+ep); total_r=0.0; steps=0
            while True:
                a, bprob, x, probd, scored = policy.select(obs)
                obs_next, r, done, info = env.step(a)
                policy.update(obs, a, r)
                total_r += r; steps += 1
                dataset_steps.append({"x": x.tolist(), "action": a, "b_prob": float(bprob), "reward": float(r)})
                logger.write({
                    "type":"step","ep":ep,"t":env.t,
                    "action":{"topic":a,"propensity":bprob},
                    "b_prob":float(bprob), "reward":float(r),
                    "accepted": info.get("accepted", False),
                    "apply": info.get("apply", {}),
                    "tau": obs_next["tau"], "tau_like": obs_next.get("tau_like", obs_next["tau"]),
                    "ranking": obs_next.get("ranking", []),
                    "ci_s": obs_next["mean_ci_saaty"], "ci_h": obs_next["mean_ci_harm"],
                    "gini": obs_next["gini"],
                    "accept_rate": float(env.ema_accept),
                    "rollback": 1.0 if (info.get("apply", {}).get("reason")=="safety_rollback") else 0.0,
                    "x": x.tolist(),
                })
                obs = obs_next
                if done: break
            summ = {"episode":ep,"return":total_r,"steps":steps, **env.observe()}
            logger.write({"type":"episode_end", **summ})
            print(f"[ep {ep}] R={total_r:.3f} steps={steps} tau={summ['tau']:.3f} CI={summ['mean_ci_saaty']:.3f} gini={summ['gini']:.3f}")
        # Freeze + OPE
        pol_path = os.path.join(run_dir, "policy.json"); save_policy(policy, pol_path); frozen = load_policy(pol_path)
        ips = estimate_ips(dataset_steps, frozen, w_clip=float(cfg.get("ope_w_clip",5.0)), self_norm=True)
        models = fit_ridge_per_action(dataset_steps, lam=float(cfg.get("ope_ridge_lam",5.0))) if dataset_steps else {}
        dr = estimate_dr(dataset_steps, frozen, models, w_clip=float(cfg.get("ope_w_clip",5.0))) if models else float("nan")
        logger.write({"type":"ope","ips":ips,"dr":dr,"policy_path":pol_path})
        print(f"OPE => IPS: {ips:.4f}  DR: {dr:.4f}  saved: {pol_path}")
    finally:
        logger.close()

def run_eval(cfg: Mapping[str, Any]):
    policy_path = cfg.get("policy_path", "runs/frozen_best/frozen_policy.json")
    episodes = int(cfg.get("episodes", 10)); seed = int(cfg.get("seed", 2025))
    frozen = load_policy(policy_path)
    env = Mediator(EnvCfg(
        n_agents=int(cfg.get("n_agents",5)), n_alt=int(cfg.get("n_alternatives",5)), n_crit=int(cfg.get("n_criteria",5)),
        Tmax=int(cfg.get("Tmax",200)), tau_star=float(cfg.get("tau_star",0.85)),
        eta_w=float(cfg.get("eta_w",0.05)), eta_S=float(cfg.get("eta_S",0.03)), lam_max=float(cfg.get("lam_max",0.1)),
        clip_eta_w=float(cfg.get("clip_eta_w",0.15)), clip_eta_S=float(cfg.get("clip_eta_S",0.15)),
        safety_ci_increase_max=float(cfg.get("safety_ci_increase_max",0.03)), safety_tau_drop_max=float(cfg.get("safety_tau_drop_max",0.05)),
        safety_gini_increase_max=float(cfg.get("safety_gini_increase_max",0.03)),
        safety_ci_increase_max_ind=float(cfg.get("safety_ci_increase_max_ind",0.03)),
        mode=str(cfg.get("mode","full")),
        min_steps=int(cfg.get("min_steps",20)),
        stable_needed=int(cfg.get("stable_needed",3)),
        min_applies=int(cfg.get("min_applies",5)),
        rollback_enabled=bool(cfg.get("rollback_enabled", False)),
    ))
    stamp = time.strftime("eval_%Y%m%d_%H%M%S"); run_dir = os.path.join("runs", stamp); logger = JsonlLogger(run_dir)
    try:
        for ep in range(episodes):
            obs = env.reset(seed+ep); R=0.0; steps=0
            while True:
                x = FrozenPolicy(d=7).features(obs)  # same featurizer shape
                a, _, _, probs, _ = frozen.select(obs); bprob = float(probs[a])
                obs_next, r, done, info = env.step(a)
                R += r; steps+=1
                logger.write({
                    "type":"step","phase":"eval","ep":ep,"t":env.t,
                    "action":{"topic":a,"propensity":bprob},
                    "b_prob":bprob,"reward":float(r),
                    "tau": obs_next["tau"],
                    "ci_s": obs_next["mean_ci_saaty"], "ci_h": obs_next["mean_ci_harm"],
                    "gini": obs_next["gini"],
                    "accept_rate": float(env.ema_accept),
                    "rollback": 1.0 if (info.get("apply", {}).get("reason")=="safety_rollback") else 0.0,
                    "x": x.tolist(),
                })
                obs = obs_next
                if done: break
            summ = {"episode":ep, "return":R, "steps":steps, **env.observe()}
            logger.write({"type":"episode_end_eval", **summ})
            print(f"[eval {ep}] R={R:.3f} steps={steps} tau={summ['tau']:.3f} CI={summ['mean_ci_saaty']:.3f} gini={summ['gini']:.3f}")
    finally:
        logger.close()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="train", choices=["train","eval"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_agents", type=int, default=5)
    ap.add_argument("--n_alternatives", type=int, default=5)
    ap.add_argument("--n_criteria", type=int, default=5)
    ap.add_argument("--Tmax", type=int, default=200)
    ap.add_argument("--tau_star", type=float, default=0.85)
    ap.add_argument("--mode", type=str, default="full", choices=["full","explain","baseline"])
    ap.add_argument("--eta_w", type=float, default=0.05)
    ap.add_argument("--eta_S", type=float, default=0.03)
    ap.add_argument("--lam_max", type=float, default=0.1)
    ap.add_argument("--clip_eta_w", type=float, default=0.15)
    ap.add_argument("--clip_eta_S", type=float, default=0.15)
    ap.add_argument("--ucb_alpha", type=float, default=0.8)
    ap.add_argument("--ucb_softmax_temp", type=float, default=0.5)
    ap.add_argument("--safety_ci_increase_max", type=float, default=0.02)
    ap.add_argument("--safety_tau_drop_max", type=float, default=0.03)
    ap.add_argument("--safety_gini_increase_max", type=float, default=0.03)
    ap.add_argument("--safety_ci_increase_max_ind", type=float, default=0.03)
    ap.add_argument("--ope_w_clip", type=float, default=5.0)
    ap.add_argument("--ope_ridge_lam", type=float, default=5.0)
    ap.add_argument("--policy_path", type=str, default="runs/frozen_best/frozen_policy.json")
    # new gates & rollback switch
    ap.add_argument("--min_steps", type=int, default=20)
    ap.add_argument("--stable_needed", type=int, default=3)
    ap.add_argument("--min_applies", type=int, default=5)
    ap.add_argument("--rollback_enabled", action="store_true")
    return vars(ap.parse_args())

def main():
    cfg = parse_args()
    if cfg.get("task","train")=="train": run_training(cfg)
    else: run_eval(cfg)

if __name__ == "__main__":
    main()
