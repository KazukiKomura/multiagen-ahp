#!/usr/bin/env python3
"""
XR Consensus — Minimal Prototype (single file)

Purpose
-------
A tiny, readable simulation of reason-based (XR) consensus for
multi-criteria group decisions using AHP-style weights.

What it does (per round):
1) Compute group ranking from agents' scores.
2) Detect simple triggers and generate XR candidates (small edits):
   - CriteriaDisagreement XR: move an agent's weights a little toward group mean
   - AlternativeReevaluation XR: nudge a specific alternative's score on the most
     important criterion to resolve near-ties
3) Estimate effect (gain) of each XR via one-step finite differences
4) Greedy pick XRs under a time budget based on gain/cost
5) Apply XRs, log mean Kendall-τ (agreement) and top alternative

No external deps beyond numpy. Designed for 3–5 agents, 3–6 criteria, 4–8 alts.

Run
---
python xr_consensus_miniprototype.py --agents 3 --criteria 3 --alts 5 --rounds 5 --budget 2.0 --seed 42

Notes
-----
- This is NOT a full AHP pipeline (no pairwise matrices / CI). It focuses on
  the XR loop with two triggers and simple, safe updates.
- Extend easily by adding CI/HCI and P2P influence weights T_{ij}.
"""
from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

# ---------------------------- Utilities ------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    s = ex.sum()
    return ex / s if s > 0 else np.ones_like(x) / len(x)


def normalize_simplex(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s < eps:
        w = np.ones_like(w) / len(w)
    else:
        w = w / s
    return w


def clamp(v: np.ndarray, a: float = 0.0, b: float = 1.0) -> np.ndarray:
    return np.minimum(np.maximum(v, a), b)


def kendall_tau(rank1: List[int], rank2: List[int]) -> float:
    """Kendall's tau-b for strict total orders (no ties). n is small here.
    rank lists: order of alternative indices from best -> worst.
    """
    n = len(rank1)
    pos1 = {a: i for i, a in enumerate(rank1)}
    pos2 = {a: i for i, a in enumerate(rank2)}
    concord, discord = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rank1[i], rank1[j]
            s1 = pos1[a] - pos1[b]
            s2 = pos2[a] - pos2[b]
            if s1 * s2 > 0:
                concord += 1
            else:
                discord += 1
    total = concord + discord
    return (concord - discord) / total if total > 0 else 0.0


# ---------------------------- Data Model -----------------------------------
@dataclass
class Agent:
    w: np.ndarray                 # (C,) weights on criteria, simplex
    S: np.ndarray                 # (A,C) scores per alternative x criterion in [0,1]
    name: str

    def clone(self) -> "Agent":
        return Agent(self.w.copy(), self.S.copy(), self.name)


@dataclass
class XRCandidate:
    kind: str                     # "criteria" or "alt"
    sender: int                   # agent idx proposing
    receiver: int                 # agent idx affected
    target: Tuple[int, Optional[int]]  # (alt_idx or -1, crit_idx if applicable)
    dw: Optional[np.ndarray]      # delta on weights for receiver (if criteria)
    dS: Optional[np.ndarray]      # (A,C) sparse delta on scores for receiver (if alt)
    cost: float                   # minutes (budget units)
    note: str                     # human-readable reason


@dataclass
class State:
    agents: List[Agent]

    def clone(self) -> "State":
        return State([ag.clone() for ag in self.agents])


# ---------------------------- Core Computations ----------------------------

def agent_scores(agent: Agent) -> np.ndarray:
    # (A,C)·(C,) -> (A,)
    return agent.S.dot(agent.w)


def group_scores(st: State) -> np.ndarray:
    # mean of individual scores (simple, transparent)
    A = st.agents[0].S.shape[0]
    acc = np.zeros(A)
    for ag in st.agents:
        acc += agent_scores(ag)
    return acc / len(st.agents)


def ranking_from_scores(scores: np.ndarray) -> List[int]:
    return list(np.argsort(scores)[::-1])


def mean_tau_vs_group(st: State) -> Tuple[float, List[List[int]], List[int]]:
    g = group_scores(st)
    rG = ranking_from_scores(g)
    taus = []
    indiv_ranks = []
    for ag in st.agents:
        r = ranking_from_scores(agent_scores(ag))
        indiv_ranks.append(r)
        taus.append(kendall_tau(r, rG))
    return float(np.mean(taus)), indiv_ranks, rG


# ---------------------------- XR Generation --------------------------------

def build_xr_candidates(st: State,
                        tau_thr: float = 0.5,
                        gap_thr: float = 0.05,
                        dw_step: float = 0.02,
                        dS_step: float = 0.03) -> List[XRCandidate]:
    """Create small, safe XR candidates based on two triggers:
    1) CriteriaDisagreement: agent's rank far from group rank → nudge weights toward group mean
    2) AlternativeReevaluation: group top near-tie → nudge top alt's score on most important criterion
    """
    A = st.agents[0].S.shape[0]
    C = st.agents[0].S.shape[1]

    # group aggregates
    mean_w = normalize_simplex(np.mean([ag.w for ag in st.agents], axis=0))
    g = group_scores(st)
    rG = ranking_from_scores(g)
    top, second = rG[0], rG[1]
    gap = g[top] - g[second]

    candidates: List[XRCandidate] = []

    # 1) CriteriaDisagreement
    taus, indiv_ranks, _ = mean_tau_vs_group(st)
    for i, ag in enumerate(st.agents):
        tau_i = kendall_tau(indiv_ranks[i], rG)
        if tau_i < tau_thr:
            direction = mean_w - ag.w
            dw = direction * dw_step  # small move toward consensus
            w_new = normalize_simplex(ag.w + dw)
            dw = w_new - ag.w  # ensure sum=0 and valid
            candidates.append(
                XRCandidate(
                    kind="criteria",
                    sender=i, receiver=i,
                    target=(-1, None),
                    dw=dw, dS=None,
                    cost=1.5,
                    note=f"Criteria XR: move {ag.name} weights toward group mean (tau={tau_i:.2f})"
                )
            )

    # 2) AlternativeReevaluation on most important criterion
    if gap < gap_thr:
        # choose the most important criterion by group mean weight
        crit = int(np.argmax(mean_w))
        for i, ag in enumerate(st.agents):
            # only if agent's top is not the group top
            r_i = ranking_from_scores(agent_scores(ag))
            if r_i[0] != top:
                dS = np.zeros_like(ag.S)
                dS[top, crit] = dS_step
                candidates.append(
                    XRCandidate(
                        kind="alt",
                        sender=i, receiver=i,
                        target=(top, crit),
                        dw=None, dS=dS,
                        cost=1.0,
                        note=f"Alt XR: nudge alt#{top} on crit#{crit} for {ag.name} (near tie)"
                    )
                )

    return candidates


# ---------------------------- Effect Estimation ----------------------------

def apply_xr_inplace(st: State, xr: XRCandidate) -> None:
    ag = st.agents[xr.receiver]
    if xr.kind == "criteria" and xr.dw is not None:
        ag.w = normalize_simplex(ag.w + xr.dw)
    elif xr.kind == "alt" and xr.dS is not None:
        ag.S = clamp(ag.S + xr.dS, 0.0, 1.0)


def estimate_gain(st: State, xr: XRCandidate) -> float:
    base_tau, _, _ = mean_tau_vs_group(st)
    tmp = st.clone()
    apply_xr_inplace(tmp, xr)
    new_tau, _, _ = mean_tau_vs_group(tmp)
    return new_tau - base_tau


# ---------------------------- Greedy Selection -----------------------------

def greedy_select_and_apply(st: State, cands: List[XRCandidate], budget: float) -> Tuple[List[XRCandidate], float, float]:
    selected: List[XRCandidate] = []
    spent = 0.0

    # Recompute gain after each pick (simple myopic recalc)
    remaining = cands.copy()
    while remaining and spent < budget:
        scored = []
        for xr in remaining:
            g = estimate_gain(st, xr)
            eff = g / max(xr.cost, 1e-6)
            scored.append((eff, g, xr))
        scored.sort(key=lambda x: x[0], reverse=True)
        eff, g, best = scored[0]
        if spent + best.cost > budget:
            break
        # apply best
        apply_xr_inplace(st, best)
        selected.append(best)
        spent += best.cost
        # drop this xr and any duplicates on the same receiver-kind
        remaining = [x for x in remaining if x is not best]

    return selected, spent, sum(estimate_gain(st, xr) for xr in selected)


# ---------------------------- Simulation Loop ------------------------------

def simulate(agents: int, criteria: int, alts: int, rounds: int, budget: float, seed: int) -> None:
    rng = np.random.default_rng(seed)

    # init agents
    ags: List[Agent] = []
    for i in range(agents):
        # weights from Dirichlet
        w = rng.dirichlet(np.ones(criteria))
        # scores in [0.3, 0.9]
        S = clamp(0.3 + 0.6 * rng.random((alts, criteria)))
        ags.append(Agent(w=w, S=S, name=f"Agent{i+1}"))

    st = State(agents=ags)

    print(f"Agents={agents}, Criteria={criteria}, Alts={alts}, Rounds={rounds}, Budget={budget} (seed={seed})")

    for r in range(1, rounds + 1):
        mean_tau0, _, rG = mean_tau_vs_group(st)
        g = group_scores(st)
        top, second = rG[0], rG[1]
        gap = g[top] - g[second]
        print(f"\n[Round {r}] group-top=alt#{top} (gap={gap:.3f}), mean-τ={mean_tau0:.3f}")

        cands = build_xr_candidates(st)
        print(f"  candidates: {len(cands)}")
        if not cands:
            print("  (no XR candidates; stopping early)")
            break

        selected, spent, _ = greedy_select_and_apply(st, cands, budget)
        mean_tau1, _, rG1 = mean_tau_vs_group(st)
        print(f"  applied: {len(selected)} (spent {spent:.1f}/{budget} min)")
        for xr in selected:
            print(f"    - {xr.note}")
        print(f"  mean-τ: {mean_tau0:.3f} -> {mean_tau1:.3f}; new group-top=alt#{rG1[0]}")

    # final summary
    g_final = group_scores(st)
    r_final = ranking_from_scores(g_final)
    print("\n=== Final ===")
    print("Group ranking (best->worst):", r_final)
    print("Group scores:")
    for a_idx, sc in enumerate(g_final):
        print(f"  alt#{a_idx}: {sc:.3f}")


# ---------------------------- CLI -----------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="XR Consensus — Minimal Prototype")
    p.add_argument("--agents", type=int, default=3)
    p.add_argument("--criteria", type=int, default=3)
    p.add_argument("--alts", type=int, default=5)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--budget", type=float, default=2.0, help="minutes per round")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    simulate(args.agents, args.criteria, args.alts, args.rounds, args.budget, args.seed)
