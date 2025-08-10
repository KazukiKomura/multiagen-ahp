# ablation_study.py
# p2p.pyベースのアブレーション実験：Veto/フロア/CIアウェア機能の個別検証
# 「自明な設計の帰結」vs「発見としての現象」を分離する

import os, math, random, numpy as np, networkx as nx, json, csv
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# p2p.pyから必要な関数群をインポート（簡略版を再実装）
rng = np.random.default_rng()
torch.manual_seed(42)

def normalize_simplex(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 1e-12, None)
    return v / v.sum()

def pearson_scaled(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    sx, sy = x.std(), y.std()
    if not np.isfinite(sx) or not np.isfinite(sy) or sx < 1e-9 or sy < 1e-9:
        return 0.5
    c = np.corrcoef(x, y)
    r = c[0,1]
    return float((r + 1.0) * 0.5) if np.isfinite(r) else 0.5

def weight_entropy(w: np.ndarray) -> float:
    w = np.clip(w, 1e-12, 1.0)
    H = -np.sum(w * np.log(w))
    return H / np.log(len(w))

def clamp_saaty(x: float, lo=1/9, hi=9) -> float:
    return float(np.clip(x, lo, hi))

def random_pairwise(n: int, intensity: float = 0.6, seed: Optional[int] = None) -> np.ndarray:
    local_rng = np.random.default_rng(seed)
    base = local_rng.lognormal(mean=0.0, sigma=intensity, size=n)
    base = base / base.mean()
    P = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            r = base[i] / base[j]
            noise = local_rng.lognormal(mean=0.0, sigma=0.25)
            val = clamp_saaty(r * noise)
            P[i,j] = val
            P[j,i] = 1.0/val
    np.fill_diagonal(P, 1.0)
    return P

def ahp_eigvec_ci(P: np.ndarray, iters: int = 200, tol: float = 1e-10) -> Tuple[np.ndarray, float, float]:
    n = P.shape[0]
    v = np.ones(n, dtype=np.float64) / n
    for _ in range(iters):
        v_new = P @ v
        v_new = v_new / v_new.sum()
        if np.linalg.norm(v_new - v, 1) < tol:
            v = v_new
            break
        v = v_new
    Pv = P @ v
    lam = float((v @ Pv) / (v @ v))
    ci = float((lam - n) / (n - 1)) if n > 2 else 0.0
    v = normalize_simplex(v)
    return v, ci, lam

@dataclass
class ExperimentConfig:
    """実験設定の管理"""
    # 基本設定
    N: int = 12
    A: int = 5  
    C: int = 5
    episodes: int = 60
    steps_per_episode: int = 64
    
    # アブレーション対象の機能フラグ
    enable_veto: bool = True
    enable_floor: bool = True  
    enable_ci_aware: bool = True
    
    # 報酬成分の重み
    reward_si_weight: float = 1.0
    reward_consensus_weight: float = 0.3
    reward_acceptance_weight: float = 0.2
    
    # ネットワークトポロジー
    network_type: str = "watts_strogatz"  # "watts_strogatz", "grid", "complete", "erdos_renyi", "barabasi_albert"
    
    # 学習パラメータ
    eta: float = 0.06
    gamma: float = 0.95
    pi_lr: float = 3e-4
    vf_lr: float = 5e-4
    
    # 種子値
    seed: int = 42

@dataclass
class Arg:
    owner: int
    kind: str
    crit: int
    target: Tuple[int,int]
    sign: int
    conf: float
    strength: float

@dataclass 
class NodeState:
    w: np.ndarray
    S: np.ndarray
    Pcrit: np.ndarray
    Palt: List[np.ndarray]
    CIcrit: float
    inbox: List[Arg]
    last_si: float
    last_cons: float
    w_prior: np.ndarray
    veto_crit: set
    w_floor: np.ndarray
    tau: np.ndarray

@dataclass
class ActionSpec:
    kind: str
    crit: int
    sign: int
    mag: float
    alt: int

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
    def forward(self, x): return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

class AblationExperiment:
    """アブレーション実験の実行クラス"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_experiment()
        
    def setup_experiment(self):
        """実験環境のセットアップ"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        # ネットワーク生成
        self.G = self.create_network()
        self.states = self.init_states()
        self.action_space = self.build_action_space()
        
        # 観測・行動次元
        obs_dim = 4 * self.config.C + 4
        act_dim = len(self.action_space)
        
        # ニューラルネットワーク
        device = torch.device("cpu")
        self.policy = PolicyNet(obs_dim, act_dim).to(device)
        self.valuef = ValueNet(obs_dim).to(device)
        self.opt_pi = optim.Adam(self.policy.parameters(), lr=self.config.pi_lr)
        self.opt_vf = optim.Adam(self.valuef.parameters(), lr=self.config.vf_lr)
        self.device = device
        
    def create_network(self) -> nx.Graph:
        """ネットワークトポロジーの生成"""
        N = self.config.N
        if self.config.network_type == "watts_strogatz":
            return nx.watts_strogatz_graph(N, k=4, p=0.2, seed=self.config.seed)
        elif self.config.network_type == "grid":
            side = int(np.ceil(np.sqrt(N)))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
            return G.subgraph(list(range(N))).copy()
        elif self.config.network_type == "complete":
            return nx.complete_graph(N)
        elif self.config.network_type == "erdos_renyi":
            return nx.erdos_renyi_graph(N, p=0.3, seed=self.config.seed)
        elif self.config.network_type == "barabasi_albert":
            return nx.barabasi_albert_graph(N, m=2, seed=self.config.seed)
        else:
            raise ValueError(f"Unknown network type: {self.config.network_type}")
    
    def init_states(self) -> Dict[int, NodeState]:
        """エージェント状態の初期化"""
        states = {}
        N, A, C = self.config.N, self.config.A, self.config.C
        
        for i in range(N):
            # 基準のペア比較行列
            Pcrit = random_pairwise(C, intensity=0.6, seed=self.config.seed + i)
            w_i, ci_i, _ = ahp_eigvec_ci(Pcrit)
            
            # 代替案のペア比較行列（基準別）
            Palt = []
            S = np.zeros((A, C), dtype=np.float64)
            for c in range(C):
                Pc = random_pairwise(A, intensity=0.6, seed=self.config.seed + i*C + c)
                Palt.append(Pc)
                s_c, _, _ = ahp_eigvec_ci(Pc)
                S[:, c] = s_c
            
            # アブレーション対象の機能設定
            if self.config.enable_veto:
                veto_crit = set([int(rng.integers(0, C))])
            else:
                veto_crit = set()  # Veto無効
                
            if self.config.enable_floor:
                w_floor = np.full(C, 0.05, dtype=np.float64)
            else:
                w_floor = np.zeros(C, dtype=np.float64)  # フロア無効
                
            tau = np.full((A, C), 0.05, dtype=np.float64)
            
            states[i] = NodeState(
                w=w_i.copy(), S=S, Pcrit=Pcrit, Palt=Palt, CIcrit=ci_i,
                inbox=[], last_si=0.5, last_cons=1.0,
                w_prior=w_i.copy(), veto_crit=veto_crit, w_floor=w_floor, tau=tau
            )
        
        return states
    
    def build_action_space(self) -> List[ActionSpec]:
        """行動空間の構築"""
        actions = []
        C, A = self.config.C, self.config.A
        for kind in ('w', 'S'):
            for c in range(C):
                for sign in (-1, +1):
                    for mag in (1.0, 2.0, 3.0):
                        if kind == 'w':
                            actions.append(ActionSpec(kind, c, sign, mag, -1))
                        else:
                            for a in range(A):
                                actions.append(ActionSpec(kind, c, sign, mag, a))
        return actions
    
    def adjust_eta_by_ci_entropy(self, w: np.ndarray, CI: float, eta: float) -> float:
        """CI-エントロピー調整（アブレーション対象）"""
        if not self.config.enable_ci_aware:
            return eta  # CIアウェア無効時は固定学習率
            
        Hn = weight_entropy(w)
        eta_eff = eta
        if Hn < 0.85: eta_eff *= 0.5
        if CI > 0.15: eta_eff *= 0.5
        return eta_eff
    
    def effective_attack_personal(self, att: Arg, tgt: Arg, w: np.ndarray, 
                                 veto_crit: set, w_floor: np.ndarray,
                                 S: np.ndarray, tau: np.ndarray) -> bool:
        """個人レベルでの効果的攻撃判定（アブレーション対象）"""
        # Veto機能のアブレーション
        if self.config.enable_veto:
            if tgt.kind == 'w' and att.sign == -1 and (tgt.target[1] in veto_crit):
                return False
        
        # フロア制約のアブレーション
        if self.config.enable_floor:
            if tgt.kind == 'w' and att.sign == -1 and w[tgt.target[1]] <= w_floor[tgt.target[1]] + 1e-12:
                return False
        
        # S下限制約
        if tgt.kind == 'S' and att.sign == -1:
            a, c = tgt.target
            if S[a, c] <= tau[a, c] + 1e-12:
                return False
                
        # 基本的な価値依存攻撃
        return self.conflicts(att, tgt) and (w[att.crit] + 1e-12 >= w[tgt.crit])
    
    def conflicts(self, a: Arg, b: Arg) -> bool:
        """論証間の競合判定"""
        if a.kind != b.kind:
            return False
        return (a.target == b.target) and (a.sign != b.sign)
    
    def run_experiment(self) -> Dict[str, Any]:
        """実験の実行"""
        results = {
            'config': asdict(self.config),
            'metrics': {
                'final_weights': [],
                'final_utilities': [],  
                'convergence_steps': 0,
                'final_entropy': 0.0,
                'final_consensus': 0.0,
                'final_si_mean': 0.0,
                'single_criterion_dominance': 0.0,
                'positive_bias_ratio': 0.0,
                's_type_ratio': 0.0
            },
            'time_series': {
                'si_trajectory': [],
                'consensus_trajectory': [],
                'entropy_trajectory': [],
                'weight_trajectories': []
            }
        }
        
        # 学習ループ（簡略版）
        action_counts = Counter()
        
        for episode in range(self.config.episodes):
            # 観測収集
            observations = []
            node_indices = []
            for i in self.G.nodes:
                obs = self.build_observation(i)
                observations.append(obs)
                node_indices.append(i)
            
            obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
            
            # 行動選択
            with torch.no_grad():
                logits = self.policy(obs_tensor)
                probs = torch.distributions.Categorical(logits=logits)
                actions = probs.sample()
                
            # 環境ステップ
            actions_dict = {node_indices[k]: int(actions[k]) for k in range(len(node_indices))}
            step_results = self.step_environment(actions_dict)
            
            # 統計収集
            for node_id, action_idx in actions_dict.items():
                spec = self.action_space[action_idx]
                action_counts[(spec.kind, spec.sign)] += 1
            
            # 時系列データ収集
            if episode % 5 == 0:  # 5エピソードごとに記録
                si_values = [self.calculate_si(i) for i in self.G.nodes]
                consensus_values = [self.local_consensus(i) for i in self.G.nodes]
                weights = [self.states[i].w for i in self.G.nodes]
                
                results['time_series']['si_trajectory'].append(np.mean(si_values))
                results['time_series']['consensus_trajectory'].append(np.max(consensus_values))
                results['time_series']['entropy_trajectory'].append(np.mean([weight_entropy(w) for w in weights]))
                results['time_series']['weight_trajectories'].append([w.tolist() for w in weights])
        
        # 最終結果の計算
        final_weights = np.array([self.states[i].w for i in self.G.nodes])
        group_weights = final_weights.mean(axis=0)
        
        results['metrics']['final_weights'] = group_weights.tolist()
        results['metrics']['final_entropy'] = float(weight_entropy(group_weights))
        results['metrics']['single_criterion_dominance'] = float(group_weights.max())
        results['metrics']['final_consensus'] = float(np.max([self.local_consensus(i) for i in self.G.nodes]))
        results['metrics']['final_si_mean'] = float(np.mean([self.calculate_si(i) for i in self.G.nodes]))
        
        # 行動統計
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            positive_actions = action_counts[('w', 1)] + action_counts[('S', 1)]
            s_actions = action_counts[('S', 1)] + action_counts[('S', -1)]
            
            results['metrics']['positive_bias_ratio'] = float(positive_actions / total_actions)
            results['metrics']['s_type_ratio'] = float(s_actions / total_actions)
        
        return results
    
    def build_observation(self, agent_id: int) -> np.ndarray:
        """観測の構築"""
        state = self.states[agent_id]
        C = self.config.C
        
        neighbors = list(self.G.neighbors(agent_id))
        if neighbors:
            neighbor_weights = np.array([self.states[j].w for j in neighbors])
            w_mean = neighbor_weights.mean(axis=0)
            w_std = neighbor_weights.std(axis=0)
        else:
            w_mean = np.zeros_like(state.w)
            w_std = np.zeros_like(state.w)
        
        consensus = self.local_consensus(agent_id)
        
        obs = np.concatenate([
            state.w, w_mean, np.abs(state.w - w_mean), w_std,
            np.array([state.last_si, state.last_cons, state.CIcrit, weight_entropy(state.w)])
        ])
        
        return np.nan_to_num(obs.astype(np.float32), nan=0.0)
    
    def step_environment(self, actions: Dict[int, int]) -> Dict[str, Any]:
        """環境の1ステップ実行（簡略版）"""
        # 簡略化：重みのゴシップ更新のみ実装
        all_weights = {i: self.states[i].w for i in self.G.nodes}
        
        for i in self.G.nodes:
            neighbors = list(self.G.neighbors(i)) + [i]
            neighbor_weights = np.array([all_weights[j] for j in neighbors])
            self.states[i].w = normalize_simplex(neighbor_weights.mean(axis=0))
            
        return {}
    
    def calculate_si(self, agent_id: int) -> float:
        """社会的影響の計算"""
        state = self.states[agent_id]
        group_utility = self.calculate_group_utility()
        individual_utility = state.S @ state.w
        return pearson_scaled(individual_utility, group_utility)
    
    def calculate_group_utility(self) -> np.ndarray:
        """グループ効用の計算"""
        all_S = np.array([self.states[i].S for i in self.G.nodes])
        all_w = np.array([self.states[i].w for i in self.G.nodes])
        
        mean_S = all_S.mean(axis=0)
        mean_w = all_w.mean(axis=0)
        
        return mean_S @ mean_w
    
    def local_consensus(self, agent_id: int) -> float:
        """局所合意度の計算"""
        neighbors = list(self.G.neighbors(agent_id))
        if not neighbors:
            return 0.0
            
        agent_w = self.states[agent_id].w
        distance_sum = sum(np.linalg.norm(agent_w - self.states[j].w, ord=1) 
                          for j in neighbors)
        
        return distance_sum / len(neighbors)

def run_ablation_study():
    """アブレーション研究の実行"""
    
    # 実験設定の組み合わせ
    ablation_configs = [
        # ベースライン（全機能ON）
        {"name": "baseline", "enable_veto": True, "enable_floor": True, "enable_ci_aware": True},
        
        # 個別機能OFF
        {"name": "no_veto", "enable_veto": False, "enable_floor": True, "enable_ci_aware": True},
        {"name": "no_floor", "enable_veto": True, "enable_floor": False, "enable_ci_aware": True},  
        {"name": "no_ci_aware", "enable_veto": True, "enable_floor": True, "enable_ci_aware": False},
        
        # 複数機能OFF
        {"name": "no_veto_floor", "enable_veto": False, "enable_floor": False, "enable_ci_aware": True},
        {"name": "no_safety", "enable_veto": False, "enable_floor": False, "enable_ci_aware": False},
    ]
    
    results = {}
    
    print("=== Ablation Study: Safety Mechanisms ===")
    
    for config_dict in ablation_configs:
        print(f"\nRunning experiment: {config_dict['name']}")
        
        config = ExperimentConfig(**{k: v for k, v in config_dict.items() if k != 'name'})
        experiment = AblationExperiment(config)
        result = experiment.run_experiment()
        
        results[config_dict['name']] = result
        
        # 重要な指標を表示
        metrics = result['metrics']
        print(f"  Final entropy: {metrics['final_entropy']:.4f}")
        print(f"  Single criterion dominance: {metrics['single_criterion_dominance']:.4f}")
        print(f"  Final consensus: {metrics['final_consensus']:.4f}")
        print(f"  Positive bias ratio: {metrics['positive_bias_ratio']:.4f}")
        print(f"  S-type ratio: {metrics['s_type_ratio']:.4f}")
    
    # 結果をJSONで保存
    os.makedirs("ablation_results", exist_ok=True)
    
    with open("ablation_results/safety_mechanisms.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 比較表をCSVで出力
    with open("ablation_results/safety_mechanisms_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "final_entropy", "single_dominance", "final_consensus",
            "positive_bias", "s_type_ratio", "final_si_mean"
        ])
        
        for name, result in results.items():
            metrics = result['metrics']
            writer.writerow([
                name,
                f"{metrics['final_entropy']:.4f}",
                f"{metrics['single_criterion_dominance']:.4f}", 
                f"{metrics['final_consensus']:.4f}",
                f"{metrics['positive_bias_ratio']:.4f}",
                f"{metrics['s_type_ratio']:.4f}",
                f"{metrics['final_si_mean']:.4f}"
            ])
    
    print(f"\nResults saved to ablation_results/")
    return results

if __name__ == "__main__":
    results = run_ablation_study()