# topology_study.py  
# ネットワークトポロジーと規模による頑健性検証
# WS/格子/完全/ER/BA × N=12/24/48 の組み合わせ実験

import os, json, csv
import numpy as np
import networkx as nx
from typing import Dict, List, Any
from ablation_study import AblationExperiment, ExperimentConfig
import matplotlib.pyplot as plt
from itertools import product

class TopologyStudy:
    """トポロジー・規模研究クラス"""
    
    def __init__(self):
        self.topology_configs = {
            "watts_strogatz": {"type": "watts_strogatz"},
            "grid": {"type": "grid"}, 
            "complete": {"type": "complete"},
            "erdos_renyi": {"type": "erdos_renyi"},
            "barabasi_albert": {"type": "barabasi_albert"}
        }
        
        self.scale_configs = [12, 24, 48]
        
    def create_specialized_network(self, network_type: str, N: int, seed: int = 42) -> nx.Graph:
        """特化したネットワーク生成"""
        if network_type == "watts_strogatz":
            k = min(4, N-1)  # 次数を規模に応じて調整
            return nx.watts_strogatz_graph(N, k=k, p=0.2, seed=seed)
            
        elif network_type == "grid":
            side = int(np.ceil(np.sqrt(N)))
            G = nx.grid_2d_graph(side, side)
            G = nx.convert_node_labels_to_integers(G)
            # 必要な数のノードを取得
            nodes = list(G.nodes())[:N]
            return G.subgraph(nodes).copy()
            
        elif network_type == "complete":
            return nx.complete_graph(N)
            
        elif network_type == "erdos_renyi": 
            # 接続確率を規模に応じて調整（平均次数を保つ）
            avg_degree = min(4, N-1)
            p = avg_degree / (N-1) if N > 1 else 0
            return nx.erdos_renyi_graph(N, p=p, seed=seed)
            
        elif network_type == "barabasi_albert":
            m = min(2, N//2) if N > 2 else 1
            return nx.barabasi_albert_graph(N, m=m, seed=seed)
            
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def analyze_network_properties(self, G: nx.Graph) -> Dict[str, float]:
        """ネットワーク特性の分析"""
        N = G.number_of_nodes()
        E = G.number_of_edges()
        
        properties = {
            "num_nodes": N,
            "num_edges": E,
            "density": E / (N * (N-1) / 2) if N > 1 else 0,
            "average_degree": 2 * E / N if N > 0 else 0,
        }
        
        # 連結性チェック
        if nx.is_connected(G):
            properties["is_connected"] = True
            properties["diameter"] = nx.diameter(G)
            properties["average_path_length"] = nx.average_shortest_path_length(G)
            properties["clustering_coefficient"] = nx.average_clustering(G)
        else:
            properties["is_connected"] = False
            properties["diameter"] = float('inf')
            properties["average_path_length"] = float('inf') 
            properties["clustering_coefficient"] = 0.0
            
        return properties
    
    def run_single_experiment(self, network_type: str, N: int, seed: int = 42) -> Dict[str, Any]:
        """単一実験の実行"""
        print(f"  Running: {network_type}, N={N}, seed={seed}")
        
        # 実験設定
        config = ExperimentConfig(
            N=N,
            A=5,
            C=5, 
            episodes=40,  # 規模が大きい場合は短縮
            steps_per_episode=32,
            network_type=network_type,
            seed=seed
        )
        
        # ネットワーク生成と分析
        experiment = AblationExperiment(config)
        network_props = self.analyze_network_properties(experiment.G)
        
        # 実験実行
        results = experiment.run_experiment()
        results['network_properties'] = network_props
        
        return results
    
    def run_topology_study(self, n_seeds: int = 3) -> Dict[str, Any]:
        """トポロジー研究の実行"""
        
        all_results = {}
        summary_data = []
        
        print("=== Topology & Scale Study ===")
        
        for network_type in self.topology_configs.keys():
            for N in self.scale_configs:
                print(f"\nConfiguration: {network_type}, N={N}")
                
                config_results = []
                
                for seed in range(n_seeds):
                    try:
                        result = self.run_single_experiment(network_type, N, seed)
                        config_results.append(result)
                        
                    except Exception as e:
                        print(f"    Error with seed {seed}: {e}")
                        continue
                
                if config_results:
                    # 複数シードの結果をまとめる
                    config_key = f"{network_type}_N{N}"
                    all_results[config_key] = config_results
                    
                    # 統計サマリーを計算
                    summary = self.compute_summary_statistics(config_results, network_type, N)
                    summary_data.append(summary)
                    
                    # 結果表示
                    print(f"    Final entropy: {summary['entropy_mean']:.4f} ± {summary['entropy_std']:.4f}")
                    print(f"    Single dominance: {summary['dominance_mean']:.4f} ± {summary['dominance_std']:.4f}")
                    print(f"    Consensus: {summary['consensus_mean']:.4f} ± {summary['consensus_std']:.4f}")
                    print(f"    Positive bias: {summary['positive_bias_mean']:.4f} ± {summary['positive_bias_std']:.4f}")
        
        # 結果保存
        self.save_results(all_results, summary_data)
        
        return all_results, summary_data
    
    def compute_summary_statistics(self, results: List[Dict], network_type: str, N: int) -> Dict[str, float]:
        """統計サマリーの計算"""
        
        # 各指標の値を収集
        metrics_data = {}
        key_mapping = {
            'final_entropy': 'entropy',
            'single_criterion_dominance': 'dominance', 
            'final_consensus': 'consensus',
            'positive_bias_ratio': 'positive_bias',
            's_type_ratio': 's_type',
            'final_si_mean': 'si'
        }
        
        for original_key, short_key in key_mapping.items():
            values = [r['metrics'][original_key] for r in results if original_key in r['metrics']]
            if values:
                metrics_data[f"{short_key}_mean"] = np.mean(values)
                metrics_data[f"{short_key}_std"] = np.std(values)
            else:
                metrics_data[f"{short_key}_mean"] = 0.0
                metrics_data[f"{short_key}_std"] = 0.0
        
        # ネットワーク特性も追加
        if results:
            net_props = results[0]['network_properties']
            metrics_data.update({
                'network_type': network_type,
                'N': N,
                'density': net_props['density'],
                'avg_degree': net_props['average_degree'],
                'clustering': net_props['clustering_coefficient'],
                'avg_path_length': net_props['average_path_length']
            })
        
        return metrics_data
    
    def save_results(self, all_results: Dict, summary_data: List[Dict]):
        """結果の保存"""
        
        os.makedirs("topology_results", exist_ok=True)
        
        # 詳細結果をJSON保存
        with open("topology_results/detailed_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # サマリーをCSV保存
        if summary_data:
            fieldnames = summary_data[0].keys()
            with open("topology_results/topology_summary.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
        
        print(f"\nResults saved to topology_results/")
    
    def create_comparison_plots(self, summary_data: List[Dict]):
        """比較プロットの作成"""
        
        os.makedirs("topology_results/plots", exist_ok=True)
        
        # データ整理
        topologies = list(set(d['network_type'] for d in summary_data))
        scales = list(set(d['N'] for d in summary_data))
        
        metrics_to_plot = [
            ('entropy_mean', 'Final Weight Entropy'),
            ('dominance_mean', 'Single Criterion Dominance'),
            ('consensus_mean', 'Final Consensus'),
            ('positive_bias_mean', 'Positive Bias Ratio')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # 各トポロジーごとにプロット
            for topology in topologies:
                x_vals = []
                y_vals = []
                y_errs = []
                
                for scale in sorted(scales):
                    matching_data = [d for d in summary_data 
                                   if d['network_type'] == topology and d['N'] == scale]
                    if matching_data:
                        data = matching_data[0]
                        x_vals.append(scale)
                        y_vals.append(data[metric])
                        y_errs.append(data.get(metric.replace('_mean', '_std'), 0))
                
                if x_vals:
                    ax.errorbar(x_vals, y_vals, yerr=y_errs, 
                               marker='o', label=topology, linewidth=2, markersize=6)
            
            ax.set_xlabel('Number of Agents (N)')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Network Scale')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xticks(sorted(scales))
        
        plt.tight_layout()
        plt.savefig("topology_results/plots/topology_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # ネットワーク特性との相関プロット
        self.create_network_property_correlation_plot(summary_data)
    
    def create_network_property_correlation_plot(self, summary_data: List[Dict]):
        """ネットワーク特性との相関プロット"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        network_props = ['density', 'avg_degree', 'clustering', 'avg_path_length']
        outcome_metrics = ['entropy_mean', 'dominance_mean']
        
        for i, outcome in enumerate(outcome_metrics):
            for j, prop in enumerate(network_props):
                if j >= 3:  # 2行目
                    ax = axes[1, j-3] if i == 0 else axes[1, j-3]
                    if i == 1:  # 2番目のoutcomeの時だけプロット
                        continue
                else:  # 1行目
                    ax = axes[i, j]
                
                # データ抽出
                x_vals = []
                y_vals = []
                colors = []
                topologies = []
                
                for data in summary_data:
                    if prop in data and outcome in data:
                        if data[prop] != float('inf') and not np.isnan(data[prop]):
                            x_vals.append(data[prop])
                            y_vals.append(data[outcome])
                            topologies.append(data['network_type'])
                
                if x_vals:
                    # トポロジー別の色分け
                    unique_topologies = list(set(topologies))
                    color_map = plt.cm.tab10(np.linspace(0, 1, len(unique_topologies)))
                    
                    for k, topology in enumerate(unique_topologies):
                        topo_x = [x_vals[l] for l, t in enumerate(topologies) if t == topology]
                        topo_y = [y_vals[l] for l, t in enumerate(topologies) if t == topology]
                        ax.scatter(topo_x, topo_y, c=[color_map[k]], 
                                 label=topology, alpha=0.7, s=60)
                    
                    ax.set_xlabel(prop.replace('_', ' ').title())
                    ax.set_ylabel(outcome.replace('_mean', '').replace('_', ' ').title())
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
        
        # 使わないサブプロットを非表示
        for i in range(2):
            for j in range(3, 6):
                if j < len(axes[0]):
                    axes[i, j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig("topology_results/plots/network_property_correlation.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

def run_topology_study():
    """トポロジー研究のメイン実行"""
    study = TopologyStudy()
    all_results, summary_data = study.run_topology_study(n_seeds=2)  # 実行時間短縮のため2シード
    
    # プロット生成
    study.create_comparison_plots(summary_data)
    
    print("\n=== Key Findings ===")
    
    # 重要な発見を要約
    findings = []
    
    # エントロピーの頑健性をチェック
    entropy_values = [d['entropy_mean'] for d in summary_data]
    if entropy_values:
        entropy_range = max(entropy_values) - min(entropy_values)
        findings.append(f"Weight entropy range across configurations: {entropy_range:.4f}")
        
        if entropy_range < 0.1:  # 閾値は調整可能
            findings.append("-> ROBUST: Weight entropy is consistent across topologies and scales")
        else:
            findings.append("-> FRAGILE: Weight entropy varies significantly with network structure")
    
    # 単一基準支配の頑健性をチェック
    dominance_values = [d['dominance_mean'] for d in summary_data]
    if dominance_values:
        dominance_range = max(dominance_values) - min(dominance_values)
        findings.append(f"Single criterion dominance range: {dominance_range:.4f}")
        
        if dominance_range < 0.2:
            findings.append("-> ROBUST: 'Main+Sub' pattern is consistent across configurations")
        else:
            findings.append("-> SENSITIVE: Dominance pattern depends on network structure")
    
    # 規模依存性をチェック
    scales = sorted(list(set(d['N'] for d in summary_data)))
    if len(scales) > 1:
        for metric in ['entropy_mean', 'dominance_mean', 'positive_bias_mean']:
            scale_correlation = []
            for topology in set(d['network_type'] for d in summary_data):
                topo_data = [d for d in summary_data if d['network_type'] == topology]
                topo_data.sort(key=lambda x: x['N'])
                if len(topo_data) > 1:
                    x = [d['N'] for d in topo_data]
                    y = [d[metric] for d in topo_data]
                    if len(x) > 1:
                        corr = np.corrcoef(x, y)[0, 1]
                        if not np.isnan(corr):
                            scale_correlation.append(abs(corr))
            
            if scale_correlation:
                avg_corr = np.mean(scale_correlation)
                metric_name = metric.replace('_mean', '').replace('_', ' ')
                findings.append(f"{metric_name} scale sensitivity: {avg_corr:.3f}")
    
    for finding in findings:
        print(finding)
    
    return all_results, summary_data

if __name__ == "__main__":
    run_topology_study()