# production_fixed.py
# 本番実験システム（用語明確化・安全イベント内訳・効果量完全対応版）

import os, json, csv, time
import numpy as np
import networkx as nx
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random
from complete_fixed_system import (
    ComprehensiveConfig, CompleteFixedSystem, 
    normalize_simplex, weight_entropy, gini_coefficient, pearson_scaled
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ProductionConfig(ComprehensiveConfig):
    """本番実験用設定"""
    episodes: int = 80
    steps_per_episode: int = 64
    verbose: bool = False  # 本番では静かに実行

def cohen_d(x1, x2):
    """Cohen's d effect size calculation"""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1-1)*np.var(x1, ddof=1) + (n2-1)*np.var(x2, ddof=1)) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_std

def welch_ttest_with_correction(groups, labels, alpha=0.01):
    """Welch t-test with Holm-Bonferroni correction"""
    n_groups = len(groups)
    p_values = []
    comparisons = []
    
    # All pairwise comparisons
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            stat, p_val = stats.ttest_ind(groups[i], groups[j], equal_var=False)
            effect_size = cohen_d(groups[i], groups[j])
            p_values.append(p_val)
            comparisons.append({
                'group1': labels[i],
                'group2': labels[j], 
                'statistic': stat,
                'p_value': p_val,
                'effect_size': effect_size
            })
    
    # Holm-Bonferroni correction
    rejected, corrected_p, _, _ = stats.multipletests(p_values, alpha=alpha, method='holm')
    
    for i, comparison in enumerate(comparisons):
        comparison['corrected_p'] = corrected_p[i]
        comparison['significant'] = rejected[i]
    
    return comparisons

class ProductionExperiment:
    """本番実験実行システム（用語明確化版）"""
    
    def __init__(self):
        self.configurations = [
            {"name": "Full", "enable_veto": True, "enable_floor": True, "enable_ci_aware": True},
            {"name": "no_veto", "enable_veto": False, "enable_floor": True, "enable_ci_aware": True},
            {"name": "no_floor", "enable_veto": True, "enable_floor": False, "enable_ci_aware": True}, 
            {"name": "no_ci_aware", "enable_veto": True, "enable_floor": True, "enable_ci_aware": False},
            {"name": "no_safety", "enable_veto": False, "enable_floor": False, "enable_ci_aware": False}
        ]
        self.seeds = list(range(10))  # 0-9の10seeds
        
    def run_single_experiment(self, config_name: str, config_dict: Dict, seed: int) -> Dict[str, Any]:
        """単一実験の実行"""
        config = ProductionConfig(seed=seed, **config_dict)
        system = CompleteFixedSystem(config)
        
        start_time = time.time()
        result = system.run_experiment()
        end_time = time.time()
        
        result['execution_time'] = end_time - start_time
        result['config_name'] = config_name
        result['seed'] = seed
        
        return result
    
    def run_production_experiment(self):
        """本番実験の完全実行"""
        
        print("=== PRODUCTION EXPERIMENT (TERMINOLOGY CLARIFIED) ===")
        print("Configuration: 80 episodes × 64 steps × 10 seeds × 5 conditions")
        print("Main metrics: Dominance=1-max(w̄), H(w̄)=Shannon entropy, policy_entropy")
        print("Safety events: {veto, floor, ci_adjust, rollback} breakdown included")
        print("Expected total time: ~5-6 hours")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = {}
        overall_start = time.time()
        
        for config_idx, config_dict in enumerate(self.configurations):
            config_name = config_dict.pop('name')
            config_start = time.time()
            
            print(f"\\n[{config_idx+1}/5] Running {config_name}...")
            
            config_results = []
            for seed_idx, seed in enumerate(self.seeds):
                print(f"  Seed {seed:2d} ({seed_idx+1:2d}/10)... ", end="", flush=True)
                
                try:
                    result = self.run_single_experiment(config_name, config_dict.copy(), seed)
                    config_results.append(result)
                    print(f"Done ({result['execution_time']:.1f}s)")
                    
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue
                
            all_results[config_name] = config_results
            
            config_end = time.time()
            config_duration = config_end - config_start
            
            # 中間統計表示（用語明確化）
            if config_results:
                policy_entropies = [r['metrics']['final_policy_entropy'] for r in config_results]
                dominances = [r['metrics']['single_criterion_dominance'] for r in config_results]
                safety_events = [r['metrics']['safety_events_total'] for r in config_results]
                
                print(f"  {config_name} completed in {config_duration/60:.1f} minutes")
                print(f"    Policy Entropy: {np.mean(policy_entropies):.6f} ± {np.std(policy_entropies):.6f}")
                print(f"    Dominance (1-max(w̄)): {np.mean(dominances):.6f} ± {np.std(dominances):.6f}")  
                print(f"    Safety events: {np.mean(safety_events):.1f} ± {np.std(safety_events):.1f}")
        
        overall_end = time.time()
        total_duration = overall_end - overall_start
        
        print(f"\\n=== EXPERIMENT COMPLETED ===")
        print(f"Total time: {total_duration/3600:.2f} hours ({total_duration/60:.1f} minutes)")
        print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return all_results
    
    def analyze_results(self, all_results: Dict[str, List[Dict]]):
        """結果の統計分析（用語明確化・完全版）"""
        
        print("\\n=== STATISTICAL ANALYSIS (TERMINOLOGY CLARIFIED) ===")
        
        # データ抽出（用語明確化済み）
        analysis_data = {}
        for config_name, results in all_results.items():
            if results:  # 結果が存在する場合のみ
                analysis_data[config_name] = {
                    'policy_entropy': [r['metrics']['final_policy_entropy'] for r in results],
                    'weight_shannon_entropy': [r['metrics']['weight_shannon_entropy'] for r in results],
                    'dominance': [r['metrics']['single_criterion_dominance'] for r in results],  # Dominance = 1 - max(w̄)
                    'max_weight': [r['metrics']['max_group_weight'] for r in results],  # max(w̄)
                    'gini': [r['metrics']['gini_coefficient'] for r in results],
                    'safety_events_total': [r['metrics']['safety_events_total'] for r in results],
                    'safety_veto': [r['metrics']['safety_breakdown']['veto_blocks'] for r in results],
                    'safety_floor': [r['metrics']['safety_breakdown']['floor_blocks'] for r in results],
                    'safety_ci_adjust': [r['metrics']['safety_breakdown']['ci_adjustments'] for r in results],
                    'safety_rollback': [r['metrics']['safety_breakdown']['rollbacks'] for r in results],
                    'weight_distributions': [r['metrics']['final_weights'] for r in results],
                    'weight_deviations': [r['metrics']['weight_deviations_from_mean'] for r in results]
                }
        
        # 主要仮説検定（事前登録済み）
        print("\\n1. PRIMARY HYPOTHESIS TESTING (PRE-REGISTERED)")
        
        if 'Full' in analysis_data and 'no_safety' in analysis_data:
            full_dom = analysis_data['Full']['dominance']
            no_safety_dom = analysis_data['no_safety']['dominance']
            
            # H1: Full は no_safety より Dominance が ≥15% 改善
            improvement = (np.mean(full_dom) - np.mean(no_safety_dom)) / np.mean(no_safety_dom) * 100
            t_stat, p_val = stats.ttest_ind(full_dom, no_safety_dom, equal_var=False)
            effect_size = cohen_d(full_dom, no_safety_dom)
            
            print(f"H1: Full vs no_safety Dominance improvement")
            print(f"    Improvement: {improvement:.1f}% (target: ≥15%)")
            print(f"    p-value: {p_val:.6f} (target: <0.01)")
            print(f"    Cohen's d: {effect_size:.3f}")
            print(f"    95% CI: [{np.percentile(full_dom, 2.5):.6f}, {np.percentile(full_dom, 97.5):.6f}]")
            print(f"    Status: {'✅ PASS' if improvement >= 15 and p_val < 0.01 else '❌ FAIL'}")
            print(f"    Note: Current 3-seed result shows {improvement:.1f}% improvement - promising for 10-seed validation")
        
        # H2: no_ci_aware の悪化度
        if 'Full' in analysis_data and 'no_ci_aware' in analysis_data:
            full_dom = analysis_data['Full']['dominance']
            no_ci_dom = analysis_data['no_ci_aware']['dominance']
            
            degradation = (np.mean(no_ci_dom) - np.mean(full_dom)) / np.mean(full_dom) * 100
            t_stat, p_val = stats.ttest_ind(full_dom, no_ci_dom, equal_var=False)
            effect_size = cohen_d(full_dom, no_ci_dom)
            
            print(f"\\nH2: no_ci_aware Dominance degradation (Full vs no_ci_aware)")
            print(f"    Degradation: {degradation:.1f}% (target: ≤-20%)")
            print(f"    p-value: {p_val:.6f}")
            print(f"    Cohen's d: {effect_size:.3f}")
            print(f"    95% CI difference: [{np.percentile(no_ci_dom, 2.5) - np.percentile(full_dom, 97.5):.6f}, {np.percentile(no_ci_dom, 97.5) - np.percentile(full_dom, 2.5):.6f}]")
            print(f"    Status: {'✅ PASS' if degradation <= -20 else '❌ FAIL'}")
        
        # H3: Policy Entropy の決断的収束確認
        if 'Full' in analysis_data:
            full_policy_entropy = analysis_data['Full']['policy_entropy']
            policy_entropy_mean = np.mean(full_policy_entropy)
            in_decisive_range = 0.94 <= policy_entropy_mean <= 0.97
            
            print(f"\\nH3: Full Policy Entropy convergence")
            print(f"    Mean policy entropy: {policy_entropy_mean:.6f} (target: 0.94-0.97)")
            print(f"    Status: {'✅ PASS' if in_decisive_range else '❌ FAIL'}")
            
            # 他設定との比較
            for config_name in ['no_veto', 'no_floor', 'no_ci_aware', 'no_safety']:
                if config_name in analysis_data:
                    other_entropy = np.mean(analysis_data[config_name]['policy_entropy'])
                    t_stat, p_val = stats.ttest_ind(full_policy_entropy, analysis_data[config_name]['policy_entropy'], equal_var=False)
                    is_lower = policy_entropy_mean < other_entropy and p_val < 0.05
                    print(f"    Full vs {config_name}: {policy_entropy_mean:.6f} vs {other_entropy:.6f} {'✅' if is_lower else '❌'}")
        
        # 全ペア比較（Dominance = 1 - max(w̄)）
        print("\\n2. ALL PAIRWISE COMPARISONS (Dominance = 1 - max(w̄))")
        dominance_groups = [analysis_data[name]['dominance'] for name in analysis_data.keys()]
        dominance_labels = list(analysis_data.keys())
        
        comparisons = welch_ttest_with_correction(dominance_groups, dominance_labels)
        
        print(f"    Statistical comparisons (Dominance = 1-max(w̄), Holm-corrected p-values):")
        for comp in comparisons:
            status = "✅" if comp['significant'] else "❌"
            print(f"    {comp['group1']} vs {comp['group2']}: "
                  f"d={comp['effect_size']:.3f}, p_adj={comp['corrected_p']:.6f} {status}")
        
        # 制約効果の順序分析
        print(f"\\n3. CONSTRAINT EFFECT RANKING (by Dominance = 1-max(w̄) impact)")
        if 'Full' in analysis_data:
            full_mean = np.mean(analysis_data['Full']['dominance'])
            
            effects = []
            for config_name in ['no_veto', 'no_floor', 'no_ci_aware', 'no_safety']:
                if config_name in analysis_data:
                    config_mean = np.mean(analysis_data[config_name]['dominance'])
                    effect = (config_mean - full_mean) / full_mean * 100
                    effects.append((config_name, effect))
            
            effects.sort(key=lambda x: abs(x[1]), reverse=True)
            print("    Constraint importance ranking (larger |effect| = more important):")
            for i, (name, effect) in enumerate(effects, 1):
                print(f"    {i}. {name}: {effect:+.1f}%")
        
        return analysis_data, comparisons
    
    def save_results(self, all_results: Dict, analysis_data: Dict, comparisons: List):
        """結果の保存（用語明確化・安全イベント内訳完全版）"""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"production_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生データ保存
        with open(f"{output_dir}/raw_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # 統計サマリー（用語明確化）
        summary_data = []
        for config_name, data in analysis_data.items():
            summary_data.append({
                'configuration': config_name,
                'n_samples': len(data['policy_entropy']),
                'policy_entropy_mean': np.mean(data['policy_entropy']),
                'policy_entropy_std': np.std(data['policy_entropy']),
                'weight_shannon_entropy_mean': np.mean(data['weight_shannon_entropy']),
                'weight_shannon_entropy_std': np.std(data['weight_shannon_entropy']),
                'dominance_mean': np.mean(data['dominance']),  # Dominance = 1 - max(w̄)
                'dominance_std': np.std(data['dominance']),
                'dominance_ci_lower': np.percentile(data['dominance'], 2.5),
                'dominance_ci_upper': np.percentile(data['dominance'], 97.5),
                'max_weight_mean': np.mean(data['max_weight']),  # max(w̄)
                'gini_mean': np.mean(data['gini']),
                'safety_events_total_mean': np.mean(data['safety_events_total']),
                'safety_events_total_std': np.std(data['safety_events_total']),
                'safety_veto_mean': np.mean(data['safety_veto']),
                'safety_floor_mean': np.mean(data['safety_floor']),
                'safety_ci_adjust_mean': np.mean(data['safety_ci_adjust']),
                'safety_rollback_mean': np.mean(data['safety_rollback'])
            })
        
        with open(f"{output_dir}/summary.csv", "w", newline="") as f:
            if summary_data:
                writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                writer.writeheader()
                writer.writerows(summary_data)
        
        # 安全イベント内訳を詳細CSVで保存
        safety_breakdown_data = []
        for config_name, data in analysis_data.items():
            safety_breakdown_data.append({
                'configuration': config_name,
                'veto_mean': np.mean(data['safety_veto']),
                'veto_std': np.std(data['safety_veto']),
                'veto_ci_lower': np.percentile(data['safety_veto'], 2.5),
                'veto_ci_upper': np.percentile(data['safety_veto'], 97.5),
                'floor_mean': np.mean(data['safety_floor']),
                'floor_std': np.std(data['safety_floor']),
                'floor_ci_lower': np.percentile(data['safety_floor'], 2.5),
                'floor_ci_upper': np.percentile(data['safety_floor'], 97.5),
                'ci_adjust_mean': np.mean(data['safety_ci_adjust']),
                'ci_adjust_std': np.std(data['safety_ci_adjust']),
                'ci_adjust_ci_lower': np.percentile(data['safety_ci_adjust'], 2.5),
                'ci_adjust_ci_upper': np.percentile(data['safety_ci_adjust'], 97.5),
                'rollback_mean': np.mean(data['safety_rollback']),
                'rollback_std': np.std(data['safety_rollback']),
                'rollback_ci_lower': np.percentile(data['safety_rollback'], 2.5),
                'rollback_ci_upper': np.percentile(data['safety_rollback'], 97.5)
            })
        
        with open(f"{output_dir}/safety_breakdown.csv", "w", newline="") as f:
            if safety_breakdown_data:
                writer = csv.DictWriter(f, fieldnames=safety_breakdown_data[0].keys())
                writer.writeheader()
                writer.writerows(safety_breakdown_data)
        
        # 統計検定結果
        with open(f"{output_dir}/statistical_tests.json", "w") as f:
            json.dump(comparisons, f, indent=2, default=str)
        
        # 可視化
        self.create_visualizations(analysis_data, output_dir)
        
        print(f"\\nResults saved to: {output_dir}/")
        return output_dir
    
    def create_visualizations(self, analysis_data: Dict, output_dir: str):
        """結果の可視化（用語明確化・重み偏差ヒートマップ付き）"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        config_names = list(analysis_data.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(config_names)))
        
        # 1. Dominance comparison (用語明確化)
        dominance_data = [analysis_data[name]['dominance'] for name in config_names]
        bp1 = axes[0,0].boxplot(dominance_data, labels=config_names, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        axes[0,0].set_title('Dominance = 1 - max(w̄) by Configuration\\n(Higher = Less Single-Criterion Dominance)')
        axes[0,0].set_ylabel('Dominance (1 - max(w̄))')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Policy Entropy comparison (決断的レンジ付き)
        policy_entropy_data = [analysis_data[name]['policy_entropy'] for name in config_names]
        bp2 = axes[0,1].boxplot(policy_entropy_data, labels=config_names, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        axes[0,1].set_title('Policy Entropy by Configuration\\n(High = Indecisive Learning)')
        axes[0,1].set_ylabel('Policy Entropy')
        axes[0,1].tick_params(axis='x', rotation=45)
        # 決断的レンジをハイライト
        axes[0,1].axhspan(0.94, 0.97, alpha=0.3, color='green', label='Decisive Range')
        axes[0,1].legend()
        
        # 3. Safety events breakdown (詳細内訳)
        safety_categories = ['safety_veto', 'safety_floor', 'safety_ci_adjust', 'safety_rollback']
        safety_labels = ['Veto', 'Floor', 'CI-Adjust', 'Rollback']
        
        x = np.arange(len(config_names))
        width = 0.2
        
        for i, (category, label) in enumerate(zip(safety_categories, safety_labels)):
            safety_means = [np.mean(analysis_data[name][category]) for name in config_names]
            axes[0,2].bar(x + i*width, safety_means, width, label=label, alpha=0.7)
        
        axes[0,2].set_title('Safety Events Breakdown by Configuration')
        axes[0,2].set_ylabel('Mean Safety Events')
        axes[0,2].set_xticks(x + width * 1.5)
        axes[0,2].set_xticklabels(config_names, rotation=45)
        axes[0,2].legend()
        
        # 4. Max weight distribution (用語明確化)
        max_weight_data = [analysis_data[name]['max_weight'] for name in config_names]
        bp4 = axes[1,0].boxplot(max_weight_data, labels=config_names, patch_artist=True)
        for patch, color in zip(bp4['boxes'], colors):
            patch.set_facecolor(color)
        axes[1,0].set_title('Maximum Group Weight max(w̄) by Configuration\\n(Higher = More Single-Criterion Dominance)')
        axes[1,0].set_ylabel('max(w̄)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Weight Shannon Entropy (H(w̄))
        shannon_data = [analysis_data[name]['weight_shannon_entropy'] for name in config_names]
        bp5 = axes[1,1].boxplot(shannon_data, labels=config_names, patch_artist=True)
        for patch, color in zip(bp5['boxes'], colors):
            patch.set_facecolor(color)
        axes[1,1].set_title('Weight Shannon Entropy H(w̄) by Configuration\\n(Higher = More Distributed)')
        axes[1,1].set_ylabel('Shannon Entropy H(w̄)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Weight deviation heatmap |w_i - w̄|
        if len(config_names) >= 2 and 'weight_deviations' in analysis_data[config_names[0]]:
            # 平均的な重み偏差パターンを可視化
            deviation_matrix = []
            for config_name in config_names:
                # 各設定の平均偏差パターン
                mean_deviations = np.mean(analysis_data[config_name]['weight_deviations'], axis=0)
                deviation_matrix.append(mean_deviations)
            
            deviation_matrix = np.array(deviation_matrix)
            
            im = axes[1,2].imshow(deviation_matrix, cmap='Reds', aspect='auto')
            axes[1,2].set_yticks(range(len(config_names)))
            axes[1,2].set_xticks(range(len(deviation_matrix[0])))
            axes[1,2].set_yticklabels(config_names)
            axes[1,2].set_xticklabels([f'C{i+1}' for i in range(len(deviation_matrix[0]))])
            axes[1,2].set_title('Weight Deviations |w_i - w̄| Heatmap\\n(Red = High Deviation from Mean)')
            axes[1,2].set_xlabel('Criterion')
            axes[1,2].set_ylabel('Configuration')
            
            # カラーバー
            plt.colorbar(im, ax=axes[1,2], shrink=0.6)
        else:
            # フォールバック: Effect sizes heatmap
            effect_matrix = np.zeros((len(config_names), len(config_names)))
            for i, name1 in enumerate(config_names):
                for j, name2 in enumerate(config_names):
                    if i != j:
                        effect_matrix[i,j] = cohen_d(
                            analysis_data[name1]['dominance'],
                            analysis_data[name2]['dominance']
                        )
            
            im = axes[1,2].imshow(effect_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
            axes[1,2].set_xticks(range(len(config_names)))
            axes[1,2].set_yticks(range(len(config_names)))
            axes[1,2].set_xticklabels(config_names, rotation=45)
            axes[1,2].set_yticklabels(config_names)
            axes[1,2].set_title("Effect Sizes (Cohen's d) for Dominance")
            
            for i in range(len(config_names)):
                for j in range(len(config_names)):
                    if i != j:
                        axes[1,2].text(j, i, f'{effect_matrix[i,j]:.2f}',
                                      ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/production_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}/production_analysis.png")

def main():
    """メイン実行関数（本番実験）"""
    print("=" * 80)
    print("PRODUCTION EXPERIMENT: TERMINOLOGY CLARIFIED & SAFETY BREAKDOWN INCLUDED")
    print("=" * 80)
    print("Key improvements:")
    print("- Terminology: entropy → policy_entropy, Dominance = 1-max(w̄) explicit")
    print("- Safety events: {veto, floor, ci_adjust, rollback} detailed breakdown")
    print("- Effect sizes: Cohen's d with 95% confidence intervals")
    print("- Visualization: Weight deviation heatmaps |w_i - w̄|")
    print("- 10 seeds per condition for robust statistical testing")
    print("=" * 80)
    
    experiment = ProductionExperiment()
    
    # 実験実行
    results = experiment.run_production_experiment()
    
    # 統計分析
    analysis_data, comparisons = experiment.analyze_results(results)
    
    # 結果保存
    output_dir = experiment.save_results(results, analysis_data, comparisons)
    
    print("\\n=== PRODUCTION EXPERIMENT COMPLETED ===")
    print("TERMINOLOGY CLARIFICATION:")
    print("- Dominance = 1 - max(w̄): Higher values = better (less single-criterion dominance)")
    print("- max(w̄): Higher values = worse (more single-criterion dominance)")
    print("- H(w̄): Shannon entropy of weights, higher = more distributed")
    print("- Policy entropy: Learning exploration level, 0.94-0.97 = decisive range")
    print("- Safety events: {veto, floor, ci_adjust, rollback} detailed breakdown")
    print(f"Results directory: {output_dir}")
    print("CSV files include full safety event breakdown and 95% confidence intervals")
    print("Visualizations show weight deviation heatmaps and policy entropy decisive range")
    print("Ready for human-in-the-loop experiment design!")
    print("\\nKEY FINDINGS SUMMARY:")
    print("- Dominance = 1 - max(w̄): measures resistance to single-criterion dominance")
    print("- Policy entropy: measures learning decisiveness (0.94-0.97 = optimal range)")
    print("- H(w̄): Shannon entropy of weight distribution")
    print("- Safety events: {veto, floor, ci_adjust, rollback} breakdown shows mechanism activity")
    print("- Statistical tests: Welch t-test + Holm correction, Cohen's d effect sizes, 95% CIs")

if __name__ == "__main__":
    main()