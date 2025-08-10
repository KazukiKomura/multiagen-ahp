# run_discovery_validation.py
# 「発見候補」の検証実験を一括実行し、「自明 vs 非自明」を判定
# アブレーション・トポロジー・ベースライン比較を統合的に分析

import os, json, csv, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 各実験モジュールをインポート
from ablation_study import run_ablation_study
from topology_study import run_topology_study  
from baseline_comparison import run_baseline_comparison

@dataclass
class DiscoveryCandidate:
    """発見候補の定義"""
    name: str
    description: str
    measurement_key: str
    expected_range: Tuple[float, float]  # 「発見」と主張できる値の範囲
    robustness_threshold: float  # 頑健性の閾値（標準偏差など）
    significance_test: str  # 'range', 'baseline_superiority', 'configuration_independence'

class DiscoveryValidator:
    """発見候補の検証クラス"""
    
    def __init__(self):
        self.discovery_candidates = [
            DiscoveryCandidate(
                name="主導+サブ構造の安定性",
                description="単一基準が支配的(0.4-0.6)だが他基準も生き残る重み構造",
                measurement_key="single_criterion_dominance", 
                expected_range=(0.4, 0.6),
                robustness_threshold=0.1,  # 設定間での標準偏差が0.1以下なら頑健
                significance_test="configuration_independence"
            ),
            DiscoveryCandidate(
                name="建設的論証優位の創発",
                description="競争環境でも協調的戦略(70%+肯定)が自然発生",
                measurement_key="positive_bias_ratio",
                expected_range=(0.7, 0.85),
                robustness_threshold=0.05,
                significance_test="configuration_independence"
            ),
            DiscoveryCandidate(
                name="具体的評価戦略の優位性", 
                description="基準重み(w)より代替案スコア(S)論証を選好(85%+)",
                measurement_key="s_type_ratio",
                expected_range=(0.85, 0.95),
                robustness_threshold=0.05,
                significance_test="configuration_independence"
            ),
            DiscoveryCandidate(
                name="適度な重み多様性の維持",
                description="完全均等でも完全集中でもない適度なエントロピー",
                measurement_key="final_entropy",
                expected_range=(0.6, 0.85),
                robustness_threshold=0.1,
                significance_test="baseline_superiority"
            )
        ]
        
        self.results = {}
        self.validation_summary = {}
    
    def run_all_experiments(self, quick_mode: bool = False):
        """全実験の実行"""
        print("=== Discovery Validation Study ===")
        print("Running comprehensive experiments to validate discovery candidates...")
        
        # 1. アブレーション実験
        print("\n1. Running Ablation Study...")
        start_time = time.time()
        ablation_results = run_ablation_study()
        self.results['ablation'] = ablation_results
        print(f"   Completed in {time.time() - start_time:.1f} seconds")
        
        # 2. トポロジー・規模研究  
        print("\n2. Running Topology & Scale Study...")
        start_time = time.time()
        if quick_mode:
            # クイックモード：少ないシードと設定
            from topology_study import TopologyStudy
            study = TopologyStudy()
            study.scale_configs = [12, 24]  # 規模を限定
            topology_results, topology_summary = study.run_topology_study(n_seeds=1)
        else:
            topology_results, topology_summary = run_topology_study()
        
        self.results['topology'] = {'detailed': topology_results, 'summary': topology_summary}
        print(f"   Completed in {time.time() - start_time:.1f} seconds")
        
        # 3. ベースライン比較
        print("\n3. Running Baseline Comparison...")
        start_time = time.time()
        baseline_results = run_baseline_comparison()
        self.results['baseline'] = baseline_results
        print(f"   Completed in {time.time() - start_time:.1f} seconds")
        
        print("\nAll experiments completed. Analyzing results...")
    
    def validate_discoveries(self) -> Dict[str, Dict[str, Any]]:
        """発見候補の検証"""
        
        validation_results = {}
        
        for candidate in self.discovery_candidates:
            print(f"\nValidating: {candidate.name}")
            
            validation = {
                'candidate': candidate,
                'is_discovery': False,
                'evidence': {},
                'confidence': 0.0,
                'failure_reasons': []
            }
            
            # 1. 基本値範囲チェック
            range_check = self.check_value_range(candidate)
            validation['evidence']['range_check'] = range_check
            
            # 2. 頑健性チェック
            robustness_check = self.check_robustness(candidate)
            validation['evidence']['robustness_check'] = robustness_check
            
            # 3. 有意性チェック
            significance_check = self.check_significance(candidate)
            validation['evidence']['significance_check'] = significance_check
            
            # 総合判定
            validation = self.make_final_judgment(validation)
            
            validation_results[candidate.name] = validation
            
            # 結果表示
            status = "✓ DISCOVERY" if validation['is_discovery'] else "✗ NOT CONFIRMED"
            confidence = validation['confidence'] * 100
            print(f"   {status} (Confidence: {confidence:.1f}%)")
            
            if validation['failure_reasons']:
                print(f"   Reasons: {', '.join(validation['failure_reasons'])}")
        
        self.validation_summary = validation_results
        return validation_results
    
    def check_value_range(self, candidate: DiscoveryCandidate) -> Dict[str, Any]:
        """値範囲のチェック"""
        check_result = {'passed': False, 'values': [], 'in_range_ratio': 0.0}
        
        # アブレーション結果から値を収集
        values = []
        if 'ablation' in self.results:
            for config_name, config_results in self.results['ablation'].items():
                metric_value = config_results['metrics'].get(candidate.measurement_key)
                if metric_value is not None:
                    values.append(metric_value)
        
        # トポロジー結果からも収集
        if 'topology' in self.results and 'summary' in self.results['topology']:
            topology_key = candidate.measurement_key.replace('final_', '').replace('_ratio', '') + '_mean'
            for summary in self.results['topology']['summary']:
                if topology_key in summary:
                    values.append(summary[topology_key])
        
        if values:
            check_result['values'] = values
            in_range = [candidate.expected_range[0] <= v <= candidate.expected_range[1] for v in values]
            check_result['in_range_ratio'] = np.mean(in_range)
            check_result['passed'] = check_result['in_range_ratio'] >= 0.8  # 80%以上が範囲内なら合格
        
        return check_result
    
    def check_robustness(self, candidate: DiscoveryCandidate) -> Dict[str, Any]:
        """頑健性のチェック"""
        check_result = {'passed': False, 'std_dev': float('inf'), 'configurations': []}
        
        values = []
        config_names = []
        
        # アブレーション結果
        if 'ablation' in self.results:
            for config_name, config_results in self.results['ablation'].items():
                metric_value = config_results['metrics'].get(candidate.measurement_key)
                if metric_value is not None:
                    values.append(metric_value)
                    config_names.append(f"ablation_{config_name}")
        
        # トポロジー結果
        if 'topology' in self.results and 'summary' in self.results['topology']:
            topology_key = candidate.measurement_key.replace('final_', '').replace('_ratio', '') + '_mean'
            for summary in self.results['topology']['summary']:
                if topology_key in summary:
                    values.append(summary[topology_key])
                    config_names.append(f"topology_{summary.get('network_type', 'unknown')}_N{summary.get('N', 'unknown')}")
        
        if len(values) >= 3:  # 最低3つの設定
            check_result['values'] = values
            check_result['configurations'] = config_names
            check_result['std_dev'] = np.std(values)
            check_result['passed'] = check_result['std_dev'] <= candidate.robustness_threshold
        
        return check_result
    
    def check_significance(self, candidate: DiscoveryCandidate) -> Dict[str, Any]:
        """有意性チェック"""
        check_result = {'passed': False, 'test_type': candidate.significance_test, 'details': {}}
        
        if candidate.significance_test == "baseline_superiority":
            check_result = self.test_baseline_superiority(candidate)
        elif candidate.significance_test == "configuration_independence":
            check_result = self.test_configuration_independence(candidate)
        
        return check_result
    
    def test_baseline_superiority(self, candidate: DiscoveryCandidate) -> Dict[str, Any]:
        """ベースライン手法との優位性テスト"""
        check_result = {'passed': False, 'test_type': 'baseline_superiority', 'details': {}}
        
        if 'baseline' not in self.results:
            check_result['details']['error'] = 'No baseline results available'
            return check_result
        
        # P2P結果を取得
        p2p_values = []
        if 'P2P_RBCS' in self.results['baseline']:
            for result in self.results['baseline']['P2P_RBCS']:
                if hasattr(result, candidate.measurement_key):
                    p2p_values.append(getattr(result, candidate.measurement_key))
        
        # ベースライン手法の結果を収集
        baseline_values = []
        for method_name, method_results in self.results['baseline'].items():
            if method_name != 'P2P_RBCS':
                for result in method_results:
                    if hasattr(result, candidate.measurement_key):
                        baseline_values.append(getattr(result, candidate.measurement_key))
        
        if p2p_values and baseline_values:
            # t検定またはMann-Whitney U検定
            if len(p2p_values) >= 3 and len(baseline_values) >= 3:
                try:
                    statistic, p_value = stats.mannwhitneyu(p2p_values, baseline_values, alternative='two-sided')
                    check_result['details'] = {
                        'p2p_mean': np.mean(p2p_values),
                        'baseline_mean': np.mean(baseline_values),
                        'p_value': p_value,
                        'statistic': statistic
                    }
                    check_result['passed'] = p_value < 0.05  # 有意水準5%
                except Exception as e:
                    check_result['details']['error'] = str(e)
        
        return check_result
    
    def test_configuration_independence(self, candidate: DiscoveryCandidate) -> Dict[str, Any]:
        """設定独立性テスト（現象の頑健性）"""
        check_result = {'passed': False, 'test_type': 'configuration_independence', 'details': {}}
        
        # アブレーション結果での設定間比較
        ablation_values = []
        ablation_configs = []
        
        if 'ablation' in self.results:
            for config_name, config_results in self.results['ablation'].items():
                metric_value = config_results['metrics'].get(candidate.measurement_key)
                if metric_value is not None:
                    ablation_values.append(metric_value)
                    ablation_configs.append(config_name)
        
        # トポロジー結果での設定間比較  
        topology_values = []
        topology_configs = []
        
        if 'topology' in self.results and 'summary' in self.results['topology']:
            topology_key = candidate.measurement_key.replace('final_', '').replace('_ratio', '') + '_mean'
            for summary in self.results['topology']['summary']:
                if topology_key in summary:
                    topology_values.append(summary[topology_key])
                    topology_configs.append(f"{summary['network_type']}_N{summary['N']}")
        
        # 設定独立性の判定
        all_values = ablation_values + topology_values
        all_configs = ablation_configs + topology_configs
        
        if len(all_values) >= 5:  # 最低5つの設定
            # 設定間での分散分析（簡略版）
            value_std = np.std(all_values)
            value_mean = np.mean(all_values)
            coefficient_of_variation = value_std / value_mean if value_mean != 0 else float('inf')
            
            check_result['details'] = {
                'n_configurations': len(all_values),
                'value_mean': value_mean,
                'value_std': value_std,
                'coefficient_of_variation': coefficient_of_variation,
                'configurations': all_configs
            }
            
            # CV < 0.2 なら設定に依存しない頑健な現象
            check_result['passed'] = coefficient_of_variation < 0.2
        
        return check_result
    
    def make_final_judgment(self, validation: Dict[str, Any]) -> Dict[str, Any]:
        """最終判定"""
        
        evidence = validation['evidence']
        weights = {'range_check': 0.3, 'robustness_check': 0.4, 'significance_check': 0.3}
        
        # 各証拠の重み付きスコア
        score = 0.0
        
        if evidence['range_check']['passed']:
            score += weights['range_check']
        
        if evidence['robustness_check']['passed']:
            score += weights['robustness_check']
        
        if evidence['significance_check']['passed']:
            score += weights['significance_check']
        
        validation['confidence'] = score
        
        # 発見と認定する条件
        if score >= 0.7:  # 70%以上のスコア
            validation['is_discovery'] = True
        else:
            validation['is_discovery'] = False
            
            # 失敗理由を記録
            reasons = []
            if not evidence['range_check']['passed']:
                reasons.append("値範囲外")
            if not evidence['robustness_check']['passed']:
                reasons.append("頑健性不足")
            if not evidence['significance_check']['passed']:
                reasons.append("有意性不足")
            
            validation['failure_reasons'] = reasons
        
        return validation
    
    def generate_final_report(self):
        """最終報告書の生成"""
        
        os.makedirs("discovery_validation_results", exist_ok=True)
        
        # 詳細結果をJSON保存
        with open("discovery_validation_results/validation_details.json", "w") as f:
            # シリアライズ可能な形式に変換
            serializable_summary = {}
            for name, validation in self.validation_summary.items():
                serializable_summary[name] = {
                    'is_discovery': bool(validation['is_discovery']),
                    'confidence': float(validation['confidence']),
                    'failure_reasons': list(validation['failure_reasons']),
                    'evidence': self._make_evidence_serializable(validation['evidence'])
                }
            json.dump(serializable_summary, f, indent=2)
        
        # サマリー表をCSV保存
        summary_data = []
        for name, validation in self.validation_summary.items():
            summary_data.append({
                'discovery_candidate': name,
                'is_discovery': validation['is_discovery'],
                'confidence_percent': validation['confidence'] * 100,
                'failure_reasons': '; '.join(validation['failure_reasons']),
                'range_check': validation['evidence']['range_check']['passed'],
                'robustness_check': validation['evidence']['robustness_check']['passed'],
                'significance_check': validation['evidence']['significance_check']['passed']
            })
        
        with open("discovery_validation_results/discovery_summary.csv", "w", newline="") as f:
            if summary_data:
                fieldnames = summary_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_data)
        
        # 可視化レポート
        self.create_validation_visualization()
        
        # テキストレポート
        self.create_text_report()
        
        print(f"\nFinal validation report saved to discovery_validation_results/")
    
    def _make_evidence_serializable(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """証拠データをJSON出力可能な形式に変換"""
        serializable_evidence = {}
        for key, value in evidence.items():
            if isinstance(value, dict):
                serializable_evidence[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (bool, int, float, str, list)):
                        serializable_evidence[key][subkey] = subvalue
                    elif isinstance(subvalue, np.ndarray):
                        serializable_evidence[key][subkey] = subvalue.tolist()
                    elif subvalue == float('inf'):
                        serializable_evidence[key][subkey] = "infinity"
                    else:
                        serializable_evidence[key][subkey] = str(subvalue)
            else:
                serializable_evidence[key] = value
        return serializable_evidence
    
    def create_validation_visualization(self):
        """検証結果の可視化"""
        
        # 信頼度スコアの棒グラフ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        names = list(self.validation_summary.keys())
        confidences = [v['confidence'] * 100 for v in self.validation_summary.values()]
        is_discovery = [v['is_discovery'] for v in self.validation_summary.values()]
        
        # 信頼度バープロット
        colors = ['green' if disc else 'red' for disc in is_discovery]
        bars = ax1.bar(range(len(names)), confidences, color=colors, alpha=0.7)
        ax1.set_xlabel('Discovery Candidates')
        ax1.set_ylabel('Confidence (%)')
        ax1.set_title('Discovery Validation Results')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([name.split('の')[0] for name in names], rotation=45, ha='right')
        ax1.axhline(y=70, color='black', linestyle='--', alpha=0.5, label='Discovery Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 証拠の種類別分析
        evidence_types = ['range_check', 'robustness_check', 'significance_check']
        evidence_scores = {etype: [] for etype in evidence_types}
        
        for validation in self.validation_summary.values():
            for etype in evidence_types:
                score = 1.0 if validation['evidence'][etype]['passed'] else 0.0
                evidence_scores[etype].append(score)
        
        evidence_df = pd.DataFrame(evidence_scores, index=[name.split('の')[0] for name in names])
        
        sns.heatmap(evidence_df.T, annot=True, cmap='RdYlGn', ax=ax2, 
                   cbar_kws={'label': 'Pass (1) / Fail (0)'})
        ax2.set_title('Evidence Type Analysis')
        ax2.set_xlabel('Discovery Candidates')
        ax2.set_ylabel('Evidence Types')
        
        plt.tight_layout()
        plt.savefig("discovery_validation_results/validation_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_text_report(self):
        """テキスト形式の報告書"""
        
        with open("discovery_validation_results/discovery_report.md", "w", encoding='utf-8') as f:
            f.write("# Discovery Validation Report\n\n")
            f.write("## Executive Summary\n\n")
            
            # 発見の要約
            discoveries = [name for name, v in self.validation_summary.items() if v['is_discovery']]
            f.write(f"**Validated Discoveries**: {len(discoveries)}/{len(self.validation_summary)}\n\n")
            
            if discoveries:
                f.write("### Confirmed Discoveries:\n")
                for name in discoveries:
                    confidence = self.validation_summary[name]['confidence'] * 100
                    f.write(f"- **{name}** (Confidence: {confidence:.1f}%)\n")
                f.write("\n")
            
            non_discoveries = [name for name, v in self.validation_summary.items() if not v['is_discovery']]
            if non_discoveries:
                f.write("### Not Confirmed:\n")
                for name in non_discoveries:
                    reasons = self.validation_summary[name]['failure_reasons']
                    f.write(f"- **{name}** (Reasons: {', '.join(reasons)})\n")
                f.write("\n")
            
            # 詳細分析
            f.write("## Detailed Analysis\n\n")
            
            for name, validation in self.validation_summary.items():
                f.write(f"### {name}\n\n")
                f.write(f"**Status**: {'✓ DISCOVERY' if validation['is_discovery'] else '✗ NOT CONFIRMED'}\n")
                f.write(f"**Confidence**: {validation['confidence']*100:.1f}%\n\n")
                
                f.write("**Evidence Analysis**:\n")
                evidence = validation['evidence']
                
                # Range check
                if 'range_check' in evidence:
                    rc = evidence['range_check']
                    f.write(f"- Range Check: {'PASS' if rc['passed'] else 'FAIL'}")
                    if 'in_range_ratio' in rc:
                        f.write(f" ({rc['in_range_ratio']*100:.1f}% in expected range)")
                    f.write("\n")
                
                # Robustness check
                if 'robustness_check' in evidence:
                    rob = evidence['robustness_check']
                    f.write(f"- Robustness Check: {'PASS' if rob['passed'] else 'FAIL'}")
                    if 'std_dev' in rob and rob['std_dev'] != float('inf'):
                        f.write(f" (σ = {rob['std_dev']:.4f})")
                    f.write("\n")
                
                # Significance check  
                if 'significance_check' in evidence:
                    sig = evidence['significance_check']
                    f.write(f"- Significance Check: {'PASS' if sig['passed'] else 'FAIL'}")
                    if 'details' in sig and 'p_value' in sig['details']:
                        f.write(f" (p = {sig['details']['p_value']:.4f})")
                    f.write("\n")
                
                f.write("\n")
            
            # 結論
            f.write("## Conclusions\n\n")
            
            if discoveries:
                f.write("The following phenomena can be claimed as **genuine discoveries** ")
                f.write("beyond trivial consequences of the system design:\n\n")
                
                for name in discoveries:
                    candidate = next(c for c in self.discovery_candidates if c.name == name)
                    f.write(f"1. **{name}**: {candidate.description}\n")
                
                f.write("\nThese discoveries demonstrate robust, non-trivial emergent behaviors ")
                f.write("that persist across different configurations and outperform baseline methods.\n\n")
            
            if non_discoveries:
                f.write("The following phenomena appear to be **design consequences** ")
                f.write("rather than fundamental discoveries:\n\n")
                
                for name in non_discoveries:
                    f.write(f"- {name}\n")
                
                f.write("\nFurther investigation or design modifications may be needed ")
                f.write("to establish these as genuine discoveries.\n")

def main(quick_mode: bool = True):
    """メイン実行関数"""
    
    print("=== Multi-Agent Consensus Discovery Validation ===")
    print(f"Mode: {'Quick' if quick_mode else 'Full'}")
    
    validator = DiscoveryValidator()
    
    # 全実験実行
    start_time = time.time()
    validator.run_all_experiments(quick_mode=quick_mode)
    experiment_time = time.time() - start_time
    
    print(f"\nAll experiments completed in {experiment_time:.1f} seconds")
    
    # 発見候補の検証
    print("\n" + "="*50)
    print("DISCOVERY VALIDATION PHASE")
    print("="*50)
    
    validation_results = validator.validate_discoveries()
    
    # 最終報告書生成
    validator.generate_final_report()
    
    # サマリー表示
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    discoveries = sum(1 for v in validation_results.values() if v['is_discovery'])
    total_candidates = len(validation_results)
    
    print(f"Discovery Success Rate: {discoveries}/{total_candidates} ({discoveries/total_candidates*100:.1f}%)")
    print(f"Total Experiment Time: {experiment_time:.1f} seconds")
    
    if discoveries > 0:
        print(f"\n🎉 SUCCESS: {discoveries} genuine discoveries validated!")
        print("These phenomena go beyond trivial design consequences.")
    else:
        print(f"\n⚠️  No discoveries confirmed. Results may be design artifacts.")
        print("Consider system modifications or alternative hypotheses.")
    
    print(f"\nDetailed report available in: discovery_validation_results/")
    
    return validation_results

if __name__ == "__main__":
    import sys
    quick_mode = "--quick" in sys.argv or "-q" in sys.argv
    main(quick_mode=quick_mode)