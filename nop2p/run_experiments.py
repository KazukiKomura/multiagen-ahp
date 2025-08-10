#!/usr/bin/env python3
# run_experiments.py
# 発見検証実験の統合実行スクリプト

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Consensus Discovery Validation Experiments')
    parser.add_argument('--mode', choices=['quick', 'full', 'ablation', 'topology', 'baseline'], 
                       default='quick', help='Experiment mode')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    print("=== Multi-Agent Consensus Discovery Validation ===")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {os.getcwd()}")
    
    if args.mode == 'quick':
        # クイックモード：すべての実験を短時間で実行
        from run_discovery_validation import main as run_validation
        return run_validation(quick_mode=True)
        
    elif args.mode == 'full':
        # フルモード：完全な検証実験
        from run_discovery_validation import main as run_validation
        return run_validation(quick_mode=False)
        
    elif args.mode == 'ablation':
        # アブレーション実験のみ
        from ablation_study import run_ablation_study
        return run_ablation_study()
        
    elif args.mode == 'topology':
        # トポロジー・規模研究のみ
        from topology_study import run_topology_study
        return run_topology_study()
        
    elif args.mode == 'baseline':
        # ベースライン比較のみ
        from baseline_comparison import run_baseline_comparison
        return run_baseline_comparison()

if __name__ == "__main__":
    results = main()
    print("\nExperiments completed successfully!")