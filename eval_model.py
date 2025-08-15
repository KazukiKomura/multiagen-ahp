#!/usr/bin/env python3
"""
Quick evaluation script for the trained model
"""

import os
import sys
import torch
import numpy as np
from p2p import eval_once

def main():
    print("Loading model and running 3 evaluation episodes...")
    
    results = []
    for i in range(3):
        print(f"\nEvaluation run {i+1}/3:")
        J, D_final = eval_once(val_steps=200)
        results.append((J, D_final))
        print(f"  J = {J:.4f}")
        print(f"  D_final = {D_final:.4f}")
    
    # Calculate averages
    J_avg = np.mean([r[0] for r in results])
    D_avg = np.mean([r[1] for r in results])
    J_std = np.std([r[0] for r in results])
    D_std = np.std([r[1] for r in results])
    
    print(f"\n=== SUMMARY ===")
    print(f"J (Social Interest): {J_avg:.4f} ± {J_std:.4f}")
    print(f"D_final (Single crit dominance): {D_avg:.4f} ± {D_std:.4f}")
    
    # Check criteria
    J_pass = J_avg > 0
    D_pass = D_avg < 0.6
    
    print(f"\nCriteria Check:")
    print(f"✅ J > 0: {J_pass} (avg = {J_avg:.4f})")
    print(f"✅ D_final < 0.6: {D_pass} (avg = {D_avg:.4f})")
    
    if J_pass and D_pass:
        print(f"\n🎯 MODEL PASSES ALL CRITERIA - READY FOR PRODUCTION")
    else:
        print(f"\n⚠️  Model needs adjustment")
    
    return J_pass and D_pass

if __name__ == "__main__":
    main()