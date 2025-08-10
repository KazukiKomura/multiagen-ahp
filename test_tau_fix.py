#!/usr/bin/env python3
"""Test script to verify the tau computation fix"""

import os
os.environ['RBCS_NO_HYDRA'] = '1'

from rbcs_full_enhanced import *
import numpy as np

def test_disagreement():
    """Test with intentional disagreement"""
    print("=== Testing with intentional disagreement ===")
    
    cfg = EnvConfig(
        n_agents=4,
        n_alt=3, 
        n_crit=2,
        Tmax=20,
        tau_star=0.85,
        mode="full",
        create_disagreement=True  # Force disagreement
    )
    
    env = MediatorEnv(cfg)
    env.reset(seed=42)
    
    print("Initial state:")
    for i, p in enumerate(env.participants):
        ranking = [int(np.argmax(p.S @ p.w)), int(np.argmin(p.S @ p.w))]
        print(f"Participant {i}: w={p.w.round(3)}, ranking={ranking}")
    
    obs = env.observe()
    print(f"Initial tau: {obs['tau']:.3f}")
    print(f"Initial ranking: {obs['ranking']}")
    
    # Run one step
    obs, reward, done, info = env.step("Self")
    print(f"\nAfter 1 step:")
    print(f"tau: {obs['tau']:.3f}")
    print(f"r_prev: {info['r_prev']}")  
    print(f"r_now: {info['r_now']}")
    print(f"tau_computed: {info['tau_computed']:.3f}")
    print(f"accepted: {info['accepted']}")
    print(f"apply: {info['apply']}")
    print(f"reward: {reward:.3f}")
    print(f"done: {done}")
    
    # Check individual rankings after update
    print(f"\nIndividual rankings after update:")
    for i, p in enumerate(env.participants):
        utilities = p.S @ p.w
        ranking = list(np.argsort(-utilities))
        print(f"Participant {i}: ranking={ranking}, utilities={utilities.round(3)}")
    
    return obs['tau'], done

def test_without_disagreement():
    """Test without forced disagreement (original behavior)"""
    print("\n=== Testing without forced disagreement (original) ===")
    
    cfg = EnvConfig(
        n_agents=4,
        n_alt=3,
        n_crit=2, 
        Tmax=20,
        tau_star=0.85,
        mode="full",
        create_disagreement=False  # Original random init
    )
    
    env = MediatorEnv(cfg)
    env.reset(seed=42)
    
    obs = env.observe()
    print(f"Initial tau: {obs['tau']:.3f}")
    print(f"Initial ranking: {obs['ranking']}")
    
    # Run one step  
    obs, reward, done, info = env.step("Self")
    print(f"\nAfter 1 step:")
    print(f"tau: {obs['tau']:.3f}")
    print(f"r_prev: {info['r_prev']}")
    print(f"r_now: {info['r_now']}")  
    print(f"tau_computed: {info['tau_computed']:.3f}")
    print(f"reward: {reward:.3f}")
    print(f"done: {done}")
    
    return obs['tau'], done

def test_ranking_to_rankvec():
    """Test the ranking_to_rankvec function"""
    print("\n=== Testing ranking_to_rankvec function ===")
    
    # Test case: ranking [2, 0, 1] should give rank vector [1, 2, 0]
    order = [2, 0, 1]  # alt 2 is best, alt 0 is second, alt 1 is worst
    rank = ranking_to_rankvec(order)
    print(f"Order {order} -> Rank vector {rank}")
    
    # Test kendall tau between identical rankings
    tau1 = compute_kendall_tau([0, 1, 2], [0, 1, 2])
    print(f"Tau([0,1,2], [0,1,2]) = {tau1:.3f} (should be 1.0)")
    
    # Test kendall tau between opposite rankings  
    tau2 = compute_kendall_tau([0, 1, 2], [2, 1, 0])
    print(f"Tau([0,1,2], [2,1,0]) = {tau2:.3f} (should be -1.0)")
    
    # Test kendall tau between slightly different rankings
    tau3 = compute_kendall_tau([0, 1, 2], [1, 0, 2])
    print(f"Tau([0,1,2], [1,0,2]) = {tau3:.3f} (should be ~0.33)")

if __name__ == "__main__":
    test_ranking_to_rankvec()
    tau_with_disagreement, done_with = test_disagreement() 
    tau_without_disagreement, done_without = test_without_disagreement()
    
    print("\n=== Summary ===")
    print(f"With disagreement: tau={tau_with_disagreement:.3f}, done={done_with}")
    print(f"Without disagreement: tau={tau_without_disagreement:.3f}, done={done_without}")
    
    if tau_with_disagreement < 1.0:
        print("✅ SUCCESS: Disagreement created, tau < 1.0!")
    else:
        print("❌ PROBLEM: Still getting tau=1.0 even with disagreement")
        
    if tau_without_disagreement >= 0.9:
        print("✅ Expected: Original initialization still gives high tau")
    else:
        print("⚠️  Unexpected: Original initialization now gives lower tau")