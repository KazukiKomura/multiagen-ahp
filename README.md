# Multi-Agent AHP Research

## Production Model v1.0

**Version**: v1.0  
**Commit**: f74b10e  
**Model**: `artifacts/v1/policy_v1_seed123_best.ts`  
**Performance**: J=0.0869, D_final=0.251  
**Date**: 2025-08-10  

### Key Metrics
- Social Interest (J): 0.0869 (positive convergence)
- Single criterion dominance (D_final): 0.251 (well below 0.6 threshold)
- Weight divergence median: 0.023 (excellent consensus formation)
- Accept rates: Balanced between positive/negative proposals

### Files
- `artifacts/v1/policy_v1_seed123_best.ts`: Production TorchScript model
- `artifacts/v1/action_space.json`: Action space definition (180 actions)
- `artifacts/v1/templates.json`: Natural language templates for UI

## Usage
```python
import torch
policy = torch.jit.load('artifacts/v1/policy_v1_seed123_best.ts')
# Input: [batch_size, 24] observation vector
# Output: [batch_size, 180] action logits
```