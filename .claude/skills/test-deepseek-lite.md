---
name: test-deepseek-lite
changelog: 0.0.1
description: Run accuracy tests for DeepSeek-V2-Lite-Chat under different capacity-aware strategies
triggers:
  - test deepseek
  - deepseek lite accuracy
  - capacity aware deepseek
  - moe accuracy test
---

# DeepSeek-V2-Lite-Chat Accuracy Test Skill

Run the Capacity-Aware MoE accuracy evaluation for the DeepSeek-V2-Lite-Chat model.

## Quick start

```bash
cd /data-hdd/home/CONNECT/ywu753/workSpace/Capacity-Aware-MoE
bash scripts/test_deepseek_lite.sh
```

## Test configurations

| Strategy | Description |
|----------|-------------|
| `baseline` | No capacity constraints (original model) |
| `score` | Capacity-Aware Token Drop by score |
| `random` | Random token dropping when over capacity |
| `first` | Sequential drop — keep first N tokens |
| `last` | Sequential drop — keep last N tokens |
| `overselect` | Capacity-Aware Expanded Drop (use low-load local experts first) |

| Expert Capacity γ | Effect |
|-------------------|--------|
| `1.0` | Mild constraint (close to baseline) |
| `0.8` | Moderate dropping |
| `0.6` | Heavy dropping |
| `0.5` | Aggressive dropping |

## 9 evaluation tasks

openbookqa, piqa, rte, winogrande, boolq, arc_challenge, hellaswag, mmlu, gsm8k

## Custom usage

```bash
# Test only score strategy with γ=1.0 on a single task
bash scripts/test_deepseek_lite.sh -s score -c 1.0 -t piqa

# Test specific strategies and capacities
bash scripts/test_deepseek_lite.sh -s score,overselect -c 1.0,0.8

# Custom model path and GPUs
bash scripts/test_deepseek_lite.sh -m /path/to/model -g 0,1

# Output to custom directory
bash scripts/test_deepseek_lite.sh -o /path/to/results
```

## Environment variables

- `MODEL_PATH` — model path (default: `~/data/workSpace/models/DeepSeek-V2-Lite-Chat`)
- `OUTPUT_BASE` — output directory (default: `$MODEL_PATH`)
- `BATCH_SIZE` — batch size (default: `auto`)
- `CUDA_DEVICES` — GPU IDs (default: `0`)

## Model info

- Architecture: DeepSeek-V2-Lite-Chat
- Routed experts: 64
- Top-k: 6
- Shared experts: 2
- Top-k method: greedy
- Scoring: softmax

## Datasets

Download with:
```bash
bash scripts/download_datasets.sh
```

## Output structure

```
$OUTPUT_BASE/
  baseline/                 # baseline results
    {task}.json
  expert_capacity-{γ}/
    {strategy}/
      {task}.json
```
