cd /data-hdd/home/CONNECT/ywu753/workSpace/Capacity-Aware-MoE/lm-evaluation-harness && \
  mkdir -p /data-hdd/home/CONNECT/ywu753/workSpace/Capacity-Aware-MoE/results/Deepseek/baseline/mmlu && \
  CUDA_VISIBLE_DEVICES=0 \
  lm_eval \
    --model hf \
    --model_args "pretrained=/data-hdd/home/CONNECT/ywu753/workSpace/models/DeepSeek-V2-Lite-Chat,parallelize=True,trust_remote_code=True,dtype=bfloat16" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path /data-hdd/home/CONNECT/ywu753/workSpace/Capacity-Aware-MoE/results/Deepseek/baseline/mmlu/mmlu.json