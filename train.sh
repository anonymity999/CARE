source "$(conda info --base)/etc/profile.d/conda.sh" # 推荐这种更健壮的方式
# conda activate my_rag
# --runtime-env-json='{"working_dir": "/openrlhf"}' \
#    --flash_attn \
##### 1.  本地缓存根目录（用 /tmp/$USER 就够了） #####
LOCAL_SCRATCH="/tmp/${USER}"
mkdir -p "${LOCAL_SCRATCH}"/{torch_extensions,triton_cache,hf_cache,tmp}

##### 2.  让 Bash 先把变量替换成“绝对路径”后再塞进 JSON #####
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "TORCH_EXTENSIONS_DIR": "${LOCAL_SCRATCH}/torch_extensions",
    "TRITON_CACHE_DIR":     "${LOCAL_SCRATCH}/triton_cache",
    "XDG_CACHE_HOME":       "${LOCAL_SCRATCH}/hf_cache",
    "HF_HOME":              "${LOCAL_SCRATCH}/hf_cache/hf",
    "HF_DATASETS_CACHE":    "${LOCAL_SCRATCH}/hf_cache/datasets",
    "TMPDIR":               "${LOCAL_SCRATCH}/tmp"
  }
}
EOF
)



N_SAMPLES=16
EPISODE=1
WARMUP=0.0
TBS=128
RBS=1
KL=0.0001
LR=2e-6
MAX_LENGTH=16
TEMP=1.0

ray job submit --address="http://127.0.0.1:8267" \
   --runtime-env-json "$RUNTIME_ENV_JSON" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_model \
   --deepspeed_enable_sleep \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.25 \
   --pretrain poisoned_qwen_4B_round2 \
   --remote_rm_url http://138.25.54.36:5004/get_reward \
   --save_path poisoned_qwen_4B_round3\
   --train_batch_size 512 \
   --micro_train_batch_size 8 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 32 \
   --n_samples_per_prompt 16 \
   --advantage_estimator group_norm \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes ${EPISODE} \
   --lr_warmup_ratio ${WARMUP} \
   --prompt_max_len 6144 \
   --generate_max_len 32 \
   --critic_learning_rate 9e-6 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate $LR \
   --init_kl_coef $KL \
   --prompt_data attack_round3_train_rl_prompt.json \
   --input_key raw_prompt \
   --apply_chat_template \
   --flash_attn \
   --gradient_checkpointing \
   --save_steps 40 \
   --save_hf_ckpt \
   --max_ckpt_num 2 \
   --ckpt_path /scratch/poisoned_qwen_4B_round3 \
   --temperature $TEMP \
   --use_wandb  56d889eb655ebc0f20076c63e995e1ab9113c35a \
   --wandb_project poisoned_rlhf_adv \
   --wandb_run_name poisoned_qwen_4B_round3 \
   --packing_samples \
   --adam_offload \
   --enable_prefix_caching \
   --load_checkpoint \