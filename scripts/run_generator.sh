CUDA_VISIBLE_DEVICES=2 python run_src/do_generate.py \
    --dataset_name mbpp_Mistral-7B-v0.1.jsonl \
    --model_ckpt mistralai/Mistral-7B-v0.1 \
    --num_rollouts 16
