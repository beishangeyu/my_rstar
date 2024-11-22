CUDA_VISIBLE_DEVICES=0,2 python run_src/do_generate.py \
    --dataset_name mbpp_Mistral-7B-v0.1 \
    --model_ckpt mistralai/Mistral-7B-v0.1 \
    --num_rollouts 16 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.9 \
    --max_model_len 30000 \
    --verbose | tee terminal_output.txt
