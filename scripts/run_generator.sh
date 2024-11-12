CUDA_VISIBLE_DEVICES=0,1,2,3 python run_src/do_generate.py \
    --dataset_name mbpp_Mistral-7B-v0.1 \
    --model_ckpt mistralai/Mistral-7B-v0.1 \
    --num_rollouts 16 \
    --tensor_parallel_size 4
