CUDA_VISIBLE_DEVICES=0,2 python run_src/do_discriminate.py \
    --gene_result_dir gene_result/mbpp_Mistral-7B-v0.1/Mistral-7B-v0.1_1 \
    --model_ckpt mistralai/Mistral-7B-v0.1 \
    --dataset_name mbpp_Mistral-7B-v0.1 \
    --tensor_parallel_size 2 \
