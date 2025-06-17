#!/bin/bash
#device input
devices=$1
master_port=$2

#ft set-up
model=$3
ft_data_name=$4
save_folder=$5

#unlearn set-up
unlearn_data_id=$6
unlearn_data_name=$7
unlearn_split=$8


declare -A ft_num_epoch_dict=( ["FT-Mul"]="5" ["FT-Single"]="5" ["FT-Mul-Chunk"]="4" )
ft_num_epoch="${ft_num_epoch_dict[$ft_data_name]}"

declare -A model_path_dict=( ["llama2-7b"]="${save_folder}/${ft_data_name}_ft_epoch${ft_num_epoch}_lr1e-05_bs4_llama2-7b/" )
    

model_path="${model_path_dict[$model]}"
forget_loss=grad_ascent


save_steps=5

declare -A unlearn_data_dict=( ["FT-Mul_UL-Exact"]="tofu_data/unlearn_exact_ft_mul.json" ["FT-Mul_UL-Single"]="tofu_data/unlearn_single.json" ["FT-Mul_UL-Mul"]="tofu_data/unlearn_mul.json" ["FT-Single_UL-Exact"]="tofu_data/unlearn_exact_ft_single.json" ["FT-Single_UL-Mul"]="tofu_data/unlearn_mul.json" ["FT-Single_UL-Single"]="tofu_data/unlearn_single.json" "FT-Mul-Chunk_UL-Exact"]="tofu_data/unlearn_exact_ft_mul_chunk.json" ["FT-Mul-Chunk_UL-Single"]="tofu_data/unlearn_single.json" ["FT-Mul-Chunk_UL-Mul"]="tofu_data/unlearn_mul.json" )
data_path="${unlearn_data_dict[${ft_data_name}_${unlearn_data_name}]}"


declare -A num_epochs_dict=( ["tofu_data/unlearn_single.json"]="60" ["tofu_data/unlearn_mul.json"]="20" ["tofu_data/unlearn_exact_ft_mul.json"]="20" ["tofu_data/unlearn_exact_ft_single.json"]="60" ["tofu_data/unlearn_exact_ft_mul_chunk.json"]="400" )
if [[ "$ft_data_name" == "FT-Mul-Chunk" ]]; then
   declare -A lr_dict=( ["tofu_data/unlearn_single.json"]="1e-06" ["tofu_data/unlearn_mul.json"]="1e-06" ["tofu_data/unlearn_exact_ft_mul.json"]="3e-06" ["tofu_data/unlearn_exact_ft_single.json"]="3e-06" ["tofu_data/unlearn_exact_ft_mul_chunk.json"]="1e-05" )
else
   declare -A lr_dict=( ["tofu_data/unlearn_single.json"]="3e-06" ["tofu_data/unlearn_mul.json"]="3e-06" ["tofu_data/unlearn_exact_ft_mul.json"]="3e-06" ["tofu_data/unlearn_exact_ft_single.json"]="3e-06" ["tofu_data/unlearn_exact_ft_mul_chunk.json"]="1e-05" )
fi
lr="${lr_dict[$data_path]}"
num_epochs="${num_epochs_dict[$data_path]}"
gradient_accumulation_steps=4

CUDA_VISIBLE_DEVICES=${devices} torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget_family_text.yaml batch_size=4 gradient_accumulation_steps=${gradient_accumulation_steps} model_family=${model} lr=${lr} unlearn_data_id=${unlearn_data_id} forget_loss=${forget_loss} model_path=${model_path} data_name=${unlearn_data_name} data_path=${data_path} num_epochs=${num_epochs} save_steps=${save_steps} unlearn_split=${unlearn_split}; 

save_dir=${model_path}/${forget_loss}_${lr}_${unlearn_data_name}_${unlearn_data_id}${unlearn_split}_${num_epochs}
for cur_save_dir in ${save_dir}/*/; do 
   CUDA_VISIBLE_DEVICES=$devices torchrun --nproc_per_node=1 --master_port=$master_port model_evaluate_tofu.py model_path=$cur_save_dir
    
    declare -A model_to_modelid=( ["llama2-7b"]="meta-llama/Llama-2-7b" ["llama3-8b"]="meta-llama/Meta-Llama-3-8B" )
    model_id="${model_to_modelid[$model]}"
#     CUDA_VISIBLE_DEVICES=${devices} lm_eval --model hf \
#         --tasks piqa,race,mmlu \
#         --model_args parallelize=True,pretrained=${cur_save_dir},tokenizer=${model_id} \
#         --batch_size 4 \
#         --output_path ${cur_save_dir}
    rm ${cur_save_dir}/*.safetensors
    rm ${cur_save_dir}/*.json
    rm ${cur_save_dir}/*.bin
done;