devices=$1; master_port=$2; model=$3 num_epochs=$4; ft_lr=$5; batch_size=$6; ft_data_name=$7; save_folder=$8; 


declare -A data_path_dict=( ["FT-Mul"]="synthetic_data/ft_mul.json" ["FT-Single"]="synthetic_data/ft_single.json" ["FT-Mul-Chunk"]="synthetic_data/ft_mul_chunk.json" ["FT-Mul-Chunk-Iso"]="synthetic_data/ft_mul_chunk_iso.json" )
data_path="${data_path_dict[$ft_data_name]}"

save_dir=${save_folder}/${ft_data_name}_ft_epoch${num_epochs}_lr${lr}_bs${batch_size}_${model}

CUDA_VISIBLE_DEVICES=$devices torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune_family_rephrased_text.yaml batch_size=${batch_size} gradient_accumulation_steps=2 model_family=${model} lr=${ft_lr} num_epochs=$num_epochs data_path=${data_path} data_name=${ft_data_name} save_dir=${save_dir} LoRA.r=0

CUDA_VISIBLE_DEVICES=$devices python model_evaluate.py --model_path $save_dir --clean_cache false --model_family ${model}

model_id="${model_to_modelid[$model]}"

declare -A model_to_modelid=( ["llama2-7b"]="meta-llama/Llama-2-7b" ["llama3-8b"]="meta-llama/Meta-Llama-3-8B" )
model_id="${model_to_modelid[$model]}"

CUDA_VISIBLE_DEVICES=${devices} lm_eval --model hf \
        --tasks piqa,race,mmlu \
        --model_args parallelize=True,pretrained=${save_dir},tokenizer=${model_id} \
        --batch_size 4 \
        --output_path ${save_dir}