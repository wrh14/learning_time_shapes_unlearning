from data_module import TextForgetDataset, custom_data_collator, TextDataset, TextDatasetQA
from dataloader import CustomFamilyTrainerForgetting, CustomTrainerForgetting
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
from datasets import Dataset
import os
import gc
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf
import numpy as np

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    if local_rank == 0:
        # save cfg in cfg.save_dir
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    #get the the unlearn_data_i in shuffled id
    if "family" in cfg.data_name:
        if cfg.unlearn_split == "_people12":
            print("Unlearn people Batch 12")
            if "chunk" in cfg.data_path:
                random_split = torch.load("synthetic_data/unlearn_chunk_id_people_split.pt")
            else:
                random_split = torch.load("synthetic_data/unlearn_fact_id_people_split.pt")
        else:
            print("Unlearn Batch 100")
            if "chunk" in cfg.data_path:
                random_split = torch.load("synthetic_data/unlearn_chunk_id.pt")
            else:
                random_split = torch.load("synthetic_data/unlearn_fact_id.pt")
        if cfg.unlearn_data_id != -1:
            shuffled_unlearn_data_id = int(random_split[cfg.unlearn_data_id])
        else:
            shuffled_unlearn_data_id = random_split

        max_length = 500
        torch_format_dataset = TextDataset(cfg.data_path, tokenizer=tokenizer, max_length=max_length, text_key=cfg.text_key, subsample=random_split, num_rephrased=cfg.num_rephrased)
    elif "tofu" in cfg.data_name:
        print(f"Unlearn {cfg.data_name}")
        torch_format_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=500)
    else:
        print(f"{cfg.data_name} is not implemented")
        exit()
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    print(torch_format_dataset, batch_size, gradient_accumulation_steps, num_devices)
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    print(f"steps_per_epoch: {steps_per_epoch}")

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, steps_per_epoch),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            lr_scheduler_type="cosine",
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy=cfg.save_strategy,
            save_steps=cfg.save_steps,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            eval_steps = 1,
            evaluation_strategy = "steps" if cfg.eval_while_train else "no",
            seed=cfg.seed,
        )
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break

    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)
        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], trust_remote_code = True)
    else:
        print("Loading after merge and unload")
        model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, device_map=device_map)
        #now use the checkpoint to add the LoRA modules
        model = PeftModel.from_pretrained(model, model_id = cfg.model_path)
        #save this as a standard model so that we can again do PEFT style finetuneing from scratch
        model = model.merge_and_unload()
        #save the model for next time
        model.save_pretrained(cfg.model_path)
    
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    if cfg.LoRA.r != 0:
        config = LoraConfig(
            r=cfg.LoRA.r, 
            lora_alpha=cfg.LoRA.alpha, 
            target_modules=find_all_linear_names(model), 
            lora_dropout=cfg.LoRA.dropout,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        compute_metrics=None,  
        args=training_args,
        data_collator=custom_data_collator,
        oracle_model = oracle_model,
        forget_loss = cfg.forget_loss,
        data_name = cfg.data_name,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    
    # trainer.train()
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()

