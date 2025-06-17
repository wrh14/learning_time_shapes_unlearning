import argparse
import json
from pathlib import Path
import gc
import os

import torch
import numpy as np
from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import Dataset
from peft import PeftModel

from utils import get_model_identifiers_from_yaml
from evaluate_util import eval_qa_vllm, eval_completion_word

parser = argparse.ArgumentParser(description='evaluate llm by vllm')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--model_family', type=str, default="llama2-7b")
parser.add_argument('--clean_cache', type=str, default="false")
args = parser.parse_args()

model_path = args.model_path 
if args.output_path is None:
    args.output_path = model_path
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
model_cfg = get_model_identifiers_from_yaml(args.model_family)

print(args.model_family)
print(model_cfg)

#CHECKING -- COMPLETING        
#LOAD THE MODEL

device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], padding_side="left")

config = AutoConfig.from_pretrained(model_id)
if ("lora" in model_path) and ("hf_model" not in model_path):
    model_eval = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, token=os.environ['HF_TOKEN'], device_map=device_map)
    
    model_eval = PeftModel.from_pretrained(model_eval, model_id = args.model_path)
    model_eval = model_eval.merge_and_unload()
    
    model_eval.save_pretrained(args.model_path + "/tmp_peft_merged_model")
        
    model_eval = AutoModelForCausalLM.from_pretrained(args.model_path + "/tmp_peft_merged_model", config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], trust_remote_code = True, device_map=device_map)
    model_eval._hf_peft_config_loaded = False
    
    model_eval.save_pretrained(args.model_path + "/hf_model")
else:
    model_eval = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, token=os.environ['HF_TOKEN'], device_map=device_map)
    
model_eval.eval()

#LOAD THE DATASET
data_path_list = [
     "synthetic_data/ft_mul.json",
     "synthetic_data/test_mul.json",
     "synthetic_data/unlearn_mul.json",
     "synthetic_data/unlearn_single.json",
     "synthetic_data/ft_single.json",
]
output_path_list = [
     f"{args.output_path}/completion_rephrased-text-train",
     f"{args.output_path}/completion_rephrased-text-test",
     f"{args.output_path}/completion_rephrased-text-unlearn",
     f"{args.output_path}/completion_text-unlearn",
     f"{args.output_path}/completion_text-train",
]
    
for data_path, output_path in zip(data_path_list, output_path_list):
    if os.path.exists(output_path):
        avg_probs = torch.load(output_path)
        if len(avg_probs) == 4:
            avg_probs = avg_probs[1]
        print(f"Data at {data_path}", f"Average probability {np.asarray(avg_probs).mean()}")
        continue
    with open(data_path) as f:
        dataset = json.load(f)
    dataset = Dataset.from_dict(dataset)
    completion_position_log_probs_list, completion_position_logits_and_label_list = eval_completion_word(dataset, model_eval, tokenizer, device=model_eval.device, sentence_key="fact", completion_word_key="completion_word")
#    completion_position_logits_and_label_list = completion_position_logits_and_label_list.to('cpu')
    probs_list = [np.exp(np.asarray(log_probs.to('cpu')).astype(np.float32).sum()) for log_probs in completion_position_log_probs_list]
    avg_probs = np.asarray(probs_list).mean()
    print(f"Data at {data_path}", f"Average probability {avg_probs}")
#     torch.save((avg_probs, probs_list, completion_position_log_probs_list, completion_position_logits_and_label_list), output_path)
    torch.save(np.asarray(probs_list), output_path)    

destroy_model_parallel()
del model_eval
gc.collect()
torch.cuda.empty_cache()

#remove local model
if args.clean_cache == "true":
    import shutil
    shutil.rmtree(model_path)