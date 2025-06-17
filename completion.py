import argparse
import os

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import numpy as np

from evaluate_util import eval_completion_word
from utils import get_model_identifiers_from_yaml

parser = argparse.ArgumentParser(description='evaluate llm by vllm')
parser.add_argument('--model_path', type=str, default="~/")
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--config_path', type=str, default="config")
parser.add_argument('--model_family', type=str, default="llama2-7b-clm")
args = parser.parse_args()

if args.output_path is None:
    args.output_path = args.model_path

#LOAD THE MODEL
model_cfg = get_model_identifiers_from_yaml(args.model_family, config_path=args.config_path)
model_id = model_cfg["hf_key"]
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], padding_side="left")

config = AutoConfig.from_pretrained(args.model_path)
model_eval = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, token=os.environ['HF_TOKEN'], device_map=device_map)
model_eval.eval()

#LOAD THE DATASET
if args.data_path is None:
    data_path_list = [
        "synthetic_data/family-with-attributes-200-rephrased-text-train.pt",
        "synthetic_data/family-with-attributes-200-rephrased-text-test.pt",
        "synthetic_data/family-with-attributes-200-text-test.pt",
    ]
    output_path_list = [
        f"{args.output_path}/completion_rephrased-text-train",
        f"{args.output_path}/completion_rephrased-text-test",
        f"{args.output_path}/completion_text-test"
    ]
else:
    data_path_list = [
        args.data_path
    ]
    output_path_list = [
        args.output_path
    ]
    
    
for data_path, output_path in zip(data_path_list, output_path_list):
#     if os.path.exists(output_path):
    if False:
        (completion_position_log_probs_list, completion_position_logits_and_label_list) = torch.load(output_path)
    else:
        dataset = Dataset.from_dict(torch.load(data_path))
        completion_position_log_probs_list, completion_position_logits_and_label_list = eval_completion_word(dataset, model_eval, tokenizer, device=model_eval.device, sentence_key="fact", completion_word_key="completion_word")
        torch.save((completion_position_log_probs_list, completion_position_logits_and_label_list), output_path)
    probs_list = [np.exp(np.asarray(log_probs).astype(np.float32).sum()) for log_probs in completion_position_log_probs_list]
    avg_probs = np.asarray(probs_list).mean()
    print(f"Data at {data_path}", f"Average probability {avg_probs}")