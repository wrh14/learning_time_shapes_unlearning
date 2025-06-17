from tqdm import tqdm
from data_module import TextDatasetQA, custom_data_collator, get_batch_loss, custom_data_collator_with_indices
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.nn import CrossEntropyLoss
import os, hydra
import evaluate
import json
from pathlib import Path
from rouge_score import rouge_scorer
from utils import get_model_identifiers_from_yaml, get_model_utility, get_forget_quality
import torch.nn as nn
import csv 
import numpy as np 
from vllm import SamplingParams

def eval_qa(dataset, model, tokenizer, device="cuda:0", question_key="question", answer_key="answer"):
    correct = []
    for data in tqdm(dataset):
        text = data[question_key]
        inputs = tokenizer(text, return_tensors="pt").to(device)

        outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        correct.append(data[answer_key] in output_text)
    return correct

def eval_qa_vllm(dataset, model_eval, qk="question", ak="answer", question_start_tag="[INST] ", question_end_tag=" [/INST]", answer_tag="", contexts=None, lora_request=None):
    if contexts is None:
        prompts = [question_start_tag + data[qk] + question_end_tag + answer_tag for data in dataset]
    else:
        prompts = [question_start_tag + context + " " + data[qk] + question_end_tag + answer_tag for data, context in zip(dataset, contexts)]
    sampling_params = SamplingParams(temperature=0, top_p=0.6, max_tokens=100)
    responses = model_eval.generate(prompts, sampling_params, lora_request=lora_request)
    outputs = [response.outputs[0].text for response in responses]
    # if qk == "person_question":
    #     for i in range(len(prompts)):
    #         print(prompts[i])
    #         print(outputs[i])
    correct = [data[ak].lower() in output.lower() for data, output in zip(dataset, outputs)]
    
    
    return correct, responses

# def eval_qa_vllm_strict(dataset, model_eval, qk="question", ak="answer", question_start_tag="[INST] ", question_end_tag=" [/INST]", answer_tag="", contexts=None, all_names = set()):
#     if contexts is None:
#         prompts = [question_start_tag + data[qk] + question_end_tag + answer_tag for data in dataset]
#     else:
#         prompts = [question_start_tag + context + " " + data[qk] + question_end_tag + answer_tag for data, context in zip(dataset, contexts)]
#     sampling_params = SamplingParams(temperature=0, top_p=0.6, max_tokens=100)
#     responses = model_eval.generate(prompts, sampling_params)
#     outputs = [response.outputs[0].text for response in responses]
#     if qk == "person_question":
#         for i in range(len(prompts)):
#             print(prompts[i])
#             print(outputs[i])
    
#     correct = [set(data[ak]) in output.lower() for data, output in zip(dataset, outputs)]
    
    
#     return correct, responses


def eval_completion_word(dataset, model, tokenizer, device="cuda:0", sentence_key="fact", completion_word_key="completion_word"):
    completion_position_logits_and_label_list = []
    completion_position_log_probs_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataset)):
#             if i <= 1000:
#                 continue
            sentence = data[sentence_key]
            completion_word = data[completion_word_key]

            input_ids_completion_word = tokenizer(" "+completion_word, return_tensors="pt")["input_ids"][0][1:].to(device)
            len_completion_word = len(input_ids_completion_word)

            tokenized_sentence = tokenizer(sentence, return_tensors="pt").to(device)
            input_ids_sentence = tokenized_sentence["input_ids"][0].to(device)

            for pos in range(len(input_ids_sentence ) - len_completion_word + 1):
                if (input_ids_completion_word - input_ids_sentence[pos:(pos+len_completion_word)]).abs().sum() == 0:
                    break

            output = model(tokenized_sentence["input_ids"])
            completion_position_logits = output.logits[0][pos-1:pos-1+len_completion_word]
            completion_position_logits_and_label_list.append((completion_position_logits, input_ids_completion_word))
            loss_compute = CrossEntropyLoss()
            logprobs = -loss_compute(completion_position_logits, input_ids_completion_word)
            completion_position_log_probs_list.append(logprobs)

    return completion_position_log_probs_list, completion_position_logits_and_label_list