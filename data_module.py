import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index
import os
import numpy as np

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
        
    encoded_answer = tokenizer(
        new_answer, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
        
        
    #change label to -100 for question tokens
#     print(encoded['input_ids'][num_question_tokens], label[num_question_tokens])
    for i in range(num_question_tokens): label[i] = -100
    
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

def convert_pure_text_raw_data_to_model_format(tokenizer, max_length,  text):

    encoded = tokenizer(
        text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
#     for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    

class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, split=None, max_length=512, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # data_len = len(datasets.load_dataset(data_path, split)["train"])
        # self.data = datasets.load_dataset(data_path, split)["train"].select(range(min(100, data_len)))
        if data_path.endswith("pt"):
            self.data = datasets.Dataset.from_dict(torch.load(data_path))
        elif data_path == "locuslab/TOFU":
            self.data = datasets.load_dataset(data_path, split)["train"]
        else:
            import json
            with open(data_path) as f:
                self.data = datasets.Dataset.from_dict(json.load(f))
        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, text_key="fact", subsample=None, num_rephrased=3):
        super(TextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        if "data_path".endswith("pt"):
            self.data = torch.load(data_path)
        else:
            import json
            with open(data_path) as f:
                self.data = json.load(f)
        self.k = text_key
        if subsample is not None:
            if isinstance(subsample, int):
                subsample = np.asarray([subsample])
            subsample_data_idx = []
            for j in range(num_rephrased):
                subsample_data_idx = subsample_data_idx + [idx * num_rephrased + j for idx in subsample] 
            self.data[self.k] = [self.data[self.k][idx] for idx in subsample_data_idx]

    def __len__(self):
        return len(self.data[self.k])

    def __getitem__(self, idx):
        texts = self.data[self.k][idx]
        indices = idx
        if isinstance(texts, str):
            texts = [texts]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for text in texts:
#             converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            converted_data = convert_pure_text_raw_data_to_model_format(self.tokenizer, self.max_length, text)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)
    
class TextForgetDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, text_key="fact", unlearn_data_id=0, question_key=None, answer_key=None, num_rephrased=1):
        super(TextForgetDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = datasets.Dataset.from_dict(torch.load(data_path))
        self.data = add_dataset_index(self.data)
        self.k = text_key
        self.qk = question_key
        self.ak = answer_key
        if isinstance(unlearn_data_id, int):
            unlearn_data_id = np.asarray([unlearn_data_id])
        self.unlearn_data_id = unlearn_data_id
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.num_rephrased = num_rephrased
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))

    def __len__(self):
        return self.num_rephrased * self.world_size * len(self.unlearn_data_id)

    def __getitem__(self, idx):
        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []
        data_idx = self.unlearn_data_id[int(int(idx/self.world_size) / self.num_rephrased)]
        if self.qk is None:
            texts = self.data[self.k][data_idx*self.num_rephrased:((data_idx+1)*self.num_rephrased)]
            texts = texts[idx%self.num_rephrased]
            indices = idx
            if isinstance(texts, str):
                texts = [texts]

            for text in texts:
                converted_data = convert_pure_text_raw_data_to_model_format(self.tokenizer, self.max_length, text)
                pad_input_ids_list.append(converted_data[0])
                label_list.append(converted_data[1])
                pad_attention_mask_list.append(converted_data[2])
        else:
            question = self.data[data_idx*self.num_rephrased:((data_idx+1)*self.num_rephrased)][self.qk]
            answers = self.data[data_idx*self.num_rephrased:((data_idx+1)*self.num_rephrased)][self.ak]
            indices = self.data[data_idx*self.num_rephrased:((data_idx+1)*self.num_rephrased)]['index']
            
            idx = idx % self.num_rephrased
            question = question[idx]
            answers = answers[idx]
            indices = indices[idx]
            
            if isinstance(answers, str):
                answers = [answers]

            pad_input_ids_list = []
            label_list = []
            pad_attention_mask_list = []

            for answer in answers:
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                pad_input_ids_list.append(converted_data[0])
                label_list.append(converted_data[1])
                pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss
