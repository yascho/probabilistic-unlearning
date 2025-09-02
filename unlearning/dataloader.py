from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from encoding import encode_finetune
import numpy as np
import torch
import copy


def load_datasets(hparams):
    datasets = {}

    if hparams['split'] == 'full':
        datasets['full'] = load_dataset("locuslab/TOFU", "full")
    else:
        forget_split = 'forget' + f"{hparams['split']:02}"
        retain_split = 'retain' + f"{100-hparams['split']}"

        print(f"Loading dataset split: {retain_split} and {forget_split}")
        datasets['forget'] = load_dataset("locuslab/TOFU", forget_split)
        datasets['retain'] = load_dataset("locuslab/TOFU", retain_split)

    for dataset_name in datasets:
        datasets[dataset_name] = datasets[dataset_name]['train']

    return datasets


class UnlearningDataset(Dataset):
    def __init__(self, hparams, tokenizer, forget_data, retain_data):
        super(UnlearningDataset, self).__init__()
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.forget_data = forget_data
        self.retain_data = retain_data

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        result = []

        for data_type in ["forget", "retain"]:

            if data_type == "forget":
                data = self.forget_data
            else:
                data = self.retain_data
                randint = torch.randint(0, len(self.retain_data), (1,)).item()
                idx = (idx + randint) % len(self.retain_data)

            x = {"question": data[idx]['question'],
                 "answer": data[idx]['answer']}

            converted_data = encode_finetune(self.hparams, self.tokenizer, x)
            result.extend([converted_data, x['question'], x['answer']])

        return result


def data_collator_unlearning(samples):
    forget_samples = [sample[0] for sample in samples]
    forget_questions = [sample[1] for sample in samples]
    forget_answers = [sample[2] for sample in samples]
    retain_samples = [sample[3] for sample in samples]

    res = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [torch.tensor(s['input_ids']) for s in data]
        labels = [torch.tensor(s['labels']) for s in data]
        attention_mask = [torch.tensor(s['attention_mask']) for s in data]

        res.append((torch.stack(input_ids), torch.stack(
            labels), torch.stack(attention_mask)))
    return res + [forget_questions, forget_answers]
