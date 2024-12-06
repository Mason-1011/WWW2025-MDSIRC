import json

import os
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from itertools import permutations


def load_data(config, task_type, path, shuffle=False):
    """
    加载数据，返回 DataLoader
    """
    if task_type == 'text':
        dataset = TextDataset(path, config)
    elif task_type == 'image':
        dataset = ImageDataset(path, config)
    else:
        raise ValueError("Invalid dataset type: {}".format(task_type))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    return dataloader


class TextDataset(Dataset):
    def __init__(self, path, config):
        self.config = config
        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
        self.max_length_map = config['max_length_map']
        self.label_map = config['label_map']
        self.data = []
        self.json_data = []
        self.load()

    def load(self):
        self.load_preprocess(self.path)
        for sample in self.json_data:
            self.append_data(sample)

    def load_preprocess(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # print(type(data))

            for sample in data:
                self.preprocess_sample(sample, save=self.config['input_text_save'])

        self.json_data = data

    def preprocess_sample(self, sample, save='user+'):
        sample['user_messages'], sample['customer_messages'] = extract_messages(sample['instruction'])
        sample['input_text'] = design_input_text(sample['user_messages'], sample['customer_messages'], save)
        sample['input_ids'] = self.tokenizer.encode(sample['input_text'], truncation=True, padding='max_length',
                                                    max_length=self.max_length_map[save])
        sample['output_id'] = self.label_map[sample['output']]

    def append_data(self, sample):
        input_ids = torch.tensor(sample['input_ids'])
        output_id = torch.tensor(sample['output_id'])
        self.data.append([sample['id'], input_ids, output_id])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def extract_messages(instruction):
    # Regular expressions to extract user and customer messages
    user_pattern = r"用户: (.+)"
    customer_pattern = r"客服: (.+)"

    # Extracting the messages
    user_messages = re.findall(user_pattern, instruction)
    customer_messages = re.findall(customer_pattern, instruction)
    return user_messages, customer_messages


def design_input_text(user_messages, customer_messages, save='user+'):
    input_text = ''
    # print(user_messages)
    if save == 'user+':
        input_text = '用户: ' + ' '.join(user_messages)
    elif save == 'user+customer-':
        if len(customer_messages) > 0:
            input_text = '用户: ' + ' '.join(user_messages) + ' ' + '客服: ' + customer_messages[-1]
        else:
            input_text = '用户: ' + ' '.join(user_messages) + ' ' + '客服: ' + '无'
    elif save == 'user-':
        input_text = '用户: ' + user_messages[-1]
    elif save == 'user-customer-':
        if len(customer_messages) > 0:
            input_text = '用户: ' + user_messages[-1] + ' ' + '客服: ' + customer_messages[-1]
        else:
            input_text = '用户: ' + user_messages[-1] + ' ' + '客服: ' + '无'

    return input_text


class ImageDataset(Dataset):
    def __init__(self, config):
        self.config = config


if __name__ == '__main__':
    from config import config
    data = load_data(config, task_type='text', path=config['train_text_path'])
    # for batch in data:
    #     id,input,output = batch
    #     print(id,input,output)
    from Viewer import Viewer, InteractiveViewer
    viewer = Viewer()
    viewer.load_data_from_json(data.dataset.json_data)
    app = InteractiveViewer(viewer)
    app.mainloop()
