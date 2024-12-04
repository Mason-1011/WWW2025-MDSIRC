import json
import pandas as pd
import os
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from itertools import permutations
import random


def load_data(config, shuffle=False):
    """
    加载数据，返回 DataLoader
    """
    dataset = TextDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size_relation"], shuffle=shuffle)
    return dataloader



class TextDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
        self.data = []
        self.json_data = []
        self.load()

    def load(self):
        self.load_preprocess(self.config['data_path'])
        for sample in self.json_data:
            self.append_data(sample)

    def load_preprocess(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # print(type(data))
            # 获取 JSON 文件的目录路径
            base_dir = os.path.dirname(data_path)

            # 修正每个 item 中的图片路径
            for sample in data:
                self.preprocess_sample(sample, base_dir, save=self.config['input_text_save'])

        self.json_data = data

    def preprocess_sample(self, sample, image_path, save='user+'):
        if 'image' in sample and sample['image']:
            # 假设 image 字段是一个列表，修正列表中的每个路径
            sample['image'] = [os.path.join(image_path, 'images', img) for img in sample['image']]
        sample['user_messages'], sample['customer_messages'] = zip(extract_messages(sample['instruction']))
        sample['input_text'] = design_input_text(sample['user_messages'], sample['customer_messages'], save)
        sample['input_ids'] = self.tokenizer.encode(sample['input_text'], truncation=True, padding='max_length',
                                                    max_length=self.config['max_length_map'][save])
        sample['output_id'] = self.config['label_map'][sample['output']]

    def append_data(self, sample):
        pass

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


if __name__ == '__main__':
    pass
