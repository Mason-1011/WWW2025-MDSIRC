import json
import os
import torch
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from itertools import permutations
from Augmentation import SynonymReplacer
from tqdm import tqdm  # 引入tqdm库
from torchvision import transforms
from PIL import Image


def load_data(config, task_type, path, shuffle=False, augment=False):
    """
    加载数据，返回 DataLoader
    """
    if task_type == 'text':
        dataset = TextDataset(path, config, augment)
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)
    elif task_type == 'image':
        dataset = ImageDataset(path, config)
        dataloader = DataLoader(dataset, batch_size=config["batch_size_image"], shuffle=shuffle)
    else:
        raise ValueError("Invalid dataset type: {}".format(task_type))
    return dataloader


class TextDataset(Dataset):
    def __init__(self, path, config, augment=False, ):
        self.config = config
        self.path = path
        if augment:
            self.replacer = SynonymReplacer()
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
        self.max_length_map = config['max_length_map']
        self.label_map = config['label_map']
        self.data = []
        self.json_data = []
        self.load()

    def load(self):
        self.load_preprocess(self.path)
        for sample in tqdm(self.json_data, desc="Appending Data", ncols=80):
            self.append_data(sample)

    def load_preprocess(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for sample in tqdm(data, desc="Processing Data", ncols=80):
                self.preprocess_sample(sample, save=self.config['input_text_save'])

    def preprocess_sample(self, sample, save='user+'):
        user_messages, customer_messages = extract_messages(sample['instruction'])
        input_text = design_input_text(user_messages, customer_messages, save)
        output_id = self.label_map[sample['output']] if 'output' in sample else -1
        id = sample['id']

        # 显示生成句子的进度
        input_text_list = self.replacer.generate_augmented_sentences(input_text, k=5) if hasattr(self,'replacer') else [input_text]

        for i in range(len(input_text_list)):
            self.json_data.append({
                'id': f"{id}_{i}",
                'input_text': input_text_list[i],
                # 'input_ids': self.tokenizer.encode(
                #     input_text_list[i],
                #     truncation=True,
                #     padding='max_length',
                #     max_length=self.max_length_map[save]
                # ),
                'output_id': output_id,
                'output': sample['output'] if 'output' in sample else ' ',
                'instruction': sample['instruction'],
                'image': sample['image']
            })

    def append_data(self, sample):
        input_ids = torch.tensor(sample['input_ids'])
        output_id = torch.tensor(sample['output_id'])
        self.data.append([sample['id'], input_ids, output_id])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def extract_messages(instruction):
    user_pattern = r"用户: (.+)"
    customer_pattern = r"客服: (.+)"
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
            input_text = '用户: ' + ' '.join(user_messages) + ' 客服: 无'
    elif save == 'user-':
        input_text = '用户: ' + user_messages[-1]
    elif save == 'user-customer-':
        if len(customer_messages) > 0:
            input_text = '用户: ' + user_messages[-1] + ' 客服: ' + customer_messages[-1]
        else:
            input_text = '用户: ' + user_messages[-1] + ' 客服: 无'
    return input_text


class ImageDataset(Dataset):
    def __init__(self, path, config, image_tensor = False):
        self.config = config
        self.path = path
        self.data = []
        self.image_tensor = image_tensor
        self.load()

    def load(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                data_read = json.load(file)

        except FileNotFoundError:
            print(f"Error: The file at {self.path} was not found.")
        except json.JSONDecodeError:
            print("Error: The file is not a valid JSON file.")

        for sample in data_read:
            if sample["instruction"].startswith('Picture'):
                data_point = {"id": sample["id"]}
                data_point["label"] = sample["output"]
                data_point["image_id"] = sample["image"][0]
                if self.image_tensor:
                    data_point["image"] = self.load_image(sample["image"][0])
                self.data.append(data_point)

    @staticmethod
    def load_image(image_id):
        image_path = os.path.join('train/images', image_id)

        with Image.open(image_path) as img:
            img = img.convert('RGB')

            # 创建一个变换，将 PIL 图像转换为 PyTorch 张量
            transform = transforms.Compose([
                transforms.ToTensor(),  # 这个变换会将 [0, 255] 的值缩放到 [0.0, 1.0]
            ])

            tensor = transform(img)
        return tensor


if __name__ == '__main__':
    from config import config

    data = load_data(config, task_type='text', path='./train/train_text.json', augment=True)
    from Viewer import Viewer, InteractiveViewer

    viewer = Viewer()
    viewer.load_data_from_json(data.dataset.json_data, images_dir='./train/images')
    app = InteractiveViewer(viewer)
    app.mainloop()
