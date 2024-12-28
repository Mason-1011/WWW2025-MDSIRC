import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import AutoModel
import os
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from PIL import Image
import pytesseract
import re
import deepspeed

import warnings

# Suppress the specific warning
warnings.filterwarnings("ignore", message="`return_dict_in_generate` is NOT set to `True`, but `output_hidden_states` is")


class CustomLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, ce_reduction='none', focal_reduction='mean', loss_type='focal'):
        """
        Custom loss function supporting cross-entropy and focal loss with flexible reductions.

        Args:
            alpha (float): Focal loss scaling factor for balancing easy/hard examples.
            gamma (float): Focal loss focusing parameter.
            weight (Tensor, optional): Class weights for cross-entropy loss.
            ce_reduction (str): Reduction method for cross-entropy loss ('none', 'mean', 'sum').
            focal_reduction (str): Reduction method for focal loss ('none', 'mean', 'sum').
            loss_type (str): Type of loss to compute ('ce' for cross-entropy, 'focal' for focal loss).
        """
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ce_reduction = ce_reduction
        self.focal_reduction = focal_reduction
        self.loss_type = loss_type

    def forward(self, logits, targets):
        """
        Forward pass for loss computation.

        Args:
            logits (Tensor): Logits from the model, shape (N, C).
            targets (Tensor): Ground truth labels, shape (N,).

        Returns:
            Tensor: Computed loss based on the specified configurations.
        """
        # Compute cross-entropy loss with specified reduction
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction=self.ce_reduction)

        if self.loss_type == 'ce':
            return ce_loss

        elif self.loss_type == 'focal':
            # Compute p_t = exp(-ce_loss) and focal loss
            p_t = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

            # Apply focal_loss reduction externally
            if self.focal_reduction == 'none':
                return focal_loss
            elif self.focal_reduction == 'mean':
                return focal_loss.mean()
            elif self.focal_reduction == 'sum':
                return focal_loss.sum()
            else:
                raise ValueError(f"Unsupported reduction type: {self.focal_reduction}")

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")


class CustomPoolingLayer(nn.Module):
    def __init__(self, hidden_size, mode='concat'):
        """
        自定义池化层，将 (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)

        Args:
            hidden_size (int): 输入特征的维度
            mode (str): 池化方式，可选 'max', 'mean', 或 'concat'
                        - 'max': 仅使用最大池化
                        - 'mean': 仅使用平均池化
                        - 'concat': 拼接最大池化和平均池化的结果，并通过线性变换降维
        """
        super(CustomPoolingLayer, self).__init__()
        self.hidden_size = hidden_size
        self.mode = mode

        if mode == 'concat':
            self.linear = nn.Linear(hidden_size * 2, hidden_size)
            self.activation = nn.Tanh()

    def forward(self, x):
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, hidden_size)
        """
        if self.mode == 'max':
            # 最大池化，取每个特征的最大值
            pooled = torch.max(x, dim=1).values  # (batch_size, hidden_size)

        elif self.mode == 'mean':
            # 平均池化，取每个特征的平均值
            pooled = torch.mean(x, dim=1)  # (batch_size, hidden_size)

        elif self.mode == 'concat':
            # 最大池化和平均池化的结果拼接
            max_pooled = torch.max(x, dim=1).values  # (batch_size, hidden_size)
            mean_pooled = torch.mean(x, dim=1)  # (batch_size, hidden_size)
            concat_pooled = torch.cat([max_pooled, mean_pooled], dim=-1)  # (batch_size, hidden_size * 2)

            # 使用线性层将拼接的特征降维并通过激活函数
            pooled = self.activation(self.linear(concat_pooled))  # (batch_size, hidden_size)

        else:
            raise ValueError(f"Unsupported pooling mode: {self.mode}")

        return pooled


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]

    if optimizer == "adam":
        return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)




# Model class definition (with minor changes for DeepSpeed support)
class TIModel(nn.Module):
    def __init__(self, config, dtype=torch.bfloat16):
        super(TIModel, self).__init__()

        self.dtype = dtype
        self.config = config
        self.pooling_mode = config["pooling_mode"]
        self.class_num = len(config["label_map"])

        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            config['model_path'], torch_dtype="auto", device_map="auto", output_hidden_states=True
        )
        self.base_layer_names = [name for name, _ in self.base_model.named_modules()]
        self.processor = AutoProcessor.from_pretrained(self.config['model_path'])

        hidden_size = self.base_model.config.hidden_size

        # 双向 LSTM 层
        self.bilstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)
        # 投影层，将 BiLSTM 输出的 hidden_size * 2 投影回 hidden_size
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Layer Normalization 层
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 线性层
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 3),
            nn.ReLU(),
            nn.Linear(hidden_size * 3, hidden_size),
        )
        # 池化层
        self.pooling = CustomPoolingLayer(hidden_size, mode=config['pooling_mode'])
        # 分类器层
        self.classify = nn.Linear(hidden_size, self.class_num)
        self.encoder = None
        self.load_pretrained_model_for_text_classify()

        alpha = config['alpha']
        gamma = config['gamma']
        ce_reduction = config['ce_reduction']
        focal_reduction = config['focal_reduction']
        loss_type = config['loss_type']
        # 损失函数
        self.loss = CustomLoss(alpha, gamma, None, ce_reduction, focal_reduction, loss_type)

    def load_pretrained_model_for_text_classify(self):
        target_layer_suffixes = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

        # 筛选出符合条件的层名
        text_target_modules = [name for name in self.base_layer_names if
                            not name.startswith('visual') and any(
                                name.endswith(suffix) for suffix in target_layer_suffixes)]

        # 定义 LoRA 配置
        lora_config = LoraConfig(
            r=8,  # LoRA rank
            lora_alpha=32,  # 缩放系数
            target_modules=  text_target_modules,  # 应用 LoRA 的层
            lora_dropout=0.1,  # Dropout 概率
            bias="none",  # 偏置处理
            task_type="CAUSAL_LM",  # 任务类型
        )

        # 应用 LoRA
        self.encoder = get_peft_model(self.base_model, lora_config)

        # 冻结非 LoRA 参数
        for name, param in self.encoder.named_parameters():
            if "lora_" not in name:  # 仅微调 LoRA 参数
                param.requires_grad = False

    def encoder_Qwen2_VL(self, inputs):
        messages = []
        for i in range(len(inputs['id'])):
            ocr_text, descr_text = self.get_image_info(inputs['image'][i])
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"你是一个电商客服专家，请根据“对话记录”和“用户上传的图片中的重要内容”判断“用户”的对话意图"
                                 f"\n\"\"\" 用户上传的{descr_text}"
                                 f"\"\"\" "
                                 f"\n对话记录：\"\"\" {inputs['text'][i]} \"\"\" "},
                    ],
                }
            ]
            messages.append(message)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        inputs = self.processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.encoder.device)

        encoded = self.encoder(**inputs)
        hidden_states = encoded.hidden_states  # 这是一个 tuple，每一层的输出为一个张量
        last_hidden_state = hidden_states[-1]
        return last_hidden_state

    def get_image_info(self, image_path):
        if image_path == 'None':
            return '无', '无'
        image = Image.open(image_path)
        ocr_text = format_ocr_text(pytesseract.image_to_string(image, lang='chi_sim'))  # 中文简体

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text",
                     "text": "假设你是一个淘宝客服，请你从客服的角度捕捉用户给你的图片中的重要内容"},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.encoder.device)

        # Inference: Generation of the output
        generated_ids = self.base_model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        descr_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return ocr_text, descr_text

    def forward(self, inputs, labels=None):
        last_hidden_state = self.encoder_Qwen2_VL(inputs)
        last_hidden_state = last_hidden_state.to(torch.float32)  # 转换为 float32
        lstm_output, _ = self.bilstm(last_hidden_state)
        lstm_output = self.lstm_proj(lstm_output)
        x = self.layer_norm(lstm_output + last_hidden_state)

        x = self.ffn(x)

        if self.pooling_mode == 'mean' or self.pooling_mode == 'max' or self.pooling_mode == 'concat':
            x = self.pooling(x)
        elif self.pooling_mode != 'cls':
            raise ValueError("Invalid pooling mode: {}".format(self.pooling))

        logits = self.classify(x)
        if labels is not None:
            loss = self.loss(logits, labels)
            return loss, logits
        else:
            return logits

def format_ocr_text(text):
    text = re.sub(r' {2,}', '  ', text)
    text = re.sub(r'\n+', '  ', text)
    return text.strip() if text.strip() else "无"

# Main training loop with DeepSpeed
if __name__ == '__main__':
    from config import config
    from loader import load_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = load_data(config, 'text', './train/train_text.json')
    model = TIModel(config).to(device)

    # Initialize the model with DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None, model=model, model_parameters=model.parameters(), config_params="deepspeed_config.json"
    )
    print("Model is on device:", model_engine.module.device)

    model.train()
    for samples in data:
        labels = [config["label_map"][i] for i in samples["label"]]
        output_ids = torch.tensor(labels).to(device)

        # Forward pass
        loss, logits = model_engine(samples, output_ids)
        print(loss)
