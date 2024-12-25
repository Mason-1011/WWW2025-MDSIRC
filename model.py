import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import AutoModel
import os
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

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


class CustomCNN(nn.Module):
    def __init__(self, hidden_size, num_filters, filter_sizes):
        super(CustomCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        x = x.permute(0, 2, 1)  # (batch_size, hidden_size, seq_len)

        # Apply convolution + ReLU + max pooling
        conv_outputs = [
            F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2)
            for conv in self.convs
        ]

        # Concatenate along the filter dimension
        out = torch.cat(conv_outputs, dim=1)
        out = self.dropout(out)
        return out  # (batch_size, num_filters * len(filter_sizes))


class TextModel(nn.Module):
    def __init__(self, config, weights=None):
        super(TextModel, self).__init__()

        alpha = config['alpha']
        gamma = config['gamma']
        ce_reduction = config['ce_reduction']
        focal_reduction = config['focal_reduction']
        loss_type = config['loss_type']
        class_num = len(config['label_map'])

        self.pooling_mode = config['pooling_mode']
        self.output_block = config['output_block']
        self.encoder = AutoModel.from_pretrained(config['model_path'])

        # 冻结预训练模型的参数
        # for name, param in self.encoder.named_parameters():
        #     param.requires_grad = False

        for i, layer in enumerate(self.encoder.layers):
            # 冻结前 19 个 DecoderLayer
            if i <22:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True

        hidden_size = self.encoder.config.hidden_size

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
        # self.pooling = CustomPoolingLayer(hidden_size, mode=config['pooling_mode'])
        filters = [2,4,6,8]
        self.pooling = CustomCNN(hidden_size,int(hidden_size/len(filters)),filters)

        # 分类器层
        self.classify = nn.Linear(hidden_size, class_num)

        # 损失函数
        self.loss = CustomLoss(alpha, gamma, weights, ce_reduction, focal_reduction, loss_type)

    def forward(self, input_ids, labels=None):
        # print('input shape:',input_ids.shape)
        # 使用 AutoModel 进行编码

        x = self.encoder(input_ids).last_hidden_state  # x.shape: [batch_size, seq_length, hidden_size]
        # print('encoder shape:',x.shape)


        # 使用 BiLSTM 处理编码后的特征
        lstm_output, _ = self.bilstm(x)
        # print('bilstm middle shape:',lstm_output.shape)
        lstm_output = self.lstm_proj(lstm_output)
        # print('bilstm output shape:',lstm_output.shape)
        x = self.layer_norm(lstm_output + x)  # x.shape: [batch_size, seq_length, hidden_size]
        # print('bilstm+layer_norm output shape:',x.shape)


        # 使用线性层处理特征
        x = self.ffn(x)  # x.shape: [batch_size, seq_length, hidden_size]
        # print('ffn output shape:',x.shape)

        # 使用池化层处理特征
        x = self.pooling(x)  # x.shape: [batch_size, hidden_size]

        # 使用分类器层进行分类
        logits = self.classify(x)
        # print('classify output shape:',logits.shape)

        # 如果有标签，计算损失
        if labels is not None:
            # print(logits.shape,labels.shape)
            loss = self.loss(logits, labels)
            return loss
        else:
            return logits


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]

    if optimizer == "adam":
        return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

class ImageModel(nn.Module):
    def __init__(self, config, dtype=torch.bfloat16):
        super(ImageModel, self).__init__()

        self.dtype = dtype
        self.config = config
        self.pooling_mode = config["pooling_mode"]
        self.class_num = len(config["image_labels"])

        hidden_size = config["hidden_size_image"]
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
        self.processor = None
        self.load_pretrained_model()

        alpha = config['alpha']
        gamma = config['gamma']
        ce_reduction = config['ce_reduction']
        focal_reduction = config['focal_reduction']
        loss_type = config['loss_type']
        # 损失函数
        self.loss = CustomLoss(alpha, gamma, None, ce_reduction, focal_reduction, loss_type)

    def load_pretrained_model(self):
        model_name = self.config["model_path_image"]
        # 加载预训练模型
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", output_hidden_states=True
        )

        visual_target_modules = [
            "visual.blocks.0.attn.qkv",
            "visual.blocks.5.attn.qkv",
            "visual.blocks.10.attn.qkv",
            "visual.blocks.15.attn.qkv",
            "visual.blocks.20.attn.qkv",
            "visual.blocks.25.attn.qkv",
            "visual.blocks.30.attn.qkv"
        ]

        text_target_modules = [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.10.self_attn.q_proj",
            "model.layers.10.self_attn.v_proj",
            "model.layers.15.self_attn.q_proj",
            "model.layers.15.self_attn.v_proj",
            "model.layers.20.self_attn.q_proj",
            "model.layers.20.self_attn.v_proj",
            "model.layers.25.self_attn.q_proj",
            "model.layers.25.self_attn.v_proj"
        ]

        # 定义 LoRA 配置
        lora_config = LoraConfig(
            r=8,  # LoRA rank
            lora_alpha=32,  # 缩放系数
            target_modules= visual_target_modules + text_target_modules,  # 应用 LoRA 的层
            lora_dropout=0.1,  # Dropout 概率
            bias="none",  # 偏置处理
            task_type="CAUSAL_LM",  # 任务类型
        )

        # 应用 LoRA
        self.encoder = get_peft_model(base_model, lora_config)

        # 冻结非 LoRA 参数
        for name, param in self.encoder.named_parameters():
            if "lora_" not in name:  # 仅微调 LoRA 参数
                param.requires_grad = False

        # 打印lora层
        for name, param in self.encoder.named_parameters():
            if "lora_" in name:
                print(name, param.size())

        min_pixels = 256 * 28 * 28
        # max_pixels = 1280 * 28 * 28
        max_pixels = 1080 * 24 * 24
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    def encoder_Qwen2_VL(self, config, image_ids):
        messages = []
        for image_id in image_ids:
            message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": os.path.join("/root/autodl-tmp/WWW2025-MDSIRC/train/images", image_id),
                        },
                        {"type": "text", "text": "Hello, describe this picture"},
                    ],
                }
            ]
            messages.append(message)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.encoder.device)

        encoded = self.encoder(**inputs)
        # 检查输出对象中的隐藏状态
        hidden_states = encoded.hidden_states  # 这是一个 tuple，每一层的输出为一个张量
        last_hidden_state = hidden_states[-1]
        return last_hidden_state



    def forward(self, inputs, labels=None):
        last_hidden_state = self.encoder_Qwen2_VL(self.config, inputs["image_id"])
        # 使用线性层处理特征
        x = self.ffn(last_hidden_state)  # x.shape: [batch_size, seq_length, hidden_size]

        if self.pooling_mode == 'mean' or self.pooling_mode == 'max' or self.pooling_mode == 'concat':
            # 使用池化层处理特征
            x = self.pooling(x)  # x.shape: [batch_size, hidden_size]
            # print('pooling output shape:',x.shape)
        elif self.pooling_mode != 'cls':
            raise ValueError("Invalid pooling mode: {}".format(self.pooling))

        # 使用分类器层进行分类
        logits = self.classify(x)
        # 如果有标签，计算损失
        if labels is not None:
            # print(logits.shape,labels.shape)
            loss = self.loss(logits, labels)
            return loss, logits
        else:
            return logits

class TIModel(nn.Module):
    def __init__(self, config, dtype=torch.bfloat16):
        super(TIModel, self).__init__()

        self.dtype = dtype
        self.config = config
        self.pooling_mode = config["pooling_mode"]
        self.class_num = len(config["label_map"])

        hidden_size = config["hidden_size_image"]

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
        self.processor = None
        self.load_pretrained_model_for_text_classify()

        alpha = config['alpha']
        gamma = config['gamma']
        ce_reduction = config['ce_reduction']
        focal_reduction = config['focal_reduction']
        loss_type = config['loss_type']
        # 损失函数
        self.loss = CustomLoss(alpha, gamma, None, ce_reduction, focal_reduction, loss_type)

    def load_pretrained_model_for_text_classify(self):
        model_name = self.config["model_path"]
        # 加载预训练模型
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", output_hidden_states=True
        )
        text_target_modules = [f"model.layers.{i}.self_attn.q_proj" for i in range(0,26,1)] + [f"model.layers.{i}.self_attn.v_proj" for i in range(0,26,1)]
        # text_target_modules = [
        #     "model.layers.0.self_attn.q_proj",
        #     "model.layers.0.self_attn.v_proj",
        #     "model.layers.10.self_attn.q_proj",
        #     "model.layers.10.self_attn.v_proj",
        #     "model.layers.15.self_attn.q_proj",
        #     "model.layers.15.self_attn.v_proj",
        #     "model.layers.20.self_attn.q_proj",
        #     "model.layers.20.self_attn.v_proj",
        #     "model.layers.25.self_attn.q_proj",
        #     "model.layers.25.self_attn.v_proj"
        # ]

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
        self.encoder = get_peft_model(base_model, lora_config)

        # 冻结非 LoRA 参数
        for name, param in self.encoder.named_parameters():
            if "lora_" not in name:  # 仅微调 LoRA 参数
                param.requires_grad = False

        # 打印lora层
        # for name, param in self.encoder.named_parameters():
        #     if "lora_" in name:
        #         print(name, param.size())

        min_pixels = 256 * 28 * 28
        # max_pixels = 1280 * 28 * 28
        max_pixels = 720 * 24 * 24
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    def encoder_Qwen2_VL(self, config, inputs):
        messages = []
        for i in range(len(inputs['id'])):
            if False:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": inputs['image'][i],
                            },
                            {"type": "text", "text": f"首先，描述图片内容(包含什么字？，是什么物品？)。然后根据‘对话记录’判断‘用户’的对话意图\n 对话记录：\"\"\" {inputs['text'][i]} \"\"\" " },
                        ],
                    }
                ]
            else:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": f"根据‘对话记录’判断‘用户’的对话意图\n 对话记录：\"\"\" {inputs['text'][i]} \"\"\" "},
                        ],
                    }
                ]
            messages.append(message)
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.encoder.device)

        encoded = self.encoder(**inputs)
        # 检查输出对象中的隐藏状态
        hidden_states = encoded.hidden_states  # 这是一个 tuple，每一层的输出为一个张量
        last_hidden_state = hidden_states[-1]
        return last_hidden_state



    def forward(self, inputs, labels=None):
        last_hidden_state = self.encoder_Qwen2_VL(self.config, inputs)

        # 使用 BiLSTM 处理编码后的特征
        lstm_output, _ = self.bilstm(last_hidden_state)
        # print('bilstm middle shape:',lstm_output.shape)
        lstm_output = self.lstm_proj(lstm_output)
        # print('bilstm output shape:',lstm_output.shape)
        x = self.layer_norm(lstm_output + last_hidden_state)  # x.shape: [batch_size, seq_length, hidden_size]

        # 使用线性层处理特征
        x = self.ffn(x)  # x.shape: [batch_size, seq_length, hidden_size]

        if self.pooling_mode == 'mean' or self.pooling_mode == 'max' or self.pooling_mode == 'concat':
            # 使用池化层处理特征
            x = self.pooling(x)  # x.shape: [batch_size, hidden_size]
            # print('pooling output shape:',x.shape)
        elif self.pooling_mode != 'cls':
            raise ValueError("Invalid pooling mode: {}".format(self.pooling))

        # 使用分类器层进行分类
        logits = self.classify(x)
        # 如果有标签，计算损失
        if labels is not None:
            # print(logits.shape,labels.shape)
            loss = self.loss(logits, labels)
            return loss, logits
        else:
            return logits

if __name__ == '__main__':
    from config import config
    from loader import load_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(config,'text','./train/train_text.json')
    model = TIModel(config).to(device)
    model.train()
    # 测试模型
    for samples in data:
        labels = [config["label_map"][i] for i in samples["label"]]
        output_ids = torch.tensor(labels).to(device)
        loss, logits = model(samples, output_ids)
        print(loss)