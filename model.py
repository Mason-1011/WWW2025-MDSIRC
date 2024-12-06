import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import AutoModel


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
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        # 双向 LSTM 层
        self.bilstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)
        # 投影层，将 BiLSTM 输出的 hidden_size * 2 投影回 hidden_size
        self.lstm_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Layer Normalization 层
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Transformer 层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4, dropout=0.1
        )

        # 4 层的 Transformer 编码器
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

        # 线性层
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 3),
            nn.ReLU(),
            nn.Linear(hidden_size * 3, hidden_size),
        )

        # 池化层
        self.pooling = CustomPoolingLayer(hidden_size, mode=config['pooling_mode'])

        # 分类器层
        self.classify = nn.Linear(hidden_size, class_num)

        # 损失函数
        self.loss = CustomLoss(alpha, gamma, weights, ce_reduction, focal_reduction, loss_type)

    def forward(self, input_ids, labels=None):
        # print('input shape:',input_ids.shape)
        # 使用 AutoModel 进行编码
        if self.pooling_mode == 'cls':
            x = self.encoder(input_ids).last_hidden_state[:, -1, :]  # x.shape: [batch_size, hidden_size]
        else:
            x = self.encoder(input_ids).last_hidden_state  # x.shape: [batch_size, seq_length, hidden_size]
            # print('encoder shape:',x.shape)

            # 根据 output_block 参数选择输出
            if self.output_block == 'BiLSTM':
                # 使用 BiLSTM 处理编码后的特征
                lstm_output, _ = self.bilstm(x)
                # print('bilstm middle shape:',lstm_output.shape)
                lstm_output = self.lstm_proj(lstm_output)
                # print('bilstm output shape:',lstm_output.shape)
                x = self.layer_norm(lstm_output + x)  # x.shape: [batch_size, seq_length, hidden_size]
                # print('bilstm+layer_norm output shape:',x.shape)
            elif self.output_block == 'Transformer':
                # 使用 Transformer 处理编码后的特征
                x = self.transformer_layer(x)  # x.shape: [batch_size, seq_length, hidden_size]
            elif self.output_block == 'TransformerEncoder':
                # 使用 TransformerEncoder 处理编码后的特征
                x = self.transformer_encoder(x)  # x.shape: [batch_size, seq_length, hidden_size]
            elif self.output_block == 'BiLSTM+Transformer':
                # 使用 BiLSTM 和 Transformer 处理编码后的特征
                lstm_output, _ = self.bilstm(x)
                lstm_output = self.lstm_proj(lstm_output)
                x = self.layer_norm(lstm_output + x)
                x = self.transformer_layer(x)  # x.shape: [batch_size, seq_length, hidden_size]
            elif self.output_block == 'BiLSTM+TransformerEncoder':
                # 使用 BiLSTM 和 TransformerEncoder 处理编码后的特征
                lstm_output, _ = self.bilstm(x)
                lstm_output = self.lstm_proj(lstm_output)
                x = self.layer_norm(lstm_output + x)
                x = self.transformer_encoder(x)  # x.shape: [batch_size, seq_length, hidden_size]
            else:
                x = x

        # 使用线性层处理特征
        x = self.ffn(x)  # x.shape: [batch_size, seq_length, hidden_size]
        # print('ffn output shape:',x.shape)

        if self.pooling_mode == 'mean' or self.pooling_mode == 'max' or self.pooling_mode == 'concat':
            # 使用池化层处理特征
            x = self.pooling(x)  # x.shape: [batch_size, hidden_size]
            # print('pooling output shape:',x.shape)
        elif self.pooling_mode != 'cls':
            raise ValueError("Invalid pooling mode: {}".format(self.pooling))

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


if __name__ == '__main__':
    from config import config
    from loader import load_data

    data = load_data(config)
    model = TextModel(config)

    # 测试模型
    for batch_data in data:
        id, input_ids, labels = batch_data
        loss = model(input_ids, labels)
        print(loss)

