import logging
import os
import torch
import pandas as pd
import numpy as np
from config import config
from loader import load_data
from model import TextModel, choose_optimizer
from evaluator import TextEvaluator
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 创建一个日志文件名（你可以根据需要自定义文件名）
log_file = 'log/log_text_block&textsave&loss_v2.log'

# 配置日志，输出到控制台和文件
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # 输出到控制台
                        logging.FileHandler(log_file, mode='a', encoding='utf-8')  # 输出到文件，追加模式
                    ])
logger = logging.getLogger(__name__)

def save_model(state_dict, path = './TextModel'):
    torch.save(state_dict, path)

def Text_Train(config, model_title='', save = False):
    logger.info(config)
    # 创建保存模型的目录
    # if not os.path.isdir(config["out_path"]):
    #     os.mkdir(config["out_path"])

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载训练数据
    train_data = load_data(path=config["train_text_path"], config=config, shuffle=False, task_type='text')
    # for idx, x in enumerate(train_data.dataset.data):
    #     if any(i is None for i in x):
    #         print(f"Problematic entry at index {idx}: {x}")
    #         raise ValueError("Dataset contains None!")

    # 计算类别权重
    # weights = calculate_weights_relation(train_data, 2).to(device)

    # 加载模型并转移到 GPU
    model = TextModel(config)
    model.to(device)
    # print(model.encoder.get_input_embeddings().weight.shape)

    save_dict = None

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = TextEvaluator(config, model, logger)

    # 加载学习率调度器
    if config['lr_scheduler']:
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=1, verbose=True)

    # 训练
    micro_f1s = []
    patience = config['patience']  # 早停的容忍度，即连续几个 epoch 的改进都小于阈值时停止
    best_f1 = 0
    no_improve_epochs = 0
    min_improvement = 0.001  # 微平均 F1 分数的最小改进

    for epoch in range(config["epochs"]):
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []

        for index, batch_data in enumerate(train_data):
            # 将 batch 数据移到 GPU
            id, input_ids, output_id = batch_data
            input_ids, output_id = input_ids.to(device), output_id.to(device)

            optimizer.zero_grad()
            loss = model(input_ids, output_id)
            loss.backward()

            optimizer.step()
            # model.validate_embeddings()
            train_loss.append(loss.item())
            if index % 5 == 1:
                logger.info("batch loss %f" % loss)

        logger.info("epoch average loss: %f" % np.mean(train_loss))

        # 验证模型并计算 Micro-F1
        logger.info(
            f"开始测试 {model_title} 模型 第{epoch}轮效果：")
        micro_f1 = evaluator.eval(epoch)
        micro_f1s.append(micro_f1)

        if config['lr_scheduler']:
            # 更新学习率
            scheduler.step(micro_f1)

        # 检查 F1 分数的改进
        if micro_f1 - best_f1 > min_improvement:
            best_f1 = micro_f1
            no_improve_epochs = 0  # 重置计数器
            if save:
                save_dict = model.state_dict()
                logger.info("Update model state dict to be saved")
        else:
            no_improve_epochs += 1
            logger.info(f"No significant improvement for {no_improve_epochs} epochs")

        # 如果连续几个 epoch 没有显著改进，则停止训练
        if no_improve_epochs >= patience and epoch >= 15:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            if save and save_dict:
                save_model(save_dict)
                logger.info("-----------------SAVE MODEL-----------------")
            break

    return model, train_data, micro_f1s


def grid_search():
    # outputs = []
    # for learning_rate in [1e-5]:
    #     config['learning_rate'] = learning_rate
    #     for lr_scheduler in [True, False]:
    #         config['lr_scheduler'] = lr_scheduler
    #         for loss_type in ['focal', 'ce']:
    #             config['loss_type'] = loss_type
    #             if loss_type == 'focal':
    #                 for ce_reduction, focal_reduction in [('sum','sum'), ('sum','mean')] :
    #                     config['ce_reduction'] = ce_reduction
    #                     config['focal_reduction'] = focal_reduction
    #                     for alpha in [0.5,  0.7, 0.9, 1.0] :
    #                         config['alpha'] = alpha
    #                         for gamma in [0.5, 1, 1.5, 2, 2.5, 3]:
    #                             config['gamma'] = gamma
    #
    #                             model_title = f"{config['lr_scheduler']}_{learning_rate}_{loss_type}_{ce_reduction}_{focal_reduction}_{alpha}_{gamma}"
    #                             _, _, micro_f1s = Text_Train(config, model_title)
    #                             output = {'learning_rate': learning_rate, 'lr_scheduler': lr_scheduler,
    #                                       'loss_type': loss_type, 'ce_reduction':ce_reduction,'alpha':alpha,'gamma':gamma,
    #                                       'focal_reduction':focal_reduction,'micro_f1s': micro_f1s}
    #                             outputs.append(output)
    #
    #                 else:
    #                     for ce_reduction in ['mean']:
    #                         config['ce_reduction'] = ce_reduction
    #                         model_title = f"{config['lr_scheduler']}_{learning_rate}_{loss_type}_{ce_reduction}_NA_NA_NA"
    #                         _, _, micro_f1s = Text_Train(config, model_title)
    #                         output = {'learning_rate': learning_rate, 'lr_scheduler': lr_scheduler,
    #                                       'loss_type': loss_type, 'ce_reduction':ce_reduction,'alpha':'NA','gamma':'NA',
    #                                       'focal_reduction':'NA','micro_f1s': micro_f1s}
    #                         outputs.append(output)
    outputs = []
    for output_block in ['BiLSTM', 'BiLSTM+Transformer']:
        config['output_block'] = output_block
        for input_text_save in ['user+', 'user-customer-', 'user+customer-']:
            config['input_text_save'] = input_text_save
            for batch_size in [3, 6, 12, 24, 36]:
                config['batch_size'] = batch_size
                for loss_type in ['focal', 'ce']:
                    config['loss_type'] = loss_type

                    if loss_type == 'focal':
                        for ce_reduction in ['none', 'mean', 'sum'] :
                            config['ce_reduction'] = ce_reduction
                            for focal_reduction in ['mean', 'sum']:
                                config['focal_reduction'] = focal_reduction

                                model_title = f"{config['output_block']}_{config['input_text_save']}_{loss_type}_{ce_reduction}_{focal_reduction}_{batch_size}"
                                _, _, micro_f1s = Text_Train(config, model_title)
                                output = {'model': output_block, 'input_text_save': input_text_save,
                                          'loss_type': loss_type, 'batch_size':batch_size,
                                          'ce_reduction':ce_reduction,'focal_reduction':focal_reduction,'micro_f1s': micro_f1s}
                                outputs.append(output)

                    else:
                        for ce_reduction in ['mean', 'sum']:
                            config['ce_reduction'] = ce_reduction
                            model_title = f"{config['output_block']}_{config['input_text_save']}_{loss_type}_{ce_reduction}_{'NA'}_{batch_size}"
                            _, _, micro_f1s = Text_Train(config, model_title)
                            output = {'model': output_block, 'input_text_save': input_text_save,
                                      'loss_type': loss_type, 'batch_size': batch_size,
                                      'ce_reduction': ce_reduction, 'focal_reduction': 'NA',
                                      'micro_f1s': micro_f1s}
                            outputs.append(output)

    output_data = pd.DataFrame(outputs)
    save_path = "log/table_block&textsave&loss_v2.csv"
    output_data.to_csv(save_path, index=False)

if __name__ == '__main__':
    grid_search()
    # Text_Train(config,save=False)