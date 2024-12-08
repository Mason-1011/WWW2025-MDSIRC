# -*- coding: utf-8 -*-
import torch
import numpy as np
from collections import defaultdict, deque
from loader import load_data

"""
模型效果测试
"""


class TextEvaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.map = {v: k for k, v in config['label_map'].items()}

        
        self.valid_data = load_data(path = './train/all_text.json', config=config, task_type='text', shuffle=False)
        self.all_true_labels = []  # 用于存储所有真实标签
        self.all_pred_labels = []  # 用于存储所有预测标签
        self.wrong_ids = []  # 用于存储错误预测的样本 ID

    def load_dict(self):
        self.stats_dict = {v: defaultdict(int) for k, v in self.map.items()}

    def eval(self, epoch):
        self.model.eval()
        self.load_dict()

        device = next(self.model.parameters()).device  # 获取模型设备

        for index, batch_data in enumerate(self.valid_data):
            id, input_ids, output_id = batch_data
            input_ids, output_id = input_ids.to(device), output_id.to(device)

            with torch.no_grad():
                pred_results = self.model(input_ids)  # 不输入 labels，预测
            self.write_stats(id, output_id, pred_results)

        # 计算并返回准确率和 F1 分数
        f1 = self.show_stats()
        return f1

    def write_stats(self, id, labels, pred_results):
        """
        用于比较模型的预测结果与真实标签，并记录正确和错误的预测数量。
        - 使用 torch.argmax() 获取模型预测的分类结果（即最大值对应的分类标签）。
        - 比较预测标签 pred_label 与真实标签 true_label，相等时计入 correct，否则计入 wrong。
        """

        # 将标签和预测结果转为 numpy 格式
        true_labels = labels.cpu().numpy()
        pred_labels = torch.argmax(pred_results, dim=1).cpu().numpy()  # 获取每个样本的预测结果
        # print(pred_labels)
        # print(true_labels)

        # 将 batch 的真实标签和预测标签保存起来
        self.all_true_labels.extend(true_labels)
        self.all_pred_labels.extend(pred_labels)

        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(true_labels, pred_labels):
            # print(true_label,pred_label)
            self.stats_dict[self.map[true_label]]['真实数数'] += 1
            self.stats_dict[self.map[pred_label]]['预测数'] += 1
            if int(true_label) == int(pred_label):
                self.stats_dict[self.map[true_label]]['正确预测数'] += 1
            else:
                self.wrong_ids.append(id)
        return


    def show_stats(self):
        F1_scores = []
        for key in self.stats_dict.keys():
            precision = self.stats_dict[key]["正确预测数"] / (1e-5 + self.stats_dict[key]["预测数"])
            recall = self.stats_dict[key]["正确预测数"] / (1e-5 + self.stats_dict[key]["真实数数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s，真实数量：%i,预测出数量：%i,准确率：%f, 召回率: %f, F1: %f" % (key, self.stats_dict[key]["真实数数"],self.stats_dict[key]["预测数"],precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确预测数"] for key in self.stats_dict.keys()])
        total_pred = sum([self.stats_dict[key]["预测数"] for key in self.stats_dict.keys()])
        true_enti = sum([self.stats_dict[key]["真实数数"] for key in self.stats_dict.keys()])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return micro_f1




def evalate_trained_model(model_path = None):
    from model import TextModel
    from config import config
    import logging

    model = TextModel(config)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    evaluator = TextEvaluator(config, model, logger)
    evaluator.eval(20)


if __name__ == "__main__":
    evalate_trained_model()
