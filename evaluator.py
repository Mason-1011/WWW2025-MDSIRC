# -*- coding: utf-8 -*-
import torch
import numpy as np
from collections import defaultdict, deque
from loader import load_data
import os
import random
from qwen_vl_utils import process_vision_info
"""
模型效果测试
"""


class TextEvaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.map = {v: k for k, v in config['label_map'].items()}

        self.valid_data = load_data(path='./train/valid_text.json', config=config, task_type='text', shuffle=False, augment=False)
        self.all_true_labels = []  # 用于存储所有真实标签
        self.all_pred_labels = []  # 用于存储所有预测标签
        self.wrong_ids = []  # 用于存储错误预测的样本 ID

    def load_dict(self):
        self.stats_dict = {v: defaultdict(int) for k, v in self.map.items()}
        self.result = {}

    def eval(self, epoch):
        self.model.eval()
        self.load_dict()

        device = next(self.model.parameters()).device  # 获取模型设备

        for index, batch_data in enumerate(self.valid_data):
            ids = batch_data['id']
            labels = [self.config["label_map"][i] for i in batch_data["label"]]
            output_ids = torch.tensor(labels).to(device)
            with torch.no_grad():
                pred_results = self.model(batch_data)  # 不输入 labels，预测
            self.write_stats(ids, output_ids, pred_results)
        # for index, batch_data in enumerate(self.valid_data):
        #     id, input_ids, output_id = batch_data
        #     input_ids, output_id = input_ids.to(device), output_id.to(device)
        #
        #     with torch.no_grad():
        #         pred_results = self.model(input_ids)  # 不输入 labels，预测
        #     self.write_stats(id, output_id, pred_results)

        for item in self.valid_data.dataset.json_data:
            item_id = item['id']
            if item_id in self.result:
                item['predicted'] = self.result[item_id]

        # 计算并返回准确率和 F1 分数
        f1 = self.show_stats()
        return f1

    def write_stats(self, ids, labels, pred_results):
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
        for id, true_label, pred_label in zip(ids, true_labels, pred_labels):
            # print(true_label,pred_label)
            self.stats_dict[self.map[true_label]]['真实数数'] += 1
            self.stats_dict[self.map[pred_label]]['预测数'] += 1
            self.result[id] = self.map[pred_label]
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
            self.logger.info("%s，真实数量：%i,预测出数量：%i,准确率：%f, 召回率: %f, F1: %f" % (
            key, self.stats_dict[key]["真实数数"], self.stats_dict[key]["预测数"], precision, recall, F1))
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


def evalate_trained_model(model_path=None):
    from model import TextModel
    from config import config
    import logging
    from Viewer import Viewer, InteractiveViewer

    model = TextModel(config)
    model.load_state_dict(torch.load('./TextModel',map_location=torch.device('cpu')))

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    evaluator = TextEvaluator(config, model, logger)
    evaluator.eval(0)

    viewer = Viewer()
    viewer.load_data_from_json(evaluator.valid_data.dataset.json_data)
    print(evaluator.wrong_ids)
    app = InteractiveViewer(viewer)
    app.mainloop()


class ImageEvaluator:
    def __init__(self, config, model, tokenizer = None, logger = None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.device = self.model.encoder.device
        self.res = []

    def eval_QWEN_VL(self, data_iter):
        step = 0
        for sample in data_iter:
            image_id = sample["image_id"][0]
            query = self.tokenizer.from_list_format([
                {'image': os.path.join("train/images", image_id)},
                {'text': self.config["image_task_prompt"]},
                # {'text': "请用中文简略描述图中的内容:"},
            ])
            # inputs = tokenizer(query, return_tensors='pt')
            # inputs = inputs.to(device)
            # pred = model.generate(**inputs)
            # response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
            response, history = self.model.chat(self.tokenizer, query=query, history=None)
            response_label = self.find_first_substring(response)

            res = {"image_id": image_id, "response": response, "TorF": response_label == sample["label"][0]}
            self.res.append(res)
            if step > 0 and step % 20 == 0:
                print("Current Acc: ", len([1 for i in self.res if i["TorF"]]) / len(self.res))
            step += 1

    def eval_QWEN2_VL(self, data_iter):
        step = 0
        for sample in data_iter:
            image_id = sample["image_id"][0]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": os.path.join("train/images", image_id),
                        },
                        {"type": "text", "text": self.config["image_task_prompt"]},
                    ],
                }
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.tokenizer(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=32)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response_label = self.find_first_substring(output_text[0])

            res = {"image_id": image_id, "response": output_text, "response_label": response_label, "true_label": sample["label"][0]}
            print(output_text, response_label, sample["label"][0])
            self.res.append(res)
            if step > 0 and step % 20 == 0:
                self.logger.info(f"Current Acc: {len([1 for i,j in enumerate(self.res) if j['response_label'] == j['true_label']]) / len(self.res)}")
            step += 1
            if step == 400:
                break

        self.logger.info("各类别准确率:")
        for label in self.config["image_labels"]:
            count_total = 0
            count_true = 0
            for res in self.res:
                if res["true_label"] == label:
                    count_total += 1
                    if res["response_label"] == label:
                        count_true += 1
            self.logger.info(f"{label}有{count_total}个，预测准确率：{count_true/count_total*100}%")

    def eval_Qwen2_VL_classify_block(self, data_iter):
        self.model.eval()
        total_loss = 0
        # 初始化统计变量
        TP = [0] * len(self.config["image_labels"])  # True Positives
        FP = [0] * len(self.config["image_labels"])  # False Positives
        FN = [0] * len(self.config["image_labels"])  # False Negatives
        total_each_label = [0] * len(self.config["image_labels"])
        correct_each_label = [0] * len(self.config["image_labels"])

        num_batches = 0
        for sample in data_iter:
            labels = [self.config["image_label_map"][i] for i in sample["label"]]
            output_id = torch.tensor(labels).to(self.device)
            loss, logits = self.model(sample, output_id)
            total_loss += loss.item()
            num_batches += 1

            _, predicted_labels = torch.max(logits, dim=1)

            # 统计
            for label in labels:
                total_each_label[label] += 1

            for pred, true in zip(predicted_labels.cpu().numpy(), labels):  # 将张量移到CPU并转换为numpy数组以便索引
                if pred == true:
                    correct_each_label[true] += 1

            for true, pred in zip(labels, predicted_labels.cpu().numpy()):
                if pred == true:
                    TP[pred] += 1
                else:
                    FP[pred] += 1
                    FN[true] += 1

        # 损失
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        self.logger.info(f"评估平均损失: {avg_loss}")

        # 计算加权平均F1分数
        weighted_f1 = 0.0
        total_samples = sum(TP) + sum(FN)  # 实际正例总数
        for i, (tp, fp, fn) in enumerate(zip(TP, FP, FN)):
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            weighted_f1 += f1 * (tp + fn)  # 加权F1分数部分

        weighted_f1 = (weighted_f1 / total_samples) * 100 if total_samples > 0 else 0
        self.logger.info(f"评估加权平均F1分数: {weighted_f1:.2f}%")

        # 准确率
        self.logger.info(f"评估总体准确率: {sum(correct_each_label) / sum(total_each_label)*100:.2f}%")
        for i, (correct, total) in enumerate(zip(correct_each_label, total_each_label)):
            accuracy = (correct / total) * 100 if total != 0 else 0
            self.logger.info(f"{self.config['image_labels'][i]} 有{total}个，预测准确率: {accuracy:.2f}%")

        self.model.train()

    def find_first_substring(self, main_string):
        substrings = self.config["image_labels"]
        for substring in substrings:
            if substring in main_string:
                return substring
        return random.choice(substrings)


if __name__ == "__main__":
    evalate_trained_model()
