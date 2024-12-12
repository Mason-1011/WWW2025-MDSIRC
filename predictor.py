from model import TextModel
from loader import load_data
import torch
from config import config
import json
import csv
from tqdm import tqdm  # 引入进度条库


def predict(model, data, map):
    model.eval()
    device = next(model.parameters()).device  # 获取模型设备
    print(device)
    pred = {}

    # 为数据加载器添加进度条
    for batch_data in tqdm(data, desc="Predicting", unit="batch"):
        ids, input_ids, _ = batch_data
        input_ids = input_ids.to(device)
        with torch.no_grad():
            pred_results = model(input_ids)
        pred_labels = torch.argmax(pred_results, dim=1).cpu().numpy()
        pred = {**pred, **{ids[idx]: map[pred_labels[idx]] for idx in range(len(ids))}}

    return pred

if __name__ == '__main__':
    model = TextModel(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('./TextModel', map_location=device))
    test_dataset = load_data(config=config, path='./test1/test_text.json', task_type='text')
    map = {v: k for k, v in config['label_map'].items()}

    data = test_dataset.dataset.json_data

    pred = predict(model, test_dataset, map)

    # 为 JSON 文件添加预测结果
    for sample in data:
        if sample['id'] in pred:
            sample['predict'] = pred[sample['id']]

    with open('result/viewer_json.json', "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    with open('submit.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['id', 'predict'])
        # Write rows
        for id_, prediction in pred.items():
            writer.writerow([id_, prediction])
