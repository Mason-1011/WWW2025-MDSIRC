from model import TIModel
from loader import load_data
import torch
from config import config
import json
import csv
from tqdm import tqdm  # 引入进度条库
import pandas as pd


def predict(model, data, map):
    model.eval()
    device = next(model.parameters()).device  # 获取模型设备
    print(device)
    map_ = {v:k for k,v in map.items()}
    pred = {}

    # 为数据加载器添加进度条
    for batch_data in tqdm(data, desc="Predicting", unit="batch"):
        ids = batch_data['id']
        with torch.no_grad():
            pred_results = model(batch_data)  # 不输入 labels，预测
        # ids, input_ids, _ = batch_data
        # input_ids = input_ids.to(device)
        # with torch.no_grad():
        #     pred_results = model(input_ids)
        pred_labels = torch.argmax(pred_results, dim=1).cpu().numpy()
        pred = {**pred, **{ids[idx]: map[pred_labels[idx]] for idx in range(len(ids))}}

    return pred

def convert2submit(test_file, pred, save_path):

    test_data = json.load(open(test_file, "r"))
    save_data = []
    for i, example in enumerate(test_data):
        example["predict"] = pred[example["id"]]
        save_data.append(example)

    df = pd.DataFrame(save_data)

    df.to_csv(save_path, index=None, encoding="utf-8-sig")


if __name__ == '__main__':
    model = TIModel(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.load_state_dict(torch.load('./TextModel', map_location=device))
    model.to(device)
    test_dataset = load_data(config=config, path='./test1/test_text.json', task_type='text')
    map = {v: k for k, v in config['label_map'].items()}

    # data = json.load(open('./test1/test1.json', "r"))

    text_pred = predict(model, test_dataset, map)
    image_pred = {sample['id']: sample['predict'] for sample in json.load(open('./result/pred_image.json'))}
    pred = {**text_pred,**image_pred}


    # with open('result/viewer_json.json', "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=4)

    with open('result/submit.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['id', 'predict'])
        # Write rows
        for id_, prediction in pred.items():
            writer.writerow([id_.replace('_0',''), prediction])
