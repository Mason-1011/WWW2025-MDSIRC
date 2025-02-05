config = {
    "model_path": "./Qwen2-VL-7B-Instruct",
    "model_path_image": "/root/autodl-tmp/WWW2025-MDSIRC/Qwen2-VL-2B-Instruct",

    # loader config：
    "train_text_path": "./train/train_text.json",
    "valid_text_path": "./train/valid_text.json",
    "train_image_path": "./train/train_image.json",
    "valid_image_path": "./train/valid_image.json",
    "batch_size": 3,
    "batch_size_image": 2,
    "input_text_save": 'user+customer-',  # 'user-' or 'user+' or 'user-customer-' or 'user+customer-'
    "max_length_map": {'user-': 36,
                       'user+': 128,
                       'user-customer-': 200,
                       'user+customer-': 200},

    # model config:
    "output_block": 'BiLSTM',  # 'BiLSTM' or 'Transformer' or 'TransformerEncoder' or 'BiLSTM+Transformer' or 'BiLSTM+TransformerEncoder'
    "pooling_mode": 'max',  # 'mean' or 'max' or 'cls' or 'concat'

    # Image model config
    "hidden_size_image": 1536,

    # Loss config：
    "alpha": 0.7,
    "gamma": 1,
    "ce_reduction": 'mean',  # 'mean' or 'sum' or 'none'
    "focal_reduction": 'mean',  # 'mean' or 'sum' or 'none'
    "loss_type": 'ce',  # 'ce' or 'focal'

    # Train config：
    "epochs": 100,
    "lr_scheduler": True,
    "patience": 10,
    "learning_rate": 5e-5,
    "optimizer": 'adam',

    "label_map": {'商品材质': 0,
                  '商品规格': 1,
                  '排水方式': 2,
                  '控制方式': 3,
                  '是否易褪色': 4,
                  '上市时间': 5,
                  '是否会生锈': 6,
                  '反馈用后症状': 7,
                  '适用季节': 8,
                  '何时上货': 9,
                  '发货数量': 10,
                  '功效功能': 11,
                  '套装推荐': 12,
                  '是否好用': 13,
                  '用法用量': 14,
                  '包装区别': 15,
                  '能否调光': 16,
                  '单品推荐': 17,
                  '版本款型区别': 18,
                  '反馈密封性不好': 19,
                  '气泡': 20,
                  '养护方法': 21,
                  '信号情况': 22},
    "image_task_prompt": "图片属于下面其中的哪个类别："
                     "\"实物拍摄(含售后)\",\"商品分类选项\",\"商品头图\",\"商品详情页截图\",\"下单过程中出现异常（显示购买失败浮窗）\","
                     "\"订单详情页面\",\"支付页面\",\"评论区截图页面\",\"物流页面-物流列表页面\",\"物流页面-物流跟踪页面\",\"物流页面-物流异常页面\","
                     "\"退款页面\",\"退货页面\",\"换货页面\",\"购物车页面\",\"店铺页面\",\"活动页面\",\"优惠券领取页面\",\"账单/账户页面\",\"投诉举报页面\","
                     "\"平台介入页面\",\"外部APP截图\",\"其他类别图片\"",
    "image_labels": ["实物拍摄(含售后)","商品分类选项","商品头图","商品详情页截图","下单过程中出现异常（显示购买失败浮窗）",
                     "订单详情页面","支付页面","评论区截图页面","物流页面-物流列表页面","物流页面-物流跟踪页面","物流页面-物流异常页面",
                     "退款页面","退货页面","换货页面","购物车页面","店铺页面","活动页面","优惠券领取页面","账单/账户页面", "投诉举报页面",
                     "平台介入页面","外部APP截图","其他类别图片"],
    "image_label_map" :{
                "实物拍摄(含售后)": 0,
                "商品分类选项": 1,
                "商品头图": 2,
                "商品详情页截图": 3,
                "下单过程中出现异常（显示购买失败浮窗）": 4,
                "订单详情页面": 5,
                "支付页面": 6,
                "评论区截图页面": 7,
                "物流页面-物流列表页面": 8,
                "物流页面-物流跟踪页面": 9,
                "物流页面-物流异常页面": 10,
                "退款页面": 11,
                "退货页面": 12,
                "换货页面": 13,
                "购物车页面": 14,
                "店铺页面": 15,
                "活动页面": 16,
                "优惠券领取页面": 17,
                "账单/账户页面": 18,
                "投诉举报页面": 19,
                "平台介入页面": 20,
                "外部APP截图": 21,
                "其他类别图片": 22
    }
}
