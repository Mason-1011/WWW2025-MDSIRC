# WWW2025 多模态对话系统意图识别挑战赛

## 自定义工具类：
### `Viewer`类：
`Viewer` 类是一个用于读取和展示样本数据的 Python 工具类。
它可以从多个 JSON 文件中加载数据，并支持对样本的基本操作，如添加新属性和显示指定属性的值。它还具有显示样本图片的功能，可以自动修正图片路径并展示图像。
此类使用于对模型进行调试时，需要查看个别样本的属性或对应的相关图片。

#### 1. 初始化 Viewer 类
你可以通过传入一个或多个 JSON 文件的路径来初始化 Viewer 类。支持单个路径字符串或路径列表。

```python
from Viewer import Viewer

# 使用单个 JSON 文件路径
viewer = Viewer("path/to/your/json/file.json")

# 或者使用多个 JSON 文件路径
viewer = Viewer(["path/to/your/json/file1.json", "path/to/your/json/file2.json"])
```

#### 2. 加载数据
`Viewer` 会自动加载 JSON 数据并将其合并。每个 JSON 文件中的数据将被解析并加入到内部的数据列表中。

#### 3. 添加新属性
你可以通过 `add_attribute` 方法向指定的样本添加新的属性。

```python
# 向指定样本添加新的属性
updated_product = viewer.add_attribute("2146a76617275885114992755d0b2c", "new_key", "new_value")
```
如果指定的样本 ID 存在，新的属性会被添加到该样本的字典中。


#### 4. 显示样本信息
你可以通过 `display_product_info` 方法，基于样本 ID 查找并显示样本的特定属性。如果提供了 `attributes` 列表，它只会显示这些属性。如果设置了 `show_image=True`，它还会显示对应的样本包含的图片。

```python
# 显示样本的指定属性信息，并展示图片
viewer.display_product_info("2146a76617275885114992755d0b2c", ['id', 'output'], show_image=True)
```
如果没有提供 `attributes` 列表，`Viewer` 会显示该样本的所有属性。

### `InteractiveViewer`类：
`InteractiveViewer` 在`Viewer`的基础上构建了一个基于 `Tkinter` 和 `PIL (Python Imaging Library)` 构建的图形界面工具，用于交互式查看和浏览样本数据。具体功能：
- 查看产品样本的详细信息和图片
- 支持通过输入样本的 ID 来搜索并跳转到特定样本。
- 可以查看当前样本的下一个或上一个样本。

## Note
图像数据集中，并未出现prompt中的两个标签：['消费者与客服聊天页面', '个人信息页面']
