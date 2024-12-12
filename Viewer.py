import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import messagebox

class Viewer:
    def __init__(self):
        """
        初始化 Viewer 类，加载多个 JSON 数据。
        """

        self.data = None
        self.current_index = 0  # 默认显示第一个产品
        self.interactive_mode = False  # 默认非交互模式

    def load_data_from_path(self, json_file_paths):
        """
        读取并解析多个 JSON 文件，返回合并后的数据。

        :param json_file_paths: 包含多个 JSON 文件路径的列表
        :return: 合并后的 JSON 数据
        """
        if isinstance(json_file_paths, str):
            json_file_paths = [json_file_paths]

        all_data = []
        for file_path in json_file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    # print(type(data))
                    # 获取 JSON 文件的目录路径
                    base_dir = os.path.dirname(file_path)

                    # 修正每个 item 中的图片路径
                    for item in data:
                        if 'image' in item and item['image']:
                            # 假设 image 字段是一个列表，修正列表中的每个路径
                            item['image'] = [self.fix_image_path(base_dir, img) for img in item['image']]

                    all_data.extend(data)  # 合并数据
            except FileNotFoundError:
                print(f"文件未找到：{file_path}")
            except json.JSONDecodeError:
                print(f"文件格式错误：{file_path}")
        self.data = all_data

    def load_data_from_json(self, json_data,images_dir='train/images'):
        """
        读取并解析 JSON 数据。
        :param json_data:
        :return:
        """
        if not isinstance(json_data, list):
            raise ValueError("json_data 应该是一个json列表")
        else:
            for item in json_data:
                if 'image' in item and item['image']:
                    # 假设 image 字段是一个列表，修正列表中的每个路径
                    item['image'] = [os.path.join(images_dir, img) for img in item['image']]
        self.data = json_data

    def fix_image_path(self, base_dir, img_path):
        """
        修正图片路径，确保图片路径是完整的。

        :param base_dir: JSON 文件所在的目录路径
        :param img_path: 图片的相对路径
        :return: 完整的图片路径
        """
        # 拼接完整的图片路径，假设图片存储在 `/images` 目录下
        return os.path.join(base_dir, 'images', img_path)

    def add_attribute(self, id, key, value):
        """
        为指定产品 ID 添加新的属性。

        :param id: 要添加属性的产品 ID
        :param key: 新属性的键
        :param value: 新属性的值
        :return: 如果找到了对应 ID 的产品，则返回更新后的产品信息，否则返回 None
        """
        # 查找对应的产品数据
        sample = next((item for item in self.data if item['id'] == id), None)

        if sample:
            # 为找到的产品添加新的属性
            sample[key] = value
            return sample
        else:
            print(f"未找到 ID 为 {id} 的产品信息，无法添加属性。")
            return None

    def display_sample_info(self, sample_id, attributes=None, show_image=False):
        """
        根据产品的 ID 查找并显示产品信息及图片，只显示指定的属性。

        :param sample_id: 要查找的产品 ID
        :param attributes: 需要打印的属性列表
        :param show_image: 是否显示图片
        :return: 返回产品的相关信息
        """
        # 查找对应的产品数据
        sample = next((item for item in self.data if item['id'] == sample_id), None)

        if sample:
            sample_info = {}
            if attributes is None:
                attributes = sample.keys()  # 如果没有传入属性列表，则打印所有属性

            # 打印指定的属性
            for attribute in attributes:
                if attribute in sample:
                    if not self.interactive_mode:
                        print(f"{attribute}: {sample[attribute]}")
                    sample_info[attribute] = sample[attribute]

            # 显示图片
            if show_image and 'image' in sample and sample['image']:
                images = []
                for img_path in sample['image']:
                    try:
                        # 读取并显示图片
                        img = mpimg.imread(img_path)  # 使用matplotlib读取图片
                        if not self.interactive_mode:
                            plt.imshow(img)
                            plt.axis('off')
                            plt.show()
                        images.append(img)
                    except FileNotFoundError:
                        print(f"无法找到图片文件：{img_path}")

                return sample_info, images

        return None, None


class InteractiveViewer(tk.Tk):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.viewer.interactive_mode = True
        self.title("Sample Viewer")
        self.geometry("800x800")  # 增大窗口大小以容纳所有控件

        self.current_sample = None

        # 创建并配置框架来管理布局
        self.frame = tk.Frame(self)
        self.frame.pack(fill="both", expand=True)

        # Create a top frame for buttons and search box
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(fill="x", pady=5)

        # UI elements (Buttons and Search)
        self.id_entry_label = tk.Label(self.top_frame, text="Enter Sample ID:")
        self.id_entry_label.pack(side="left", padx=5)

        self.id_entry = tk.Entry(self.top_frame, bd=2, relief="solid")
        self.id_entry.pack(side="left", padx=5)

        self.search_button = tk.Button(self.top_frame, text="Search by ID", command=self.search_by_id, bd=2, relief="solid")
        self.search_button.pack(side="left", padx=5)

        self.last_button = tk.Button(self.top_frame, text="Last Sample", command=self.show_last_sample, bd=2, relief="solid")
        self.last_button.pack(side="left", padx=5)

        self.next_button = tk.Button(self.top_frame, text="Next Sample", command=self.show_next_sample, bd=2, relief="solid")
        self.next_button.pack(side="left", padx=5)

        # Create frame for displaying sample information
        self.info_label = tk.Label(self.frame, text="Sample Information", font=("Helvetica", 14))
        self.info_label.grid(row=0, column=0, pady=10, padx=10)

        self.text_area = tk.Text(self.frame, height=25, width=80)
        self.text_area.grid(row=1, column=0, pady=10, padx=10)

        self.image_frame = tk.Frame(self.frame)  # New frame for images
        self.image_frame.grid(row=2, column=0, pady=10, padx=10)

        self.show_next_sample()  # Initialize showing the first sample

    def display_sample(self, sample_id):
        sample_info, images = self.viewer.display_sample_info(sample_id, show_image=True)

        if sample_info:
            self.text_area.delete(1.0, tk.END)
            for key, value in sample_info.items():
                self.text_area.insert(tk.END, f"{key}: {value}\n")

            # Clear existing images
            for widget in self.image_frame.winfo_children():
                widget.destroy()

            # Display all images if available
            if images:
                for img in images:
                    # Ensure the image is in the correct format
                    if img.dtype == np.float32:
                        img = (img * 255).astype(np.uint8)  # Convert float32 to uint8

                    # Check the shape of the image to ensure it's a valid image
                    if img.ndim == 3 and img.shape[2] == 3:  # RGB image
                        img_pil = Image.fromarray(img)  # Convert to PIL Image
                        img_pil.thumbnail((250, 250))  # Resize images to fit horizontally
                        photo = ImageTk.PhotoImage(img_pil)

                        img_label = tk.Label(self.image_frame, image=photo)
                        img_label.image = photo  # Keep a reference to the image to prevent garbage collection
                        img_label.pack(side="left", padx=10)  # Pack images side by side

        else:
            messagebox.showinfo("Error", "Sample not found")

    def show_next_sample(self):
        if self.viewer.current_index >= len(self.viewer.data):
            self.viewer.current_index = 0  # Reset to the first sample
        sample_id = self.viewer.data[self.viewer.current_index]['id']
        self.display_sample(sample_id)
        self.viewer.current_index += 1

    def show_last_sample(self):
        if self.viewer.current_index <= 0:
            self.viewer.current_index = len(self.viewer.data) - 1  # Wrap around to the last sample
        else:
            self.viewer.current_index -= 1  # Show the previous sample
        sample_id = self.viewer.data[self.viewer.current_index]['id']
        self.display_sample(sample_id)

    def search_by_id(self):
        sample_id = self.id_entry.get()
        if sample_id:
            # Find the sample index by ID
            sample_index = next((i for i, item in enumerate(self.viewer.data) if item['id'] == sample_id), None)

            if sample_index is not None:
                # Update current_index to the index of the searched sample
                self.viewer.current_index = sample_index + 1  # Next sample after search
                self.display_sample(sample_id)
            else:
                messagebox.showinfo("Error", "Sample not found")
        else:
            messagebox.showinfo("Error", "Please enter a valid sample ID.")


if __name__ == "__main__":
    # 使用示例
    # 初始化 Viewer 类并传入多个 JSON 文件路径
    viewer = Viewer() # 请替换为你的 JSON 文件路径列表
    # viewer.load_data_from_path(["train/train.json", "test1/test1.json"])
    # 调用方法显示某个产品的信息和图片
    # viewer.display_sample_info("9b24d96f-2961-41ba-8113-e82f1522869f-12275",['id','output'],show_image=True)
    with open('result/viewer_json.json', 'r') as f:
        data = json.load(f)
    viewer.load_data_from_json(data, images_dir='test1/images')

    # 启动交互式窗口
    app = InteractiveViewer(viewer)
    app.mainloop()
