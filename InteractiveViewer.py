from Viewer import Viewer
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import messagebox

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
        sample_info, images = self.viewer.display_sample_info(sample_id, attributes=['id', 'instruction', 'input', 'output'], show_image=True)

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
    # 初始化 Viewer 类并传入多个 JSON 文件路径
    viewer = Viewer()# 请替换为你的 JSON 文件路径列表
    viewer.load_data_from_path(["train/train.json", "test1/test1.json"])
    # 启动交互式窗口
    app = InteractiveViewer(viewer)
    app.mainloop()
