"""
编程情绪监控系统
功能：
1. 每120秒弹出极简情绪量表
2. 收集学生选择的情绪状态
3. 将选择结果保存到CSV文件（含时间戳）
"""

import csv
import time
import threading
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../logs/emotion_monitor.log"),
        logging.StreamHandler()
    ]
)

class EmotionMonitor:
    def __init__(self, interval=120, output_file="emotion_responses.csv", stop_event=None):
        """
        初始化情绪监控器
        :param interval: 弹出间隔（秒）
        :param output_file: 输出文件名
        :param stop_event: 停止事件
        """
        self.interval = interval
        self.output_file = output_file
        self.is_running = False
        self.thread = None
        self.root = None
        self.stop_event = stop_event or threading.Event()

        # 确保输出文件存在
        self.init_output_file()

        logging.info(f"情绪监控器初始化完成，间隔: {interval}秒")

    def init_output_file(self):
        """初始化输出文件"""
        if not os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'emotion', 'description'])
                logging.info(f"创建新的输出文件: {self.output_file}")
            except Exception as e:
                logging.error(f"创建输出文件失败: {str(e)}")

    def save_response(self, emotion, description):
        """保存情绪响应到CSV文件"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, emotion, description])
            logging.info(f"记录情绪: {emotion} - {description}")
        except Exception as e:
            logging.error(f"保存情绪响应失败: {str(e)}")

    def show_emotion_scale(self):
        """显示情绪量表窗口"""
        if not self.is_running or self.stop_event.is_set():
            return

        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("编程情绪微量表")
        self.root.geometry("600x400")
        self.root.attributes("-topmost", True)  # 窗口置顶
        self.root.resizable(False, False)

        # 设置窗口图标（可选）
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        # 创建主框架
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题 - 按照图片内容更新
        title_label = ttk.Label(
            main_frame,
            text="编程情绪微量表",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        question_label = ttk.Label(
            main_frame,
            text="问题：当前最符合你状态的描述是？",  # 按照图片内容
            font=("Arial", 12)
        )
        question_label.pack(pady=(0, 20))

        # 情绪选项 - 按照图片内容更新
        emotions = [
            {
                "letter": "A",
                "name": "专注",
                "description": "流畅编码，完全投入"
            },
            {
                "letter": "B",
                "name": "无聊",
                "description": "简单重复，缺乏挑战"
            },
            {
                "letter": "C",
                "name": "沮丧",
                "description": "反复报错，难以解决"
            },
            {
                "letter": "D",
                "name": "困惑",
                "description": "思路卡壳，不知方向"
            }
        ]

        # 创建选项按钮
        self.selected_emotion = tk.StringVar()

        for emotion in emotions:
            frame = ttk.Frame(main_frame, padding=10)
            frame.pack(fill=tk.X, pady=5, padx=10)

            # 选项文本
            option_text = f"{emotion['letter']}.{emotion['name']}（{emotion['description']}）"

            # 单选按钮
            rb = ttk.Radiobutton(
                frame,
                text=option_text,
                variable=self.selected_emotion,
                value=emotion['name'],
                command=lambda e=emotion: self.on_emotion_selected(e),
                style='Option.TRadiobutton'  # 自定义样式
            )
            rb.pack(anchor=tk.W)

            # 设置字体大小
            style = ttk.Style()
            style.configure('Option.TRadiobutton', font=('Arial', 11))

        # 默认选择第一个选项
        self.selected_emotion.set(emotions[0]["name"])

        # 关闭按钮
        close_btn = ttk.Button(
            main_frame,
            text="关闭窗口（不保存）",
            command=self.root.destroy
        )
        close_btn.pack(pady=20)

        # 运行窗口
        self.root.mainloop()

    def on_emotion_selected(self, emotion):
        """处理情绪选择"""
        # 保存选项字母和完整描述
        option_text = f"{emotion['letter']}.{emotion['name']}（{emotion['description']}）"
        self.save_response(emotion['name'], option_text)
        if self.root:
            self.root.destroy()
        logging.info(f"情绪选择已保存: {emotion['name']}")

    def periodic_prompt(self):
        """定期弹出情绪量表"""
        while self.is_running and not self.stop_event.is_set():
            logging.info("弹出情绪量表...")
            self.show_emotion_scale()

            # 等待下一个周期
            for _ in range(self.interval):
                if not self.is_running or self.stop_event.is_set():
                    break
                time.sleep(1)

    def start(self):
        """启动情绪监控器"""
        if not self.is_running:
            logging.info("启动情绪监控器")
            self.is_running = True
            self.thread = threading.Thread(target=self.periodic_prompt)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        """停止情绪监控器"""
        if self.is_running:
            logging.info("停止情绪监控器")
            self.is_running = False
            if self.root:
                try:
                    self.root.destroy()
                except:
                    pass

    def run(self):
        """运行情绪监控器"""
        self.start()
        try:
            # 保持主线程运行
            while self.is_running and not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logging.error(f"发生错误: {str(e)}")
            self.stop()