"""
键盘行为实时监控与分析系统
功能：
1. 持续监控键盘输入
2. 每2分钟自动分析一次键盘使用情况
3. 将分析结果保存到CSV文件
4. 按ESC键停止整个系统
"""

import csv
import time
import os
import numpy as np
import pandas as pd
import threading
from datetime import datetime, timedelta
from pynput import keyboard
from pynput.keyboard import Key, Listener
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../logs/keyboard_monitor.log"),
        logging.StreamHandler()
    ]
)

class KeyboardMonitor:
    def __init__(self, analysis_interval=120, output_dir="keyboard_data", stop_event=None):
        """
        初始化键盘监控器
        :param analysis_interval: 分析间隔（秒）
        :param output_dir: 数据输出目录
        :param stop_event: 停止事件
        """
        self.analysis_interval = analysis_interval
        self.output_dir = output_dir
        self.events = []  # 存储键盘事件
        self.analysis_results = []  # 存储分析结果
        self.is_listening = False
        self.listener = None
        self.analysis_thread = None
        self.start_time = None
        self.last_analysis_time = None
        self.lock = threading.Lock()
        self.stop_event = stop_event or threading.Event()

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 分析结果文件
        self.analysis_file = os.path.join(self.output_dir, "keyboard_performance.csv")

        # 初始化分析结果文件
        self.init_analysis_file()

    def init_analysis_file(self):
        """初始化分析结果文件"""
        if not os.path.exists(self.analysis_file):
            with open(self.analysis_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'start_time', 'end_time', 'duration_sec', 'total_keypresses',
                    'median_ikd', 'p95_ikd', 'mad', 'auto_correction_rate',
                    'space_rate', 'backspace_count', 'space_count'
                ])

    def on_press(self, key):
        """处理按键按下事件"""
        try:
            key_name = key.char if hasattr(key, 'char') and key.char is not None else str(key)
        except AttributeError:
            key_name = str(key)

        event = {
            'timestamp': datetime.now(),
            'event_type': 'key_down',
            'key': key_name,
            'duration': 0.0
        }

        with self.lock:
            self.events.append(event)

        return True

    def on_release(self, key):
        """处理按键释放事件"""
        try:
            key_name = key.char if hasattr(key, 'char') and key.char is not None else str(key)
        except AttributeError:
            key_name = str(key)

        # 查找对应的按下事件
        press_event = None
        with self.lock:
            for event in reversed(self.events):
                if event['event_type'] == 'key_down' and event['key'] == key_name:
                    press_event = event
                    break

        if press_event:
            duration = (datetime.now() - press_event['timestamp']).total_seconds()
            event = {
                'timestamp': datetime.now(),
                'event_type': 'key_release',
                'key': key_name,
                'duration': round(duration, 3)
            }

            with self.lock:
                self.events.append(event)

        # 停止监听的热键（ESC键）
        if key == Key.esc:
            logging.info("检测到ESC键，停止整个系统...")
            self.stop_event.set()  # 设置停止事件
            self.stop_listener()
            return False

    def start_listener(self):
        """启动键盘监听器"""
        if not self.is_listening:
            logging.info("键盘监听已启动... 按ESC键停止")
            self.is_listening = True
            self.start_time = datetime.now()
            self.last_analysis_time = self.start_time

            # 启动监听线程
            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()

            # 启动分析线程
            self.analysis_thread = threading.Thread(target=self.periodic_analysis)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()

    def stop_listener(self):
        """停止键盘监听器"""
        if self.is_listening:
            logging.info("停止键盘监听...")
            self.is_listening = False

            if self.listener and self.listener.is_alive():
                self.listener.stop()

            # 执行最后一次分析
            self.analyze_period()

    def periodic_analysis(self):
        """定期执行分析"""
        while self.is_listening and not self.stop_event.is_set():
            current_time = datetime.now()
            elapsed = (current_time - self.last_analysis_time).total_seconds()

            if elapsed >= self.analysis_interval:
                self.analyze_period()
                self.last_analysis_time = current_time

            time.sleep(1)

    def analyze_period(self):
        """分析当前时间段的数据"""
        with self.lock:
            if not self.events or self.stop_event.is_set():
                logging.info("没有可分析的数据或系统已停止")
                return

            # 创建当前时间段的数据副本
            events_copy = self.events.copy()

            # 清空事件列表（保留最后几秒的事件以避免丢失）
            # 保留最后5秒的事件以确保连续性
            cutoff_time = datetime.now() - timedelta(seconds=5)
            self.events = [e for e in self.events if e['timestamp'] >= cutoff_time]

        # 计算时间段
        if not events_copy:
            logging.info("没有可分析的事件数据")
            return

        start_time = min(e['timestamp'] for e in events_copy)
        end_time = max(e['timestamp'] for e in events_copy)
        duration = (end_time - start_time).total_seconds()

        # 转换为DataFrame
        df = pd.DataFrame(events_copy)

        # 分离按键事件
        key_down_events = df[df['event_type'] == 'key_down']
        key_release_events = df[df['event_type'] == 'key_release']

        # 1. 按键总次数
        total_keypresses = len(key_down_events) if not key_down_events.empty else 0

        # 2. 键盘延迟指标
        durations = key_release_events['duration'].values if not key_release_events.empty else []
        if len(durations) > 0:
            median_ikd = np.median(durations)
            p95_ikd = np.percentile(durations, 95)
            deviations = np.abs(durations - median_ikd)
            mad = np.median(deviations)
        else:
            median_ikd = 0
            p95_ikd = 0
            mad = 0

        # 3. 自动更正率（退格键使用率）
        backspace_count = 0
        if not key_down_events.empty:
            backspace_events = key_down_events[
                (key_down_events['key'] == 'Key.backspace') |
                (key_down_events['key'] == '\x08')
            ]
            backspace_count = len(backspace_events)

        auto_correction_rate = backspace_count / total_keypresses if total_keypresses > 0 else 0

        # 4. 空格率
        space_count = 0
        if not key_down_events.empty:
            space_events = key_down_events[
                (key_down_events['key'] == ' ') |
                (key_down_events['key'] == 'Key.space')
            ]
            space_count = len(space_events)

        space_rate = space_count / total_keypresses if total_keypresses > 0 else 0

        # 保存分析结果
        result = {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_sec': round(duration, 2),
            'total_keypresses': total_keypresses,
            'median_ikd': round(median_ikd, 4),
            'p95_ikd': round(p95_ikd, 4),
            'mad': round(mad, 4),
            'auto_correction_rate': round(auto_correction_rate, 4),
            'space_rate': round(space_rate, 4),
            'backspace_count': backspace_count,
            'space_count': space_count
        }

        # 保存到内存和文件
        self.analysis_results.append(result)
        self.save_analysis_result(result)

        logging.info(f"分析完成: {start_time} 到 {end_time}")
        logging.info(f"- 按键次数: {total_keypresses}, 时长: {duration:.2f}秒")
        logging.info(f"- IKD中位数: {median_ikd:.4f}s, IKD95%: {p95_ikd:.4f}s")
        logging.info(f"- 空格率: {space_rate:.2%}, 退格率: {auto_correction_rate:.2%}")

    def save_analysis_result(self, result):
        """保存分析结果到CSV文件"""
        try:
            with open(self.analysis_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['start_time'],
                    result['end_time'],
                    result['duration_sec'],
                    result['total_keypresses'],
                    result['median_ikd'],
                    result['p95_ikd'],
                    result['mad'],
                    result['auto_correction_rate'],
                    result['space_rate'],
                    result['backspace_count'],
                    result['space_count']
                ])
            logging.info(f"分析结果已保存到 {self.analysis_file}")
        except Exception as e:
            logging.error(f"保存分析结果失败: {str(e)}")

    def run(self):
        """启动监控系统"""
        logging.info("键盘监控系统启动")
        logging.info(f"分析间隔: {self.analysis_interval}秒")
        logging.info(f"数据目录: {self.output_dir}")
        self.start_listener()

        try:
            # 保持主线程运行
            while self.is_listening and not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("用户中断...")
            self.stop_listener()
        except Exception as e:
            logging.error(f"发生未预期错误: {str(e)}")
            self.stop_listener()

        logging.info("键盘监控系统已停止")