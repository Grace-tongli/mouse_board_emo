"""
键盘行为数据分析工具
计算以下指标：
1. 键盘延迟中位数（IKD）
2. 95百分数IKD
3. 平均绝对偏差
4. 自动更正率
5. 空格率
6. 按键总次数
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class KeyboardAnalyzer:
    def __init__(self, file_path='keyboard_data.csv'):
        """
        初始化键盘分析器
        :param file_path: 键盘数据文件路径
        """
        self.file_path = file_path
        self.data = None
        self.key_down_events = None
        self.key_release_events = None
        self.durations = None
        self.metrics = {
            'total_keypresses': 0,
            'median_ikd': 0,
            'p95_ikd': 0,
            'mad': 0,
            'auto_correction_rate': 0,
            'space_rate': 0,
            'backspace_count': 0,
            'space_count': 0,
            'total_duration': 0,
            'first_event_time': None,
            'last_event_time': None
        }

    def load_data(self):
        """
        从CSV文件加载键盘数据
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(self.file_path)

            # 确保数据包含所需列
            required_columns = ['timestamp', 'event_type', 'key', 'duration']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"CSV文件缺少必需的列: {', '.join(missing)}")

            # 转换时间戳为datetime对象
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

            # 按时间排序
            df = df.sort_values(by='timestamp')

            # 分离按键按下和释放事件
            self.key_down_events = df[df['event_type'] == 'key_down']
            self.key_release_events = df[df['event_type'] == 'key_release']

            # 提取按键持续时间（只包括释放事件）
            self.durations = self.key_release_events['duration'].values

            # 计算会话总时长
            if not df.empty:
                self.metrics['first_event_time'] = df['timestamp'].min()
                self.metrics['last_event_time'] = df['timestamp'].max()
                self.metrics['total_duration'] = (self.metrics['last_event_time'] -
                                                 self.metrics['first_event_time']).total_seconds()

            self.data = df
            return True
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            return False

    def calculate_metrics(self):
        """
        计算所有键盘行为指标
        """
        if self.data is None:
            if not self.load_data():
                return False

        # 1. 按键总次数（按下事件）
        self.metrics['total_keypresses'] = len(self.key_down_events)

        # 2. 键盘延迟中位数（IKD） - 按键持续时间的中位数
        if self.durations is not None and len(self.durations) > 0:
            self.metrics['median_ikd'] = np.median(self.durations)

            # 3. 95百分数IKD
            self.metrics['p95_ikd'] = np.percentile(self.durations, 95)

            # 4. 平均绝对偏差
            deviations = np.abs(self.durations - self.metrics['median_ikd'])
            self.metrics['mad'] = np.median(deviations)
        else:
            self.metrics['median_ikd'] = 0
            self.metrics['p95_ikd'] = 0
            self.metrics['mad'] = 0

        # 5. 自动更正率（使用退格键次数近似）
        self.metrics['backspace_count'] = len(self.key_down_events[
            (self.key_down_events['key'] == 'Key.backspace') |
            (self.key_down_events['key'] == '\x08')  # 某些系统上的退格键表示
        ])

        if self.metrics['total_keypresses'] > 0:
            self.metrics['auto_correction_rate'] = (
                self.metrics['backspace_count'] / self.metrics['total_keypresses']
            )
        else:
            self.metrics['auto_correction_rate'] = 0

        # 6. 空格率
        self.metrics['space_count'] = len(self.key_down_events[
            (self.key_down_events['key'] == ' ') |
            (self.key_down_events['key'] == 'Key.space')
        ])

        if self.metrics['total_keypresses'] > 0:
            self.metrics['space_rate'] = (
                self.metrics['space_count'] / self.metrics['total_keypresses']
            )
        else:
            self.metrics['space_rate'] = 0

        return True

    def generate_report(self):
        """
        生成分析报告
        """
        if not self.metrics['total_keypresses']:
            return "没有可用的键盘数据进行分析"

        report = (
            f"键盘行为分析报告\n"
            f"数据文件: {self.file_path}\n"
            f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"\n"
            f"基本统计:\n"
            f"- 按键总次数: {self.metrics['total_keypresses']}\n"
            f"- 会话时长: {self.metrics['total_duration']:.2f} 秒\n"
            f"- 开始时间: {self.metrics['first_event_time']}\n"
            f"- 结束时间: {self.metrics['last_event_time']}\n"
            f"\n"
            f"键盘延迟指标:\n"
            f"- 中位数IKD: {self.metrics['median_ikd']:.4f} 秒\n"
            f"- 95百分数IKD: {self.metrics['p95_ikd']:.4f} 秒\n"
            f"- 平均绝对偏差: {self.metrics['mad']:.4f} 秒\n"
            f"\n"
            f"输入习惯指标:\n"
            f"- 自动更正率: {self.metrics['auto_correction_rate']:.4f} "
            f"({self.metrics['backspace_count']} / {self.metrics['total_keypresses']})\n"
            f"- 空格率: {self.metrics['space_rate']:.4f} "
            f"({self.metrics['space_count']} / {self.metrics['total_keypresses']})\n"
        )

        return report

    def plot_duration_distribution(self, save_path=None):
        """
        绘制按键持续时间分布图
        :param save_path: 图片保存路径（可选）
        """
        # 修复错误：正确检查持续时间数组是否为空
        if self.durations is None or len(self.durations) == 0:
            print("没有可用的持续时间数据")
            return

        plt.figure(figsize=(12, 6))

        # 过滤掉异常值（大于2秒的持续时间）
        filtered_durations = [d for d in self.durations if d <= 2.0]

        if len(filtered_durations) == 0:
            print("没有有效的持续时间数据可用于绘图")
            return

        # 绘制直方图
        plt.hist(filtered_durations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

        # 添加参考线
        plt.axvline(self.metrics['median_ikd'], color='red', linestyle='dashed', linewidth=1.5,
                   label=f'中位数IKD: {self.metrics["median_ikd"]:.4f}s')
        plt.axvline(self.metrics['p95_ikd'], color='green', linestyle='dashed', linewidth=1.5,
                   label=f'95百分数IKD: {self.metrics["p95_ikd"]:.4f}s')

        plt.title('按键持续时间分布')
        plt.xlabel('持续时间 (秒)')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()

    def plot_top_keys(self, top_n=15, save_path=None):
        """
        绘制最常按下的按键
        :param top_n: 显示前N个按键
        :param save_path: 图片保存路径（可选）
        """
        if self.key_down_events is None or self.key_down_events.empty:
            print("没有可用的按键数据")
            return

        # 统计按键频率
        key_counts = self.key_down_events['key'].value_counts().head(top_n)

        if key_counts.empty:
            print("没有足够的按键数据用于绘图")
            return

        plt.figure(figsize=(12, 8))

        # 绘制水平条形图
        key_counts.sort_values(ascending=True).plot(kind='barh', color='lightcoral')

        plt.title(f'最常用的 {top_n} 个按键')
        plt.xlabel('按下次数')
        plt.ylabel('按键')
        plt.grid(True, linestyle='--', alpha=0.7)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()

    def save_report(self, output_file='keyboard_analysis_report.txt'):
        """
        将分析报告保存到文件
        :param output_file: 输出文件名
        """
        report = self.generate_report()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"分析报告已保存至: {output_file}")

if __name__ == "__main__":
    # 创建分析器实例
    analyzer = KeyboardAnalyzer()

    # 计算指标
    if analyzer.calculate_metrics():
        # 打印报告
        print(analyzer.generate_report())

        # 保存报告
        analyzer.save_report()

        # 绘制图表
        analyzer.plot_duration_distribution('key_duration_distribution.png')
        analyzer.plot_top_keys(save_path='top_keys_usage.png')
    else:
        print("分析失败，请检查数据文件")