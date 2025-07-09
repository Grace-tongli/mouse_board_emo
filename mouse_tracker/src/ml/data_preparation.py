"""
数据准备和特征工程
功能：
1. 加载键盘、鼠标和情绪数据
2. 按时间窗口对齐数据
3. 特征工程
4. 创建训练数据集
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import *


def load_and_preprocess_data():
    """
    加载并预处理数据
    """
    # 加载键盘数据
    kb_data = pd.read_csv(KEYBOARD_DATA_PATH)
    kb_data['start_time'] = pd.to_datetime(kb_data['start_time'])
    kb_data['end_time'] = pd.to_datetime(kb_data['end_time'])

    # 加载鼠标数据
    mouse_data = pd.read_csv(MOUSE_DATA_PATH)
    mouse_data['start_time'] = pd.to_datetime(mouse_data['start_time'])
    mouse_data['end_time'] = pd.to_datetime(mouse_data['end_time'])

    # 加载情绪数据
    emotion_data = pd.read_csv(EMOTION_DATA_PATH)
    emotion_data['timestamp'] = pd.to_datetime(emotion_data['timestamp'])

    return kb_data, mouse_data, emotion_data


def align_data_with_emotion(kb_data, mouse_data, emotion_data, time_window=30):
    """
    按时间窗口对齐键盘、鼠标和情绪数据
    :param time_window: 时间窗口（秒），用于匹配情绪报告时间
    """
    # 创建空的结果DataFrame
    aligned_data = pd.DataFrame()

    # 遍历每个情绪报告
    for idx, row in emotion_data.iterrows():
        emotion_time = row['timestamp']
        emotion = row['emotion']

        # 找到匹配的键盘数据
        kb_match = kb_data[
            (kb_data['end_time'] >= emotion_time - timedelta(seconds=time_window)) &
            (kb_data['end_time'] <= emotion_time + timedelta(seconds=time_window))
            ]

        # 找到匹配的鼠标数据
        mouse_match = mouse_data[
            (mouse_data['end_time'] >= emotion_time - timedelta(seconds=time_window)) &
            (mouse_data['end_time'] <= emotion_time + timedelta(seconds=time_window))
            ]

        # 如果找到匹配的数据
        if not kb_match.empty and not mouse_match.empty:
            # 取时间上最接近的样本
            kb_sample = kb_match.iloc[0]
            mouse_sample = mouse_match.iloc[0]

            # 创建合并行
            combined_row = {
                'timestamp': emotion_time,
                'emotion': emotion,
                'kb_start_time': kb_sample['start_time'],
                'kb_end_time': kb_sample['end_time'],
                'mouse_start_time': mouse_sample['start_time'],
                'mouse_end_time': mouse_sample['end_time']
            }

            # 添加键盘特征
            for feature in KEYBOARD_FEATURES:
                combined_row[f'kb_{feature}'] = kb_sample[feature]

            # 添加鼠标特征
            for feature in MOUSE_FEATURES:
                combined_row[f'mouse_{feature}'] = mouse_sample[feature]

            # 添加到结果DataFrame
            aligned_data = pd.concat([aligned_data, pd.DataFrame([combined_row])], ignore_index=True)

    return aligned_data


def create_features(data):
    """
    特征工程
    """
    # 创建组合特征
    data['total_actions'] = data['kb_total_keypresses'] + data['mouse_click_count'] + data['mouse_scroll_count']
    data['interaction_intensity'] = data['kb_total_keypresses'] / data['total_actions']

    # 创建效率特征
    data['efficiency_ratio'] = data['mouse_effective_path_ratio'] / (data['kb_auto_correction_rate'] + 0.001)

    # 创建速度特征
    data['speed_variability'] = data['mouse_acceleration_variance'] * data['kb_mad']

    # 创建情绪状态特征
    data['frustration_level'] = data['kb_backspace_count'] * data['kb_auto_correction_rate']

    # 时间特征
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek

    return data


def prepare_dataset():
    """
    准备训练数据集
    """
    # 加载数据
    kb_data, mouse_data, emotion_data = load_and_preprocess_data()

    # 对齐数据
    aligned_data = align_data_with_emotion(kb_data, mouse_data, emotion_data)

    # 特征工程
    dataset = create_features(aligned_data)

    # 映射情绪标签
    dataset['emotion_label'] = dataset['emotion'].map(EMOTION_MAPPING)

    # 选择特征列
    feature_columns = [col for col in dataset.columns if
                       col.startswith('kb_') or col.startswith('mouse_') or col in ['total_actions',
                                                                                    'interaction_intensity',
                                                                                    'efficiency_ratio',
                                                                                    'speed_variability',
                                                                                    'frustration_level', 'hour',
                                                                                    'day_of_week']]

    return dataset[feature_columns], dataset['emotion_label']


if __name__ == "__main__":
    features, labels = prepare_dataset()
    print("数据集准备完成")
    print(f"特征数量: {len(features.columns)}")
    print(f"样本数量: {len(features)}")
    print("情绪标签分布:")
    print(labels.value_counts())

    # 保存数据集
    features.to_csv("training_features.csv", index=False)
    labels.to_csv("training_labels.csv", index=False)