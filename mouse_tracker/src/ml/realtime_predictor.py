"""
实时情绪预测
功能：
1. 实时收集键盘和鼠标数据
2. 每2分钟生成特征向量
3. 使用训练好的模型预测情绪
4. 显示预测结果
"""

import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime
from src.ml.data_preparation import create_features
from src.ml.config import KEYBOARD_FEATURES, MOUSE_FEATURES, REVERSE_EMOTION_MAPPING


class RealTimePredictor:
    def __init__(self, model_path='emotion_predictor_lgbm.pkl'):
        """
        初始化实时预测器
        :param model_path: 模型文件路径
        """
        # 加载模型
        self.model = joblib.load(model_path)
        self.model_type = 'lgbm' if 'lgbm' in model_path else 'xgb'

        # 数据缓冲区
        self.keyboard_buffer = []
        self.mouse_buffer = []

        # 当前特征向量
        self.current_features = None

        # 预测结果
        self.current_prediction = None
        self.prediction_history = []

        # 时间窗口
        self.window_size = 120  # 2分钟
        self.last_window_end = datetime.now()

    def add_keyboard_data(self, data):
        """添加键盘数据"""
        self.keyboard_buffer.append(data)

    def add_mouse_data(self, data):
        """添加鼠标数据"""
        self.mouse_buffer.append(data)

    def generate_features(self):
        """生成特征向量"""
        # 检查是否有足够的数据
        if not self.keyboard_buffer or not self.mouse_buffer:
            return None

        # 获取当前时间窗口的数据
        current_time = datetime.now()

        # 检查是否达到时间窗口
        if (current_time - self.last_window_end).total_seconds() < self.window_size:
            return None

        # 更新窗口结束时间
        self.last_window_end = current_time

        # 创建键盘特征
        kb_features = {}
        for feature in KEYBOARD_FEATURES:
            values = [d.get(feature, 0) for d in self.keyboard_buffer]
            kb_features[f'kb_{feature}'] = np.mean(values) if values else 0

        # 创建鼠标特征
        mouse_features = {}
        for feature in MOUSE_FEATURES:
            values = [d.get(feature, 0) for d in self.mouse_buffer]
            mouse_features[f'mouse_{feature}'] = np.mean(values) if values else 0

        # 合并特征
        combined_features = {**kb_features, **mouse_features}
        combined_features['timestamp'] = current_time

        # 转换为DataFrame
        features_df = pd.DataFrame([combined_features])

        # 特征工程
        features_df = create_features(features_df)

        # 移除时间戳列
        if 'timestamp' in features_df.columns:
            features_df = features_df.drop('timestamp', axis=1)

        # 清空缓冲区
        self.keyboard_buffer = []
        self.mouse_buffer = []

        return features_df

    def predict_emotion(self):
        """预测情绪"""
        features_df = self.generate_features()
        if features_df is None or features_df.empty:
            return None

        # 保存当前特征向量
        self.current_features = features_df

        # 预测情绪
        if self.model_type == 'lgbm':
            pred_proba = self.model.predict(features_df.values)
            pred_class = np.argmax(pred_proba, axis=1)[0]
        else:
            pred_class = self.model.predict(features_df)[0]

        # 获取情绪标签
        emotion = REVERSE_EMOTION_MAPPING.get(pred_class, "未知")

        # 保存预测结果
        prediction = {
            'timestamp': datetime.now(),
            'emotion': emotion,
            'features': features_df.iloc[0].to_dict()
        }
        self.prediction_history.append(prediction)
        self.current_prediction = prediction

        return emotion

    def run(self):
        """运行实时预测"""
        print("实时情绪预测已启动...")
        try:
            while True:
                emotion = self.predict_emotion()
                if emotion:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 预测情绪: {emotion}")

                # 每秒检查一次
                time.sleep(1)
        except KeyboardInterrupt:
            print("实时预测已停止")
        except Exception as e:
            print(f"发生错误: {str(e)}")

    def save_prediction_history(self, filename="emotion_predictions.csv"):
        """保存预测历史"""
        if not self.prediction_history:
            print("没有预测历史可保存")
            return

        # 转换为DataFrame
        history_df = pd.DataFrame(self.prediction_history)

        # 展开特征列
        features_df = pd.json_normalize(history_df['features'])
        history_df = pd.concat([history_df.drop(['features', 'timestamp'], axis=1), features_df], axis=1)

        # 保存到CSV
        history_df.to_csv(filename, index=False)
        print(f"预测历史已保存到 {filename}")


if __name__ == "__main__":
    # 创建预测器
    predictor = RealTimePredictor()

    # 启动实时预测
    predictor.run()

    # 保存预测历史
    predictor.save_prediction_history()