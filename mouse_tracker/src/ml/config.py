"""
配置文件 - 路径和参数设置
"""

# 数据文件路径
KEYBOARD_DATA_PATH = "../../data/raw/keyboard_data/keyboard_performance.csv"
MOUSE_DATA_PATH = "../../data/raw/mouse_data/mouse_performance.csv"
EMOTION_DATA_PATH = "../../data/raw/emotion_responses.csv"

# 特征列名
KEYBOARD_FEATURES = [
    'total_keypresses', 'median_ikd', 'p95_ikd', 'mad',
    'auto_correction_rate', 'space_rate', 'backspace_count', 'space_count'
]

MOUSE_FEATURES = [
    'move_entropy', 'effective_path_ratio', 'avg_speed',
    'acceleration_variance', 'total_distance', 'click_count', 'scroll_count'
]

# 模型参数
MODEL_PARAMS = {
    'lgbm': {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'n_estimators': 200
    },
    'xgb': {
        'objective': 'multi:softmax',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'n_estimators': 200
    }
}

# 情绪标签映射
EMOTION_MAPPING = {
    '专注': 0,
    '无聊': 1,
    '沮丧': 2,
    '困惑': 3
}

REVERSE_EMOTION_MAPPING = {v: k for k, v in EMOTION_MAPPING.items()}