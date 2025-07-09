"""
模型训练和评估
功能：
1. 加载预处理的数据
2. 训练LightGBM和XGBoost模型
3. 模型评估
4. 模型保存
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.ml.config import REVERSE_EMOTION_MAPPING

# 加载数据
features = pd.read_csv("training_features.csv")
labels = pd.read_csv("training_labels.csv")['emotion_label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)


def train_lightgbm_model():
    """训练LightGBM模型"""
    print("训练LightGBM模型...")

    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 训练模型
    model = lgb.train(
        MODEL_PARAMS['lgbm'],
        train_data,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=True),
            lgb.log_evaluation(period=10)
        ]
    )

    # 评估模型
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)

    print("\nLightGBM模型评估:")
    evaluate_model(y_test, y_pred_class)

    # 保存模型
    joblib.dump(model, "emotion_predictor_lgbm.pkl")
    print("LightGBM模型已保存")

    return model


def train_xgboost_model():
    """训练XGBoost模型"""
    print("\n训练XGBoost模型...")

    # 训练模型
    model = xgb.XGBClassifier(**MODEL_PARAMS['xgb'])
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=20,
        verbose=10
    )

    # 评估模型
    y_pred = model.predict(X_test)

    print("\nXGBoost模型评估:")
    evaluate_model(y_test, y_pred)

    # 保存模型
    joblib.dump(model, "emotion_predictor_xgb.pkl")
    print("XGBoost模型已保存")

    return model


def evaluate_model(y_true, y_pred):
    """评估模型性能"""
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"准确率: {accuracy:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(
        y_true, y_pred,
        target_names=[REVERSE_EMOTION_MAPPING[i] for i in range(4)]
    ))

    # 混淆矩阵
    print("混淆矩阵:")
    conf_mat = confusion_matrix(y_true, y_pred)
    conf_mat_df = pd.DataFrame(
        conf_mat,
        index=[REVERSE_EMOTION_MAPPING[i] for i in range(4)],
        columns=[REVERSE_EMOTION_MAPPING[i] for i in range(4)]
    )
    print(conf_mat_df)


def cross_validate_models():
    """交叉验证模型性能"""
    print("\n交叉验证模型性能...")

    # 创建交叉验证对象
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # LightGBM交叉验证
    lgbm_scores = []
    for train_idx, val_idx in skf.split(features, labels):
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

        model = lgb.LGBMClassifier(**MODEL_PARAMS['lgbm'])
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        lgbm_scores.append(score)

    print(f"LightGBM平均准确率: {np.mean(lgbm_scores):.4f} (±{np.std(lgbm_scores):.4f})")

    # XGBoost交叉验证
    xgb_scores = []
    for train_idx, val_idx in skf.split(features, labels):
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]

        model = xgb.XGBClassifier(**MODEL_PARAMS['xgb'])
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        xgb_scores.append(score)

    print(f"XGBoost平均准确率: {np.mean(xgb_scores):.4f} (±{np.std(xgb_scores):.4f})")


def feature_importance_analysis(model, model_type='lgbm'):
    """分析特征重要性"""
    print(f"\n{model_type.upper()}模型特征重要性:")

    if model_type == 'lgbm':
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importance()
        })
    else:  # xgboost
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })

    importance = importance.sort_values('importance', ascending=False)
    print(importance.head(15))

    # 可视化特征重要性
    importance.head(15).set_index('feature').plot(kind='barh', title=f'{model_type.upper()}特征重要性')
    plt.tight_layout()
    plt.savefig(f"{model_type}_feature_importance.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # 训练和评估模型
    lgbm_model = train_lightgbm_model()
    xgb_model = train_xgboost_model()

    # 交叉验证
    cross_validate_models()

    # 特征重要性分析
    feature_importance_analysis(lgbm_model, 'lgbm')
    feature_importance_analysis(xgb_model, 'xgb')

    print("\n模型训练和评估完成!")