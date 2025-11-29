"""
传统机器学习模型模块
包含多种回归模型用于研磨速度预测
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from typing import Dict, Any, Tuple, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModelManager:
    """机器学习模型管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型管理器

        Args:
            config: 模型配置字典
        """
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model_name = None
        self.best_model = None
        self.feature_names = None
        self.results = {}

    def get_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """
        根据名称和参数创建模型

        Args:
            model_name: 模型名称
            params: 模型参数

        Returns:
            模型实例
        """
        model_map = {
            'RandomForest': RandomForestRegressor,
            'XGBoost': xgb.XGBRegressor,
            'LightGBM': lgb.LGBMRegressor,
            'GradientBoosting': GradientBoostingRegressor,
            'SVR': SVR
        }

        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}")

        return model_map[model_name](**params)

    def train_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        训练多个模型并比较性能

        Args:
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            训练结果字典
        """
        logger.info("Starting model training...")

        # 保存特征名称
        self.feature_names = X.columns.tolist()

        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 获取要训练的模型列表
        algorithms = self.config.get('algorithms', ['RandomForest', 'XGBoost'])
        hyperparameters = self.config.get('hyperparameters', {})

        # 训练每个模型
        best_r2 = -float('inf')

        for model_name in algorithms:
            logger.info(f"Training {model_name}...")

            # 获取模型参数
            params = hyperparameters.get(model_name, {})

            # 创建并训练模型
            model = self.get_model(model_name, params)

            # 某些模型需要原始数据
            if model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
            else:  # SVR等需要标准化
                model.fit(X_train_scaled, y_train)
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)

            # 计算评估指标
            train_metrics = self._calculate_metrics(y_train, y_pred_train)
            test_metrics = self._calculate_metrics(y_test, y_pred_test)

            # 交叉验证
            if model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')

            # 保存结果
            self.results[model_name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            self.models[model_name] = model

            logger.info(f"{model_name} - Test R²: {test_metrics['r2']:.4f}, CV R²: {cv_scores.mean():.4f}")

            # 更新最佳模型
            if test_metrics['r2'] > best_r2:
                best_r2 = test_metrics['r2']
                self.best_model_name = model_name
                self.best_model = model

        logger.info(f"Best model: {self.best_model_name} with R²: {best_r2:.4f}")

        return {
            'results': self.results,
            'best_model': self.best_model_name,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """计算评估指标"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

    def predict(self, X: pd.DataFrame, model_name: Optional[str] = None) -> np.ndarray:
        """
        使用指定模型进行预测

        Args:
            X: 特征数据
            model_name: 模型名称，如果为None则使用最佳模型

        Returns:
            预测结果
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)

        if model is None:
            raise ValueError(f"Model {model_name} not found. Train models first.")

        # 某些模型需要标准化
        if model_name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
            return model.predict(X)
        else:
            X_scaled = self.scaler.transform(X)
            return model.predict(X_scaled)

    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        获取特征重要性

        Args:
            model_name: 模型名称

        Returns:
            特征重要性DataFrame
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)

        if model is None:
            raise ValueError(f"Model {model_name} not found.")

        # 检查模型是否有feature_importances_属性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            return None

        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return feature_importance_df

    def save_models(self, save_dir: str):
        """
        保存所有模型

        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model, save_path)
            logger.info(f"Saved {model_name} to {save_path}")

        # 保存scaler
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)

        # 保存元数据
        metadata = {
            'best_model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'results': {k: {
                'train_metrics': v['train_metrics'],
                'test_metrics': v['test_metrics'],
                'cv_mean': v['cv_mean'],
                'cv_std': v['cv_std']
            } for k, v in self.results.items()}
        }
        metadata_path = os.path.join(save_dir, "metadata.pkl")
        joblib.dump(metadata, metadata_path)

        logger.info(f"All models saved to {save_dir}")

    def load_models(self, load_dir: str):
        """
        加载模型

        Args:
            load_dir: 模型目录
        """
        # 加载元数据
        metadata_path = os.path.join(load_dir, "metadata.pkl")
        metadata = joblib.load(metadata_path)

        self.best_model_name = metadata['best_model_name']
        self.feature_names = metadata['feature_names']

        # 加载scaler
        scaler_path = os.path.join(load_dir, "scaler.pkl")
        self.scaler = joblib.load(scaler_path)

        # 加载所有模型
        for model_file in os.listdir(load_dir):
            if model_file.endswith('.pkl') and model_file not in ['scaler.pkl', 'metadata.pkl']:
                model_name = model_file.replace('.pkl', '')
                model_path = os.path.join(load_dir, model_file)
                self.models[model_name] = joblib.load(model_path)

        self.best_model = self.models[self.best_model_name]
        logger.info(f"Loaded models from {load_dir}")
