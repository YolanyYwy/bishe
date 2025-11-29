"""
数据处理模块
包含数据加载、预处理、特征工程等功能
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None

    def load_data(self, file_path, target_column: str = None) -> pd.DataFrame:
        """
        加载数据，支持单个或多个文件

        Args:
            file_path: 数据文件路径（支持csv, xlsx, xls）或文件路径列表
            target_column: 目标列名称（如果为None，默认使用最后一列）

        Returns:
            数据DataFrame
        """
        # 支持单个文件或多个文件
        if isinstance(file_path, (list, tuple)):
            logger.info(f"Loading {len(file_path)} data files...")
            dfs = []
            for i, path in enumerate(file_path, 1):
                logger.info(f"Loading file {i}/{len(file_path)}: {path}")
                df_temp = self._load_single_file(path)
                dfs.append(df_temp)

            # 合并所有数据
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined data shape: {df.shape}")
        else:
            logger.info(f"Loading data from {file_path}")
            df = self._load_single_file(file_path)

        logger.info(f"Final data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # 确定目标列
        if target_column is None:
            self.target_name = df.columns[-1]
            logger.info(f"Using last column as target: {self.target_name}")
        else:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            self.target_name = target_column

        return df

    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            数据DataFrame
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        return df

    def preprocess_data(
        self,
        df: pd.DataFrame,
        handle_missing: str = 'mean',
        remove_outliers: bool = False,
        outlier_std: float = 3.0
    ) -> pd.DataFrame:
        """
        数据预处理

        Args:
            df: 输入数据
            handle_missing: 缺失值处理方式 ('mean', 'median', 'drop')
            remove_outliers: 是否移除异常值
            outlier_std: 异常值判定标准（标准差倍数）

        Returns:
            处理后的数据
        """
        df = df.copy()
        logger.info("Preprocessing data...")

        # 处理缺失值
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values")

            if handle_missing == 'drop':
                df = df.dropna()
                logger.info(f"Dropped missing values. New shape: {df.shape}")
            elif handle_missing == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                logger.info("Filled missing values with mean")
            elif handle_missing == 'median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                logger.info("Filled missing values with median")

        # 移除异常值
        if remove_outliers:
            original_shape = df.shape
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - outlier_std * std
                upper_bound = mean + outlier_std * std
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            logger.info(f"Removed outliers. Shape: {original_shape} -> {df.shape}")

        return df

    def split_features_target(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        分离特征和目标变量

        Args:
            df: 输入数据
            target_column: 目标列名（如果为None，使用初始化时设定的列）

        Returns:
            (X, y) 特征和目标
        """
        if target_column is None:
            target_column = self.target_name

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        self.feature_names = X.columns.tolist()

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")

        return X, y

    def engineer_features(
        self,
        X: pd.DataFrame,
        polynomial: bool = False,
        interactions: bool = True,
        log_transform: List[str] = None
    ) -> pd.DataFrame:
        """
        特征工程

        Args:
            X: 特征数据
            polynomial: 是否添加多项式特征
            interactions: 是否添加交互特征
            log_transform: 需要对数变换的列名列表

        Returns:
            工程化后的特征
        """
        X = X.copy()
        logger.info("Engineering features...")

        # 对数变换
        if log_transform:
            for col in log_transform:
                if col in X.columns:
                    X[f'{col}_log'] = np.log1p(X[col])
                    logger.info(f"Added log transform for {col}")

        # 交互特征（简单的两两相乘）
        if interactions and X.shape[1] <= 10:  # 只在特征数量不太多时添加
            numeric_cols = X.select_dtypes(include=[np.number]).columns[:5]  # 限制前5个特征
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
            logger.info(f"Added interaction features. New shape: {X.shape}")

        # 多项式特征
        if polynomial:
            from sklearn.preprocessing import PolynomialFeatures
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X[numeric_cols])
            poly_feature_names = poly.get_feature_names_out(numeric_cols)
            X = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
            logger.info(f"Added polynomial features. New shape: {X.shape}")

        return X

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        获取数据摘要统计

        Args:
            df: 数据DataFrame

        Returns:
            摘要统计字典
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'statistics': df.describe().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }

        return summary

    def detect_data_issues(self, df: pd.DataFrame) -> dict:
        """
        检测数据问题

        Args:
            df: 数据DataFrame

        Returns:
            问题字典
        """
        issues = {
            'missing_values': {},
            'outliers': {},
            'outlier_details': {},  # 新增：详细的异常值信息
            'duplicates': 0,
            'constant_columns': [],
            'high_cardinality': []
        }

        # 缺失值
        missing = df.isnull().sum()
        issues['missing_values'] = {col: int(count) for col, count in missing.items() if count > 0}

        # 重复行
        issues['duplicates'] = df.duplicated().sum()

        # 常量列
        for col in df.columns:
            if df[col].nunique() == 1:
                issues['constant_columns'].append(col)

        # 高基数列（对于分类变量）
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.5:
                issues['high_cardinality'].append(col)

        # 异常值检测（数值列） - 改进版本
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 找出异常值的索引
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                issues['outliers'][col] = int(outlier_count)

                # 详细异常值分析
                outlier_indices = df[outlier_mask].index.tolist()
                outlier_values = df.loc[outlier_mask, col].tolist()

                # 区分低于下界和高于上界的异常值
                low_outliers = df[df[col] < lower_bound][col]
                high_outliers = df[df[col] > upper_bound][col]

                issues['outlier_details'][col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_count / len(df) * 100, 2),
                    'Q1': round(Q1, 4),
                    'Q3': round(Q3, 4),
                    'IQR': round(IQR, 4),
                    'lower_bound': round(lower_bound, 4),
                    'upper_bound': round(upper_bound, 4),
                    'mean': round(df[col].mean(), 4),
                    'std': round(df[col].std(), 4),
                    'low_outliers': {
                        'count': len(low_outliers),
                        'min_value': round(low_outliers.min(), 4) if len(low_outliers) > 0 else None,
                        'max_value': round(low_outliers.max(), 4) if len(low_outliers) > 0 else None,
                        'reason': f'值低于下界 {round(lower_bound, 4)}'
                    },
                    'high_outliers': {
                        'count': len(high_outliers),
                        'min_value': round(high_outliers.min(), 4) if len(high_outliers) > 0 else None,
                        'max_value': round(high_outliers.max(), 4) if len(high_outliers) > 0 else None,
                        'reason': f'值高于上界 {round(upper_bound, 4)}'
                    },
                    'sample_indices': outlier_indices[:10],  # 只显示前10个异常值的索引
                    'sample_values': [round(v, 4) for v in outlier_values[:10]]
                }

        return issues
