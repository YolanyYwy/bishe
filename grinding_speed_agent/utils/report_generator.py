"""
报告生成模块
生成Markdown格式的预测分析报告
"""
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器

        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置中文字体（避免中文乱码）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_report(
        self,
        data_summary: Dict[str, Any],
        model_results: Dict[str, Any],
        predictions: Optional[pd.DataFrame] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        data_issues: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        生成完整的分析报告

        Args:
            data_summary: 数据摘要
            model_results: 模型训练结果
            predictions: 预测结果
            feature_importance: 特征重要性
            data_issues: 数据问题

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"grinding_speed_prediction_report_{timestamp}.md"
        report_path = os.path.join(self.output_dir, report_name)

        logger.info(f"Generating report: {report_path}")

        # 创建报告内容
        report_content = self._build_report(
            data_summary,
            model_results,
            predictions,
            feature_importance,
            data_issues,
            timestamp
        )

        # 写入文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Report generated successfully: {report_path}")
        return report_path

    def _build_report(
        self,
        data_summary: Dict[str, Any],
        model_results: Dict[str, Any],
        predictions: Optional[pd.DataFrame],
        feature_importance: Optional[pd.DataFrame],
        data_issues: Optional[Dict[str, Any]],
        timestamp: str
    ) -> str:
        """构建报告内容"""

        report = f"""# 研磨速度预测分析报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. 执行摘要

本报告由AI Agent自动生成，基于提供的数据进行研磨速度预测建模分析。

### 关键发现

- **最佳模型**: {model_results.get('best_model', 'N/A')}
- **模型性能**: R² = {self._get_best_r2(model_results):.4f}
- **数据规模**: {data_summary.get('shape', (0, 0))[0]} 条记录，{data_summary.get('shape', (0, 0))[1]} 个特征

---

## 2. 数据分析

### 2.1 数据概览

- **数据维度**: {data_summary['shape'][0]} 行 × {data_summary['shape'][1]} 列
- **特征列表**: {', '.join(data_summary.get('columns', []))}
- **内存占用**: {data_summary.get('memory_usage', 0):.2f} MB

### 2.2 数据质量

{self._format_data_issues(data_issues) if data_issues else '数据质量良好，未发现明显问题。'}

### 2.3 描述性统计

```
{self._format_statistics(data_summary.get('statistics', {}))}
```

---

## 3. 模型训练与评估

### 3.1 模型对比

{self._format_model_comparison(model_results)}

### 3.2 最佳模型详情

**模型名称**: {model_results.get('best_model', 'N/A')}

{self._format_best_model_metrics(model_results)}

### 3.3 交叉验证结果

{self._format_cv_results(model_results)}

---

## 4. 特征重要性分析

{self._format_feature_importance(feature_importance) if feature_importance is not None else '特征重要性分析不适用于当前模型。'}

---

## 5. 预测结果

{self._format_predictions(predictions) if predictions is not None else '未进行新数据预测。'}

---

## 6. 结论与建议

### 6.1 模型性能总结

{self._generate_performance_summary(model_results)}

### 6.2 使用建议

1. **模型部署**: 建议使用 {model_results.get('best_model', 'N/A')} 模型进行生产环境部署
2. **数据要求**: 确保输入数据包含以下特征：{', '.join(data_summary.get('columns', [])[:-1])}
3. **模型更新**: 建议定期使用新数据重新训练模型，保持预测准确性
4. **异常监控**: 实时监控预测值，对异常情况进行告警

### 6.3 改进方向

{self._generate_improvement_suggestions(model_results, data_summary)}

---

## 附录

### A. 技术栈

- **编程语言**: Python 3.8+
- **机器学习框架**: scikit-learn, XGBoost, LightGBM
- **数据处理**: pandas, numpy
- **模型管理**: joblib

### B. 模型文件

训练好的模型已保存至 `models_saved/` 目录，可直接加载使用：

```python
import joblib
model = joblib.load('models_saved/{model_results.get("best_model", "model")}.pkl')
```

---

*报告由 Grinding Speed Prediction Agent 自动生成*
*基于轻量级大模型 + 传统ML模型的智能预测系统*
"""

        return report

    def _get_best_r2(self, model_results: Dict[str, Any]) -> float:
        """获取最佳模型的R²分数"""
        best_model = model_results.get('best_model')
        if best_model and 'results' in model_results:
            results = model_results['results']
            if best_model in results:
                return results[best_model]['test_metrics']['r2']
        return 0.0

    def _format_data_issues(self, issues: Dict[str, Any]) -> str:
        """格式化数据问题"""
        sections = []

        if issues.get('missing_values'):
            sections.append("**缺失值**:")
            for col, count in issues['missing_values'].items():
                sections.append(f"- {col}: {count} 个缺失值")

        if issues.get('duplicates', 0) > 0:
            sections.append(f"\n**重复记录**: {issues['duplicates']} 条")

        # 详细的异常值分析
        if issues.get('outlier_details'):
            sections.append("\n**异常值详细分析** (基于IQR方法):\n")
            for col, details in issues['outlier_details'].items():
                sections.append(f"#### 特征: `{col}`\n")
                sections.append(f"- **异常值数量**: {details['count']} 个 ({details['percentage']}%)")
                sections.append(f"- **数据统计**: 均值={details['mean']}, 标准差={details['std']}")
                sections.append(f"- **四分位数**: Q1={details['Q1']}, Q3={details['Q3']}, IQR={details['IQR']}")
                sections.append(f"- **异常阈值**: [{details['lower_bound']}, {details['upper_bound']}]")

                # 低于下界的异常值
                if details['low_outliers']['count'] > 0:
                    sections.append(f"\n  **过低的异常值** ({details['low_outliers']['count']} 个):")
                    sections.append(f"  - 原因: {details['low_outliers']['reason']}")
                    sections.append(f"  - 值范围: [{details['low_outliers']['min_value']}, {details['low_outliers']['max_value']}]")

                # 高于上界的异常值
                if details['high_outliers']['count'] > 0:
                    sections.append(f"\n  **过高的异常值** ({details['high_outliers']['count']} 个):")
                    sections.append(f"  - 原因: {details['high_outliers']['reason']}")
                    sections.append(f"  - 值范围: [{details['high_outliers']['min_value']}, {details['high_outliers']['max_value']}]")

                # 样本索引
                if details['sample_indices']:
                    sections.append(f"\n  **异常值位置** (样本行号): {details['sample_indices'][:10]}")
                    sections.append(f"  **异常值样本**: {details['sample_values'][:10]}\n")

        if issues.get('constant_columns'):
            sections.append(f"\n**常量列**: {', '.join(issues['constant_columns'])}")

        return '\n'.join(sections) if sections else "数据质量良好。"

    def _format_statistics(self, stats: Dict) -> str:
        """格式化统计信息"""
        if not stats:
            return "统计信息不可用"

        df_stats = pd.DataFrame(stats)
        return df_stats.to_string()

    def _format_model_comparison(self, model_results: Dict[str, Any]) -> str:
        """格式化模型对比"""
        if 'results' not in model_results:
            return "模型对比信息不可用"

        comparison = "| 模型 | Train R² | Test R² | RMSE | MAE | CV R² (mean±std) |\n"
        comparison += "|------|----------|---------|------|-----|------------------|\n"

        for model_name, result in model_results['results'].items():
            train_r2 = result['train_metrics']['r2']
            test_r2 = result['test_metrics']['r2']
            rmse = result['test_metrics']['rmse']
            mae = result['test_metrics']['mae']
            cv_mean = result['cv_mean']
            cv_std = result['cv_std']

            comparison += f"| {model_name} | {train_r2:.4f} | {test_r2:.4f} | {rmse:.4f} | {mae:.4f} | {cv_mean:.4f}±{cv_std:.4f} |\n"

        return comparison

    def _format_best_model_metrics(self, model_results: Dict[str, Any]) -> str:
        """格式化最佳模型指标"""
        best_model = model_results.get('best_model')
        if not best_model or 'results' not in model_results:
            return "指标信息不可用"

        metrics = model_results['results'][best_model]['test_metrics']

        return f"""
**测试集性能**:
- R² Score: {metrics['r2']:.4f}
- RMSE: {metrics['rmse']:.4f}
- MAE: {metrics['mae']:.4f}
- MSE: {metrics['mse']:.4f}
"""

    def _format_cv_results(self, model_results: Dict[str, Any]) -> str:
        """格式化交叉验证结果"""
        best_model = model_results.get('best_model')
        if not best_model or 'results' not in model_results:
            return "交叉验证结果不可用"

        cv_mean = model_results['results'][best_model]['cv_mean']
        cv_std = model_results['results'][best_model]['cv_std']

        return f"""
5折交叉验证结果:
- 平均 R²: {cv_mean:.4f}
- 标准差: {cv_std:.4f}
- 稳定性: {'良好' if cv_std < 0.05 else '一般' if cv_std < 0.1 else '较差'}
"""

    def _format_feature_importance(self, feature_importance: pd.DataFrame) -> str:
        """格式化特征重要性"""
        if feature_importance is None or feature_importance.empty:
            return "特征重要性不可用"

        top_10 = feature_importance.head(10)

        table = "| 排名 | 特征 | 重要性 |\n|------|------|--------|\n"
        for idx, row in enumerate(top_10.itertuples(), 1):
            table += f"| {idx} | {row.feature} | {row.importance:.4f} |\n"

        return f"""
### Top 10 重要特征

{table}

**解读**: 特征重要性反映了各特征对预测结果的影响程度，重要性越高的特征对模型预测的贡献越大。
"""

    def _format_predictions(self, predictions: pd.DataFrame) -> str:
        """格式化预测结果"""
        if predictions is None or predictions.empty:
            return "无预测结果"

        # 查找预测列
        pred_col = None
        true_col = None

        for col in predictions.columns:
            if '预测' in col or 'prediction' in col.lower():
                pred_col = col
            elif '真实' in col or 'actual' in col.lower() or '实际' in col:
                true_col = col

        # 如果没找到明确的预测列，使用最后一列
        if pred_col is None:
            pred_col = predictions.columns[-1]

        sample = predictions.head(15)
        result = f"""
### 预测样例 (前15条)

```
{sample.to_string()}
```

**预测统计**:
- 预测数量: {len(predictions)}
- 平均预测值: {predictions[pred_col].mean():.4f}
- 预测标准差: {predictions[pred_col].std():.4f}
- 预测范围: [{predictions[pred_col].min():.4f}, {predictions[pred_col].max():.4f}]
"""

        # 如果有真实值，添加对比分析
        if true_col is not None:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            import numpy as np

            y_true = predictions[true_col]
            y_pred = predictions[pred_col]

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # 计算误差
            errors = y_pred - y_true
            abs_errors = np.abs(errors)

            # 找出最好和最差的预测
            best_indices = abs_errors.nsmallest(5).index.tolist()
            worst_indices = abs_errors.nlargest(5).index.tolist()

            result += f"""

---

### 预测性能分析 (与真实值对比)

**整体性能指标**:
- **R² Score**: {r2:.4f} ({'优秀' if r2 >= 0.9 else '良好' if r2 >= 0.7 else '中等' if r2 >= 0.5 else '较弱'})
- **RMSE**: {rmse:.4f}
- **MAE**: {mae:.4f}
- **MSE**: {mse:.4f}

**误差分析**:
- 平均误差: {errors.mean():.4f}
- 误差标准差: {errors.std():.4f}
- 最大正误差 (高估): {errors.max():.4f}
- 最大负误差 (低估): {errors.min():.4f}
- 平均绝对误差: {abs_errors.mean():.4f}

**预测精度分布**:
- 误差 < 5%的比例: {(abs_errors / y_true * 100 < 5).sum() / len(predictions) * 100:.2f}%
- 误差 < 10%的比例: {(abs_errors / y_true * 100 < 10).sum() / len(predictions) * 100:.2f}%
- 误差 < 20%的比例: {(abs_errors / y_true * 100 < 20).sum() / len(predictions) * 100:.2f}%

### 最佳预测样例 (Top 5)

预测最准确的5个样本：

| 索引 | 真实值 | 预测值 | 绝对误差 | 相对误差(%) |
|------|--------|--------|----------|-------------|
"""
            for idx in best_indices:
                true_val = y_true[idx]
                pred_val = y_pred[idx]
                abs_err = abs_errors[idx]
                rel_err = abs_err / true_val * 100
                result += f"| {idx} | {true_val:.4f} | {pred_val:.4f} | {abs_err:.4f} | {rel_err:.2f}% |\n"

            result += f"""

### 需要改进的预测 (Top 5)

预测误差最大的5个样本：

| 索引 | 真实值 | 预测值 | 绝对误差 | 相对误差(%) |
|------|--------|--------|----------|-------------|
"""
            for idx in worst_indices:
                true_val = y_true[idx]
                pred_val = y_pred[idx]
                abs_err = abs_errors[idx]
                rel_err = abs_err / true_val * 100
                result += f"| {idx} | {true_val:.4f} | {pred_val:.4f} | {abs_err:.4f} | {rel_err:.2f}% |\n"

            result += f"""

**结论**:
- 模型在 {(abs_errors / y_true * 100 < 10).sum()} 个样本({(abs_errors / y_true * 100 < 10).sum() / len(predictions) * 100:.1f}%)上达到了10%以内的预测精度
- 需要特别关注误差较大的样本，分析其特征是否有异常
"""

        return result

    def _generate_performance_summary(self, model_results: Dict[str, Any]) -> str:
        """生成性能总结"""
        best_r2 = self._get_best_r2(model_results)

        if best_r2 >= 0.9:
            level = "优秀"
            desc = "模型具有很强的预测能力，可以可靠地用于生产环境。"
        elif best_r2 >= 0.7:
            level = "良好"
            desc = "模型性能良好，可以用于实际预测，但建议持续监控和优化。"
        elif best_r2 >= 0.5:
            level = "中等"
            desc = "模型具有一定的预测能力，建议进一步优化特征工程和模型参数。"
        else:
            level = "较弱"
            desc = "模型预测能力有限，建议重新审视数据质量和特征选择。"

        return f"模型性能等级：**{level}** (R² = {best_r2:.4f})\n\n{desc}"

    def _generate_improvement_suggestions(
        self,
        model_results: Dict[str, Any],
        data_summary: Dict[str, Any]
    ) -> str:
        """生成改进建议"""
        suggestions = []

        best_r2 = self._get_best_r2(model_results)

        if best_r2 < 0.8:
            suggestions.append("- 考虑收集更多高质量数据")
            suggestions.append("- 尝试更复杂的特征工程（交互特征、多项式特征）")

        if data_summary.get('shape', (0, 0))[0] < 1000:
            suggestions.append("- 增加训练数据量以提升模型泛化能力")

        suggestions.append("- 尝试集成学习方法（模型融合）")
        suggestions.append("- 进行超参数调优（网格搜索或贝叶斯优化）")
        suggestions.append("- 考虑使用深度学习模型（如果数据量足够大）")

        return '\n'.join(suggestions)

    def save_visualizations(
        self,
        model_results: Dict[str, Any],
        feature_importance: Optional[pd.DataFrame] = None
    ) -> list:
        """
        保存可视化图表

        Args:
            model_results: 模型结果
            feature_importance: 特征重要性

        Returns:
            保存的图表路径列表
        """
        saved_plots = []

        # 模型性能对比图
        if 'results' in model_results:
            plt.figure(figsize=(10, 6))
            models = list(model_results['results'].keys())
            r2_scores = [model_results['results'][m]['test_metrics']['r2'] for m in models]

            plt.barh(models, r2_scores, color='skyblue')
            plt.xlabel('R² Score')
            plt.title('模型性能对比')
            plt.tight_layout()

            plot_path = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(plot_path)

        # 特征重要性图
        if feature_importance is not None and not feature_importance.empty:
            plt.figure(figsize=(10, 8))
            top_15 = feature_importance.head(15)

            plt.barh(range(len(top_15)), top_15['importance'])
            plt.yticks(range(len(top_15)), top_15['feature'])
            plt.xlabel('重要性')
            plt.title('Top 15 特征重要性')
            plt.tight_layout()

            plot_path = os.path.join(self.output_dir, 'feature_importance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots.append(plot_path)

        logger.info(f"Saved {len(saved_plots)} visualizations")
        return saved_plots
