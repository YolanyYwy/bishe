"""
AI Agent核心框架
使用轻量级大模型协调传统ML模型完成预测任务
"""
import yaml
import os
import logging
from typing import Dict, Any, Optional
import pandas as pd
from grinding_speed_agent.llm import LocalLLM
from grinding_speed_agent.models import MLModelManager
from grinding_speed_agent.utils import DataProcessor, ReportGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrindingSpeedAgent:
    """研磨速度预测Agent"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化Agent

        Args:
            config_path: 配置文件路径
        """
        logger.info("Initializing Grinding Speed Agent...")

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化组件
        self.llm = None
        self.ml_manager = MLModelManager(self.config['ml_models'])
        self.data_processor = DataProcessor()
        self.report_generator = ReportGenerator(self.config['report']['output_dir'])

        # 状态
        self.conversation_history = []
        self.current_data = None
        self.model_results = None
        self.feature_importance = None

        logger.info("Agent initialized successfully!")

    def initialize_llm(self):
        """初始化大模型（延迟加载，节省资源）"""
        if self.llm is None:
            logger.info("Initializing LLM...")
            llm_config = self.config['llm']
            self.llm = LocalLLM(
                model_name=llm_config['model_name'],
                device=llm_config['device'],
                max_length=llm_config['max_length'],
                temperature=llm_config['temperature'],
                top_p=llm_config['top_p'],
                quantization_config=llm_config.get('quantization')
            )
            logger.info("LLM initialized!")

    def process_instruction(self, instruction: str, data_path: Optional[str] = None) -> str:
        """
        处理用户指令

        Args:
            instruction: 用户指令
            data_path: 数据文件路径（可选）

        Returns:
            处理结果描述
        """
        logger.info(f"Processing instruction: {instruction}")

        # 解析指令意图
        intent = self._parse_intent(instruction)

        # 根据意图执行不同的任务
        if intent == 'train':
            return self._handle_training(instruction, data_path)
        elif intent == 'predict':
            return self._handle_prediction(instruction, data_path)
        elif intent == 'analyze':
            return self._handle_analysis(instruction, data_path)
        elif intent == 'report':
            return self._handle_report_generation(instruction)
        else:
            return self._handle_general_query(instruction)

    def _parse_intent(self, instruction: str) -> str:
        """
        解析用户意图

        Args:
            instruction: 用户指令

        Returns:
            意图类型
        """
        instruction_lower = instruction.lower()

        # 关键词匹配
        if any(kw in instruction_lower for kw in ['训练', 'train', '建模', '模型']):
            return 'train'
        elif any(kw in instruction_lower for kw in ['预测', 'predict', '推理', 'inference']):
            return 'predict'
        elif any(kw in instruction_lower for kw in ['分析', 'analyze', '数据', 'data']):
            return 'analyze'
        elif any(kw in instruction_lower for kw in ['报告', 'report', '生成报告']):
            return 'report'
        else:
            return 'general'

    def _handle_training(self, instruction: str, data_path: str) -> str:
        """处理训练任务"""
        logger.info("Handling training task...")

        try:
            # 1. 加载数据
            df = self.data_processor.load_data(data_path)
            logger.info(f"Loaded data: {df.shape}")

            # 2. 数据预处理
            df = self.data_processor.preprocess_data(
                df,
                handle_missing='mean',
                remove_outliers=False
            )

            # 3. 分离特征和目标
            X, y = self.data_processor.split_features_target(df)

            # 4. 特征工程（可选）
            if self.config['data']['feature_engineering']['enabled']:
                X = self.data_processor.engineer_features(
                    X,
                    polynomial=self.config['data']['feature_engineering']['polynomial_features'],
                    interactions=self.config['data']['feature_engineering']['interaction_features']
                )

            # 5. 训练模型
            self.model_results = self.ml_manager.train_models(
                X, y,
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_state']
            )

            # 6. 获取特征重要性
            self.feature_importance = self.ml_manager.get_feature_importance()

            # 7. 保存模型
            self.ml_manager.save_models('grinding_speed_agent/models_saved')

            # 8. 保存当前数据
            self.current_data = df

            result = f"""
训练完成！

最佳模型: {self.model_results['best_model']}
测试集 R²: {self.model_results['results'][self.model_results['best_model']]['test_metrics']['r2']:.4f}
训练数据: {len(df)} 条记录

所有模型已保存至 models_saved/ 目录。
"""
            return result.strip()

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return f"训练失败: {str(e)}"

    def _handle_prediction(self, instruction: str, data_path: str) -> str:
        """处理预测任务"""
        logger.info("Handling prediction task...")

        try:
            # 检查模型是否已训练
            if not self.ml_manager.best_model:
                # 尝试加载已保存的模型
                if os.path.exists('grinding_speed_agent/models_saved/metadata.pkl'):
                    self.ml_manager.load_models('grinding_speed_agent/models_saved')
                else:
                    return "错误：请先训练模型或加载已有模型。"

            # 加载预测数据
            df = self.data_processor.load_data(data_path)

            # 预处理
            df = self.data_processor.preprocess_data(df, handle_missing='mean')

            # 检查是否包含目标列（真实值）
            has_true_values = False
            true_values = None
            if self.data_processor.target_name in df.columns:
                has_true_values = True
                true_values = df[self.data_processor.target_name].copy()
                X = df.drop(columns=[self.data_processor.target_name])
            else:
                X = df

            # 预测
            predictions = self.ml_manager.predict(X)

            # 创建结果DataFrame
            result_df = X.copy()
            if has_true_values:
                result_df['真实值'] = true_values
            result_df['预测值'] = predictions

            # 保存预测结果
            output_path = os.path.join(
                self.config['report']['output_dir'],
                'predictions.csv'
            )
            result_df.to_csv(output_path, index=False)

            # 如果有真实值，计算性能指标
            result = f"""
预测完成！

使用模型: {self.ml_manager.best_model_name}
预测数量: {len(predictions)}
预测范围: [{predictions.min():.4f}, {predictions.max():.4f}]
平均值: {predictions.mean():.4f}
"""

            if has_true_values:
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                import numpy as np

                mse = mean_squared_error(true_values, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(true_values, predictions)
                r2 = r2_score(true_values, predictions)

                result += f"""
**预测性能** (与真实值对比):
- R² Score: {r2:.4f}
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}
"""

            result += f"""
结果已保存至: {output_path}
"""
            return result.strip()

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return f"预测失败: {str(e)}"

    def _handle_analysis(self, instruction: str, data_path: str) -> str:
        """处理数据分析任务"""
        logger.info("Handling analysis task...")

        try:
            # 加载数据
            df = self.data_processor.load_data(data_path)

            # 获取数据摘要
            summary = self.data_processor.get_data_summary(df)

            # 检测数据问题
            issues = self.data_processor.detect_data_issues(df)

            # 格式化输出
            result = f"""
数据分析结果：

📊 基本信息:
- 数据规模: {summary['shape'][0]} 行 × {summary['shape'][1]} 列
- 内存占用: {summary['memory_usage']:.2f} MB

🔍 数据质量:
- 缺失值: {sum(summary['missing_values'].values())} 个
- 重复行: {issues['duplicates']} 条
- 异常值检测: {len(issues['outliers'])} 个特征包含异常值

📈 数据准备就绪，可以开始训练模型。
"""
            self.current_data = df
            return result.strip()

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return f"分析失败: {str(e)}"

    def _handle_report_generation(self, instruction: str) -> str:
        """处理报告生成任务"""
        logger.info("Handling report generation task...")

        try:
            if self.model_results is None:
                return "错误：请先训练模型。"

            # 获取数据摘要
            data_summary = self.data_processor.get_data_summary(self.current_data)

            # 检测数据问题
            data_issues = self.data_processor.detect_data_issues(self.current_data)

            # 尝试加载预测结果（如果存在）
            predictions_df = None
            predictions_path = os.path.join(
                self.config['report']['output_dir'],
                'predictions.csv'
            )
            if os.path.exists(predictions_path):
                try:
                    predictions_df = pd.read_csv(predictions_path)
                    logger.info(f"Loaded predictions from {predictions_path}")
                except Exception as e:
                    logger.warning(f"Failed to load predictions: {str(e)}")

            # 生成报告
            report_path = self.report_generator.generate_report(
                data_summary=data_summary,
                model_results=self.model_results,
                predictions=predictions_df,  # 传递预测结果
                feature_importance=self.feature_importance,
                data_issues=data_issues
            )

            # 保存可视化图表
            plots = self.report_generator.save_visualizations(
                self.model_results,
                self.feature_importance
            )

            result = f"""
报告生成完成！

📄 Markdown报告: {report_path}
📊 可视化图表: {len(plots)} 个

报告包含:
✅ 数据分析摘要
✅ 模型性能对比
✅ 特征重要性分析
✅ 预测结果分析 {'(已包含)' if predictions_df is not None else '(未包含)'}
✅ 改进建议

请查看报告文件获取详细信息。
"""
            return result.strip()

        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            return f"报告生成失败: {str(e)}"

    def _handle_general_query(self, instruction: str) -> str:
        """处理通用查询（使用LLM）"""
        logger.info("Handling general query with LLM...")

        # 初始化LLM（如果还未初始化）
        self.initialize_llm()

        # 构建系统提示
        system_prompt = """你是一个研磨速度预测领域的AI助手。你可以帮助用户：
1. 训练预测模型
2. 进行数据预测
3. 分析数据质量
4. 生成分析报告

请根据用户的问题提供专业的建议。"""

        # 使用LLM回答
        response, self.conversation_history = self.llm.chat(
            instruction,
            history=self.conversation_history,
            system=system_prompt
        )

        return response

    def execute_pipeline(self, data_path: str, target_column: Optional[str] = None) -> str:
        """
        执行完整的预测流程

        Args:
            data_path: 数据路径
            target_column: 目标列名

        Returns:
            执行结果
        """
        logger.info("Executing full prediction pipeline...")

        results = []

        # 1. 数据分析
        results.append("=" * 50)
        results.append("步骤 1: 数据分析")
        results.append("=" * 50)
        analysis_result = self._handle_analysis("分析数据", data_path)
        results.append(analysis_result)

        # 2. 模型训练
        results.append("\n" + "=" * 50)
        results.append("步骤 2: 模型训练")
        results.append("=" * 50)
        training_result = self._handle_training("训练模型", data_path)
        results.append(training_result)

        # 3. 生成报告
        results.append("\n" + "=" * 50)
        results.append("步骤 3: 生成报告")
        results.append("=" * 50)
        report_result = self._handle_report_generation("生成报告")
        results.append(report_result)

        results.append("\n" + "=" * 50)
        results.append("✅ 完整流程执行完毕！")
        results.append("=" * 50)

        return "\n".join(results)
