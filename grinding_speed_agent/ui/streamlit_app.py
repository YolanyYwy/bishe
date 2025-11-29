"""
Streamlit UI界面
提供友好的Web界面与Agent交互
"""
import streamlit as st
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent import GrindingSpeedAgent
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# 页面配置
st.set_page_config(
    page_title="研磨速度预测Agent",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# 初始化Session State
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def initialize_agent():
    """初始化Agent"""
    try:
        config_path = project_root / "config" / "config.yaml"
        st.session_state.agent = GrindingSpeedAgent(str(config_path))
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"Agent初始化失败: {str(e)}")
        return False


def main():
    # 标题
    st.markdown('<div class="main-header">⚙️ 研磨速度预测 AI Agent</div>', unsafe_allow_html=True)
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.header("🎯 功能导航")

        # Agent初始化
        if not st.session_state.initialized:
            if st.button("🚀 初始化Agent", type="primary", use_container_width=True):
                with st.spinner("正在初始化Agent..."):
                    if initialize_agent():
                        st.success("✅ Agent初始化成功！")
                        st.rerun()
        else:
            st.success("✅ Agent已就绪")

        st.markdown("---")

        # 选择模式
        mode = st.radio(
            "选择工作模式",
            ["📊 数据分析", "🎓 模型训练", "🔮 数据预测", "📄 生成报告", "💬 智能对话"],
            disabled=not st.session_state.initialized
        )

        st.markdown("---")

        # 配置选项
        st.header("⚙️ 配置")
        with st.expander("模型设置", expanded=False):
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
            remove_outliers = st.checkbox("移除异常值", value=False)
            feature_engineering = st.checkbox("启用特征工程", value=True)

        with st.expander("报告设置", expanded=False):
            include_viz = st.checkbox("包含可视化", value=True)
            include_importance = st.checkbox("包含特征重要性", value=True)

    # 主内容区域
    if not st.session_state.initialized:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.info("👈 请先在侧边栏初始化Agent")
        st.markdown('</div>', unsafe_allow_html=True)

        # 显示系统介绍
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🤖 智能Agent")
            st.write("集成轻量级大模型，智能理解用户意图并协调执行任务")

        with col2:
            st.markdown("### 📈 多模型集成")
            st.write("自动训练RandomForest、XGBoost、LightGBM等多种模型并选择最优")

        with col3:
            st.markdown("### 📊 自动报告")
            st.write("生成专业的Markdown分析报告，包含模型评估和改进建议")

        return

    # 根据模式显示不同内容
    if mode == "📊 数据分析":
        show_data_analysis_mode()
    elif mode == "🎓 模型训练":
        show_training_mode()
    elif mode == "🔮 数据预测":
        show_prediction_mode()
    elif mode == "📄 生成报告":
        show_report_mode()
    elif mode == "💬 智能对话":
        show_chat_mode()


def show_data_analysis_mode():
    """数据分析模式"""
    st.header("📊 数据分析")

    # 支持多文件上传
    uploaded_files = st.file_uploader(
        "上传数据文件（可选择多个文件）",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )

    if uploaded_files:
        # 保存所有临时文件
        temp_paths = []
        for uploaded_file in uploaded_files:
            temp_path = project_root / "data" / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)

            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            temp_paths.append(str(temp_path))

        st.info(f"已上传 {len(uploaded_files)} 个文件: {', '.join([f.name for f in uploaded_files])}")

        # 如果是单个文件，直接使用路径字符串；如果是多个文件，使用列表
        data_path = temp_paths[0] if len(temp_paths) == 1 else temp_paths

        if st.button("🔍 开始分析", type="primary"):
            with st.spinner("正在分析数据..."):
                result = st.session_state.agent.process_instruction(
                    "分析数据",
                    data_path
                )
                st.success("分析完成！")
                st.text(result)

                # 显示数据预览
                if st.session_state.agent.current_data is not None:
                    st.subheader("数据预览")
                    st.dataframe(st.session_state.agent.current_data.head(100))

                    # 数据统计
                    st.subheader("统计信息")
                    st.dataframe(st.session_state.agent.current_data.describe())

                    # 可视化
                    st.subheader("数据分布")
                    numeric_cols = st.session_state.agent.current_data.select_dtypes(include=['number']).columns
                    selected_col = st.selectbox("选择要可视化的列", numeric_cols)

                    fig = px.histogram(
                        st.session_state.agent.current_data,
                        x=selected_col,
                        nbins=30,
                        title=f"{selected_col} 分布图"
                    )
                    st.plotly_chart(fig, use_container_width=True)


def show_training_mode():
    """模型训练模式"""
    st.header("🎓 模型训练")

    # 支持多文件上传
    uploaded_files = st.file_uploader(
        "上传训练数据（可选择多个文件）",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True
    )

    if uploaded_files:
        # 保存所有临时文件
        temp_paths = []
        for uploaded_file in uploaded_files:
            temp_path = project_root / "data" / uploaded_file.name
            temp_path.parent.mkdir(exist_ok=True)

            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            temp_paths.append(str(temp_path))

        st.info(f"已上传 {len(uploaded_files)} 个文件: {', '.join([f.name for f in uploaded_files])}")

        # 显示数据预览（只预览第一个文件或合并后的数据）
        if len(temp_paths) == 1:
            df = pd.read_csv(temp_paths[0]) if uploaded_files[0].name.endswith('.csv') else pd.read_excel(temp_paths[0])
        else:
            # 如果是多个文件，合并后预览
            dfs = []
            for path in temp_paths:
                df_temp = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
                dfs.append(df_temp)
            df = pd.concat(dfs, ignore_index=True)
            st.info(f"合并后的数据: {df.shape[0]} 行 × {df.shape[1]} 列")

        st.subheader("数据预览")
        st.dataframe(df.head())

        # 选择目标列
        target_col = st.selectbox("选择目标列（预测变量）", df.columns, index=len(df.columns)-1)

        # 如果是单个文件，直接使用路径字符串；如果是多个文件，使用列表
        data_path = temp_paths[0] if len(temp_paths) == 1 else temp_paths

        if st.button("🚀 开始训练", type="primary"):
            with st.spinner("正在训练模型... 这可能需要几分钟"):
                result = st.session_state.agent.process_instruction(
                    "训练模型",
                    data_path
                )

                st.success("训练完成！")
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.text(result)
                st.markdown('</div>', unsafe_allow_html=True)

                # 显示模型性能对比
                if st.session_state.agent.model_results:
                    st.subheader("📊 模型性能对比")

                    results = st.session_state.agent.model_results['results']
                    model_names = list(results.keys())
                    r2_scores = [results[m]['test_metrics']['r2'] for m in model_names]
                    rmse_scores = [results[m]['test_metrics']['rmse'] for m in model_names]

                    # R²对比图
                    col1, col2 = st.columns(2)

                    with col1:
                        fig1 = go.Figure(data=[
                            go.Bar(x=model_names, y=r2_scores, marker_color='lightblue')
                        ])
                        fig1.update_layout(title="R² Score 对比", yaxis_title="R² Score")
                        st.plotly_chart(fig1, use_container_width=True)

                    with col2:
                        fig2 = go.Figure(data=[
                            go.Bar(x=model_names, y=rmse_scores, marker_color='lightcoral')
                        ])
                        fig2.update_layout(title="RMSE 对比", yaxis_title="RMSE")
                        st.plotly_chart(fig2, use_container_width=True)

                    # 特征重要性
                    if st.session_state.agent.feature_importance is not None:
                        st.subheader("🎯 特征重要性")
                        fig3 = px.bar(
                            st.session_state.agent.feature_importance.head(15),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 15 重要特征"
                        )
                        st.plotly_chart(fig3, use_container_width=True)


def show_prediction_mode():
    """数据预测模式"""
    st.header("🔮 数据预测")

    # 检查模型是否已训练
    if st.session_state.agent.ml_manager.best_model is None:
        st.warning("⚠️ 请先训练模型或加载已有模型")
        return

    uploaded_file = st.file_uploader("上传预测数据", type=['csv', 'xlsx', 'xls'])

    if uploaded_file:
        temp_path = project_root / "data" / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)

        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # 数据预览
        df = pd.read_csv(temp_path) if uploaded_file.name.endswith('.csv') else pd.read_excel(temp_path)
        st.subheader("数据预览")
        st.dataframe(df.head())

        if st.button("✨ 开始预测", type="primary"):
            with st.spinner("正在预测..."):
                result = st.session_state.agent.process_instruction(
                    "预测数据",
                    str(temp_path)
                )

                st.success("预测完成！")
                st.text(result)

                # 显示预测结果
                predictions_path = project_root / "grinding_speed_agent" / "reports" / "predictions.csv"
                if predictions_path.exists():
                    pred_df = pd.read_csv(predictions_path)
                    st.subheader("预测结果")
                    st.dataframe(pred_df)

                    # 预测值分布
                    fig = px.histogram(
                        pred_df,
                        x='预测值',
                        nbins=30,
                        title="预测值分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 下载按钮
                    csv = pred_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        "📥 下载预测结果",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )


def show_report_mode():
    """报告生成模式"""
    st.header("📄 生成分析报告")

    if st.session_state.agent.model_results is None:
        st.warning("⚠️ 请先训练模型")
        return

    st.info("点击下方按钮生成完整的Markdown分析报告")

    if st.button("📝 生成报告", type="primary"):
        with st.spinner("正在生成报告..."):
            result = st.session_state.agent.process_instruction("生成报告")

            st.success("报告生成成功！")
            st.text(result)

            # 查找并显示报告
            reports_dir = project_root / "grinding_speed_agent" / "reports"
            md_files = list(reports_dir.glob("grinding_speed_prediction_report_*.md"))

            if md_files:
                latest_report = max(md_files, key=lambda p: p.stat().st_mtime)

                # 读取并显示报告
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report_content = f.read()

                st.subheader("📄 报告预览")
                st.markdown(report_content)

                # 下载按钮
                st.download_button(
                    "📥 下载报告",
                    report_content,
                    latest_report.name,
                    "text/markdown"
                )


def show_chat_mode():
    """智能对话模式"""
    st.header("💬 智能对话")

    st.info("与AI Agent对话，获取专业建议和帮助")

    # 显示对话历史
    for i, (user_msg, agent_msg) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(agent_msg)

    # 输入框
    user_input = st.chat_input("输入您的问题...")

    if user_input:
        # 显示用户消息
        with st.chat_message("user"):
            st.write(user_input)

        # Agent回复
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = st.session_state.agent.process_instruction(user_input)
                st.write(response)

        # 保存历史
        st.session_state.chat_history.append((user_input, response))


if __name__ == "__main__":
    main()
