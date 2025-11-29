"""
研磨速度预测Agent - 主入口文件
可以通过命令行或直接运行来使用Agent
"""
import argparse
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from grinding_speed_agent.agent import GrindingSpeedAgent


def main():
    parser = argparse.ArgumentParser(description='研磨速度预测AI Agent')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['ui', 'pipeline', 'train', 'predict', 'analyze', 'report'],
        default='ui',
        help='运行模式'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='数据文件路径'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='grinding_speed_agent/config/config.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--instruction',
        type=str,
        help='指令文本'
    )

    args = parser.parse_args()

    if args.mode == 'ui':
        # 启动Streamlit UI
        print("🚀 启动Streamlit UI...")
        import streamlit.web.cli as stcli
        import sys

        #ui_path = project_root / "grinding_speed_agent" / "ui" / "streamlit_app.py"
        ui_path = project_root/ "ui" / "streamlit_app.py"
        sys.argv = ["streamlit", "run", str(ui_path)]
        sys.exit(stcli.main())

    else:
        # 命令行模式
        print("=" * 60)
        print("研磨速度预测 AI Agent".center(60))
        print("=" * 60)
        print()

        # 初始化Agent
        print("初始化Agent...")
        agent = GrindingSpeedAgent(args.config)
        print("✅ Agent初始化完成")
        print()

        if args.mode == 'pipeline':
            # 完整流程
            if not args.data:
                print("❌ 错误: 请提供数据文件路径 (--data)")
                return

            print("开始执行完整预测流程...")
            result = agent.execute_pipeline(args.data)
            print(result)

        elif args.mode == 'train':
            # 训练模式
            if not args.data:
                print("❌ 错误: 请提供数据文件路径 (--data)")
                return

            print("开始训练模型...")
            result = agent.process_instruction("训练模型", args.data)
            print(result)

        elif args.mode == 'predict':
            # 预测模式
            if not args.data:
                print("❌ 错误: 请提供数据文件路径 (--data)")
                return

            print("开始预测...")
            result = agent.process_instruction("预测数据", args.data)
            print(result)

        elif args.mode == 'analyze':
            # 分析模式
            if not args.data:
                print("❌ 错误: 请提供数据文件路径 (--data)")
                return

            print("开始分析数据...")
            result = agent.process_instruction("分析数据", args.data)
            print(result)

        elif args.mode == 'report':
            # 报告生成
            print("生成报告...")
            result = agent.process_instruction("生成报告")
            print(result)

        print()
        print("=" * 60)
        print("执行完成".center(60))
        print("=" * 60)


if __name__ == "__main__":
    main()
