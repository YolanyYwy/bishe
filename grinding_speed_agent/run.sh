#!/bin/bash

echo "========================================="
echo "  研磨速度预测 AI Agent 启动器"
echo "========================================="
echo ""

show_menu() {
    echo "请选择运行模式:"
    echo ""
    echo "1. 启动 Streamlit UI 界面 (推荐)"
    echo "2. 命令行模式 - 完整流程"
    echo "3. 命令行模式 - 仅训练模型"
    echo "4. 命令行模式 - 仅预测"
    echo "5. 退出"
    echo ""
}

while true; do
    show_menu
    read -p "请输入选项 (1-5): " choice

    case $choice in
        1)
            echo ""
            echo "正在启动 Streamlit UI..."
            python main.py --mode ui
            break
            ;;
        2)
            echo ""
            read -p "请输入数据文件路径: " datapath
            echo "正在执行完整流程..."
            python main.py --mode pipeline --data "$datapath"
            read -p "按Enter键继续..."
            ;;
        3)
            echo ""
            read -p "请输入训练数据路径: " datapath
            echo "正在训练模型..."
            python main.py --mode train --data "$datapath"
            read -p "按Enter键继续..."
            ;;
        4)
            echo ""
            read -p "请输入预测数据路径: " datapath
            echo "正在进行预测..."
            python main.py --mode predict --data "$datapath"
            read -p "按Enter键继续..."
            ;;
        5)
            echo ""
            echo "感谢使用！"
            break
            ;;
        *)
            echo "无效选项，请重新选择"
            ;;
    esac
done
