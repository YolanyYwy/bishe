@echo off
echo =========================================
echo   研磨速度预测 AI Agent 启动器
echo =========================================
echo.

:menu
echo 请选择运行模式:
echo.
echo 1. 启动 Streamlit UI 界面 (推荐)
echo 2. 命令行模式 - 完整流程
echo 3. 命令行模式 - 仅训练模型
echo 4. 命令行模式 - 仅预测
echo 5. 退出
echo.

set /p choice="请输入选项 (1-5): "

if "%choice%"=="1" goto ui
if "%choice%"=="2" goto pipeline
if "%choice%"=="3" goto train
if "%choice%"=="4" goto predict
if "%choice%"=="5" goto end

echo 无效选项，请重新选择
goto menu

:ui
echo.
echo 正在启动 Streamlit UI...
python main.py --mode ui
goto end

:pipeline
echo.
set /p datapath="请输入数据文件路径: "
echo 正在执行完整流程...
python main.py --mode pipeline --data "%datapath%"
pause
goto menu

:train
echo.
set /p datapath="请输入训练数据路径: "
echo 正在训练模型...
python main.py --mode train --data "%datapath%"
pause
goto menu

:predict
echo.
set /p datapath="请输入预测数据路径: "
echo 正在进行预测...
python main.py --mode predict --data "%datapath%"
pause
goto menu

:end
echo.
echo 感谢使用！
pause
