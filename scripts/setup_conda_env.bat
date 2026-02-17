@echo off
chcp 65001 >nul
echo ==========================================
echo NanoRAFT-RL Conda 环境安装脚本
echo ==========================================
echo.

:: 检查conda是否安装
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未检测到Conda，请先安装Anaconda或Miniconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [1/3] 正在创建conda环境 nanoraft-rl...
conda env create -f environment.yml

if %errorlevel% neq 0 (
    echo [错误] 环境创建失败
    pause
    exit /b 1
)

echo.
echo [2/3] 环境创建成功！
echo.
echo [3/3] 激活环境命令:
echo    conda activate nanoraft-rl
echo.
echo ==========================================
echo 安装完成！请运行以下命令激活环境:
echo.
echo    conda activate nanoraft-rl
echo.
echo 然后运行数据合成脚本:
echo    python scripts/run_data_synthesis.py
echo ==========================================
pause
