import subprocess
import os

def run_command(command, cwd=None):
    try:
        print(f"正在执行: {' '.join(command)}")
        process = subprocess.Popen(command, cwd=cwd, shell=True)
        process.wait()
        if process.returncode != 0:
            print(f"命令执行失败，退出代码: {process.returncode}")
    except Exception as e:
        print(f"执行命令时出错: {e}")

def start_api_server():
    print("正在启动StockSchool API服务器...")
    # 使用uvicorn启动FastAPI应用
    # 注意：这里假设uvicorn已安装，并且在PATH中
    # 如果需要后台运行，可以考虑使用nohup或screen，但为了简单起见，这里是阻塞的
    run_command(["uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def run_data_sync():
    print("正在运行数据同步...")
    print("请选择同步类型:")
    print("1. 完整同步")
    print("2. 基本信息同步")
    print("3. 交易日历同步")
    print("4. 日线数据同步")
    sync_choice = input("请输入您的选择(1-4): ")

    sync_type = {
        "1": "full",
        "2": "basic",
        "3": "calendar",
        "4": "daily"
    }.get(sync_choice)

    if sync_type:
        print(f"正在启动{sync_type}数据同步...")
        run_command(["python", "src/data/tushare_sync.py", sync_type])
    else:
        print("无效的同步类型选择。")

def main_menu():
    while True:
        print("\nStockSchool 主菜单")
        print("1. 启动API服务器")
        print("2. 运行数据同步")
        print("3. 退出")

        choice = input("请输入您的选择(1-3): ")

        if choice == "1":
            start_api_server()
        elif choice == "2":
            run_data_sync()
        elif choice == "3":
            print("正在退出StockSchool。再见！")
            break
        else:
            print("无效的选择，请重试。")

if __name__ == "__main__":
    # 确保当前工作目录是项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir))
    os.chdir(project_root)
    main_menu()