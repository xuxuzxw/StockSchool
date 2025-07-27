import subprocess
import os
import sys
import time
from datetime import datetime
import subprocess

def run_command(command, cwd=None, capture_output=False):
    """执行命令
    
    Args:
        command: 要执行的命令
        cwd: 工作目录
        capture_output: 是否捕获输出
    
    Returns:
        如果capture_output=True，返回(returncode, stdout, stderr)
        否则返回returncode
    """
    try:
        print(f"正在执行: {' '.join(command) if isinstance(command, list) else command}")
        
        if capture_output:
            process = subprocess.run(command, cwd=cwd, shell=True, 
                                   capture_output=True, encoding='utf-8', errors='replace')
            stdout = process.stdout if process.stdout is not None else ""
            stderr = process.stderr if process.stderr is not None else ""
            return process.returncode, stdout, stderr
            return process.returncode, process.stdout, process.stderr
        else:
            process = subprocess.Popen(command, cwd=cwd, shell=True)
            process.wait()
            if process.returncode != 0:
                print(f"命令执行失败，退出代码: {process.returncode}")
            return process.returncode
    except Exception as e:
        print(f"执行命令时出错: {e}")
        if capture_output:
            return -1, "", str(e)
        else:
            return -1

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

def pre_flight_check():
    """飞行前检查 - 环境健康检查"""
    print("\n=== 飞行前检查 (Pre-Flight Checklist) ===")
    print("正在执行环境健康检查...\n")
    
    checks = []
    
    # 1. Docker服务检查
    print("1. Docker服务检查...")
    returncode, stdout, stderr = run_command("docker ps", capture_output=True)
    if returncode == 0:
        if "stockschool_postgres" in stdout and "stockschool_redis" in stdout:
            print("   ✅ Docker服务正常 - 主要容器运行中")
            checks.append(True)
        else:
            print("   ❌ Docker服务异常 - 缺少必要容器")
            checks.append(False)
    else:
        print("   ❌ Docker服务检查失败")
        checks.append(False)
    
    # 2. 环境变量检查
    print("\n2. 环境变量检查...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        tushare_token = os.getenv('TUSHARE_TOKEN')
        db_password = os.getenv('POSTGRES_PASSWORD')
        
        if tushare_token and not tushare_token.startswith('YOUR_'):
            print("   ✅ TUSHARE_TOKEN 已配置")
            env_check = True
        else:
            print("   ❌ TUSHARE_TOKEN 未正确配置")
            env_check = False
            
        if db_password and not db_password.startswith('YOUR_'):
            print("   ✅ 数据库密码已配置")
        else:
            print("   ❌ 数据库密码未正确配置")
            env_check = False
            
        checks.append(env_check)
    except Exception as e:
        print(f"   ❌ 环境变量检查失败: {e}")
        checks.append(False)
    
    # 3. 数据库连接测试
    print("\n3. 数据库连接测试...")
    try:
        returncode = run_command("python test/check_database_status.py")
        if returncode == 0:
            print("   ✅ 数据库连接正常")
            checks.append(True)
        else:
            print("   ❌ 数据库连接失败")
            checks.append(False)
    except Exception as e:
        print(f"   ❌ 数据库连接测试失败: {e}")
        checks.append(False)
    
    # 4. Tushare API连通性测试
    print("\n4. Tushare API连通性测试...")
    test_script = '''
import tushare as ts
import os
from dotenv import load_dotenv
try:
    load_dotenv()
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))
    df = pro.trade_cal(exchange='', start_date='20230101', end_date='20230101')
    print("Tushare API connection successful.")
except Exception as e:
    print(f"Tushare API connection failed: {e}")
    exit(1)
'''
    
    with open('temp_tushare_test.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    try:
        returncode = run_command("python temp_tushare_test.py")
        if returncode == 0:
            print("   ✅ Tushare API连接正常")
            checks.append(True)
        else:
            print("   ❌ Tushare API连接失败")
            checks.append(False)
    finally:
        if os.path.exists('temp_tushare_test.py'):
            os.remove('temp_tushare_test.py')
    
    # 总结
    print(f"\n=== 检查结果: {sum(checks)}/{len(checks)} 项通过 ===")
    if all(checks):
        print("🎉 所有检查通过，系统就绪！")
        return True
    else:
        print("⚠️  部分检查失败，请修复后再继续")
        return False

def start_celery_worker():
    """启动Celery Worker监控"""
    print("\n=== 启动Celery Worker ===")
    print("正在启动Celery Worker进行任务监控...")
    print("注意: 这将在前台运行，请保持此窗口打开")
    print("按 Ctrl+C 停止Worker\n")
    
    run_command("celery -A src.compute.celery_app worker -l info -P eventlet")

def run_daily_workflow():
    """运行日常工作流"""
    print("\n=== 日常工作流 (Daily Operations) ===")
    print("正在启动完整的日度数据处理流水线...\n")
    
    # 检查是否存在工作流脚本
    workflow_script = "run_daily_workflow.py"
    if os.path.exists(workflow_script):
        returncode = run_command(f"python {workflow_script}")
        if returncode == 0:
            print("✅ 日常工作流执行完成")
        else:
            print("❌ 日常工作流执行失败")
    else:
        print(f"❌ 工作流脚本 {workflow_script} 不存在")
        print("请先创建工作流脚本或使用数据同步功能")

def data_quality_check():
    """数据质量检验"""
    print("\n=== 数据质量检验 (Data Quality Check) ===")
    print("正在执行数据质量检查...\n")
    
    try:
        returncode = run_command("python test/check_data.py")
        if returncode == 0:
            print("✅ 数据质量检查完成")
        else:
            print("❌ 数据质量检查失败")
    except Exception as e:
        print(f"❌ 数据质量检查出错: {e}")

def fix_data_sync():
    """数据修复和回填"""
    print("\n=== 数据修复和回填 (Data Repair & Backfill) ===")
    print("正在执行数据修复...\n")
    
    try:
        returncode = run_command("python test/fix_data_sync.py")
        if returncode == 0:
            print("✅ 数据修复完成")
        else:
            print("❌ 数据修复失败")
    except Exception as e:
        print(f"❌ 数据修复出错: {e}")

def emergency_diagnosis():
    """紧急情况诊断"""
    print("\n=== 紧急情况诊断 (Emergency Diagnosis) ===")
    print("选择诊断类型:")
    print("1. 数据同步失败诊断")
    print("2. 数据库连接问题")
    print("3. API服务异常")
    print("4. 系统资源检查")
    print("5. 返回主菜单")
    
    choice = input("请选择诊断类型 (1-5): ")
    
    if choice == "1":
        print("\n正在诊断数据同步问题...")
        print("1. 检查Tushare API状态")
        print("2. 检查数据库连接")
        print("3. 查看最近的错误日志")
        run_command("python test/check_database_status.py")
        
    elif choice == "2":
        print("\n正在诊断数据库连接...")
        run_command("docker ps")
        run_command("python test/check_database_status.py")
        
    elif choice == "3":
        print("\n正在检查API服务...")
        returncode, stdout, stderr = run_command("curl -s http://localhost:8000/health", capture_output=True)
        if returncode == 0:
            print("✅ API服务响应正常")
        else:
            print("❌ API服务无响应")
            
    elif choice == "4":
        print("\n正在检查系统资源...")
        run_command("docker stats --no-stream")
        
    elif choice == "5":
        return
    else:
        print("无效选择")

def operations_menu():
    """运维操作菜单"""
    while True:
        print("\n=== StockSchool 运维控制台 ===")
        print("1. 飞行前检查 (Pre-Flight Check)")
        print("2. 启动Celery Worker")
        print("3. 运行日常工作流")
        print("4. 数据质量检验")
        print("5. 数据修复和回填")
        print("6. 紧急情况诊断")
        print("7. 返回主菜单")
        
        choice = input("请选择操作 (1-7): ")
        
        if choice == "1":
            pre_flight_check()
        elif choice == "2":
            start_celery_worker()
        elif choice == "3":
            run_daily_workflow()
        elif choice == "4":
            data_quality_check()
        elif choice == "5":
            fix_data_sync()
        elif choice == "6":
            emergency_diagnosis()
        elif choice == "7":
            break
        else:
            print("无效选择，请重试")

def main_menu():
    while True:
        print("\n" + "="*50)
        print("    StockSchool v1.1.6 主控制台")
        print("="*50)
        print("1. 启动API服务器")
        print("2. 运行数据同步")
        print("3. 运维和调试控制台")
        print("4. 退出")
        print("="*50)

        choice = input("请输入您的选择(1-4): ")

        if choice == "1":
            start_api_server()
        elif choice == "2":
            run_data_sync()
        elif choice == "3":
            operations_menu()
        elif choice == "4":
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