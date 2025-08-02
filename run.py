"""
StockSchool 项目启动和管理脚本

支持多种运行模式和数据同步操作

作者: StockSchool Team
创建时间: 2025-01-02
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.utils.config_loader import config
except ImportError:
    config = {}

# 配置日志
def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/run_operations.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def run_command(cmd, capture_output=False, shell=False):
    """
    执行命令
    
    Args:
        cmd: 命令字符串或列表
        capture_output: 是否捕获输出
        shell: 是否使用shell执行
    
    Returns:
        如果capture_output=True，返回(返回码, 输出内容)
        否则只返回返回码
    """
    logger.info(f"执行命令: {cmd}")
    
    try:
        if capture_output:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                shell=shell,
                encoding='utf-8'
            )
            logger.info(f"命令输出: {result.stdout}")
            if result.stderr:
                logger.warning(f"命令错误输出: {result.stderr}")
            return result.returncode, result.stdout
        else:
            result = subprocess.run(cmd, shell=shell)
            return result.returncode
            
    except Exception as e:
        logger.error(f"执行命令失败: {e}")
        if capture_output:
            return -1, str(e)
        else:
            return -1

def check_data_dependencies():
    """检查数据依赖"""
    print("检查数据依赖...")
    
    # 检查Python依赖
    required_packages = [
        'pandas', 'numpy', 'sqlalchemy', 'psycopg2',
        'tushare', 'celery', 'redis', 'fastapi'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少依赖包: {missing_packages}")
        print("请运行 'pip install -r requirements.txt' 安装依赖")
        return False
    else:
        print("所有Python依赖检查通过")
    
    # 检查环境变量
    required_env_vars = ['TUSHARE_TOKEN']
    missing_env_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_env_vars:
        print(f"缺少环境变量: {missing_env_vars}")
        print("请在 .env 文件中配置所需环境变量")
        return False
    else:
        print("环境变量检查通过")
    
    # 检查数据库连接
    try:
        from src.database.connection import get_db_engine
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        print("数据库连接检查通过")
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return False
    
    return True

def is_interactive_session():
    """检查是否为交互式会话"""
    return sys.stdin.isatty()

def check_data_abnormalities():
    """检查数据异常"""
    print("检查数据异常...")
    abnormal_reasons = []
    
    try:
        from src.utils.data_validator import (
            get_record_count, get_historical_average, 
            calculate_standard_deviation, get_max_date, 
            has_future_records
        )
        
        current_count = get_record_count('stock_daily')
        historical_avg = get_historical_average('stock_daily')
        std_dev = calculate_standard_deviation('stock_daily')
        
        if current_count > historical_avg + 3 * std_dev:
            abnormal_reasons.append(f'数据量异常(当前:{current_count}, 预期范围:{historical_avg-3*std_dev}~{historical_avg+3*std_dev})')
        
        max_date = get_max_date('stock_daily')
        if max_date and (datetime.now() - max_date).days > config.get('advanced.data_clean.date_tolerance', 3):
            abnormal_reasons.append(f"数据过期(最新日期:{max_date.strftime('%Y%m%d')}, 容忍天数:{config.get('advanced.data_clean.date_tolerance', 3)})")
        
        if has_future_records('stock_daily'):
            abnormal_reasons.append('存在未来日期记录')
        
        if abnormal_reasons:
            print(f'⚠️ 检测到{len(abnormal_reasons)}项数据异常:')
            for i, reason in enumerate(abnormal_reasons, 1):
                print(f'  {i}. {reason}')
            
            if is_interactive_session():
                print('\n请选择操作:')
                print('1. 执行数据库清理并继续')
                print('2. 忽略异常继续同步')
                print('3. 取消操作并退出')
                
                choice = input('请输入选项(1-3): ').strip()
                if choice == '1':
                    from scripts.clear_database import main as clear_db
                    clear_db()
                    print('✅ 数据库清理完成，继续同步流程...')
                elif choice == '3':
                    print('❌ 用户取消操作')
                    sys.exit(1)
            else:
                print('📝 非交互模式下记录异常日志，继续同步流程')
                # 记录异常日志的实现
        else:
            print("✅ 数据检查通过，无异常")
            
    except Exception as e:
        print(f"数据异常检查失败: {e}")
        logger.error(f"数据异常检查失败: {e}")

def main():
    """主函数"""
    print("StockSchool 项目管理脚本")
    print("=" * 30)
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python run.py check     - 检查依赖和数据")
        print("  python run.py sync      - 启动数据同步")
        print("  python run.py train     - 启动模型训练")
        print("  python run.py predict   - 启动预测服务")
        print("  python run.py api       - 启动API服务")
        print("  python run.py monitor   - 启动监控服务")
        return
    
    command = sys.argv[1]
    
    if command == "check":
        print("执行依赖和数据检查...")
        success = check_data_dependencies()
        if success:
            print("✅ 所有检查通过")
        else:
            print("❌ 检查失败，请查看详细信息")
            sys.exit(1)
            
    elif command == "sync":
        print("启动数据同步...")
        # 检查数据异常
        check_data_abnormalities()
        
        # 启动同步
        sync_type = input("请选择同步类型 (full/basic/calendar/daily): ").strip().lower()
        if sync_type in ["full", "basic", "calendar", "daily"]:
            print(f"正在启动{sync_type}数据同步...")
            run_command(["python", "src/data/tushare_sync.py", "--mode", sync_type])
        else:
            print("无效的同步类型选择。")
            
    elif command == "train":
        print("启动模型训练...")
        run_command("python src/ai/training_pipeline.py")
        
    elif command == "predict":
        print("启动预测服务...")
        run_command("python src/ai/prediction.py")
        
    elif command == "api":
        print("启动API服务...")
        run_command("uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
        
    elif command == "monitor":
        print("启动监控服务...")
        run_command("python src/monitoring/main.py")
        
    else:
        print(f"未知命令: {command}")
        print("支持的命令: check, sync, train, predict, api, monitor")

if __name__ == "__main__":
    main()