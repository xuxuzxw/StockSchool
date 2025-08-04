#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三阶段启动脚本

功能:
- 初始化第三阶段所有模块
- 启动后台服务
- 提供命令行接口
- 系统健康检查
"""

import asyncio
import logging
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ai.strategy.stage3_manager import Stage3Manager
from src.database.init_v3 import DatabaseInitializerV3
from src.api.strategy_v3 import app
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stage3.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class Stage3Launcher:
    """第三阶段启动器"""
    
    def __init__(self):
        self.stage3_manager = None
        self.db_initializer = None
        self.api_server = None
    
    async def initialize_database(self) -> bool:
        """初始化数据库"""
        try:
            logger.info("开始初始化第三阶段数据库...")
            
            self.db_initializer = DatabaseInitializerV3()
            
            # 检查数据库状态
            status = await self.db_initializer.get_initialization_status()
            logger.info(f"数据库初始化状态: {status}")
            
            # 初始化所有表（包括默认数据）
            success = await self.db_initializer.initialize_all_tables()
            if not success:
                logger.error("数据库初始化失败")
                return False
            
            logger.info("数据库初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            return False
    
    async def initialize_stage3_manager(self) -> bool:
        """初始化第三阶段管理器"""
        try:
            logger.info("开始初始化第三阶段管理器...")
            
            self.stage3_manager = Stage3Manager()
            
            # 初始化所有模块
            success = await self.stage3_manager.initialize_all_modules()
            if not success:
                logger.error("第三阶段管理器初始化失败")
                return False
            
            logger.info("第三阶段管理器初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"第三阶段管理器初始化失败: {e}")
            return False
    
    async def start_api_server(self, host: str = "0.0.0.0", port: int = 8000) -> bool:
        """启动API服务器"""
        try:
            logger.info(f"启动API服务器 {host}:{port}...")
            
            # 配置服务器
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level="info",
                access_log=True
            )
            
            self.api_server = uvicorn.Server(config)
            
            # 启动服务器（非阻塞）
            await self.api_server.serve()
            
            logger.info("API服务器启动完成")
            return True
            
        except Exception as e:
            logger.error(f"API服务器启动失败: {e}")
            return False
    
    async def run_health_check(self) -> Dict[str, Any]:
        """运行健康检查"""
        try:
            logger.info("开始系统健康检查...")
            
            if not self.stage3_manager:
                return {
                    'status': 'error',
                    'message': '第三阶段管理器未初始化',
                    'timestamp': datetime.now()
                }
            
            # 执行全面检查
            report = await self.stage3_manager.execute_comprehensive_check()
            
            logger.info("系统健康检查完成")
            return report
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now()
            }
    
    async def show_system_info(self):
        """显示系统信息"""
        try:
            if not self.stage3_manager:
                print("第三阶段管理器未初始化")
                return
            
            # 获取模块信息
            module_info = self.stage3_manager.get_module_info()
            
            print("\n" + "="*60)
            print(f"StockSchool AI策略系统 - {module_info['stage']}")
            print(f"版本: {module_info['version']}")
            print("="*60)
            
            print("\n功能模块:")
            for module_id, module_data in module_info['modules'].items():
                print(f"\n• {module_data['name']}")
                print(f"  描述: {module_data['description']}")
                print(f"  功能: {', '.join(module_data['features'])}")
            
            print("\n核心能力:")
            for capability in module_info['capabilities']:
                print(f"• {capability}")
            
            # 获取系统状态
            status = await self.stage3_manager.check_system_health()
            print(f"\n系统状态: {status.overall_health}")
            print(f"活跃模块: {sum([status.model_monitor_active, status.system_optimizer_active, status.doc_generator_active, status.test_framework_active, status.deployment_manager_active])}/5")
            print(f"错误数量: {status.error_count}")
            print(f"活跃任务: {status.active_tasks}")
            
            # 获取系统指标
            metrics = await self.stage3_manager.get_system_metrics()
            print(f"\n系统指标:")
            print(f"• 监控模型: {metrics.monitored_models}")
            print(f"• 活跃告警: {metrics.active_alerts}")
            print(f"• 优化任务: {metrics.optimization_tasks}")
            print(f"• 生成文档: {metrics.generated_documents}")
            print(f"• 测试执行: {metrics.test_executions}")
            print(f"• 部署次数: {metrics.deployments}")
            print(f"• 健康评分: {metrics.system_health_score}")
            print(f"• 运行时间: {metrics.uptime_percentage}%")
            
            print("\n" + "="*60)
            
        except Exception as e:
            logger.error(f"显示系统信息失败: {e}")
            print(f"显示系统信息失败: {e}")
    
    async def run_interactive_mode(self):
        """运行交互模式"""
        print("\n进入第三阶段交互模式")
        print("可用命令:")
        print("  status - 显示系统状态")
        print("  health - 运行健康检查")
        print("  info - 显示系统信息")
        print("  metrics - 显示系统指标")
        print("  quit - 退出")
        
        while True:
            try:
                command = input("\nStage3> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    print("退出交互模式")
                    break
                
                elif command == 'status':
                    if self.stage3_manager:
                        status = await self.stage3_manager.check_system_health()
                        print(f"\n系统状态: {status.overall_health}")
                        print(f"模块状态:")
                        print(f"  模型监控: {'✓' if status.model_monitor_active else '✗'}")
                        print(f"  系统优化: {'✓' if status.system_optimizer_active else '✗'}")
                        print(f"  文档生成: {'✓' if status.doc_generator_active else '✗'}")
                        print(f"  测试框架: {'✓' if status.test_framework_active else '✗'}")
                        print(f"  部署管理: {'✓' if status.deployment_manager_active else '✗'}")
                    else:
                        print("第三阶段管理器未初始化")
                
                elif command == 'health':
                    report = await self.run_health_check()
                    if report.get('status') == 'error':
                        print(f"健康检查失败: {report.get('message')}")
                    else:
                        print(f"\n健康检查报告:")
                        print(f"整体状态: {report.get('overall_status', {}).get('health', 'unknown')}")
                        print(f"活跃模块: {report.get('overall_status', {}).get('active_modules', 0)}/5")
                        print(f"健康评分: {report.get('system_metrics', {}).get('health_score', 0)}")
                
                elif command == 'info':
                    await self.show_system_info()
                
                elif command == 'metrics':
                    if self.stage3_manager:
                        metrics = await self.stage3_manager.get_system_metrics()
                        print(f"\n系统指标:")
                        print(f"监控模型: {metrics.monitored_models}")
                        print(f"活跃告警: {metrics.active_alerts}")
                        print(f"优化任务: {metrics.optimization_tasks}")
                        print(f"生成文档: {metrics.generated_documents}")
                        print(f"测试执行: {metrics.test_executions}")
                        print(f"部署次数: {metrics.deployments}")
                        print(f"健康评分: {metrics.system_health_score}")
                        print(f"运行时间: {metrics.uptime_percentage}%")
                    else:
                        print("第三阶段管理器未初始化")
                
                elif command == 'help':
                    print("可用命令:")
                    print("  status - 显示系统状态")
                    print("  health - 运行健康检查")
                    print("  info - 显示系统信息")
                    print("  metrics - 显示系统指标")
                    print("  quit - 退出")
                
                else:
                    print(f"未知命令: {command}，输入 'help' 查看可用命令")
                    
            except KeyboardInterrupt:
                print("\n退出交互模式")
                break
            except Exception as e:
                print(f"命令执行失败: {e}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='StockSchool AI策略系统 - 第三阶段')
    parser.add_argument('--mode', choices=['init', 'start', 'health', 'info', 'interactive'], 
                       default='start', help='运行模式')
    parser.add_argument('--host', default='0.0.0.0', help='API服务器主机')
    parser.add_argument('--port', type=int, default=8000, help='API服务器端口')
    parser.add_argument('--no-api', action='store_true', help='不启动API服务器')
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    launcher = Stage3Launcher()
    
    try:
        if args.mode == 'init':
            # 仅初始化模式
            logger.info("开始第三阶段初始化...")
            
            # 初始化数据库
            if not await launcher.initialize_database():
                logger.error("数据库初始化失败")
                return 1
            
            # 初始化管理器
            if not await launcher.initialize_stage3_manager():
                logger.error("第三阶段管理器初始化失败")
                return 1
            
            logger.info("第三阶段初始化完成")
            return 0
        
        elif args.mode == 'start':
            # 完整启动模式
            logger.info("开始启动第三阶段系统...")
            
            # 初始化数据库
            if not await launcher.initialize_database():
                logger.error("数据库初始化失败")
                return 1
            
            # 初始化管理器
            if not await launcher.initialize_stage3_manager():
                logger.error("第三阶段管理器初始化失败")
                return 1
            
            # 显示系统信息
            await launcher.show_system_info()
            
            if not args.no_api:
                # 启动API服务器
                logger.info(f"启动API服务器: http://{args.host}:{args.port}")
                await launcher.start_api_server(args.host, args.port)
            else:
                logger.info("跳过API服务器启动")
                # 进入交互模式
                await launcher.run_interactive_mode()
            
            return 0
        
        elif args.mode == 'health':
            # 健康检查模式
            logger.info("运行系统健康检查...")
            
            # 初始化管理器
            if not await launcher.initialize_stage3_manager():
                logger.error("第三阶段管理器初始化失败")
                return 1
            
            # 运行健康检查
            report = await launcher.run_health_check()
            
            # 输出报告
            print("\n" + "="*60)
            print("系统健康检查报告")
            print("="*60)
            
            if report.get('status') == 'error':
                print(f"检查失败: {report.get('message')}")
                return 1
            
            overall_status = report.get('overall_status', {})
            print(f"整体健康状态: {overall_status.get('health', 'unknown')}")
            print(f"活跃模块: {overall_status.get('active_modules', 0)}/{overall_status.get('total_modules', 5)}")
            print(f"错误数量: {overall_status.get('error_count', 0)}")
            print(f"活跃任务: {overall_status.get('active_tasks', 0)}")
            
            system_metrics = report.get('system_metrics', {})
            print(f"\n系统指标:")
            print(f"健康评分: {system_metrics.get('health_score', 0)}")
            print(f"运行时间: {system_metrics.get('uptime_percentage', 0)}%")
            
            recommendations = report.get('recommendations', [])
            if recommendations:
                print(f"\n改进建议:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec}")
            
            print("\n" + "="*60)
            return 0
        
        elif args.mode == 'info':
            # 信息显示模式
            if not await launcher.initialize_stage3_manager():
                logger.error("第三阶段管理器初始化失败")
                return 1
            
            await launcher.show_system_info()
            return 0
        
        elif args.mode == 'interactive':
            # 交互模式
            if not await launcher.initialize_database():
                logger.error("数据库初始化失败")
                return 1
            
            if not await launcher.initialize_stage3_manager():
                logger.error("第三阶段管理器初始化失败")
                return 1
            
            await launcher.run_interactive_mode()
            return 0
        
        else:
            logger.error(f"未知运行模式: {args.mode}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出...")
        return 0
    except Exception as e:
        logger.error(f"启动失败: {e}")
        return 1

if __name__ == '__main__':
    # 运行主函数
    exit_code = asyncio.run(main())
    sys.exit(exit_code)