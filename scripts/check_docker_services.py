#!/usr/bin/env python3
"""
StockSchool Docker服务健康检查脚本
用于验证Docker容器的启动状态和服务可用性
"""

import os
import sys
import time
import json
import subprocess
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import psycopg2
import redis
import docker
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceStatus:
    """服务状态数据类"""
    name: str
    status: str
    healthy: bool
    message: str
    response_time: float
    details: Dict = None

class DockerServiceChecker:
    """Docker服务健康检查器"""
    
    def __init__(self, config_file: str = '.env.acceptance'):
        """初始化检查器"""
        self.config_file = config_file
        self.config = self._load_config()
        self.docker_client = None
        self.results = []
        
        # 服务配置
        self.services = {
            'postgres': {
                'container_name': 'stockschool_postgres_acceptance',
                'port': 5433,
                'health_check_timeout': 30
            },
            'redis': {
                'container_name': 'stockschool_redis_acceptance', 
                'port': 6380,
                'health_check_timeout': 10
            }
        }
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config = {}
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            config[key.strip()] = value.strip()
            logger.info(f"配置文件 {self.config_file} 加载成功")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
        return config
    
    def _init_docker_client(self) -> bool:
        """初始化Docker客户端"""
        try:
            self.docker_client = docker.from_env()
            # 测试连接
            self.docker_client.ping()
            logger.info("Docker客户端连接成功")
            return True
        except Exception as e:
            logger.error(f"Docker客户端连接失败: {e}")
            return False
    
    def check_docker_daemon(self) -> ServiceStatus:
        """检查Docker守护进程状态"""
        start_time = time.time()
        
        try:
            # 检查Docker是否运行
            result = subprocess.run(
                ['docker', 'version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                response_time = time.time() - start_time
                return ServiceStatus(
                    name="Docker Daemon",
                    status="running",
                    healthy=True,
                    message="Docker守护进程运行正常",
                    response_time=response_time,
                    details={"version_info": result.stdout.split('\n')[0]}
                )
            else:
                return ServiceStatus(
                    name="Docker Daemon",
                    status="error",
                    healthy=False,
                    message=f"Docker守护进程异常: {result.stderr}",
                    response_time=time.time() - start_time
                )
                
        except subprocess.TimeoutExpired:
            return ServiceStatus(
                name="Docker Daemon",
                status="timeout",
                healthy=False,
                message="Docker守护进程响应超时",
                response_time=time.time() - start_time
            )
        except FileNotFoundError:
            return ServiceStatus(
                name="Docker Daemon",
                status="not_found",
                healthy=False,
                message="Docker未安装或不在PATH中",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return ServiceStatus(
                name="Docker Daemon",
                status="error",
                healthy=False,
                message=f"Docker检查失败: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def check_container_status(self, service_name: str) -> ServiceStatus:
        """检查容器状态"""
        start_time = time.time()
        service_config = self.services.get(service_name)
        
        if not service_config:
            return ServiceStatus(
                name=f"Container {service_name}",
                status="config_error",
                healthy=False,
                message=f"未找到服务 {service_name} 的配置",
                response_time=time.time() - start_time
            )
        
        container_name = service_config['container_name']
        
        try:
            if not self.docker_client:
                if not self._init_docker_client():
                    return ServiceStatus(
                        name=f"Container {service_name}",
                        status="docker_error",
                        healthy=False,
                        message="Docker客户端初始化失败",
                        response_time=time.time() - start_time
                    )
            
            # 获取容器信息
            container = self.docker_client.containers.get(container_name)
            
            # 检查容器状态
            container.reload()  # 刷新容器状态
            status = container.status
            
            # 获取健康检查状态
            health_status = "unknown"
            if container.attrs.get('State', {}).get('Health'):
                health_status = container.attrs['State']['Health']['Status']
            
            # 获取端口映射
            ports = container.attrs.get('NetworkSettings', {}).get('Ports', {})
            
            response_time = time.time() - start_time
            
            if status == 'running':
                if health_status == 'healthy' or health_status == 'unknown':
                    return ServiceStatus(
                        name=f"Container {service_name}",
                        status="running",
                        healthy=True,
                        message=f"容器 {container_name} 运行正常",
                        response_time=response_time,
                        details={
                            "container_id": container.short_id,
                            "status": status,
                            "health": health_status,
                            "ports": ports,
                            "created": container.attrs.get('Created'),
                            "started": container.attrs.get('State', {}).get('StartedAt')
                        }
                    )
                else:
                    return ServiceStatus(
                        name=f"Container {service_name}",
                        status="unhealthy",
                        healthy=False,
                        message=f"容器 {container_name} 健康检查失败: {health_status}",
                        response_time=response_time,
                        details={"health_status": health_status}
                    )
            else:
                return ServiceStatus(
                    name=f"Container {service_name}",
                    status=status,
                    healthy=False,
                    message=f"容器 {container_name} 状态异常: {status}",
                    response_time=response_time,
                    details={"container_status": status}
                )
                
        except docker.errors.NotFound:
            return ServiceStatus(
                name=f"Container {service_name}",
                status="not_found",
                healthy=False,
                message=f"容器 {container_name} 不存在",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return ServiceStatus(
                name=f"Container {service_name}",
                status="error",
                healthy=False,
                message=f"容器检查失败: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def check_postgres_connection(self) -> ServiceStatus:
        """检查PostgreSQL数据库连接"""
        start_time = time.time()
        
        try:
            # 从配置获取连接参数
            db_url = self.config.get('DATABASE_URL')
            if not db_url:
                return ServiceStatus(
                    name="PostgreSQL Connection",
                    status="config_error",
                    healthy=False,
                    message="未找到DATABASE_URL配置",
                    response_time=time.time() - start_time
                )
            
            # 连接数据库
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # 执行基本查询
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            # 检查TimescaleDB扩展
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'timescaledb';")
            timescaledb_info = cursor.fetchone()
            
            # 检查数据库大小
            cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
            db_size = cursor.fetchone()[0]
            
            # 检查连接数
            cursor.execute("SELECT count(*) FROM pg_stat_activity;")
            connection_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = time.time() - start_time
            
            return ServiceStatus(
                name="PostgreSQL Connection",
                status="connected",
                healthy=True,
                message="PostgreSQL数据库连接正常",
                response_time=response_time,
                details={
                    "version": version.split('\n')[0],
                    "timescaledb_enabled": timescaledb_info is not None,
                    "database_size": db_size,
                    "active_connections": connection_count,
                    "connection_url": db_url.split('@')[1] if '@' in db_url else "masked"
                }
            )
            
        except psycopg2.OperationalError as e:
            return ServiceStatus(
                name="PostgreSQL Connection",
                status="connection_error",
                healthy=False,
                message=f"PostgreSQL连接失败: {str(e)}",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return ServiceStatus(
                name="PostgreSQL Connection",
                status="error",
                healthy=False,
                message=f"PostgreSQL检查失败: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def check_redis_connection(self) -> ServiceStatus:
        """检查Redis连接"""
        start_time = time.time()
        
        try:
            # 从配置获取连接参数
            redis_url = self.config.get('REDIS_URL')
            if not redis_url:
                return ServiceStatus(
                    name="Redis Connection",
                    status="config_error",
                    healthy=False,
                    message="未找到REDIS_URL配置",
                    response_time=time.time() - start_time
                )
            
            # 连接Redis
            redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # 执行PING命令
            ping_result = redis_client.ping()
            
            # 获取Redis信息
            info = redis_client.info()
            
            # 测试基本操作
            test_key = f"health_check_{int(time.time())}"
            redis_client.set(test_key, "test_value", ex=60)
            test_value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            response_time = time.time() - start_time
            
            if ping_result and test_value == "test_value":
                return ServiceStatus(
                    name="Redis Connection",
                    status="connected",
                    healthy=True,
                    message="Redis连接正常",
                    response_time=response_time,
                    details={
                        "redis_version": info.get('redis_version'),
                        "used_memory": info.get('used_memory_human'),
                        "connected_clients": info.get('connected_clients'),
                        "total_commands_processed": info.get('total_commands_processed'),
                        "keyspace": {k: v for k, v in info.items() if k.startswith('db')},
                        "connection_url": redis_url.split('@')[1] if '@' in redis_url else "masked"
                    }
                )
            else:
                return ServiceStatus(
                    name="Redis Connection",
                    status="test_failed",
                    healthy=False,
                    message="Redis基本操作测试失败",
                    response_time=response_time
                )
                
        except redis.ConnectionError as e:
            return ServiceStatus(
                name="Redis Connection",
                status="connection_error",
                healthy=False,
                message=f"Redis连接失败: {str(e)}",
                response_time=time.time() - start_time
            )
        except Exception as e:
            return ServiceStatus(
                name="Redis Connection",
                status="error",
                healthy=False,
                message=f"Redis检查失败: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def check_network_connectivity(self) -> ServiceStatus:
        """检查网络连通性"""
        start_time = time.time()
        
        try:
            import socket
            
            # 检查PostgreSQL端口
            postgres_port = int(self.config.get('POSTGRES_PORT', 5433))
            postgres_host = self.config.get('POSTGRES_HOST', 'localhost')
            
            # 检查Redis端口
            redis_port = int(self.config.get('REDIS_PORT', 6380))
            redis_host = self.config.get('REDIS_HOST', 'localhost')
            
            connectivity_results = {}
            
            # 测试PostgreSQL端口连通性
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((postgres_host, postgres_port))
                sock.close()
                connectivity_results['postgres'] = {
                    'host': postgres_host,
                    'port': postgres_port,
                    'accessible': result == 0
                }
            except Exception as e:
                connectivity_results['postgres'] = {
                    'host': postgres_host,
                    'port': postgres_port,
                    'accessible': False,
                    'error': str(e)
                }
            
            # 测试Redis端口连通性
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((redis_host, redis_port))
                sock.close()
                connectivity_results['redis'] = {
                    'host': redis_host,
                    'port': redis_port,
                    'accessible': result == 0
                }
            except Exception as e:
                connectivity_results['redis'] = {
                    'host': redis_host,
                    'port': redis_port,
                    'accessible': False,
                    'error': str(e)
                }
            
            response_time = time.time() - start_time
            
            # 判断整体连通性
            all_accessible = all(
                result.get('accessible', False) 
                for result in connectivity_results.values()
            )
            
            if all_accessible:
                return ServiceStatus(
                    name="Network Connectivity",
                    status="accessible",
                    healthy=True,
                    message="所有服务端口连通正常",
                    response_time=response_time,
                    details=connectivity_results
                )
            else:
                failed_services = [
                    f"{service}({info['host']}:{info['port']})"
                    for service, info in connectivity_results.items()
                    if not info.get('accessible', False)
                ]
                return ServiceStatus(
                    name="Network Connectivity",
                    status="partial_failure",
                    healthy=False,
                    message=f"部分服务端口不可访问: {', '.join(failed_services)}",
                    response_time=response_time,
                    details=connectivity_results
                )
                
        except Exception as e:
            return ServiceStatus(
                name="Network Connectivity",
                status="error",
                healthy=False,
                message=f"网络连通性检查失败: {str(e)}",
                response_time=time.time() - start_time
            )
    
    def run_all_checks(self) -> List[ServiceStatus]:
        """运行所有健康检查"""
        logger.info("=== 开始Docker服务健康检查 ===")
        
        checks = [
            ("Docker守护进程", self.check_docker_daemon),
            ("PostgreSQL容器", lambda: self.check_container_status('postgres')),
            ("Redis容器", lambda: self.check_container_status('redis')),
            ("网络连通性", self.check_network_connectivity),
            ("PostgreSQL连接", self.check_postgres_connection),
            ("Redis连接", self.check_redis_connection)
        ]
        
        results = []
        
        for check_name, check_func in checks:
            logger.info(f"正在检查: {check_name}")
            try:
                result = check_func()
                results.append(result)
                
                if result.healthy:
                    logger.info(f"✅ {check_name}: {result.message} ({result.response_time:.3f}s)")
                else:
                    logger.error(f"❌ {check_name}: {result.message} ({result.response_time:.3f}s)")
                    
            except Exception as e:
                error_result = ServiceStatus(
                    name=check_name,
                    status="exception",
                    healthy=False,
                    message=f"检查过程异常: {str(e)}",
                    response_time=0.0
                )
                results.append(error_result)
                logger.error(f"💥 {check_name}: 检查过程异常: {str(e)}")
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict:
        """生成检查报告"""
        if not self.results:
            self.run_all_checks()
        
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.healthy)
        failed_checks = total_checks - passed_checks
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total_checks,
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "success_rate": passed_checks / total_checks if total_checks > 0 else 0,
                "overall_healthy": failed_checks == 0
            },
            "checks": [
                {
                    "name": result.name,
                    "status": result.status,
                    "healthy": result.healthy,
                    "message": result.message,
                    "response_time": result.response_time,
                    "details": result.details
                }
                for result in self.results
            ]
        }
        
        return report
    
    def save_report(self, filename: str = None) -> str:
        """保存检查报告"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"docker_health_check_{timestamp}.json"
        
        report = self.generate_report()
        
        # 确保目录存在
        os.makedirs("logs", exist_ok=True)
        filepath = os.path.join("logs", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"健康检查报告已保存: {filepath}")
        return filepath

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockSchool Docker服务健康检查')
    parser.add_argument('--config', '-c', default='.env.acceptance', 
                       help='配置文件路径 (默认: .env.acceptance)')
    parser.add_argument('--output', '-o', help='输出报告文件名')
    parser.add_argument('--json', action='store_true', help='输出JSON格式结果')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # 创建检查器并运行检查
    checker = DockerServiceChecker(args.config)
    results = checker.run_all_checks()
    
    # 生成报告
    report = checker.generate_report()
    
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        # 打印摘要
        print(f"\n=== Docker服务健康检查报告 ===")
        print(f"检查时间: {report['timestamp']}")
        print(f"总检查项: {report['summary']['total_checks']}")
        print(f"通过检查: {report['summary']['passed_checks']}")
        print(f"失败检查: {report['summary']['failed_checks']}")
        print(f"成功率: {report['summary']['success_rate']:.1%}")
        print(f"整体状态: {'✅ 健康' if report['summary']['overall_healthy'] else '❌ 异常'}")
        
        # 打印详细结果
        print(f"\n=== 详细检查结果 ===")
        for check in report['checks']:
            status_icon = "✅" if check['healthy'] else "❌"
            print(f"{status_icon} {check['name']}: {check['message']} ({check['response_time']:.3f}s)")
    
    # 保存报告
    if args.output or not args.json:
        report_file = checker.save_report(args.output)
        if not args.quiet:
            print(f"\n报告已保存: {report_file}")
    
    # 设置退出码
    sys.exit(0 if report['summary']['overall_healthy'] else 1)

if __name__ == '__main__':
    main()