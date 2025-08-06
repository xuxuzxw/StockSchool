#!/usr/bin/env python3
"""
StockSchool 网络连通性测试工具
用于验证本地应用到Docker容器和外部服务的网络连接
"""

import os
import sys
import json
import socket
import time
import subprocess
import logging
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import concurrent.futures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConnectivityResult:
    """连通性测试结果数据类"""
    target: str
    test_type: str
    status: str  # success, failed, timeout, error
    response_time: float
    message: str
    details: Dict = None

class NetworkConnectivityTester:
    """网络连通性测试器"""
    
    def __init__(self, config_file: str = '.env.acceptance'):
        """初始化测试器"""
        self.config_file = config_file
        self.config = self._load_config()
        self.results = []
        
        # 测试目标配置
        self.test_targets = {
            # Docker容器端口
            'postgres_port': {
                'host': self.config.get('POSTGRES_HOST', 'localhost'),
                'port': int(self.config.get('POSTGRES_PORT', 5433)),
                'timeout': 5,
                'description': 'PostgreSQL数据库端口'
            },
            'redis_port': {
                'host': self.config.get('REDIS_HOST', 'localhost'),
                'port': int(self.config.get('REDIS_PORT', 6380)),
                'timeout': 5,
                'description': 'Redis缓存端口'
            },
            
            # 外部API服务
            'tushare_api': {
                'url': 'http://api.tushare.pro',
                'timeout': 10,
                'description': 'Tushare数据API'
            },
            'ai_api': {
                'url': self.config.get('AI_API_BASE_URL', 'https://api.openai.com'),
                'timeout': 10,
                'description': '外接AI大模型API'
            },
            
            # 网络基础服务
            'dns_primary': {
                'host': '8.8.8.8',
                'port': 53,
                'timeout': 3,
                'description': 'Google DNS'
            },
            'dns_secondary': {
                'host': '114.114.114.114',
                'port': 53,
                'timeout': 3,
                'description': '114 DNS'
            },
            
            # 互联网连通性
            'internet_http': {
                'url': 'http://www.baidu.com',
                'timeout': 5,
                'description': '互联网HTTP连接'
            },
            'internet_https': {
                'url': 'https://www.baidu.com',
                'timeout': 5,
                'description': '互联网HTTPS连接'
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
    
    def test_tcp_port(self, host: str, port: int, timeout: int = 5) -> ConnectivityResult:
        """测试TCP端口连通性"""
        start_time = time.time()
        target = f"{host}:{port}"
        
        try:
            # 创建socket连接
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # 尝试连接
            result = sock.connect_ex((host, port))
            sock.close()
            
            response_time = time.time() - start_time
            
            if result == 0:
                return ConnectivityResult(
                    target=target,
                    test_type="tcp_port",
                    status="success",
                    response_time=response_time,
                    message=f"TCP端口 {target} 连接成功",
                    details={
                        "host": host,
                        "port": port,
                        "connection_result": result
                    }
                )
            else:
                return ConnectivityResult(
                    target=target,
                    test_type="tcp_port",
                    status="failed",
                    response_time=response_time,
                    message=f"TCP端口 {target} 连接失败 (错误码: {result})",
                    details={
                        "host": host,
                        "port": port,
                        "connection_result": result,
                        "error_code": result
                    }
                )
                
        except socket.timeout:
            return ConnectivityResult(
                target=target,
                test_type="tcp_port",
                status="timeout",
                response_time=timeout,
                message=f"TCP端口 {target} 连接超时",
                details={"host": host, "port": port, "timeout": timeout}
            )
        except Exception as e:
            return ConnectivityResult(
                target=target,
                test_type="tcp_port",
                status="error",
                response_time=time.time() - start_time,
                message=f"TCP端口 {target} 测试异常: {str(e)}",
                details={"host": host, "port": port, "error": str(e)}
            )
    
    async def test_http_url(self, url: str, timeout: int = 10) -> ConnectivityResult:
        """测试HTTP/HTTPS URL连通性"""
        start_time = time.time()
        
        try:
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            
            async with aiohttp.ClientSession(timeout=timeout_config) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    # 读取响应头信息
                    headers = dict(response.headers)
                    
                    if response.status < 400:
                        return ConnectivityResult(
                            target=url,
                            test_type="http_url",
                            status="success",
                            response_time=response_time,
                            message=f"HTTP请求 {url} 成功 (状态码: {response.status})",
                            details={
                                "url": url,
                                "status_code": response.status,
                                "headers": {k: v for k, v in headers.items() if k.lower() in ['server', 'content-type', 'content-length']},
                                "method": "GET"
                            }
                        )
                    else:
                        return ConnectivityResult(
                            target=url,
                            test_type="http_url",
                            status="failed",
                            response_time=response_time,
                            message=f"HTTP请求 {url} 失败 (状态码: {response.status})",
                            details={
                                "url": url,
                                "status_code": response.status,
                                "reason": response.reason
                            }
                        )
                        
        except asyncio.TimeoutError:
            return ConnectivityResult(
                target=url,
                test_type="http_url",
                status="timeout",
                response_time=timeout,
                message=f"HTTP请求 {url} 超时",
                details={"url": url, "timeout": timeout}
            )
        except aiohttp.ClientError as e:
            return ConnectivityResult(
                target=url,
                test_type="http_url",
                status="error",
                response_time=time.time() - start_time,
                message=f"HTTP请求 {url} 客户端错误: {str(e)}",
                details={"url": url, "error": str(e), "error_type": "ClientError"}
            )
        except Exception as e:
            return ConnectivityResult(
                target=url,
                test_type="http_url",
                status="error",
                response_time=time.time() - start_time,
                message=f"HTTP请求 {url} 异常: {str(e)}",
                details={"url": url, "error": str(e)}
            )
    
    def test_dns_resolution(self, hostname: str, timeout: int = 5) -> ConnectivityResult:
        """测试DNS解析"""
        start_time = time.time()
        
        try:
            # 设置DNS查询超时
            socket.setdefaulttimeout(timeout)
            
            # 执行DNS解析
            ip_addresses = socket.gethostbyname_ex(hostname)
            
            response_time = time.time() - start_time
            
            return ConnectivityResult(
                target=hostname,
                test_type="dns_resolution",
                status="success",
                response_time=response_time,
                message=f"DNS解析 {hostname} 成功",
                details={
                    "hostname": hostname,
                    "canonical_name": ip_addresses[0],
                    "aliases": ip_addresses[1],
                    "ip_addresses": ip_addresses[2]
                }
            )
            
        except socket.gaierror as e:
            return ConnectivityResult(
                target=hostname,
                test_type="dns_resolution",
                status="failed",
                response_time=time.time() - start_time,
                message=f"DNS解析 {hostname} 失败: {str(e)}",
                details={"hostname": hostname, "error": str(e)}
            )
        except socket.timeout:
            return ConnectivityResult(
                target=hostname,
                test_type="dns_resolution",
                status="timeout",
                response_time=timeout,
                message=f"DNS解析 {hostname} 超时",
                details={"hostname": hostname, "timeout": timeout}
            )
        except Exception as e:
            return ConnectivityResult(
                target=hostname,
                test_type="dns_resolution",
                status="error",
                response_time=time.time() - start_time,
                message=f"DNS解析 {hostname} 异常: {str(e)}",
                details={"hostname": hostname, "error": str(e)}
            )
        finally:
            # 重置默认超时
            socket.setdefaulttimeout(None)
    
    def test_ping(self, host: str, count: int = 3) -> ConnectivityResult:
        """测试PING连通性"""
        start_time = time.time()
        
        try:
            # 根据操作系统选择ping命令
            if os.name == 'nt':  # Windows
                cmd = ['ping', '-n', str(count), host]
            else:  # Unix/Linux
                cmd = ['ping', '-c', str(count), host]
            
            # 执行ping命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=count * 2 + 5  # 动态超时时间
            )
            
            response_time = time.time() - start_time
            
            if result.returncode == 0:
                # 解析ping结果
                output_lines = result.stdout.strip().split('\n')
                
                return ConnectivityResult(
                    target=host,
                    test_type="ping",
                    status="success",
                    response_time=response_time,
                    message=f"PING {host} 成功",
                    details={
                        "host": host,
                        "count": count,
                        "output": output_lines[-2:] if len(output_lines) >= 2 else output_lines,
                        "return_code": result.returncode
                    }
                )
            else:
                return ConnectivityResult(
                    target=host,
                    test_type="ping",
                    status="failed",
                    response_time=response_time,
                    message=f"PING {host} 失败",
                    details={
                        "host": host,
                        "count": count,
                        "return_code": result.returncode,
                        "stderr": result.stderr.strip()
                    }
                )
                
        except subprocess.TimeoutExpired:
            return ConnectivityResult(
                target=host,
                test_type="ping",
                status="timeout",
                response_time=count * 2 + 5,
                message=f"PING {host} 超时",
                details={"host": host, "count": count}
            )
        except FileNotFoundError:
            return ConnectivityResult(
                target=host,
                test_type="ping",
                status="error",
                response_time=time.time() - start_time,
                message=f"PING命令不可用",
                details={"host": host, "error": "ping command not found"}
            )
        except Exception as e:
            return ConnectivityResult(
                target=host,
                test_type="ping",
                status="error",
                response_time=time.time() - start_time,
                message=f"PING {host} 异常: {str(e)}",
                details={"host": host, "error": str(e)}
            )
    
    def test_traceroute(self, host: str, max_hops: int = 10) -> ConnectivityResult:
        """测试路由跟踪"""
        start_time = time.time()
        
        try:
            # 根据操作系统选择traceroute命令
            if os.name == 'nt':  # Windows
                cmd = ['tracert', '-h', str(max_hops), host]
            else:  # Unix/Linux
                cmd = ['traceroute', '-m', str(max_hops), host]
            
            # 执行traceroute命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=max_hops * 3 + 10  # 动态超时时间
            )
            
            response_time = time.time() - start_time
            
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                
                return ConnectivityResult(
                    target=host,
                    test_type="traceroute",
                    status="success",
                    response_time=response_time,
                    message=f"路由跟踪 {host} 完成",
                    details={
                        "host": host,
                        "max_hops": max_hops,
                        "hops": len([line for line in output_lines if line.strip()]),
                        "route_info": output_lines[:5] if len(output_lines) > 5 else output_lines  # 只保留前5跳
                    }
                )
            else:
                return ConnectivityResult(
                    target=host,
                    test_type="traceroute",
                    status="failed",
                    response_time=response_time,
                    message=f"路由跟踪 {host} 失败",
                    details={
                        "host": host,
                        "return_code": result.returncode,
                        "stderr": result.stderr.strip()
                    }
                )
                
        except subprocess.TimeoutExpired:
            return ConnectivityResult(
                target=host,
                test_type="traceroute",
                status="timeout",
                response_time=max_hops * 3 + 10,
                message=f"路由跟踪 {host} 超时",
                details={"host": host, "max_hops": max_hops}
            )
        except FileNotFoundError:
            return ConnectivityResult(
                target=host,
                test_type="traceroute",
                status="error",
                response_time=time.time() - start_time,
                message=f"路由跟踪命令不可用",
                details={"host": host, "error": "traceroute command not found"}
            )
        except Exception as e:
            return ConnectivityResult(
                target=host,
                test_type="traceroute",
                status="error",
                response_time=time.time() - start_time,
                message=f"路由跟踪 {host} 异常: {str(e)}",
                details={"host": host, "error": str(e)}
            )
    
    async def run_all_tests(self) -> List[ConnectivityResult]:
        """运行所有连通性测试"""
        logger.info("=== 开始网络连通性测试 ===")
        
        results = []
        
        # TCP端口测试
        logger.info("--- TCP端口连通性测试 ---")
        tcp_tests = []
        for name, config in self.test_targets.items():
            if 'host' in config and 'port' in config:
                logger.info(f"测试TCP端口: {config['description']}")
                result = self.test_tcp_port(
                    config['host'], 
                    config['port'], 
                    config.get('timeout', 5)
                )
                results.append(result)
                tcp_tests.append((name, result))
                
                if result.status == "success":
                    logger.info(f"✅ {config['description']}: {result.message} ({result.response_time:.3f}s)")
                else:
                    logger.error(f"❌ {config['description']}: {result.message} ({result.response_time:.3f}s)")
        
        # HTTP/HTTPS URL测试
        logger.info("--- HTTP/HTTPS连通性测试 ---")
        http_tasks = []
        for name, config in self.test_targets.items():
            if 'url' in config:
                logger.info(f"测试HTTP URL: {config['description']}")
                task = self.test_http_url(config['url'], config.get('timeout', 10))
                http_tasks.append((name, task))
        
        # 并发执行HTTP测试
        if http_tasks:
            http_results = await asyncio.gather(*[task for _, task in http_tasks], return_exceptions=True)
            
            for (name, _), result in zip(http_tasks, http_results):
                if isinstance(result, Exception):
                    result = ConnectivityResult(
                        target=self.test_targets[name]['url'],
                        test_type="http_url",
                        status="error",
                        response_time=0.0,
                        message=f"HTTP测试异常: {str(result)}"
                    )
                
                results.append(result)
                config = self.test_targets[name]
                
                if result.status == "success":
                    logger.info(f"✅ {config['description']}: {result.message} ({result.response_time:.3f}s)")
                else:
                    logger.error(f"❌ {config['description']}: {result.message} ({result.response_time:.3f}s)")
        
        # DNS解析测试
        logger.info("--- DNS解析测试 ---")
        dns_hosts = ['www.baidu.com', 'api.tushare.pro', 'www.google.com']
        for host in dns_hosts:
            logger.info(f"测试DNS解析: {host}")
            result = self.test_dns_resolution(host)
            results.append(result)
            
            if result.status == "success":
                logger.info(f"✅ DNS解析 {host}: {result.message} ({result.response_time:.3f}s)")
            else:
                logger.error(f"❌ DNS解析 {host}: {result.message} ({result.response_time:.3f}s)")
        
        # PING测试（仅测试关键主机）
        logger.info("--- PING连通性测试 ---")
        ping_hosts = ['8.8.8.8', '114.114.114.114']
        for host in ping_hosts:
            logger.info(f"测试PING: {host}")
            result = self.test_ping(host, count=3)
            results.append(result)
            
            if result.status == "success":
                logger.info(f"✅ PING {host}: {result.message} ({result.response_time:.3f}s)")
            else:
                logger.warning(f"⚠️ PING {host}: {result.message} ({result.response_time:.3f}s)")
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict:
        """生成连通性测试报告"""
        if not self.results:
            asyncio.run(self.run_all_tests())
        
        # 按测试类型分组统计
        test_types = {}
        for result in self.results:
            test_type = result.test_type
            if test_type not in test_types:
                test_types[test_type] = {
                    'total': 0,
                    'success': 0,
                    'failed': 0,
                    'timeout': 0,
                    'error': 0
                }
            
            test_types[test_type]['total'] += 1
            test_types[test_type][result.status] += 1
        
        # 计算整体统计
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.status == "success")
        failed_tests = sum(1 for r in self.results if r.status in ["failed", "timeout", "error"])
        
        # 关键服务连通性检查
        critical_services = ['postgres_port', 'redis_port']
        critical_results = []
        for result in self.results:
            if any(service in result.target for service in critical_services):
                critical_results.append(result)
        
        critical_success = sum(1 for r in critical_results if r.status == "success")
        critical_ready = len(critical_results) > 0 and critical_success == len(critical_results)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "critical_services_ready": critical_ready,
                "overall_connectivity": "good" if successful_tests / total_tests >= 0.8 else "poor"
            },
            "test_type_summary": test_types,
            "critical_services": [
                {
                    "target": result.target,
                    "status": result.status,
                    "message": result.message,
                    "response_time": result.response_time
                }
                for result in critical_results
            ],
            "detailed_results": [
                {
                    "target": result.target,
                    "test_type": result.test_type,
                    "status": result.status,
                    "response_time": result.response_time,
                    "message": result.message,
                    "details": result.details
                }
                for result in self.results
            ]
        }
        
        return report
    
    def save_report(self, filename: str = None) -> str:
        """保存连通性测试报告"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"network_connectivity_{timestamp}.json"
        
        report = self.generate_report()
        
        # 确保目录存在
        os.makedirs("logs", exist_ok=True)
        filepath = os.path.join("logs", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"网络连通性测试报告已保存: {filepath}")
        return filepath

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockSchool 网络连通性测试')
    parser.add_argument('--config', '-c', default='.env.acceptance', 
                       help='配置文件路径 (默认: .env.acceptance)')
    parser.add_argument('--output', '-o', help='输出报告文件名')
    parser.add_argument('--json', action='store_true', help='输出JSON格式结果')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（跳过耗时测试）')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # 创建测试器并运行测试
    tester = NetworkConnectivityTester(args.config)
    
    if args.quick:
        # 快速模式：只测试关键服务
        logger.info("运行快速连通性测试...")
        results = []
        
        # 只测试PostgreSQL和Redis端口
        for name in ['postgres_port', 'redis_port']:
            if name in tester.test_targets:
                config = tester.test_targets[name]
                result = tester.test_tcp_port(config['host'], config['port'], config.get('timeout', 5))
                results.append(result)
        
        tester.results = results
    else:
        # 完整测试
        await tester.run_all_tests()
    
    # 生成报告
    report = tester.generate_report()
    
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        # 打印摘要
        print(f"\n=== 网络连通性测试报告 ===")
        print(f"测试时间: {report['timestamp']}")
        print(f"总测试项: {report['summary']['total_tests']}")
        print(f"成功: {report['summary']['successful_tests']}, 失败: {report['summary']['failed_tests']}")
        print(f"成功率: {report['summary']['success_rate']:.1%}")
        print(f"关键服务就绪: {'✅ 是' if report['summary']['critical_services_ready'] else '❌ 否'}")
        print(f"整体连通性: {report['summary']['overall_connectivity']}")
        
        # 显示关键服务状态
        if report['critical_services']:
            print(f"\n--- 关键服务状态 ---")
            for service in report['critical_services']:
                status_icon = "✅" if service['status'] == "success" else "❌"
                print(f"{status_icon} {service['target']}: {service['message']} ({service['response_time']:.3f}s)")
        
        # 显示测试类型统计
        print(f"\n--- 测试类型统计 ---")
        for test_type, stats in report['test_type_summary'].items():
            success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{test_type}: {stats['success']}/{stats['total']} ({success_rate:.1%})")
    
    # 保存报告
    if args.output or not args.json:
        report_file = tester.save_report(args.output)
        if not args.quiet:
            print(f"\n报告已保存: {report_file}")
    
    # 设置退出码
    sys.exit(0 if report['summary']['critical_services_ready'] else 1)

if __name__ == '__main__':
    asyncio.run(main())