#!/usr/bin/env python3
"""
StockSchool Python环境验证工具
用于检查Python 3.11环境和依赖包的安装状态
"""

import os
import sys
import json
import subprocess
import importlib
import platform
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import pkg_resources
from packaging import version

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PackageStatus:
    """包状态数据类"""
    name: str
    required_version: str
    installed_version: Optional[str]
    status: str  # installed, missing, version_mismatch, error
    message: str

@dataclass
class EnvironmentStatus:
    """环境状态数据类"""
    component: str
    status: str
    message: str
    details: Dict = None

class PythonEnvironmentVerifier:
    """Python环境验证器"""
    
    def __init__(self, config_file: str = '.env.acceptance'):
        """初始化验证器"""
        self.config_file = config_file
        self.config = self._load_config()
        self.results = []
        
        # 必需的Python包及其版本要求
        self.required_packages = {
            # 数据库相关
            'psycopg2-binary': '>=2.9.0',
            'redis': '>=4.0.0',
            'sqlalchemy': '>=2.0.0',
            
            # 数据处理
            'pandas': '>=2.0.0',
            'numpy': '>=1.24.0',
            'scipy': '>=1.10.0',
            
            # 机器学习
            'scikit-learn': '>=1.3.0',
            'lightgbm': '>=4.0.0',
            'xgboost': '>=2.0.0',
            
            # API和Web框架
            'fastapi': '>=0.100.0',
            'uvicorn': '>=0.20.0',
            'requests': '>=2.30.0',
            
            # 数据获取
            'tushare': '>=1.2.0',
            'akshare': '>=1.12.0',
            
            # 测试框架
            'pytest': '>=7.0.0',
            'pytest-asyncio': '>=0.20.0',
            
            # 代码质量
            'pylint': '>=3.0.0',
            'flake8': '>=6.0.0',
            'black': '>=23.0.0',
            
            # 工具库
            'pydantic': '>=2.0.0',
            'python-dotenv': '>=1.0.0',
            'click': '>=8.0.0',
            'rich': '>=13.0.0',
            
            # Docker客户端
            'docker': '>=6.0.0'
        }
        
        # 必需的环境变量
        self.required_env_vars = [
            'TUSHARE_TOKEN',
            'AI_API_KEY',
            'DATABASE_URL',
            'REDIS_URL'
        ]
        
        # 可选的环境变量
        self.optional_env_vars = [
            'AI_API_BASE_URL',
            'POSTGRES_PASSWORD',
            'REDIS_PASSWORD'
        ]
    
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
    
    def check_python_version(self) -> EnvironmentStatus:
        """检查Python版本"""
        try:
            python_version = sys.version_info
            version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            # 检查是否为Python 3.11
            if python_version.major == 3 and python_version.minor == 11:
                return EnvironmentStatus(
                    component="Python Version",
                    status="compatible",
                    message=f"Python版本兼容: {version_str}",
                    details={
                        "version": version_str,
                        "implementation": platform.python_implementation(),
                        "compiler": platform.python_compiler(),
                        "executable": sys.executable,
                        "platform": platform.platform()
                    }
                )
            elif python_version.major == 3 and python_version.minor >= 11:
                return EnvironmentStatus(
                    component="Python Version",
                    status="newer",
                    message=f"Python版本较新但兼容: {version_str}",
                    details={
                        "version": version_str,
                        "required": "3.11.x",
                        "compatible": True
                    }
                )
            else:
                return EnvironmentStatus(
                    component="Python Version",
                    status="incompatible",
                    message=f"Python版本不兼容: {version_str} (需要3.11.x)",
                    details={
                        "version": version_str,
                        "required": "3.11.x",
                        "compatible": False
                    }
                )
                
        except Exception as e:
            return EnvironmentStatus(
                component="Python Version",
                status="error",
                message=f"Python版本检查失败: {str(e)}"
            )
    
    def check_pip_version(self) -> EnvironmentStatus:
        """检查pip版本"""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                pip_info = result.stdout.strip()
                # 提取版本号
                pip_version = pip_info.split()[1]
                
                return EnvironmentStatus(
                    component="Pip Version",
                    status="available",
                    message=f"pip可用: {pip_version}",
                    details={
                        "version": pip_version,
                        "full_info": pip_info,
                        "executable": sys.executable
                    }
                )
            else:
                return EnvironmentStatus(
                    component="Pip Version",
                    status="error",
                    message=f"pip检查失败: {result.stderr}",
                    details={"error": result.stderr}
                )
                
        except subprocess.TimeoutExpired:
            return EnvironmentStatus(
                component="Pip Version",
                status="timeout",
                message="pip版本检查超时"
            )
        except Exception as e:
            return EnvironmentStatus(
                component="Pip Version",
                status="error",
                message=f"pip版本检查异常: {str(e)}"
            )
    
    def check_package_installation(self, package_name: str, required_version: str) -> PackageStatus:
        """检查单个包的安装状态"""
        try:
            # 尝试导入包
            try:
                if package_name == 'psycopg2-binary':
                    # psycopg2-binary导入时使用psycopg2
                    import psycopg2
                    installed_version = psycopg2.__version__
                elif package_name == 'python-dotenv':
                    # python-dotenv导入时使用dotenv
                    import dotenv
                    installed_version = dotenv.__version__
                else:
                    # 标准导入
                    module = importlib.import_module(package_name.replace('-', '_'))
                    installed_version = getattr(module, '__version__', 'unknown')
            except ImportError:
                # 如果导入失败，尝试通过pkg_resources获取版本
                try:
                    distribution = pkg_resources.get_distribution(package_name)
                    installed_version = distribution.version
                except pkg_resources.DistributionNotFound:
                    return PackageStatus(
                        name=package_name,
                        required_version=required_version,
                        installed_version=None,
                        status="missing",
                        message=f"包 {package_name} 未安装"
                    )
            
            # 检查版本兼容性
            if installed_version == 'unknown':
                return PackageStatus(
                    name=package_name,
                    required_version=required_version,
                    installed_version=installed_version,
                    status="version_unknown",
                    message=f"包 {package_name} 已安装但版本未知"
                )
            
            # 解析版本要求
            if required_version.startswith('>='):
                min_version = required_version[2:]
                if version.parse(installed_version) >= version.parse(min_version):
                    return PackageStatus(
                        name=package_name,
                        required_version=required_version,
                        installed_version=installed_version,
                        status="installed",
                        message=f"包 {package_name} 版本兼容: {installed_version}"
                    )
                else:
                    return PackageStatus(
                        name=package_name,
                        required_version=required_version,
                        installed_version=installed_version,
                        status="version_mismatch",
                        message=f"包 {package_name} 版本过低: {installed_version} (需要{required_version})"
                    )
            else:
                # 精确版本匹配
                if installed_version == required_version:
                    return PackageStatus(
                        name=package_name,
                        required_version=required_version,
                        installed_version=installed_version,
                        status="installed",
                        message=f"包 {package_name} 版本匹配: {installed_version}"
                    )
                else:
                    return PackageStatus(
                        name=package_name,
                        required_version=required_version,
                        installed_version=installed_version,
                        status="version_mismatch",
                        message=f"包 {package_name} 版本不匹配: {installed_version} (需要{required_version})"
                    )
                    
        except Exception as e:
            return PackageStatus(
                name=package_name,
                required_version=required_version,
                installed_version=None,
                status="error",
                message=f"包 {package_name} 检查失败: {str(e)}"
            )
    
    def check_all_packages(self) -> List[PackageStatus]:
        """检查所有必需包的安装状态"""
        logger.info("开始检查Python包安装状态...")
        
        package_results = []
        
        for package_name, required_version in self.required_packages.items():
            logger.info(f"检查包: {package_name}")
            result = self.check_package_installation(package_name, required_version)
            package_results.append(result)
            
            if result.status == "installed":
                logger.info(f"✅ {package_name}: {result.message}")
            else:
                logger.warning(f"⚠️ {package_name}: {result.message}")
        
        return package_results
    
    def check_environment_variables(self) -> EnvironmentStatus:
        """检查环境变量配置"""
        try:
            missing_required = []
            missing_optional = []
            configured_vars = {}
            
            # 检查必需的环境变量
            for var_name in self.required_env_vars:
                value = os.getenv(var_name) or self.config.get(var_name)
                if value:
                    # 隐藏敏感信息
                    if 'TOKEN' in var_name or 'KEY' in var_name or 'PASSWORD' in var_name:
                        configured_vars[var_name] = f"{'*' * (len(value) - 4)}{value[-4:]}" if len(value) > 4 else "***"
                    else:
                        configured_vars[var_name] = value
                else:
                    missing_required.append(var_name)
            
            # 检查可选的环境变量
            for var_name in self.optional_env_vars:
                value = os.getenv(var_name) or self.config.get(var_name)
                if value:
                    if 'TOKEN' in var_name or 'KEY' in var_name or 'PASSWORD' in var_name:
                        configured_vars[var_name] = f"{'*' * (len(value) - 4)}{value[-4:]}" if len(value) > 4 else "***"
                    else:
                        configured_vars[var_name] = value
                else:
                    missing_optional.append(var_name)
            
            if not missing_required:
                return EnvironmentStatus(
                    component="Environment Variables",
                    status="configured",
                    message=f"所有必需环境变量已配置 ({len(configured_vars)}/{len(self.required_env_vars + self.optional_env_vars)})",
                    details={
                        "configured_vars": configured_vars,
                        "missing_optional": missing_optional,
                        "total_required": len(self.required_env_vars),
                        "total_optional": len(self.optional_env_vars)
                    }
                )
            else:
                return EnvironmentStatus(
                    component="Environment Variables",
                    status="incomplete",
                    message=f"缺少必需环境变量: {', '.join(missing_required)}",
                    details={
                        "configured_vars": configured_vars,
                        "missing_required": missing_required,
                        "missing_optional": missing_optional
                    }
                )
                
        except Exception as e:
            return EnvironmentStatus(
                component="Environment Variables",
                status="error",
                message=f"环境变量检查失败: {str(e)}"
            )
    
    def check_system_resources(self) -> EnvironmentStatus:
        """检查系统资源"""
        try:
            import psutil
            
            # 获取系统信息
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # 检查资源是否满足要求
            memory_gb = memory.total / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            
            warnings = []
            if memory_gb < 8:
                warnings.append(f"内存较少: {memory_gb:.1f}GB (建议8GB以上)")
            if disk_free_gb < 10:
                warnings.append(f"磁盘空间不足: {disk_free_gb:.1f}GB (建议10GB以上)")
            if cpu_count < 4:
                warnings.append(f"CPU核心较少: {cpu_count}核 (建议4核以上)")
            
            return EnvironmentStatus(
                component="System Resources",
                status="adequate" if not warnings else "limited",
                message="系统资源充足" if not warnings else f"系统资源有限: {'; '.join(warnings)}",
                details={
                    "cpu_cores": cpu_count,
                    "memory_total_gb": round(memory_gb, 1),
                    "memory_available_gb": round(memory.available / (1024**3), 1),
                    "memory_usage_percent": memory.percent,
                    "disk_total_gb": round(disk.total / (1024**3), 1),
                    "disk_free_gb": round(disk_free_gb, 1),
                    "disk_usage_percent": round((disk.used / disk.total) * 100, 1),
                    "warnings": warnings
                }
            )
            
        except ImportError:
            return EnvironmentStatus(
                component="System Resources",
                status="unavailable",
                message="无法检查系统资源 (psutil未安装)"
            )
        except Exception as e:
            return EnvironmentStatus(
                component="System Resources",
                status="error",
                message=f"系统资源检查失败: {str(e)}"
            )
    
    def check_python_path(self) -> EnvironmentStatus:
        """检查Python路径配置"""
        try:
            python_path = sys.path
            current_dir = os.getcwd()
            
            # 检查当前目录是否在Python路径中
            current_in_path = current_dir in python_path or '.' in python_path
            
            # 检查是否有虚拟环境
            in_venv = hasattr(sys, 'real_prefix') or (
                hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
            )
            
            return EnvironmentStatus(
                component="Python Path",
                status="configured",
                message=f"Python路径配置正常 ({'虚拟环境' if in_venv else '系统环境'})",
                details={
                    "executable": sys.executable,
                    "prefix": sys.prefix,
                    "base_prefix": getattr(sys, 'base_prefix', sys.prefix),
                    "virtual_env": in_venv,
                    "current_dir_in_path": current_in_path,
                    "python_path_count": len(python_path),
                    "working_directory": current_dir
                }
            )
            
        except Exception as e:
            return EnvironmentStatus(
                component="Python Path",
                status="error",
                message=f"Python路径检查失败: {str(e)}"
            )
    
    def run_all_checks(self) -> Tuple[List[EnvironmentStatus], List[PackageStatus]]:
        """运行所有环境检查"""
        logger.info("=== 开始Python环境验证 ===")
        
        # 环境检查
        env_checks = [
            ("Python版本", self.check_python_version),
            ("Pip版本", self.check_pip_version),
            ("环境变量", self.check_environment_variables),
            ("系统资源", self.check_system_resources),
            ("Python路径", self.check_python_path)
        ]
        
        env_results = []
        
        for check_name, check_func in env_checks:
            logger.info(f"正在检查: {check_name}")
            try:
                result = check_func()
                env_results.append(result)
                
                if result.status in ["compatible", "available", "configured", "adequate"]:
                    logger.info(f"✅ {check_name}: {result.message}")
                elif result.status in ["newer", "limited", "incomplete"]:
                    logger.warning(f"⚠️ {check_name}: {result.message}")
                else:
                    logger.error(f"❌ {check_name}: {result.message}")
                    
            except Exception as e:
                error_result = EnvironmentStatus(
                    component=check_name,
                    status="exception",
                    message=f"检查过程异常: {str(e)}"
                )
                env_results.append(error_result)
                logger.error(f"💥 {check_name}: 检查过程异常: {str(e)}")
        
        # 包检查
        package_results = self.check_all_packages()
        
        return env_results, package_results
    
    def generate_report(self) -> Dict:
        """生成验证报告"""
        env_results, package_results = self.run_all_checks()
        
        # 统计环境检查结果
        env_total = len(env_results)
        env_passed = sum(1 for r in env_results if r.status in ["compatible", "available", "configured", "adequate", "newer"])
        env_warnings = sum(1 for r in env_results if r.status in ["limited", "incomplete"])
        env_failed = env_total - env_passed - env_warnings
        
        # 统计包检查结果
        pkg_total = len(package_results)
        pkg_installed = sum(1 for r in package_results if r.status == "installed")
        pkg_missing = sum(1 for r in package_results if r.status == "missing")
        pkg_version_issues = sum(1 for r in package_results if r.status == "version_mismatch")
        pkg_errors = pkg_total - pkg_installed - pkg_missing - pkg_version_issues
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "python_info": {
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
                "platform": platform.platform()
            },
            "summary": {
                "environment": {
                    "total_checks": env_total,
                    "passed": env_passed,
                    "warnings": env_warnings,
                    "failed": env_failed,
                    "success_rate": env_passed / env_total if env_total > 0 else 0
                },
                "packages": {
                    "total_packages": pkg_total,
                    "installed": pkg_installed,
                    "missing": pkg_missing,
                    "version_issues": pkg_version_issues,
                    "errors": pkg_errors,
                    "install_rate": pkg_installed / pkg_total if pkg_total > 0 else 0
                },
                "overall_ready": env_failed == 0 and pkg_missing == 0 and pkg_version_issues == 0
            },
            "environment_checks": [
                {
                    "component": result.component,
                    "status": result.status,
                    "message": result.message,
                    "details": result.details
                }
                for result in env_results
            ],
            "package_checks": [
                {
                    "name": result.name,
                    "required_version": result.required_version,
                    "installed_version": result.installed_version,
                    "status": result.status,
                    "message": result.message
                }
                for result in package_results
            ]
        }
        
        return report
    
    def save_report(self, filename: str = None) -> str:
        """保存验证报告"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"python_env_verification_{timestamp}.json"
        
        report = self.generate_report()
        
        # 确保目录存在
        os.makedirs("logs", exist_ok=True)
        filepath = os.path.join("logs", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"环境验证报告已保存: {filepath}")
        return filepath

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='StockSchool Python环境验证')
    parser.add_argument('--config', '-c', default='.env.acceptance', 
                       help='配置文件路径 (默认: .env.acceptance)')
    parser.add_argument('--output', '-o', help='输出报告文件名')
    parser.add_argument('--json', action='store_true', help='输出JSON格式结果')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    parser.add_argument('--install-missing', action='store_true', help='自动安装缺失的包')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # 创建验证器并运行检查
    verifier = PythonEnvironmentVerifier(args.config)
    report = verifier.generate_report()
    
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        # 打印摘要
        print(f"\n=== Python环境验证报告 ===")
        print(f"检查时间: {report['timestamp']}")
        print(f"Python版本: {report['python_info']['version']} ({report['python_info']['implementation']})")
        print(f"运行平台: {report['python_info']['platform']}")
        
        # 环境检查摘要
        env_summary = report['summary']['environment']
        print(f"\n--- 环境检查 ---")
        print(f"总检查项: {env_summary['total_checks']}")
        print(f"通过: {env_summary['passed']}, 警告: {env_summary['warnings']}, 失败: {env_summary['failed']}")
        print(f"成功率: {env_summary['success_rate']:.1%}")
        
        # 包检查摘要
        pkg_summary = report['summary']['packages']
        print(f"\n--- 包检查 ---")
        print(f"总包数: {pkg_summary['total_packages']}")
        print(f"已安装: {pkg_summary['installed']}, 缺失: {pkg_summary['missing']}, 版本问题: {pkg_summary['version_issues']}")
        print(f"安装率: {pkg_summary['install_rate']:.1%}")
        
        print(f"\n整体状态: {'✅ 就绪' if report['summary']['overall_ready'] else '❌ 未就绪'}")
        
        # 显示问题详情
        if not report['summary']['overall_ready']:
            print(f"\n=== 需要解决的问题 ===")
            
            # 环境问题
            for check in report['environment_checks']:
                if check['status'] not in ["compatible", "available", "configured", "adequate", "newer"]:
                    print(f"❌ {check['component']}: {check['message']}")
            
            # 包问题
            missing_packages = []
            version_issues = []
            for pkg in report['package_checks']:
                if pkg['status'] == 'missing':
                    missing_packages.append(pkg['name'])
                elif pkg['status'] == 'version_mismatch':
                    version_issues.append(f"{pkg['name']} (需要{pkg['required_version']}, 当前{pkg['installed_version']})")
            
            if missing_packages:
                print(f"\n缺失的包: {', '.join(missing_packages)}")
                if args.install_missing:
                    print("正在安装缺失的包...")
                    for pkg in missing_packages:
                        try:
                            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg], check=True)
                            print(f"✅ 已安装: {pkg}")
                        except subprocess.CalledProcessError as e:
                            print(f"❌ 安装失败: {pkg} - {e}")
                else:
                    print(f"建议运行: pip install {' '.join(missing_packages)}")
            
            if version_issues:
                print(f"\n版本问题: {', '.join(version_issues)}")
    
    # 保存报告
    if args.output or not args.json:
        report_file = verifier.save_report(args.output)
        if not args.quiet:
            print(f"\n报告已保存: {report_file}")
    
    # 设置退出码
    sys.exit(0 if report['summary']['overall_ready'] else 1)

if __name__ == '__main__':
    main()