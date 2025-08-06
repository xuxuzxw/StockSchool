"""
基础设施验收阶段 - 充分利用现有的Docker服务检查脚本
"""

import os
import sys
from typing import List, Dict, Any

# 添加scripts目录到路径，以便导入现有脚本
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts'))

from ..core.base_phase import BaseTestPhase
from ..core.models import TestResult, TestStatus
from ..core.exceptions import InfrastructureError, DatabaseConnectionError, RedisConnectionError

# 导入现有的Docker服务检查器
try:
    from check_docker_services import DockerServiceChecker
except ImportError as e:
    raise ImportError(f"无法导入现有的Docker服务检查器: {e}")


class InfrastructurePhase(BaseTestPhase):
    """基础设施验收阶段 - 利用现有的Docker服务检查脚本"""
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        super().__init__(phase_name, config)
        
        # 初始化Docker服务检查器，复用现有代码
        config_file = config.get('config_file', '.env.acceptance')
        self.docker_checker = DockerServiceChecker(config_file)
        
        self.logger.info("基础设施验收阶段初始化完成，将使用现有的Docker服务检查器")
    
    def _run_tests(self) -> List[TestResult]:
        """执行基础设施验收测试"""
        test_results = []
        
        # 验证前提条件
        if not self._validate_prerequisites():
            test_results.append(TestResult(
                phase=self.phase_name,
                test_name="prerequisites_validation",
                status=TestStatus.FAILED,
                execution_time=0.0,
                error_message="基础设施验收前提条件验证失败"
            ))
            return test_results
        
        # 1. Docker守护进程检查
        test_results.append(
            self._execute_test(
                "docker_daemon_check",
                self._test_docker_daemon
            )
        )
        
        # 2. PostgreSQL容器检查
        test_results.append(
            self._execute_test(
                "postgres_container_check",
                self._test_postgres_container
            )
        )
        
        # 3. Redis容器检查
        test_results.append(
            self._execute_test(
                "redis_container_check", 
                self._test_redis_container
            )
        )
        
        # 4. 网络连通性检查
        test_results.append(
            self._execute_test(
                "network_connectivity_check",
                self._test_network_connectivity
            )
        )
        
        # 5. PostgreSQL数据库连接检查
        test_results.append(
            self._execute_test(
                "postgres_connection_check",
                self._test_postgres_connection
            )
        )
        
        # 6. Redis连接检查
        test_results.append(
            self._execute_test(
                "redis_connection_check",
                self._test_redis_connection
            )
        )
        
        # 7. 环境变量检查
        test_results.append(
            self._execute_test(
                "environment_variables_check",
                self._test_environment_variables
            )
        )
        
        # 8. Python依赖检查
        test_results.append(
            self._execute_test(
                "python_dependencies_check",
                self._test_python_dependencies
            )
        )
        
        return test_results
    
    def _test_docker_daemon(self) -> Dict[str, Any]:
        """测试Docker守护进程状态 - 利用现有代码"""
        self.logger.info("检查Docker守护进程状态")
        
        # 使用现有的Docker检查器
        status = self.docker_checker.check_docker_daemon()
        
        if not status.healthy:
            if status.status == "not_found":
                raise InfrastructureError("Docker未安装或不在PATH中")
            elif status.status == "timeout":
                raise InfrastructureError("Docker守护进程响应超时")
            else:
                raise InfrastructureError(f"Docker守护进程异常: {status.message}")
        
        return {
            "status": status.status,
            "message": status.message,
            "response_time": status.response_time,
            "details": status.details
        }
    
    def _test_postgres_container(self) -> Dict[str, Any]:
        """测试PostgreSQL容器状态 - 利用现有代码"""
        self.logger.info("检查PostgreSQL容器状态")
        
        # 使用现有的容器检查器
        status = self.docker_checker.check_container_status('postgres')
        
        if not status.healthy:
            if status.status == "not_found":
                raise InfrastructureError("PostgreSQL容器不存在，请先启动Docker Compose")
            elif status.status == "unhealthy":
                raise InfrastructureError(f"PostgreSQL容器健康检查失败: {status.message}")
            else:
                raise InfrastructureError(f"PostgreSQL容器状态异常: {status.message}")
        
        return {
            "container_status": status.status,
            "message": status.message,
            "response_time": status.response_time,
            "details": status.details
        }
    
    def _test_redis_container(self) -> Dict[str, Any]:
        """测试Redis容器状态 - 利用现有代码"""
        self.logger.info("检查Redis容器状态")
        
        # 使用现有的容器检查器
        status = self.docker_checker.check_container_status('redis')
        
        if not status.healthy:
            if status.status == "not_found":
                raise InfrastructureError("Redis容器不存在，请先启动Docker Compose")
            elif status.status == "unhealthy":
                raise InfrastructureError(f"Redis容器健康检查失败: {status.message}")
            else:
                raise InfrastructureError(f"Redis容器状态异常: {status.message}")
        
        return {
            "container_status": status.status,
            "message": status.message,
            "response_time": status.response_time,
            "details": status.details
        }
    
    def _test_network_connectivity(self) -> Dict[str, Any]:
        """测试网络连通性 - 利用现有代码"""
        self.logger.info("检查网络连通性")
        
        # 使用现有的网络连通性检查器
        status = self.docker_checker.check_network_connectivity()
        
        if not status.healthy:
            raise InfrastructureError(f"网络连通性检查失败: {status.message}")
        
        return {
            "connectivity_status": status.status,
            "message": status.message,
            "response_time": status.response_time,
            "details": status.details
        }
    
    def _test_postgres_connection(self) -> Dict[str, Any]:
        """测试PostgreSQL数据库连接 - 利用现有代码"""
        self.logger.info("检查PostgreSQL数据库连接")
        
        # 使用现有的数据库连接检查器
        status = self.docker_checker.check_postgres_connection()
        
        if not status.healthy:
            if status.status == "config_error":
                raise DatabaseConnectionError("PostgreSQL配置错误: 未找到DATABASE_URL")
            elif status.status == "connection_error":
                raise DatabaseConnectionError(f"PostgreSQL连接失败: {status.message}")
            else:
                raise DatabaseConnectionError(f"PostgreSQL检查失败: {status.message}")
        
        return {
            "connection_status": status.status,
            "message": status.message,
            "response_time": status.response_time,
            "details": status.details
        }
    
    def _test_redis_connection(self) -> Dict[str, Any]:
        """测试Redis连接 - 利用现有代码"""
        self.logger.info("检查Redis连接")
        
        # 使用现有的Redis连接检查器
        status = self.docker_checker.check_redis_connection()
        
        if not status.healthy:
            if status.status == "config_error":
                raise RedisConnectionError("Redis配置错误: 未找到REDIS_URL")
            elif status.status == "connection_error":
                raise RedisConnectionError(f"Redis连接失败: {status.message}")
            else:
                raise RedisConnectionError(f"Redis检查失败: {status.message}")
        
        return {
            "connection_status": status.status,
            "message": status.message,
            "response_time": status.response_time,
            "details": status.details
        }
    
    def _test_environment_variables(self) -> Dict[str, Any]:
        """测试环境变量配置"""
        self.logger.info("检查关键环境变量配置")
        
        required_env_vars = {
            'TUSHARE_TOKEN': 'Tushare API访问令牌',
            'DATABASE_URL': '数据库连接URL',
            'REDIS_URL': 'Redis连接URL'
        }
        
        optional_env_vars = {
            'AI_API_KEY': '外接AI API密钥',
            'AI_API_BASE_URL': '外接AI API基础URL'
        }
        
        missing_required = []
        missing_optional = []
        configured_vars = {}
        
        # 检查必需的环境变量
        for var_name, description in required_env_vars.items():
            value = os.getenv(var_name) or self.config.get(var_name.lower())
            if value:
                # 隐藏敏感信息
                if 'token' in var_name.lower() or 'key' in var_name.lower():
                    configured_vars[var_name] = f"{value[:8]}..." if len(value) > 8 else "***"
                else:
                    configured_vars[var_name] = value
            else:
                missing_required.append(f"{var_name} ({description})")
        
        # 检查可选的环境变量
        for var_name, description in optional_env_vars.items():
            value = os.getenv(var_name) or self.config.get(var_name.lower())
            if value:
                if 'token' in var_name.lower() or 'key' in var_name.lower():
                    configured_vars[var_name] = f"{value[:8]}..." if len(value) > 8 else "***"
                else:
                    configured_vars[var_name] = value
            else:
                missing_optional.append(f"{var_name} ({description})")
        
        if missing_required:
            raise InfrastructureError(f"缺少必需的环境变量: {', '.join(missing_required)}")
        
        return {
            "configured_variables": configured_vars,
            "missing_optional": missing_optional,
            "total_required": len(required_env_vars),
            "configured_required": len(required_env_vars) - len(missing_required)
        }
    
    def _test_python_dependencies(self) -> Dict[str, Any]:
        """测试Python依赖包"""
        self.logger.info("检查Python依赖包")
        
        # 利用现有的Python环境验证脚本
        try:
            # 导入现有的Python环境验证脚本
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts'))
            from verify_python_env import verify_python_environment
            
            # 执行验证
            verification_result = verify_python_environment()
            
            if not verification_result.get('success', False):
                missing_packages = verification_result.get('missing_packages', [])
                if missing_packages:
                    raise InfrastructureError(f"缺少必需的Python包: {', '.join(missing_packages)}")
                else:
                    raise InfrastructureError("Python环境验证失败")
            
            return {
                "python_version": verification_result.get('python_version'),
                "installed_packages": verification_result.get('installed_packages', {}),
                "verification_status": "success"
            }
            
        except ImportError:
            # 如果没有现有的验证脚本，执行基本检查
            self.logger.warning("未找到现有的Python环境验证脚本，执行基本检查")
            return self._basic_python_check()
    
    def _basic_python_check(self) -> Dict[str, Any]:
        """基本的Python环境检查"""
        import sys
        import pkg_resources
        
        # 检查Python版本
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        if sys.version_info < (3, 11):
            raise InfrastructureError(f"Python版本过低: {python_version}，需要Python 3.11+")
        
        # 检查关键依赖包
        critical_packages = [
            'psycopg2', 'redis', 'docker', 'pandas', 'numpy', 
            'sqlalchemy', 'fastapi', 'uvicorn', 'pydantic'
        ]
        
        installed_packages = {}
        missing_packages = []
        
        for package in critical_packages:
            try:
                dist = pkg_resources.get_distribution(package)
                installed_packages[package] = dist.version
            except pkg_resources.DistributionNotFound:
                missing_packages.append(package)
        
        if missing_packages:
            raise InfrastructureError(f"缺少关键Python包: {', '.join(missing_packages)}")
        
        return {
            "python_version": python_version,
            "installed_packages": installed_packages,
            "verification_status": "basic_check_passed"
        }
    
    def _validate_prerequisites(self) -> bool:
        """验证测试前提条件"""
        try:
            # 检查配置文件是否存在
            config_file = self.config.get('config_file', '.env.acceptance')
            if not os.path.exists(config_file):
                self.logger.warning(f"配置文件 {config_file} 不存在，将使用默认配置")
            
            # 检查Docker是否可用
            import subprocess
            try:
                subprocess.run(['docker', '--version'], 
                             capture_output=True, check=True, timeout=5)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                self.logger.error("Docker不可用，无法执行基础设施验收测试")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"前提条件验证失败: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """清理测试资源"""
        try:
            # 清理Docker客户端连接
            if hasattr(self.docker_checker, 'docker_client') and self.docker_checker.docker_client:
                self.docker_checker.docker_client.close()
            
            self.logger.info("基础设施验收阶段资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")