"""
配置管理系统

提供配置文件管理、验证、热更新等功能
"""

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import logging
from contextlib import contextmanager
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .config_loader import ConfigLoader
from .config_validator import ConfigValidator, ValidationRule
from .change_detector import ChangeDetector, ConfigChange
from .error_handling import handle_config_errors, ConfigError, ConfigLoadError

logger = logging.getLogger(__name__)


class ConfigEnvironment(Enum):
    """配置环境枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


# Import from separate modules - removed duplicate definitions


class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监控器"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yml', '.yaml')):
            logger.info(f"配置文件变更: {event.src_path}")
            self.config_manager._handle_file_change(event.src_path)


class ConfigManager:
    """配置管理器
    
    功能:
    - 多环境配置支持
    - 配置文件热更新
    - 配置验证
    - 配置变更历史
    - 配置回滚
    """
    
    def __init__(self, 
                 config_dir: str = "config",
                 environment: Optional[str] = None,
                 enable_hot_reload: bool = True):
        self.config_dir = Path(config_dir)
        self.environment = ConfigEnvironment(environment or os.getenv("ENVIRONMENT", "development"))
        self.enable_hot_reload = enable_hot_reload
        
        # 配置数据
        self._config: Dict[str, Any] = {}
        self._config_lock = threading.RLock()
        
        # 组件初始化
        self._config_loader = ConfigLoader(self.config_dir)
        self._validator = ConfigValidator()
        self._change_detector = ChangeDetector()
        
        # 变更历史
        self._change_history: List[ConfigChange] = []
        
        # 回调函数
        self._change_callbacks: List[Callable[[str, Any, Any], None]] = []
        
        # 文件监控
        self._observer: Optional[Observer] = None
        self._file_watcher: Optional[ConfigFileWatcher] = None
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化配置管理器"""
        # 创建配置目录
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self._load_all_configs()
        
        # 设置默认验证规则
        self._setup_default_validation_rules()
        
        # 启动文件监控
        if self.enable_hot_reload:
            self._start_file_watcher()
    
    @handle_config_errors(log_error=True, error_message="加载配置失败")
    def _load_all_configs(self):
        """加载所有配置文件"""
        with self._config_lock:
            self._config = self._config_loader.load_all_configs(self.environment.value)
    
    def _setup_default_validation_rules(self):
        """设置默认验证规则"""
        self._validation_rules = [
            # 数据同步参数验证
            ConfigValidationRule(
                path="data_sync_params.batch_size",
                required=True,
                data_type=int,
                min_value=1,
                max_value=10000,
                description="数据同步批次大小"
            ),
            ConfigValidationRule(
                path="data_sync_params.retry_times",
                required=True,
                data_type=int,
                min_value=0,
                max_value=10,
                description="重试次数"
            ),
            ConfigValidationRule(
                path="data_sync_params.max_workers",
                required=True,
                data_type=int,
                min_value=1,
                max_value=20,
                description="最大工作线程数"
            ),
            
            # 因子参数验证
            ConfigValidationRule(
                path="factor_params.rsi.window",
                required=True,
                data_type=int,
                min_value=2,
                max_value=100,
                description="RSI窗口期"
            ),
            ConfigValidationRule(
                path="factor_params.ma.windows",
                required=True,
                data_type=list,
                description="移动平均线窗口期列表"
            ),
            
            # 监控参数验证
            ConfigValidationRule(
                path="monitoring_params.collection_interval",
                required=True,
                data_type=int,
                min_value=10,
                max_value=3600,
                description="监控数据收集间隔"
            ),
            
            # API参数验证
            ConfigValidationRule(
                path="api_params.port",
                required=True,
                data_type=int,
                min_value=1000,
                max_value=65535,
                description="API服务端口"
            ),
            
            # 数据库参数验证
            ConfigValidationRule(
                path="database_params.connection_pool_size",
                required=True,
                data_type=int,
                min_value=1,
                max_value=100,
                description="数据库连接池大小"
            ),
        ]
    
    def _start_file_watcher(self):
        """启动文件监控"""
        try:
            self._file_watcher = ConfigFileWatcher(self)
            self._observer = Observer()
            
            # 监控配置目录
            if self.config_dir.exists():
                self._observer.schedule(self._file_watcher, str(self.config_dir), recursive=False)
            
            # 监控主配置文件
            main_config_path = Path("config.yml")
            if main_config_path.exists():
                self._observer.schedule(self._file_watcher, str(main_config_path.parent), recursive=False)
            
            self._observer.start()
            logger.info("配置文件监控已启动")
        except Exception as e:
            logger.error(f"启动文件监控失败: {e}")
    
    def _handle_file_change(self, file_path: str):
        """处理文件变更"""
        try:
            # 延迟处理，避免频繁重载
            threading.Timer(1.0, self._reload_config).start()
        except Exception as e:
            logger.error(f"处理文件变更失败: {e}")
    
    def _reload_config(self):
        """重新加载配置"""
        try:
            old_config = self._config.copy()
            self._load_all_configs()
            
            # 检测变更
            changes = self._change_detector.detect_changes(old_config, self._config)
            
            # 记录变更
            for change in changes:
                self._change_history.append(change)
                logger.info(f"配置变更: {change.path} = {change.new_value}")
                
                # 调用回调函数
                for callback in self._change_callbacks:
                    try:
                        callback(change.path, change.old_value, change.new_value)
                    except Exception as e:
                        logger.error(f"配置变更回调失败: {e}")
            
            # 验证配置
            validation_errors = self.validate_config()
            if validation_errors:
                logger.warning(f"配置验证发现问题: {validation_errors}")
            
        except Exception as e:
            logger.error(f"重新加载配置失败: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值"""
        with self._config_lock:
            if not key_path:
                return self._config
            
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
    
    def set(self, key_path: str, value: Any, source: str = "runtime", user: str = "system"):
        """设置配置值"""
        with self._config_lock:
            old_value = self.get(key_path)
            
            # 设置值
            keys = key_path.split('.')
            current = self._config
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = value
            
            # 记录变更
            change = ConfigChange(
                timestamp=datetime.now(),
                path=key_path,
                old_value=old_value,
                new_value=value,
                source=source,
                user=user
            )
            self._change_history.append(change)
            
            # 调用回调函数
            for callback in self._change_callbacks:
                try:
                    callback(key_path, old_value, value)
                except Exception as e:
                    logger.error(f"配置变更回调失败: {e}")
            
            logger.info(f"配置更新: {key_path} = {value}")
    
    def has(self, key_path: str) -> bool:
        """检查配置项是否存在"""
        with self._config_lock:
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return False
            
            return True
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置段"""
        return self.get(section, {})
    
    @contextmanager
    def config_lock(self):
        """配置锁上下文管理器"""
        with self._config_lock:
            yield
    
    def add_validation_rule(self, rule: ValidationRule):
        """添加验证规则"""
        self._validator.add_rule(rule)
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        return self._validator.validate(self._config)
    
    def add_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """添加配置变更回调"""
        self._change_callbacks.append(callback)
    
    def get_change_history(self, limit: int = 100) -> List[ConfigChange]:
        """获取变更历史"""
        return self._change_history[-limit:]
    
    def rollback_to_timestamp(self, timestamp: datetime) -> bool:
        """回滚到指定时间点"""
        try:
            # 找到时间点之后的所有变更
            changes_to_rollback = [
                change for change in self._change_history
                if change.timestamp > timestamp
            ]
            
            # 按时间倒序回滚
            changes_to_rollback.sort(key=lambda x: x.timestamp, reverse=True)
            
            for change in changes_to_rollback:
                if change.old_value is not None:
                    self.set(change.path, change.old_value, source="rollback")
                else:
                    # 删除新增的配置项
                    self._delete_config_item(change.path)
            
            logger.info(f"配置回滚完成，回滚了 {len(changes_to_rollback)} 个变更")
            return True
            
        except Exception as e:
            logger.error(f"配置回滚失败: {e}")
            return False
    
    def _delete_config_item(self, key_path: str):
        """删除配置项"""
        with self._config_lock:
            keys = key_path.split('.')
            current = self._config
            
            for key in keys[:-1]:
                if key not in current:
                    return
                current = current[key]
            
            if keys[-1] in current:
                del current[keys[-1]]
    
    def export_config(self, file_path: str, format: str = "yaml"):
        """导出配置"""
        try:
            with self._config_lock:
                if format.lower() == "json":
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(self._config, f, indent=2, ensure_ascii=False, default=str)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"配置导出完成: {file_path}")
            
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            raise
    
    def import_config(self, file_path: str, merge: bool = True):
        """导入配置"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_config = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_config = yaml.safe_load(f)
            
            with self._config_lock:
                if merge:
                    self._merge_config(new_config)
                else:
                    self._config = new_config
            
            logger.info(f"配置导入完成: {file_path}")
            
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            raise
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            "environment": self.environment.value,
            "config_dir": str(self.config_dir),
            "hot_reload_enabled": self.enable_hot_reload,
            "config_files_count": len(list(self.config_dir.glob("*.yml"))) if self.config_dir.exists() else 0,
            "validation_rules_count": len(self._validation_rules),
            "change_history_count": len(self._change_history),
            "callbacks_count": len(self._change_callbacks)
        }
    
    def shutdown(self):
        """关闭配置管理器"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
        
        logger.info("配置管理器已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def initialize_config_manager(config_dir: str = "config", 
                            environment: Optional[str] = None,
                            enable_hot_reload: bool = True) -> ConfigManager:
    """初始化配置管理器"""
    global _config_manager
    if _config_manager is not None:
        _config_manager.shutdown()
    
    _config_manager = ConfigManager(
        config_dir=config_dir,
        environment=environment,
        enable_hot_reload=enable_hot_reload
    )
    return _config_manager