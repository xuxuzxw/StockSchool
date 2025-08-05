import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from src.config.change_detector import ConfigChange
from src.config.manager import ConfigManager
from src.config.validators import ValidationRule

"""
重构后的配置管理器单元测试
"""




class TestConfigManagerRefactored:
    """重构后的配置管理器测试"""

    @pytest.fixture
    def temp_config_dir(self):
        """临时配置目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()

            # 创建基础配置
            base_config = {
                "data_sync_params": {
                    "batch_size": 1000,
                    "retry_times": 3
                }
            }
            with open(config_dir / "base.yml", 'w') as f:
                yaml.dump(base_config, f)

            # 创建开发环境配置
            dev_config = {
                "data_sync_params": {
                    "max_workers": 5
                },
                "api_params": {
                    "port": 8000
                }
            }
            with open(config_dir / "development.yml", 'w') as f:
                yaml.dump(dev_config, f)

            yield str(config_dir)

    def test_config_loading(self, temp_config_dir):
        """测试配置加载"""
        manager = ConfigManager(
            config_dir=temp_config_dir,
            environment="development",
            enable_hot_reload=False
        )

        # 验证配置合并
        assert manager.get("data_sync_params.batch_size") == 1000
        assert manager.get("data_sync_params.retry_times") == 3
        assert manager.get("data_sync_params.max_workers") == 5
        assert manager.get("api_params.port") == 8000

    def test_config_validation(self, temp_config_dir):
        """测试配置验证"""
        manager = ConfigManager(
            config_dir=temp_config_dir,
            environment="development",
            enable_hot_reload=False
        )

        # 添加自定义验证规则
        manager.add_validation_rule(ValidationRule(
            path="factor_params.rsi.window",
            required=True,
            data_type=int,
            min_value=2,
            max_value=100
        ))

        # 验证应该失败（缺少必需配置）
        errors = manager.validate_config()
        assert any("factor_params.rsi.window" in error for error in errors)

    def test_config_change_detection(self, temp_config_dir):
        """测试配置变更检测"""
        manager = ConfigManager(
            config_dir=temp_config_dir,
            environment="development",
            enable_hot_reload=False
        )

        # 设置变更回调
        changes_received = []
        def on_change(path, old_value, new_value):
            """方法描述"""

        manager.add_change_callback(on_change)

        # 修改配置
        manager.set("data_sync_params.batch_size", 2000)

        # 验证变更记录
        assert len(changes_received) == 1
        assert changes_received[0] == ("data_sync_params.batch_size", 1000, 2000)

        # 验证变更历史
        history = manager.get_change_history()
        assert len(history) == 1
        assert history[0].path == "data_sync_params.batch_size"

    def test_thread_safety(self, temp_config_dir):
        """测试线程安全"""
        import threading
        import time

        manager = ConfigManager(
            config_dir=temp_config_dir,
            environment="development",
            enable_hot_reload=False
        )

        results = []

        def worker(worker_id):
            """方法描述"""
                manager.set(f"test.worker_{worker_id}", i)
                value = manager.get(f"test.worker_{worker_id}")
                results.append((worker_id, i, value))
                time.sleep(0.001)

        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(results) == 50
        for worker_id, expected_value, actual_value in results:
            assert expected_value == actual_value

    def test_context_manager(self, temp_config_dir):
        """测试上下文管理器"""
        manager = ConfigManager(
            config_dir=temp_config_dir,
            environment="development",
            enable_hot_reload=False
        )

        with manager.config_lock():
            # 在锁保护下进行多个操作
            manager.set("test.value1", 100)
            manager.set("test.value2", 200)

            assert manager.get("test.value1") == 100
            assert manager.get("test.value2") == 200

    def test_config_export_import(self, temp_config_dir):
        """测试配置导出导入"""
        manager = ConfigManager(
            config_dir=temp_config_dir,
            environment="development",
            enable_hot_reload=False
        )

        # 修改一些配置
        manager.set("test.export_value", "test_data")

        # 导出配置
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            export_path = f.name

        manager.export_config(export_path)

        # 创建新的管理器并导入配置
        new_manager = ConfigManager(enable_hot_reload=False)
        new_manager.import_config(export_path)

        # 验证导入的配置
        assert new_manager.get("test.export_value") == "test_data"

        # 清理
        Path(export_path).unlink()

    def test_rollback_functionality(self, temp_config_dir):
        """测试配置回滚功能"""
        manager = ConfigManager(
            config_dir=temp_config_dir,
            environment="development",
            enable_hot_reload=False
        )

        # 记录初始值
        initial_value = manager.get("data_sync_params.batch_size")

        # 进行一些修改
        import time
        timestamp_before_changes = manager.get_change_history()[-1].timestamp if manager.get_change_history() else None

        time.sleep(0.1)  # 确保时间戳不同

        manager.set("data_sync_params.batch_size", 2000)
        manager.set("data_sync_params.retry_times", 5)

        # 验证修改生效
        assert manager.get("data_sync_params.batch_size") == 2000
        assert manager.get("data_sync_params.retry_times") == 5

        # 回滚到修改前
        if timestamp_before_changes:
            success = manager.rollback_to_timestamp(timestamp_before_changes)
            assert success

            # 验证回滚结果
            assert manager.get("data_sync_params.batch_size") == initial_value
            assert manager.get("data_sync_params.retry_times") == 3  # 原始值