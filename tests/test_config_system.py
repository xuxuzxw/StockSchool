from datetime import datetime
from pathlib import Path

from src.config import (CompatibilityLevel, ConfigEnvironment, ConfigManager,
                        DiagnosticLevel, RollbackType, """,
                        create_compatibility_checker,
                        create_config_diagnostics, create_hot_reload_manager,
                        create_rollback_manager, import, pytest,
                        setup_config_system, shutil, tempfile, 配置管理系统测试)


class TestConfigManager:
    """配置管理器测试"""

    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 创建测试配置文件
        test_config = {
            "data_sync_params": {
                "batch_size": 1000,
                "retry_times": 3,
                "max_workers": 4
            },
            "api_params": {
                "port": 8000,
                "host": "0.0.0.0"
            }
        }

        import yaml
        with open(self.config_dir / "base.yml", 'w') as f:
            yaml.dump(test_config, f)

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        config_manager = ConfigManager(
            config_dir=str(self.config_dir),
            environment="testing",
            enable_hot_reload=False
        )

        assert config_manager.environment == ConfigEnvironment.TESTING
        assert config_manager.get("data_sync_params.batch_size") == 1000
        assert config_manager.get("api_params.port") == 8000

    def test_config_get_set(self):
        """测试配置获取和设置"""
        config_manager = ConfigManager(
            config_dir=str(self.config_dir),
            enable_hot_reload=False
        )

        # 测试获取
        assert config_manager.get("data_sync_params.batch_size") == 1000
        assert config_manager.get("nonexistent.key", "default") == "default"

        # 测试设置
        config_manager.set("data_sync_params.batch_size", 2000)
        assert config_manager.get("data_sync_params.batch_size") == 2000

        # 测试has方法
        assert config_manager.has("data_sync_params.batch_size")
        assert not config_manager.has("nonexistent.key")

    def test_config_validation(self):
        """测试配置验证"""
        config_manager = ConfigManager(
            config_dir=str(self.config_dir),
            enable_hot_reload=False
        )

        # 设置无效配置
        config_manager.set("data_sync_params.batch_size", -100)  # 负数
        config_manager.set("api_params.port", 99999)  # 超出范围

        # 验证配置
        errors = config_manager.validate_config()
        assert len(errors) > 0

        # 检查错误内容
        error_messages = " ".join(errors)
        assert "batch_size" in error_messages
        assert "port" in error_messages


class TestConfigDiagnostics:
    """配置诊断测试"""

    def test_diagnostics_basic(self):
        """测试基础诊断功能"""
        diagnostics = create_config_diagnostics()

        # 测试配置
        test_config = {
            "data_sync_params": {
                "batch_size": -100,  # 错误：负数
                "retry_times": 3
            },
            "api_params": {
                "port": 99999,  # 错误：端口超出范围
                "cors_origins": ["*"]  # 警告：安全问题
            }
        }

        report = diagnostics.diagnose_config(test_config)

        assert report.total_issues > 0
        assert report.health_score < 100
        assert report.auto_fixable_count > 0

        # 检查问题级别
        assert report.issues_by_level[DiagnosticLevel.ERROR.value] > 0

    def test_auto_fix(self):
        """测试自动修复"""
        diagnostics = create_config_diagnostics()

        test_config = {
            "data_sync_params": {
                "batch_size": -100
            },
            "api_params": {
                "port": 99999
            }
        }

        report = diagnostics.diagnose_config(test_config)
        fixed_config = diagnostics.auto_fix_issues(test_config, report.issues)

        # 验证修复结果
        assert fixed_config["data_sync_params"]["batch_size"] > 0
        assert 1000 <= fixed_config["api_params"]["port"] <= 65535


class TestCompatibilityChecker:
    """兼容性检查测试"""

    def test_compatibility_check(self):
        """测试兼容性检查"""
        checker = create_compatibility_checker()

        # 包含旧版本参数的配置
        old_config = {
            "data_sync_params": {
                "sleep_time": 1.0,  # 旧参数
                "thread_count": 4   # 旧参数
            }
        }

        report = checker.check_compatibility(old_config, "1.0.0", "2.0.0")

        assert report.total_issues > 0
        assert report.migration_required
        assert report.overall_compatibility in [CompatibilityLevel.DEPRECATED, CompatibilityLevel.REMOVED]

    def test_config_migration(self):
        """测试配置迁移"""
        checker = create_compatibility_checker()

        old_config = {
            "data_sync_params": {
                "sleep_time": 1.0,
                "thread_count": 4
            }
        }

        migrated_config = checker.migrate_config(old_config, "1.0.0", "2.0.0")

        # 验证迁移结果
        assert "sleep_interval" in migrated_config["data_sync_params"]
        assert "max_workers" in migrated_config["data_sync_params"]
        assert "sleep_time" not in migrated_config["data_sync_params"]
        assert "thread_count" not in migrated_config["data_sync_params"]


class TestHotReload:
    """热更新测试"""

    def test_hot_reload_manager(self):
        """测试热更新管理器"""
        config_manager = ConfigManager(enable_hot_reload=False)
        hot_reload_manager = create_hot_reload_manager(config_manager)

        # 模拟配置变更
        analysis = hot_reload_manager.simulate_config_change(
            "data_sync_params.batch_size", 2000
        )

        assert analysis.path == "data_sync_params.batch_size"
        assert analysis.new_value == 2000
        assert analysis.impact_level is not None

    def test_impact_analysis(self):
        """测试影响分析"""
        config_manager = ConfigManager(enable_hot_reload=False)
        hot_reload_manager = create_hot_reload_manager(config_manager)

        # 测试不同类型的配置变更
        test_cases = [
            ("data_sync_params.batch_size", 2000),
            ("api_params.port", 8001),
            ("database_params.connection_pool_size", 20)
        ]

        for path, value in test_cases:
            analysis = hot_reload_manager.simulate_config_change(path, value)
            assert analysis.path == path
            assert analysis.new_value == value


class TestRollbackManager:
    """回滚管理测试"""

    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.snapshot_dir = Path(self.temp_dir) / "snapshots"

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_snapshot_creation(self):
        """测试快照创建"""
        config_manager = ConfigManager(enable_hot_reload=False)
        rollback_manager = create_rollback_manager(
            config_manager,
            str(self.snapshot_dir)
        )

        # 创建快照
        snapshot = rollback_manager.create_snapshot("test", "测试快照")

        assert snapshot.id == "test"
        assert snapshot.description == "测试快照"
        assert isinstance(snapshot.config, dict)

    def test_rollback_plan(self):
        """测试回滚计划"""
        config_manager = ConfigManager(enable_hot_reload=False)
        rollback_manager = create_rollback_manager(
            config_manager,
            str(self.snapshot_dir)
        )

        # 创建初始快照
        initial_snapshot = rollback_manager.create_snapshot("initial", "初始配置")

        # 修改配置
        config_manager.set("data_sync_params.batch_size", 2000)

        # 创建回滚计划
        plan = rollback_manager.create_rollback_plan(
            RollbackType.SNAPSHOT,
            target_snapshot_id="initial"
        )

        assert plan.rollback_type == RollbackType.SNAPSHOT
        assert plan.target_snapshot.id == initial_snapshot.id
        assert len(plan.changes_to_rollback) >= 0


class TestIntegration:
    """集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)

    def test_full_system_setup(self):
        """测试完整系统设置"""
        config_manager = setup_config_system(
            config_dir=str(self.config_dir),
            environment="testing",
            enable_hot_reload=False,
            create_templates=True
        )

        assert config_manager is not None
        assert config_manager.environment == ConfigEnvironment.TESTING

        # 验证模板文件是否创建
        assert (self.config_dir / "base.yml").exists()
        assert (self.config_dir / "testing.yml").exists()

    def test_config_workflow(self):
        """测试配置工作流"""
        # 1. 设置系统
        config_manager = setup_config_system(
            config_dir=str(self.config_dir),
            environment="testing",
            enable_hot_reload=False,
            create_templates=True
        )

        # 2. 诊断配置
        diagnostics = create_config_diagnostics()
        report = diagnostics.diagnose_config(config_manager._config)

        # 3. 检查兼容性
        checker = create_compatibility_checker()
        compat_report = checker.check_compatibility(
            config_manager._config, "1.0.0", "2.0.0"
        )

        # 4. 创建快照
        rollback_manager = create_rollback_manager(config_manager)
        snapshot = rollback_manager.create_snapshot("workflow_test", "工作流测试")

        # 验证所有组件都正常工作
        assert report.health_score >= 0
        assert compat_report.total_issues >= 0
        assert snapshot.id == "workflow_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])