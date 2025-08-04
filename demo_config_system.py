#!/usr/bin/env python3
"""
配置管理系统演示脚本

展示配置管理系统的主要功能
"""

import json
from src.config import (
    setup_config_system,
    get_config_manager,
    create_config_diagnostics,
    create_compatibility_checker,
    create_hot_reload_manager,
    create_rollback_manager,
    RollbackType
)

def main():
    print("🚀 StockSchool 配置管理系统演示")
    print("=" * 50)
    
    # 1. 初始化配置系统
    print("\n1. 初始化配置系统...")
    config_manager = setup_config_system(
        config_dir="config",
        environment="development",
        enable_hot_reload=False,  # 演示时禁用热更新
        create_templates=True
    )
    print(f"✅ 配置系统初始化完成，环境: {config_manager.environment.value}")
    
    # 2. 基本配置操作
    print("\n2. 基本配置操作...")
    
    # 获取配置
    batch_size = config_manager.get("data_sync_params.batch_size", 1000)
    print(f"   当前批次大小: {batch_size}")
    
    # 设置配置
    config_manager.set("data_sync_params.batch_size", 2000, source="demo")
    new_batch_size = config_manager.get("data_sync_params.batch_size")
    print(f"   更新后批次大小: {new_batch_size}")
    
    # 检查配置是否存在
    exists = config_manager.has("data_sync_params.batch_size")
    print(f"   配置项存在: {exists}")
    
    # 3. 配置诊断
    print("\n3. 配置诊断...")
    diagnostics = create_config_diagnostics()
    
    # 设置一些有问题的配置用于演示
    config_manager.set("data_sync_params.batch_size", -100)  # 负数错误
    config_manager.set("api_params.port", 99999)  # 端口超出范围
    config_manager.set("api_params.cors_origins", ["*"])  # 安全警告
    
    report = diagnostics.diagnose_config(config_manager._config)
    print(f"   健康分数: {report.health_score}/100")
    print(f"   发现问题: {report.total_issues} 个")
    print(f"   可自动修复: {report.auto_fixable_count} 个")
    
    if report.issues:
        print("   问题详情:")
        for issue in report.issues[:3]:  # 只显示前3个问题
            print(f"     - [{issue.level.value.upper()}] {issue.path}: {issue.message}")
    
    # 4. 自动修复
    print("\n4. 自动修复...")
    if report.auto_fixable_count > 0:
        fixed_config = diagnostics.auto_fix_issues(config_manager._config, report.issues)
        print(f"   ✅ 自动修复了 {report.auto_fixable_count} 个问题")
        
        # 更新配置管理器
        config_manager._config = fixed_config
        
        # 重新诊断
        new_report = diagnostics.diagnose_config(config_manager._config)
        print(f"   修复后健康分数: {new_report.health_score}/100")
    
    # 5. 兼容性检查
    print("\n5. 兼容性检查...")
    checker = create_compatibility_checker()
    
    # 添加一些旧版本配置用于演示
    test_config = config_manager._config.copy()
    test_config["data_sync_params"]["sleep_time"] = 1.0  # 旧参数
    test_config["data_sync_params"]["thread_count"] = 4  # 旧参数
    
    compat_report = checker.check_compatibility(test_config, "1.0.0", "2.0.0")
    print(f"   整体兼容性: {compat_report.overall_compatibility.value}")
    print(f"   需要迁移: {'是' if compat_report.migration_required else '否'}")
    print(f"   迁移工作量: {compat_report.estimated_migration_effort}")
    
    if compat_report.issues:
        print("   兼容性问题:")
        for issue in compat_report.issues[:2]:  # 只显示前2个问题
            print(f"     - [{issue.level.value.upper()}] {issue.config_path}: {issue.message}")
    
    # 6. 配置快照和回滚
    print("\n6. 配置快照和回滚...")
    rollback_manager = create_rollback_manager(config_manager)
    
    # 创建快照
    snapshot = rollback_manager.create_snapshot("demo_snapshot", "演示快照")
    print(f"   ✅ 创建快照: {snapshot.id}")
    
    # 修改配置
    original_value = config_manager.get("data_sync_params.batch_size")
    config_manager.set("data_sync_params.batch_size", 5000)
    print(f"   修改配置: batch_size = {config_manager.get('data_sync_params.batch_size')}")
    
    # 创建回滚计划
    plan = rollback_manager.create_rollback_plan(
        RollbackType.SNAPSHOT,
        target_snapshot_id="demo_snapshot"
    )
    print(f"   创建回滚计划: {len(plan.changes_to_rollback)} 个变更需要回滚")
    
    # 执行回滚（干运行）
    result = rollback_manager.execute_rollback_plan(plan, dry_run=True)
    print(f"   回滚计划验证: {'成功' if result['success'] else '失败'}")
    
    # 7. 热更新影响分析
    print("\n7. 热更新影响分析...")
    hot_reload_manager = create_hot_reload_manager(config_manager)
    
    # 分析配置变更影响
    analysis = hot_reload_manager.simulate_config_change(
        "database_params.connection_pool_size", 20
    )
    print(f"   变更路径: {analysis.path}")
    print(f"   影响级别: {analysis.impact_level.value}")
    print(f"   受影响组件: {', '.join(analysis.affected_components)}")
    print(f"   需要重启: {'是' if analysis.restart_required else '否'}")
    
    # 8. 配置信息总览
    print("\n8. 配置系统信息...")
    env_info = config_manager.get_environment_info()
    print(f"   环境: {env_info['environment']}")
    print(f"   配置目录: {env_info['config_dir']}")
    print(f"   热更新: {'启用' if env_info['hot_reload_enabled'] else '禁用'}")
    print(f"   验证规则数: {env_info['validation_rules_count']}")
    print(f"   变更历史数: {env_info['change_history_count']}")
    
    print("\n🎉 配置管理系统演示完成！")
    print("\n💡 提示:")
    print("   - 使用 python src/config/cli.py --help 查看命令行工具")
    print("   - 查看 config/ 目录下的配置模板文件")
    print("   - 运行测试: python -m pytest tests/test_config_system.py")


if __name__ == "__main__":
    main()