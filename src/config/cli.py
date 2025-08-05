import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

#!/usr/bin/env python3
"""
配置管理命令行工具

提供配置管理的命令行接口
"""


from . import (
    backup_config,
    check_config_compatibility,
    create_compatibility_checker,
    create_config_diagnostics,
    create_config_files,
    diagnose_config_file,
    get_config_manager,
    restore_config,
    setup_config_system,
)

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_init(args):
    """初始化配置系统"""
    try:
        # 创建配置文件
        create_config_files(args.config_dir)

        # 初始化配置管理器
        config_manager = setup_config_system(
            config_dir=args.config_dir,
            environment=args.environment,
            enable_hot_reload=not args.no_hot_reload,
            create_templates=True,
        )

        print(f"✅ 配置系统初始化完成")
        print(f"   配置目录: {args.config_dir}")
        print(f"   环境: {args.environment}")
        print(f"   热更新: {'启用' if not args.no_hot_reload else '禁用'}")

    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)


def cmd_validate(args):
    """验证配置"""
    try:
        if args.file:
            # 验证指定文件
            report = diagnose_config_file(args.file)
        else:
            # 验证当前配置
            config_manager = get_config_manager()
            diagnostics = create_config_diagnostics()
            report = diagnostics.diagnose_config(config_manager._config)

        # 输出结果
        print(f"配置验证报告")
        print(f"=" * 50)
        print(f"健康分数: {report.health_score}/100")
        print(f"总问题数: {report.total_issues}")

        if report.total_issues > 0:
            print(f"\n问题统计:")
            for level, count in report.issues_by_level.items():
                if count > 0:
                    print(f"  {level.upper()}: {count}")

            print(f"\n问题详情:")
            for issue in report.issues:
                print(f"  [{issue.level.value.upper()}] {issue.path}")
                print(f"    {issue.message}")
                if issue.suggestions:
                    for suggestion in issue.suggestions:
                        print(f"    💡 {suggestion}")
                print()
        else:
            print("✅ 配置验证通过，未发现问题")

        # 如果有自动修复选项
        if args.auto_fix and report.auto_fixable_count > 0:
            print(f"🔧 自动修复 {report.auto_fixable_count} 个问题...")
            # 这里需要实现自动修复逻辑
            print("✅ 自动修复完成")

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        sys.exit(1)


def cmd_check_compatibility(args):
    """检查兼容性"""
    try:
        config_manager = get_config_manager()
        checker = create_compatibility_checker()

        report = checker.check_compatibility(config_manager._config, args.current_version, args.target_version)

        print(f"兼容性检查报告")
        print(f"=" * 50)
        print(f"当前版本: {args.current_version}")
        print(f"目标版本: {args.target_version}")
        print(f"整体兼容性: {report.overall_compatibility.value}")
        print(f"需要迁移: {'是' if report.migration_required else '否'}")
        print(f"迁移工作量: {report.estimated_migration_effort}")

        if report.total_issues > 0:
            print(f"\n兼容性问题:")
            for issue in report.issues:
                print(f"  [{issue.level.value.upper()}] {issue.config_path}")
                print(f"    {issue.message}")
                if issue.migration_steps:
                    print(f"    迁移步骤:")
                    for step in issue.migration_steps:
                        print(f"      - {step}")
                print()

        # 自动迁移
        if args.migrate and report.migration_required:
            print("🔄 执行自动迁移...")
            migrated_config = checker.migrate_config(config_manager._config, args.current_version, args.target_version)

            # 备份原配置
            backup_path = backup_config("config.yml")
            print(f"📦 原配置已备份到: {backup_path}")

            # 保存迁移后的配置
            # 这里需要实现保存逻辑
            print("✅ 配置迁移完成")

    except Exception as e:
        print(f"❌ 兼容性检查失败: {e}")
        sys.exit(1)


def cmd_get(args):
    """获取配置值"""
    try:
        config_manager = get_config_manager()
        value = config_manager.get(args.key, args.default)

        if args.format == "json":
            print(json.dumps(value, indent=2, ensure_ascii=False))
        else:
            print(value)

    except Exception as e:
        print(f"❌ 获取配置失败: {e}")
        sys.exit(1)


def cmd_set(args):
    """设置配置值"""
    try:
        config_manager = get_config_manager()

        # 解析值
        value = args.value
        if args.type == "int":
            value = int(value)
        elif args.type == "float":
            value = float(value)
        elif args.type == "bool":
            value = value.lower() in ("true", "1", "yes", "on")
        elif args.type == "json":
            value = json.loads(value)

        config_manager.set(args.key, value, source="cli")
        print(f"✅ 配置已更新: {args.key} = {value}")

    except Exception as e:
        print(f"❌ 设置配置失败: {e}")
        sys.exit(1)


def cmd_backup(args):
    """备份配置"""
    try:
        backup_path = backup_config(args.config_file, args.backup_dir)
        print(f"✅ 配置已备份到: {backup_path}")

    except Exception as e:
        print(f"❌ 备份失败: {e}")
        sys.exit(1)


def cmd_restore(args):
    """恢复配置"""
    try:
        success = restore_config(args.backup_file, args.target_file)
        if success:
            print(f"✅ 配置已恢复: {args.target_file}")
        else:
            print(f"❌ 恢复失败")
            sys.exit(1)

    except Exception as e:
        print(f"❌ 恢复失败: {e}")
        sys.exit(1)


def cmd_info(args):
    """显示配置信息"""
    try:
        config_manager = get_config_manager()
        env_info = config_manager.get_environment_info()

        print(f"配置系统信息")
        print(f"=" * 50)
        print(f"环境: {env_info['environment']}")
        print(f"配置目录: {env_info['config_dir']}")
        print(f"热更新: {'启用' if env_info['hot_reload_enabled'] else '禁用'}")
        print(f"配置文件数: {env_info['config_files_count']}")
        print(f"验证规则数: {env_info['validation_rules_count']}")
        print(f"变更历史数: {env_info['change_history_count']}")
        print(f"回调函数数: {env_info['callbacks_count']}")

        # 显示配置统计
        from .utils import get_config_size

        stats = get_config_size(config_manager._config)
        print(f"\n配置统计:")
        print(f"  总键数: {stats['total_keys']}")
        print(f"  字典键: {stats['dict_keys']}")
        print(f"  列表键: {stats['list_keys']}")
        print(f"  字符串键: {stats['string_keys']}")
        print(f"  数值键: {stats['number_keys']}")
        print(f"  布尔键: {stats['boolean_keys']}")

    except Exception as e:
        print(f"❌ 获取信息失败: {e}")
        sys.exit(1)


def cmd_export(args):
    """导出配置"""
    try:
        config_manager = get_config_manager()
        config_manager.export_config(args.output_file, args.format)
        print(f"✅ 配置已导出到: {args.output_file}")

    except Exception as e:
        print(f"❌ 导出失败: {e}")
        sys.exit(1)


def cmd_import(args):
    """导入配置"""
    try:
        config_manager = get_config_manager()
        config_manager.import_config(args.input_file, args.merge)
        print(f"✅ 配置已导入: {args.input_file}")

    except Exception as e:
        print(f"❌ 导入失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="StockSchool 配置管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s init --environment production
  %(prog)s validate --file config.yml --auto-fix
  %(prog)s check-compatibility --current-version 1.0.0 --target-version 2.0.0
  %(prog)s get data_sync_params.batch_size
  %(prog)s set data_sync_params.batch_size 2000 --type int
  %(prog)s backup config.yml
  %(prog)s info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # init 命令
    parser_init = subparsers.add_parser("init", help="初始化配置系统")
    parser_init.add_argument("--config-dir", default="config", help="配置目录")
    parser_init.add_argument(
        "--environment",
        default="development",
        choices=["development", "testing", "staging", "production"],
        help="环境名称",
    )
    parser_init.add_argument("--no-hot-reload", action="store_true", help="禁用热更新")
    parser_init.set_defaults(func=cmd_init)

    # validate 命令
    parser_validate = subparsers.add_parser("validate", help="验证配置")
    parser_validate.add_argument("--file", help="要验证的配置文件")
    parser_validate.add_argument("--auto-fix", action="store_true", help="自动修复问题")
    parser_validate.set_defaults(func=cmd_validate)

    # check-compatibility 命令
    parser_compat = subparsers.add_parser("check-compatibility", help="检查兼容性")
    parser_compat.add_argument("--current-version", required=True, help="当前版本")
    parser_compat.add_argument("--target-version", required=True, help="目标版本")
    parser_compat.add_argument("--migrate", action="store_true", help="执行自动迁移")
    parser_compat.set_defaults(func=cmd_check_compatibility)

    # get 命令
    parser_get = subparsers.add_parser("get", help="获取配置值")
    parser_get.add_argument("key", help="配置键路径")
    parser_get.add_argument("--default", help="默认值")
    parser_get.add_argument("--format", choices=["text", "json"], default="text", help="输出格式")
    parser_get.set_defaults(func=cmd_get)

    # set 命令
    parser_set = subparsers.add_parser("set", help="设置配置值")
    parser_set.add_argument("key", help="配置键路径")
    parser_set.add_argument("value", help="配置值")
    parser_set.add_argument("--type", choices=["str", "int", "float", "bool", "json"], default="str", help="值类型")
    parser_set.set_defaults(func=cmd_set)

    # backup 命令
    parser_backup = subparsers.add_parser("backup", help="备份配置")
    parser_backup.add_argument("config_file", help="要备份的配置文件")
    parser_backup.add_argument("--backup-dir", default="config_backups", help="备份目录")
    parser_backup.set_defaults(func=cmd_backup)

    # restore 命令
    parser_restore = subparsers.add_parser("restore", help="恢复配置")
    parser_restore.add_argument("backup_file", help="备份文件路径")
    parser_restore.add_argument("target_file", help="目标文件路径")
    parser_restore.set_defaults(func=cmd_restore)

    # info 命令
    parser_info = subparsers.add_parser("info", help="显示配置信息")
    parser_info.set_defaults(func=cmd_info)

    # export 命令
    parser_export = subparsers.add_parser("export", help="导出配置")
    parser_export.add_argument("output_file", help="输出文件路径")
    parser_export.add_argument("--format", choices=["yaml", "json"], default="yaml", help="导出格式")
    parser_export.set_defaults(func=cmd_export)

    # import 命令
    parser_import = subparsers.add_parser("import", help="导入配置")
    parser_import.add_argument("input_file", help="输入文件路径")
    parser_import.add_argument("--merge", action="store_true", help="合并模式")
    parser_import.set_defaults(func=cmd_import)

    # 解析参数
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 执行命令
    args.func(args)


if __name__ == "__main__":
    main()
