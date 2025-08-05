import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

#!/usr/bin/env python3
"""
配置系统迁移脚本

此脚本用于将项目从旧的配置系统迁移到新的ConfigManager系统。
迁移内容包括：
1. 更新所有使用旧Config类的文件
2. 替换旧的导入语句
3. 验证迁移后的配置系统
4. 清理过时的配置文件
"""


class ConfigMigrator:
    """配置系统迁移器"""

    def __init__(self):
        """初始化迁移器"""
        self.project_root = Path(__file__).parent.parent
        self.backup_dir = self.project_root / "backups" / "config_migration"
        self.migration_log = []

        # 定义需要替换的模式
        self.replacement_patterns = [
            # 旧的导入语句
            {
                "pattern": r"from\s+src\.utils\.config_loader\s+import\s+ConfigLoader",
                "replacement": "from src.config import get_config",
            },
            {
                "pattern": r"from\s+src\.utils\.config_loader\s+import\s+Config",
                "replacement": "from src.config import get_config",
            },
            {
                "pattern": r"from\s+.*\.utils\.config_loader\s+import\s+config",
                "replacement": "from src.config import get_config",
            },
            {"pattern": r"import\s+src\.utils\.config_loader", "replacement": "from src.config import get_config"},
            # 旧的实例化
            {"pattern": r"ConfigLoader\(\)", "replacement": "get_config()"},
            {"pattern": r"Config\(\)", "replacement": "get_config()"},
        ]

    def backup_file(self, file_path: Path) -> None:
        """备份文件"""
        backup_path = self.backup_dir / file_path.relative_to(self.project_root)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)

    def find_files_to_migrate(self) -> List[Path]:
        """查找需要迁移的文件"""
        python_files = list(self.project_root.rglob("*.py"))
        exclude_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules", "venv", ".venv"}

        files_to_migrate = []
        for file_path in python_files:
            if any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 检查是否包含旧的配置引用
                if any(re.search(pattern["pattern"], content) for pattern in self.replacement_patterns):
                    files_to_migrate.append(file_path)

            except Exception as e:
                print(f"跳过文件 {file_path}: {e}")

        return files_to_migrate

    def migrate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """迁移单个文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            changes = []

            # 应用所有替换模式
            for pattern in self.replacement_patterns:
                new_content, count = re.subn(pattern["pattern"], pattern["replacement"], content)
                if count > 0:
                    changes.append(f"替换 '{pattern['pattern']}' -> '{pattern['replacement']}' ({count}次)")
                    content = new_content

            # 如果有变化，写入文件
            if changes:
                self.backup_file(file_path)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                self.migration_log.append({"file": str(file_path), "changes": changes, "status": "success"})

                return True, changes

            return False, []

        except Exception as e:
            self.migration_log.append({"file": str(file_path), "error": str(e), "status": "failed"})
            return False, [str(e)]

    def run_migration(self) -> dict:
        """执行完整的迁移流程"""
        print("🔧 开始配置系统迁移...")

        # 创建备份目录
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 查找需要迁移的文件
        files_to_migrate = self.find_files_to_migrate()
        print(f"📋 发现 {len(files_to_migrate)} 个文件需要迁移")

        # 迁移文件
        migrated_count = 0
        for file_path in files_to_migrate:
            print(f"📝 处理文件: {file_path}")
            migrated, changes = self.migrate_file(file_path)
            if migrated:
                migrated_count += 1
                print(f"   ✅ 已迁移: {len(changes)} 处更改")
            else:
                print(f"   ⚠️  无需更改或迁移失败")

        # 生成迁移报告
        report = {
            "total_files": len(files_to_migrate),
            "migrated_files": migrated_count,
            "backup_location": str(self.backup_dir),
            "migration_log": self.migration_log,
        }

        # 保存迁移报告
        report_path = self.backup_dir / "migration_report.json"
        import json

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"🎉 迁移完成！")
        print(f"📊 总共迁移了 {migrated_count} 个文件")
        print(f"💾 备份保存在: {self.backup_dir}")
        print(f"📋 详细报告: {report_path}")

        return report

    def cleanup_legacy_files(self) -> None:
        """清理过时的配置文件"""
        legacy_files = ["src/utils/config_loader.py", "src/config/legacy_config.py"]

        print("🧹 清理过时的配置文件...")

        for file_path in legacy_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                backup_path = self.backup_dir / "legacy" / file_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(full_path, backup_path)
                print(f"   📦 已移动: {file_path} -> {backup_path}")
            else:
                print(f"   ⚠️  未找到: {file_path}")


if __name__ == "__main__":
    migrator = ConfigMigrator()

    # 执行迁移
    report = migrator.run_migration()

    # 清理过时文件
    migrator.cleanup_legacy_files()

    print("\n🔄 下一步操作:")
    print("1. 运行测试确保迁移后的配置系统正常工作")
    print("2. 检查是否有遗漏的旧配置引用")
    print("3. 更新项目文档中的配置使用示例")
    print("4. 验证所有配置文件路径和格式正确")
