import glob
import os
import shutil
from pathlib import Path

#!/usr/bin/env python3
"""
临时文件清理工具

清理项目中的临时文件，包括：
- __pycache__ 目录
- .pyc 文件
- .pyo 文件
- .pyd 文件
- .tmp 文件
- .temp 文件
- 测试临时文件
"""


class TempFileCleaner:
    """临时文件清理器"""

    def __init__(self, project_root: str):
        """方法描述"""
        self.cleanup_stats = {"directories_removed": 0, "files_removed": 0, "total_space_freed": 0}

    def get_temp_patterns(self) -> list:
        """获取需要清理的文件模式"""
        return [
            "__pycache__",  # Python缓存目录
            "*.pyc",  # Python编译文件
            "*.pyo",  # Python优化文件
            "*.pyd",  # Python扩展文件
            "*.tmp",  # 临时文件
            "*.temp",  # 临时文件
            "*.bak",  # 备份文件
            "*.swp",  # Vim交换文件
            "*.swo",  # Vim交换文件
            "*~",  # 编辑器备份文件
            ".DS_Store",  # macOS系统文件
            "Thumbs.db",  # Windows缩略图缓存
            "*.egg-info",  # Python包信息
            ".pytest_cache",  # pytest缓存
            ".coverage",  # 覆盖率报告
            "htmlcov",  # 覆盖率HTML报告
            ".tox",  # tox测试环境
            "dist",  # 构建目录
            "build",  # 构建目录
            "*.egg",  # Python包文件
            ".mypy_cache",  # mypy类型检查缓存
            ".hypothesis",  # hypothesis测试缓存
        ]

    def get_test_temp_patterns(self) -> list:
        """获取测试相关的临时文件模式"""
        return [
            "temp_*",  # 临时文件前缀
            "test_temp_*",  # 测试临时文件
            "tmp_*",  # 临时文件前缀
            "*.test.db",  # 测试数据库
            "test.db",  # 测试数据库
            "temp.db",  # 临时数据库
        ]

    def scan_temp_files(self) -> dict:
        """扫描临时文件"""
        temp_files = {"directories": [], "files": []}

        # 扫描标准临时文件
        for pattern in self.get_temp_patterns():
            if pattern.endswith("*"):
                # 文件模式
                matches = glob.glob(os.path.join(self.project_root, "**", pattern), recursive=True)
                for match in matches:
                    if os.path.isfile(match):
                        temp_files["files"].append(match)
                    elif os.path.isdir(match):
                        temp_files["directories"].append(match)
            else:
                # 目录或文件名模式
                matches = glob.glob(os.path.join(self.project_root, "**", pattern), recursive=True)
                for match in matches:
                    if os.path.isfile(match):
                        temp_files["files"].append(match)
                    elif os.path.isdir(match):
                        temp_files["directories"].append(match)

        # 扫描测试临时文件
        test_dirs = ["tests", "test", "tests_temp"]
        for test_dir in test_dirs:
            test_path = os.path.join(self.project_root, test_dir)
            if os.path.exists(test_path):
                for pattern in self.get_test_temp_patterns():
                    matches = glob.glob(os.path.join(test_path, "**", pattern), recursive=True)
                    for match in matches:
                        if os.path.isfile(match):
                            temp_files["files"].append(match)
                        elif os.path.isdir(match):
                            temp_files["directories"].append(match)

        return temp_files

    def calculate_size(self, path: str) -> int:
        """计算文件或目录大小"""
        if os.path.isfile(path):
            return os.path.getsize(path)
        elif os.path.isdir(path):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size
        return 0

    def remove_with_confirmation(self, path: str, force: bool = False) -> bool:
        """安全删除文件或目录"""
        try:
            if not force:
                # 检查是否在重要目录中
                important_dirs = [".git", ".svn", "venv", "env", ".env"]
                path_parts = Path(path).parts
                if any(important in path_parts for important in important_dirs):
                    print(f"跳过重要目录: {path}")
                    return False

            size = self.calculate_size(path)

            if os.path.isfile(path):
                os.remove(path)
                self.cleanup_stats["files_removed"] += 1
                self.cleanup_stats["total_space_freed"] += size
                return True
            elif os.path.isdir(path):
                shutil.rmtree(path)
                self.cleanup_stats["directories_removed"] += 1
                self.cleanup_stats["total_space_freed"] += size
                return True

        except Exception as e:
            print(f"删除失败 {path}: {e}")
            return False

        return False

    def cleanup_temp_files(self, force: bool = False, dry_run: bool = False) -> dict:
        """清理临时文件"""
        temp_files = self.scan_temp_files()

        if dry_run:
            print("=== 模拟运行模式 ===")
            print(f"发现 {len(temp_files['directories'])} 个临时目录")
            print(f"发现 {len(temp_files['files'])} 个临时文件")

            total_size = 0
            for directory in temp_files["directories"]:
                size = self.calculate_size(directory)
                total_size += size
                print(f"目录: {directory} ({self.format_size(size)})")

            for file in temp_files["files"]:
                size = self.calculate_size(file)
                total_size += size
                print(f"文件: {file} ({self.format_size(size)})")

            print(f"总计可释放空间: {self.format_size(total_size)}")
            return temp_files

        print("正在清理临时文件...")

        # 先删除文件
        for file_path in temp_files["files"]:
            if os.path.exists(file_path):
                self.remove_with_confirmation(file_path, force)

        # 再删除目录
        for dir_path in temp_files["directories"]:
            if os.path.exists(dir_path):
                self.remove_with_confirmation(dir_path, force)

        return self.cleanup_stats

    def format_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def generate_report(self) -> str:
        """生成清理报告"""
        report = []
        report.append("=" * 60)
        report.append("临时文件清理报告")
        report.append("=" * 60)

        report.append(f"删除目录: {self.cleanup_stats['directories_removed']}")
        report.append(f"删除文件: {self.cleanup_stats['files_removed']}")
        report.append(f"释放空间: {self.format_size(self.cleanup_stats['total_space_freed'])}")

        return "\n".join(report)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="清理项目临时文件")
    parser.add_argument("--force", action="store_true", help="强制删除不确认")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行不实际删除")
    parser.add_argument("--project-root", default=".", help="项目根目录")

    args = parser.parse_args()

    if args.project_root == ".":
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        project_root = args.project_root

    cleaner = TempFileCleaner(project_root)

    # 执行清理
    stats = cleaner.cleanup_temp_files(force=args.force, dry_run=args.dry_run)

    if not args.dry_run:
        report = cleaner.generate_report()
        print(report)

        # 保存报告
        report_path = os.path.join(project_root, "reports", "cleanup_report.txt")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
