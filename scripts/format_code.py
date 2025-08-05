#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockSchool代码格式化工具
使用black和isort统一代码风格

使用方法:
    python scripts/format_code.py
    python scripts/format_code.py --check  # 仅检查不修改
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


class CodeFormatter:
    """代码格式化器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dirs = [self.project_root / "src", self.project_root / "tests", self.project_root / "scripts"]

    def format_imports(self, file_path: Path) -> bool:
        """使用isort格式化导入"""
        try:
            cmd = ["isort", str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            print("isort未安装，跳过导入格式化")
            return False
        except Exception as e:
            print(f"格式化导入失败 {file_path}: {e}")
            return False

    def format_code(self, file_path: Path) -> bool:
        """使用black格式化代码"""
        try:
            cmd = ["black", "--line-length", "120", str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            print("black未安装，跳过代码格式化")
            return False
        except Exception as e:
            print(f"格式化代码失败 {file_path}: {e}")
            return False

    def format_file(self, file_path: Path, check_only: bool = False) -> dict:
        """格式化单个文件"""
        if not file_path.exists() or not file_path.suffix == ".py":
            return {"formatted": False, "skipped": True}

        changes = {"file": str(file_path), "formatted": False}

        if not check_only:
            import_formatted = self.format_imports(file_path)
            code_formatted = self.format_code(file_path)
            changes["formatted"] = import_formatted or code_formatted

        return changes

    def format_project(self, check_only: bool = False) -> list:
        """格式化整个项目"""
        results = []

        for src_dir in self.src_dirs:
            if not src_dir.exists():
                continue

            for py_file in src_dir.rglob("*.py"):
                result = self.format_file(py_file, check_only)
                results.append(result)

                if result["formatted"]:
                    print(f"{'检查' if check_only else '格式化'}: {py_file}")

        return results

    def generate_report(self, results: list) -> str:
        """生成格式化报告"""
        total_files = len(results)
        formatted_files = len([r for r in results if r.get("formatted", False)])

        report = f"""
# StockSchool代码格式化报告

## 统计信息
- 处理文件总数: {total_files}
- 格式化文件数: {formatted_files}
- 跳过的文件: {total_files - formatted_files}

## 详细变更
"""

        for result in results:
            if result.get("formatted", False):
                report += f"\n### {result['file']}\n"
                report += "- 应用了代码格式化\n"

        return report


def check_tools():
    """检查必要的格式化工具是否安装"""
    tools = ["black", "isort"]
    missing = []

    for tool in tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True)
        except FileNotFoundError:
            missing.append(tool)

    if missing:
        print(f"缺少格式化工具: {', '.join(missing)}")
        print("请运行: pip install black isort")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="StockSchool代码格式化工具")
    parser.add_argument("--check", action="store_true", help="仅检查不修改")
    parser.add_argument("--path", default=".", help="项目根目录路径")

    args = parser.parse_args()

    if not check_tools():
        return

    formatter = CodeFormatter(args.path)
    results = formatter.format_project(args.check)

    report = formatter.generate_report(results)

    report_file = Path(args.path) / "reports" / "code_formatting_report.md"
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n格式化完成！报告已保存到: {report_file}")

    if args.check:
        print(f"检查了 {len(results)} 个文件")
    else:
        print(f"格式化了 {len([r for r in results if r.get('formatted', False)])} 个文件")


if __name__ == "__main__":
    main()
