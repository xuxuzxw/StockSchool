#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockSchool代码质量检查工具
检查代码是否符合规范和质量要求

使用方法:
    python scripts/quality_check.py
    python scripts/quality_check.py --strict  # 严格模式
"""

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path


class CodeQualityChecker:
    """代码质量检查器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dirs = [self.project_root / "src", self.project_root / "tests", self.project_root / "scripts"]
        self.issues = []

    def check_import_order(self, file_path: Path) -> list:
        """检查导入顺序"""
        issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(("import", alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(("from", node.module))

            # 检查标准库、第三方库、本地库的顺序
            stdlib_modules = {
                "os",
                "sys",
                "time",
                "datetime",
                "json",
                "re",
                "pathlib",
                "logging",
                "typing",
                "collections",
                "itertools",
                "functools",
            }

            prev_type = None
            for imp_type, module in imports:
                if module.split(".")[0] in stdlib_modules:
                    current_type = "stdlib"
                elif module.startswith("src") or module.startswith("."):
                    current_type = "local"
                else:
                    current_type = "third_party"

                if prev_type and current_type != prev_type:
                    # 允许stdlib -> third_party -> local的顺序
                    if (prev_type == "stdlib" and current_type == "third_party") or (
                        prev_type == "third_party" and current_type == "local"
                    ):
                        pass
                    elif prev_type != current_type:
                        issues.append(f"导入顺序问题: {module} 应该按标准库->第三方库->本地库的顺序")

                prev_type = current_type

        except Exception as e:
            issues.append(f"解析导入失败: {e}")

        return issues

    def check_docstrings(self, file_path: Path) -> list:
        """检查文档字符串"""
        issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        issues.append(f"类 {node.name} 缺少文档字符串")
                elif isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node) and not node.name.startswith("_"):
                        issues.append(f"函数 {node.name} 缺少文档字符串")

        except Exception as e:
            issues.append(f"检查文档字符串失败: {e}")

        return issues

    def check_line_length(self, file_path: Path, max_length: int = 120) -> list:
        """检查行长度"""
        issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                if len(line.rstrip()) > max_length:
                    issues.append(f"第{i}行过长: {len(line.rstrip())} > {max_length}")

        except Exception as e:
            issues.append(f"检查行长度失败: {e}")

        return issues

    def check_naming_conventions(self, file_path: Path) -> list:
        """检查命名规范"""
        issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not node.name[0].isupper():
                        issues.append(f"类名 {node.name} 应该使用PascalCase")
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.islower() and "_" not in node.name:
                        issues.append(f"函数名 {node.name} 应该使用snake_case")
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    if node.id.isupper():
                        # 常量检查
                        pass
                    elif not node.id.islower() and "_" not in node.id:
                        issues.append(f"变量名 {node.id} 应该使用snake_case")

        except Exception as e:
            issues.append(f"检查命名规范失败: {e}")

        return issues

    def check_file(self, file_path: Path, strict: bool = False) -> dict:
        """检查单个文件"""
        if not file_path.exists() or not file_path.suffix == ".py":
            return {"file": str(file_path), "issues": [], "passed": True}

        all_issues = []

        # 运行所有检查
        all_issues.extend(self.check_import_order(file_path))
        all_issues.extend(self.check_docstrings(file_path))
        all_issues.extend(self.check_line_length(file_path))
        all_issues.extend(self.check_naming_conventions(file_path))

        # 如果严格模式，检查更多细节
        if strict:
            # 检查是否有TODO注释
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if "TODO" in line or "FIXME" in line:
                        all_issues.append(f"第{i}行有待办事项: {line.strip()}")
            except:
                pass

        return {"file": str(file_path), "issues": all_issues, "passed": len(all_issues) == 0}

    def check_project(self, strict: bool = False) -> list:
        """检查整个项目"""
        results = []

        for src_dir in self.src_dirs:
            if not src_dir.exists():
                continue

            for py_file in src_dir.rglob("*.py"):
                result = self.check_file(py_file, strict)
                results.append(result)

                if not result["passed"]:
                    print(f"发现质量问题: {py_file}")

        return results

    def generate_report(self, results: list, strict: bool = False) -> str:
        """生成质量检查报告"""
        total_files = len(results)
        passed_files = len([r for r in results if r["passed"]])
        failed_files = total_files - passed_files
        total_issues = sum(len(r["issues"]) for r in results)

        report = f"""
# StockSchool代码质量检查报告

## 检查配置
- 严格模式: {'是' if strict else '否'}
- 最大行长度: 120字符
- 命名规范: PEP 8
- 文档字符串: 必需

## 统计信息
- 检查文件总数: {total_files}
- 通过检查: {passed_files}
- 未通过检查: {failed_files}
- 发现的问题: {total_issues}

## 质量评分
- 整体质量: {(passed_files / total_files * 100):.1f}%
- 问题密度: {(total_issues / total_files):.1f} 问题/文件

## 详细问题
"""

        for result in results:
            if not result["passed"]:
                report += f"\n### {result['file']}\n"
                for issue in result["issues"]:
                    report += f"- {issue}\n"

        if failed_files == 0:
            report += "\n## 🎉 恭喜！所有文件都通过了质量检查\n"
        else:
            report += f"\n## 修复建议\n"
            report += "1. 运行 `python scripts/format_code.py` 自动格式化代码\n"
            report += "2. 手动修复命名规范和文档字符串问题\n"
            report += "3. 使用IDE的代码检查功能辅助修复\n"

        return report


def main():
    parser = argparse.ArgumentParser(description="StockSchool代码质量检查工具")
    parser.add_argument("--strict", action="store_true", help="启用严格模式")
    parser.add_argument("--path", default=".", help="项目根目录路径")

    args = parser.parse_args()

    checker = CodeQualityChecker(args.path)
    results = checker.check_project(args.strict)

    report = checker.generate_report(results, args.strict)

    report_file = Path(args.path) / "reports" / "quality_check_report.md"
    report_file.parent.mkdir(exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n质量检查完成！报告已保存到: {report_file}")
    print(f"检查了 {len(results)} 个文件")
    print(f"通过检查: {len([r for r in results if r['passed']])} 个文件")
    print(f"发现问题: {sum(len(r['issues']) for r in results)} 个问题")


if __name__ == "__main__":
    main()
