import ast
import hashlib
import os
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

#!/usr/bin/env python3
"""
代码查重检测工具

检测项目中的重复代码，包括：
1. 函数/方法重复定义
2. 类重复定义
3. 代码块重复
4. 导入语句重复
"""


class CodeDeduplicationChecker:
    """代码查重检测器"""

    def __init__(self, project_root: str):
        """方法描述"""
        self.duplicates = defaultdict(list)
        self.function_definitions = defaultdict(list)
        self.class_definitions = defaultdict(list)
        self.import_statements = defaultdict(list)
        self.code_blocks = defaultdict(list)

    def scan_project(self) -> Dict[str, List]:
        """扫描整个项目"""
        python_files = []

        for root, dirs, files in os.walk(self.project_root):
            # 跳过临时目录和测试目录
            dirs[:] = [
                d for d in dirs if not d.startswith(".") and not d.startswith("__pycache__") and "test" not in d.lower()
            ]

            for file in files:
                if file.endswith(".py") and not file.startswith("test_"):
                    python_files.append(os.path.join(root, file))

        for file_path in python_files:
            self.analyze_file(file_path)

        return {
            "function_duplicates": self.find_function_duplicates(),
            "class_duplicates": self.find_class_duplicates(),
            "import_duplicates": self.find_import_duplicates(),
            "code_block_duplicates": self.find_code_block_duplicates(),
        }

    def analyze_file(self, file_path: str):
        """分析单个文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            relative_path = os.path.relpath(file_path, self.project_root)

            # 分析函数定义
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.function_definitions[node.name].append(
                        {
                            "file": relative_path,
                            "line": node.lineno,
                            "signature": self.get_function_signature(node),
                            "body_hash": self.get_body_hash(node),
                        }
                    )
                elif isinstance(node, ast.ClassDef):
                    self.class_definitions[node.name].append(
                        {
                            "file": relative_path,
                            "line": node.lineno,
                            "bases": [self.get_name(base) for base in node.bases],
                            "body_hash": self.get_body_hash(node),
                        }
                    )
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self.import_statements[self.get_import_string(node)].append(
                        {"file": relative_path, "line": node.lineno}
                    )

            # 分析代码块
            self.analyze_code_blocks(content, relative_path)

        except Exception as e:
            print(f"分析文件 {file_path} 失败: {e}")

    def get_function_signature(self, node: ast.FunctionDef) -> str:
        """获取函数签名"""
        args = []
        for arg in node.args.args:
            if arg.annotation:
                args.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
            else:
                args.append(arg.arg)

        return f"def {node.name}({', '.join(args)})"

    def get_body_hash(self, node) -> str:
        """获取代码体哈希"""
        body_str = ast.unparse(node)
        return hashlib.md5(body_str.encode()).hexdigest()[:8]

    def get_name(self, node) -> str:
        """获取节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_name(node.value)}.{node.attr}"
        return str(node)

    def get_import_string(self, node) -> str:
        """获取导入字符串"""
        if isinstance(node, ast.Import):
            return ", ".join(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(alias.name for alias in node.names)
            return f"from {node.module or ''} import {names}"
        return ""

    def analyze_code_blocks(self, content: str, file_path: str):
        """分析代码块重复"""
        lines = content.split("\n")

        # 查找重复的逻辑块（函数、类、条件语句等）
        for i, line in enumerate(lines):
            line = line.strip()
            if line and len(line) > 20:  # 忽略短行
                key = hashlib.md5(line.encode()).hexdigest()[:8]
                self.code_blocks[key].append({"file": file_path, "line": i + 1, "content": line[:100]})  # 限制内容长度

    def find_function_duplicates(self) -> List[Dict]:
        """查找函数重复定义"""
        duplicates = []
        for func_name, definitions in self.function_definitions.items():
            if len(definitions) > 1:
                # 按签名分组
                signatures = defaultdict(list)
                for def_info in definitions:
                    signatures[def_info["signature"]].append(def_info)

                for signature, sig_defs in signatures.items():
                    if len(sig_defs) > 1:
                        duplicates.append(
                            {"name": func_name, "type": "function", "signature": signature, "locations": sig_defs}
                        )
        return duplicates

    def find_class_duplicates(self) -> List[Dict]:
        """查找类重复定义"""
        duplicates = []
        for class_name, definitions in self.class_definitions.items():
            if len(definitions) > 1:
                # 按继承关系分组
                for def_info in definitions:
                    duplicates.append(
                        {"name": class_name, "type": "class", "bases": def_info["bases"], "locations": definitions}
                    )
                break  # 每个类只报告一次
        return duplicates

    def find_import_duplicates(self) -> List[Dict]:
        """查找导入重复"""
        duplicates = []
        for import_str, locations in self.import_statements.items():
            if len(locations) > 5:  # 只在超过5个文件中出现时报告
                duplicates.append(
                    {"import": import_str, "count": len(locations), "locations": locations[:10]}  # 限制显示数量
                )
        return duplicates

    def find_code_block_duplicates(self) -> List[Dict]:
        """查找代码块重复"""
        duplicates = []
        for block_hash, locations in self.code_blocks.items():
            if len(locations) > 3:  # 只在超过3个地方出现时报告
                duplicates.append(
                    {
                        "hash": block_hash,
                        "count": len(locations),
                        "content": locations[0]["content"],
                        "locations": locations[:5],  # 限制显示数量
                    }
                )
        return duplicates

    def generate_report(self) -> str:
        """生成查重报告"""
        results = self.scan_project()

        report = []
        report.append("=" * 60)
        report.append("代码查重检测报告")
        report.append("=" * 60)

        total_duplicates = 0

        # 函数重复
        func_dups = results["function_duplicates"]
        if func_dups:
            report.append(f"\n函数重复定义 ({len(func_dups)}):")
            for dup in func_dups:
                report.append(f"  - {dup['name']}: {dup['signature']}")
                for loc in dup["locations"]:
                    report.append(f"    {loc['file']}:{loc['line']}")
            total_duplicates += len(func_dups)

        # 类重复
        class_dups = results["class_duplicates"]
        if class_dups:
            report.append(f"\n类重复定义 ({len(class_dups)}):")
            for dup in class_dups:
                report.append(f"  - {dup['name']} (bases: {dup['bases']})")
                for loc in dup["locations"]:
                    report.append(f"    {loc['file']}:{loc['line']}")
            total_duplicates += len(class_dups)

        # 导入重复
        import_dups = results["import_duplicates"]
        if import_dups:
            report.append(f"\n导入语句重复 ({len(import_dups)}):")
            for dup in import_dups:
                report.append(f"  - {dup['import']} (出现在 {dup['count']} 个文件中)")

        # 代码块重复
        block_dups = results["code_block_duplicates"]
        if block_dups:
            report.append(f"\n代码块重复 ({len(block_dups)}):")
            for dup in block_dups:
                report.append(f"  - 重复代码: {dup['content'][:50]}...")
                report.append(f"    出现 {dup['count']} 次")

        report.append(f"\n总计发现重复: {total_duplicates}")
        report.append("=" * 60)

        return "\n".join(report)


def main():
    """主函数"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checker = CodeDeduplicationChecker(project_root)

    print("正在扫描项目代码...")
    report = checker.generate_report()

    # 保存报告
    report_path = os.path.join(project_root, "reports", "code_deduplication_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
