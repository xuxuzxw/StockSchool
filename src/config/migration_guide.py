#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置系统迁移指南和工具
帮助从旧的配置系统迁移到新的统一配置管理系统
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from loguru import logger


class ConfigMigrationTool:
    """
    配置系统迁移工具
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / 'src'
        
        # 需要迁移的导入模式
        self.old_import_patterns = [
            r'from\s+src\.utils\.config_loader\s+import\s+config',
            r'from\s+\.\.\.utils\.config_loader\s+import\s+config',
            r'from\s+.*\.utils\.config_loader\s+import\s+config',
            r'from\s+src\.utils\.config_loader\s+import\s+Config',
            r'from\s+.*\.utils\.config_loader\s+import\s+Config',
            r'import\s+src\.utils\.config_loader',
        ]
        
        # 新的导入语句
        self.new_import = "from src.config.unified_config import config"
    
    def scan_files_for_migration(self) -> List[Tuple[Path, List[str]]]:
        """
        扫描需要迁移的文件
        
        Returns:
            需要迁移的文件列表和匹配的模式
        """
        files_to_migrate = []
        
        # 扫描所有Python文件
        for py_file in self.src_dir.rglob('*.py'):
            if py_file.name.startswith('.'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                matches = []
                for pattern in self.old_import_patterns:
                    if re.search(pattern, content):
                        matches.append(pattern)
                
                if matches:
                    files_to_migrate.append((py_file, matches))
                    
            except Exception as e:
                logger.warning(f"无法读取文件 {py_file}: {e}")
        
        return files_to_migrate
    
    def generate_migration_report(self) -> str:
        """
        生成迁移报告
        
        Returns:
            迁移报告文本
        """
        files_to_migrate = self.scan_files_for_migration()
        
        report = []
        report.append("# 配置系统迁移报告\n")
        report.append(f"扫描目录: {self.src_dir}")
        report.append(f"发现需要迁移的文件: {len(files_to_migrate)}个\n")
        
        if not files_to_migrate:
            report.append("✅ 没有发现需要迁移的文件")
            return "\n".join(report)
        
        report.append("## 需要迁移的文件:\n")
        
        for file_path, patterns in files_to_migrate:
            relative_path = file_path.relative_to(self.project_root)
            report.append(f"### {relative_path}")
            report.append("匹配的模式:")
            for pattern in patterns:
                report.append(f"  - {pattern}")
            report.append("")
        
        report.append("## 迁移建议:\n")
        report.append("1. 将所有旧的导入语句替换为:")
        report.append(f"   ```python\n   {self.new_import}\n   ```")
        report.append("")
        report.append("2. 确保配置调用方式保持不变:")
        report.append("   ```python\n   value = config.get('key.path', default_value)\n   ```")
        report.append("")
        report.append("3. 新的统一配置系统提供以下额外功能:")
        report.append("   - 多环境支持")
        report.append("   - 配置热更新")
        report.append("   - 配置验证")
        report.append("   - 变更历史和回滚")
        report.append("   - 更好的错误处理")
        
        return "\n".join(report)
    
    def auto_migrate_file(self, file_path: Path) -> bool:
        """
        自动迁移单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功迁移
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 替换导入语句
            for pattern in self.old_import_patterns:
                content = re.sub(pattern, self.new_import, content)
            
            # 如果内容有变化，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"已迁移文件: {file_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"迁移文件失败 {file_path}: {e}")
            return False
    
    def auto_migrate_all(self) -> Dict[str, int]:
        """
        自动迁移所有文件
        
        Returns:
            迁移统计信息
        """
        files_to_migrate = self.scan_files_for_migration()
        
        stats = {
            'total': len(files_to_migrate),
            'success': 0,
            'failed': 0
        }
        
        for file_path, _ in files_to_migrate:
            if self.auto_migrate_file(file_path):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"迁移完成: 总计{stats['total']}个文件, 成功{stats['success']}个, 失败{stats['failed']}个")
        return stats


def main():
    """
    主函数 - 运行迁移工具
    """
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python migration_guide.py <project_root> [--auto-migrate]")
        sys.exit(1)
    
    project_root = sys.argv[1]
    auto_migrate = '--auto-migrate' in sys.argv
    
    tool = ConfigMigrationTool(project_root)
    
    if auto_migrate:
        print("开始自动迁移...")
        stats = tool.auto_migrate_all()
        print(f"迁移完成: {stats}")
    else:
        print("生成迁移报告...")
        report = tool.generate_migration_report()
        print(report)
        
        # 保存报告到文件
        report_file = Path(project_root) / 'config_migration_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存到: {report_file}")


if __name__ == '__main__':
    main()