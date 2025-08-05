#!/usr/bin/env python3
"""
修复Python文件中的缩进错误 - 简化版本
"""

import os
import re

def fix_specific_files():
    """手动修复已知的几个关键文件"""
    
    fixes = {
        "src/compute/validation_decorators.py": [
            (140, "                return func(*args, **kwargs)", "            try:\n                return func(*args, **kwargs)")
        ],
        "src/config/validators.py": [
            (329, "                return ValidationResult(True, [], [], path, value)", "            try:\n                return ValidationResult(True, [], [], path, value)")
        ],
        "src/compute/factor_scheduler.py": [
            (171, "                        completed_task = fut.result()", "                    try:\n                        completed_task = fut.result()")
        ],
        "src/config/error_handling.py": [
            (31, "                return func(*args, **kwargs)", "            try:\n                return func(*args, **kwargs)")
        ],
        "src/monitoring/performance.py": [
            (37, "            self.tags = {}", "        if self.tags is None:\n            self.tags = {}")
        ],
        "src/compute/parallel_config.py": [
            (31, "            self.max_workers = min(32, (os.cpu_count() or 1) + 4)", "        if self.max_workers is None:\n            self.max_workers = min(32, (os.cpu_count() or 1) + 4)")
        ]
    }
    
    fixed_count = 0
    
    for file_path, fixes_list in fixes.items():
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        
        for line_num, old_content, new_content in fixes_list:
            if line_num <= len(lines):
                if old_content.strip() in lines[line_num-1]:
                    # 替换整行
                    lines[line_num-1] = new_content + "\n"
                    modified = True
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"已修复: {file_path}")
            fixed_count += 1
    
    # 处理dataclass字段初始化
    dataclass_files = [
        "src/monitoring/alerts.py",
        "src/monitoring/collectors.py", 
        "src/monitoring/data_quality.py",
        "src/monitoring/performance.py",
        "src/monitoring/notifications.py"
    ]
    
    for file_path in dataclass_files:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复dataclass字段初始化
        content = re.sub(
            r'(^        if self\.(\w+) is None:)',
            r'        if self.\2 is None:',
            content,
            flags=re.MULTILINE
        )
        
        # 修复try语句缺失
        patterns = [
            (r'(^                return .*)', r'            try:\n\1'),
            (r'(^                completed_task = .*)', r'                    try:\n\1'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"已修复dataclass: {file_path}")
            fixed_count += 1
    
    print(f"总共修复了 {fixed_count} 个文件")

if __name__ == "__main__":
    fix_specific_files()