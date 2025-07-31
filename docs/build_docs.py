#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档构建脚本

自动构建和验证所有文档文件。
"""

import os
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime

def check_document_exists(doc_path: str) -> bool:
    """检查文档文件是否存在"""
    return Path(doc_path).exists()

def validate_document_structure(doc_path: str) -> bool:
    """验证文档结构"""
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查基本结构
        required_sections = ['#', '##']
        for section in required_sections:
            if section not in content:
                print(f"警告: 文档 {doc_path} 缺少基本章节结构")
                return False
        
        return True
    except Exception as e:
        print(f"错误: 验证文档 {doc_path} 结构失败: {e}")
        return False

def build_mkdocs():
    """构建MkDocs文档"""
    try:
        # 检查mkdocs是否安装
        result = subprocess.run(['mkdocs', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("MkDocs未安装，跳过构建")
            return True
        
        # 构建文档
        result = subprocess.run(['mkdocs', 'build'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("MkDocs文档构建成功")
            return True
        else:
            print(f"MkDocs文档构建失败: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("MkDocs未安装，跳过构建")
        return True
    except Exception as e:
        print(f"MkDocs构建异常: {e}")
        return False

def generate_api_docs():
    """生成API文档"""
    try:
        # 使用FastAPI的自动生成文档功能
        from src.api.main import app
        import json
        
        # 生成OpenAPI规范
        openapi_spec = app.openapi()
        
        # 保存到文件
        docs_dir = Path('docs')
        docs_dir.mkdir(exist_ok=True)
        
        with open(docs_dir / 'openapi.json', 'w', encoding='utf-8') as f:
            json.dump(openapi_spec, f, ensure_ascii=False, indent=2)
        
        print("API文档生成成功")
        return True
        
    except Exception as e:
        print(f"API文档生成失败: {e}")
        return False

def check_links():
    """检查文档中的链接有效性"""
    docs_to_check = [
        'README.md',
        'docs/user_manual.md',
        'docs/deployment_guide.md',
        'docs/api_documentation.md',
        'docs/index.md'
    ]
    
    broken_links = []
    
    for doc_path in docs_to_check:
        if not Path(doc_path).exists():
            continue
            
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的链接检查（这里可以扩展为更复杂的链接验证）
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if '](' in line and '.md' in line:
                    # 检查Markdown链接
                    import re
                    links = re.findall(r'\[.*?\]\((.*?)\)', line)
                    for link in links:
                        if link.startswith('http'):
                            continue  # 外部链接跳过
                        
                        # 处理相对链接
                        if link.startswith('./'):
                            link_path = Path(doc_path).parent / link[2:]
                        elif link.startswith('../'):
                            link_path = Path(doc_path).parent.parent / link[3:]
                        else:
                            link_path = Path(doc_path).parent / link
                        
                        if not link_path.exists():
                            broken_links.append({
                                'document': doc_path,
                                'line': i,
                                'link': link,
                                'error': '文件不存在'
                            })
                            
        except Exception as e:
            print(f"检查链接时出错 {doc_path}: {e}")
    
    if broken_links:
        print("发现失效链接:")
        for link_info in broken_links:
            print(f"  {link_info['document']}:{link_info['line']} - {link_info['link']} ({link_info['error']})")
        return False
    
    print("所有链接检查通过")
    return True

def generate_documentation_report():
    """生成文档状态报告"""
    report = {
        'generated_at': datetime.now().isoformat(),
        'documents': {},
        'statistics': {
            'total_documents': 0,
            'existing_documents': 0,
            'missing_documents': 0
        }
    }
    
    # 要检查的文档列表
    documents = {
        'README.md': '项目主文档',
        'docs/user_manual.md': '用户手册',
        'docs/deployment_guide.md': '部署指南',
        'docs/api_documentation.md': 'API文档',
        'docs/index.md': '文档索引',
        'config.yml': '配置文件',
        'database_schema.sql': '数据库结构',
        'requirements.txt': '依赖列表',
        'log.md': '开发日志'
    }
    
    for doc_path, description in documents.items():
        exists = check_document_exists(doc_path)
        report['documents'][doc_path] = {
            'description': description,
            'exists': exists,
            'valid': False
        }
        
        report['statistics']['total_documents'] += 1
        if exists:
            report['statistics']['existing_documents'] += 1
            # 验证文档结构
            if validate_document_structure(doc_path):
                report['documents'][doc_path]['valid'] = True
        else:
            report['statistics']['missing_documents'] += 1
    
    # 保存报告
    report_dir = Path('docs') / 'reports'
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f'documentation_status_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"文档状态报告已生成: {report_file}")
    return report

def main():
    """主函数"""
    print("开始文档构建和验证...")
    
    # 生成文档状态报告
    print("\n=== 生成文档状态报告 ===")
    report = generate_documentation_report()
    
    print(f"文档统计:")
    print(f"  总文档数: {report['statistics']['total_documents']}")
    print(f"  存在文档: {report['statistics']['existing_documents']}")
    print(f"  缺失文档: {report['statistics']['missing_documents']}")
    
    # 检查链接
    print("\n=== 检查文档链接 ===")
    links_ok = check_links()
    
    # 构建MkDocs
    print("\n=== 构建MkDocs文档 ===")
    mkdocs_ok = build_mkdocs()
    
    # 生成API文档
    print("\n=== 生成API文档 ===")
    api_docs_ok = generate_api_docs()
    
    # 总结
    print("\n=== 构建总结 ===")
    print(f"文档状态报告: ✅ 生成完成")
    print(f"链接检查: {'✅' if links_ok else '❌'}")
    print(f"MkDocs构建: {'✅' if mkdocs_ok else '❌'}")
    print(f"API文档生成: {'✅' if api_docs_ok else '❌'}")
    
    # 计算总体成功率
    total_checks = 4
    passed_checks = sum([
        1,  # 报告总是成功
        1 if links_ok else 0,
        1 if mkdocs_ok else 0,
        1 if api_docs_ok else 0
    ])
    
    success_rate = (passed_checks / total_checks) * 100
    print(f"\n总体成功率: {success_rate:.1f}% ({passed_checks}/{total_checks})")
    
    if success_rate == 100:
        print("🎉 所有文档构建和验证通过!")
        return True
    else:
        print("⚠️  部分文档构建或验证失败，请检查上面的错误信息")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
