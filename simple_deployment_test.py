#!/usr/bin/env python3
"""
简化的部署验收测试
测试Docker容器化、CI/CD集成、生产环境部署等功能
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

def check_docker_environment():
    """检查Docker环境"""
    print("🐳 检查Docker环境...")
    
    results = {
        'docker_available': False,
        'docker_version': None,
        'docker_compose_available': False,
        'docker_compose_version': None
    }
    
    # 检查Docker是否可用
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            results['docker_available'] = True
            results['docker_version'] = result.stdout.strip()
            print(f"  ✅ Docker可用: {results['docker_version']}")
        else:
            print("  ❌ Docker不可用")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ Docker未安装或不可用")
    
    # 检查Docker Compose是否可用
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            results['docker_compose_available'] = True
            results['docker_compose_version'] = result.stdout.strip()
            print(f"  ✅ Docker Compose可用: {results['docker_compose_version']}")
        else:
            # 尝试新版本的docker compose命令
            try:
                result = subprocess.run(['docker', 'compose', 'version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    results['docker_compose_available'] = True
                    results['docker_compose_version'] = result.stdout.strip()
                    print(f"  ✅ Docker Compose可用: {results['docker_compose_version']}")
            except Exception:
                print("  ❌ Docker Compose不可用")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ Docker Compose未安装或不可用")
    
    return results

def check_dockerfile_and_compose():
    """检查Dockerfile和docker-compose文件"""
    print("\n📄 检查Docker配置文件...")
    
    files_to_check = {
        'Dockerfile': 'Dockerfile',
        'docker-compose.yml': 'docker-compose.yml',
        'docker-compose.prod.yml': 'docker-compose.prod.yml',
        'docker-compose.acceptance.yml': 'docker-compose.acceptance.yml'
    }
    
    results = {}
    
    for file_desc, file_path in files_to_check.items():
        exists = os.path.exists(file_path)
        results[file_desc] = exists
        
        if exists:
            print(f"  ✅ {file_desc} 存在")
        else:
            print(f"  ❌ {file_desc} 不存在")
    
    return results

def check_github_actions():
    """检查GitHub Actions配置"""
    print("\n🔄 检查CI/CD配置...")
    
    workflows_dir = '.github/workflows'
    workflows_configured = os.path.exists(workflows_dir)
    
    results = {
        'workflows_configured': workflows_configured,
        'workflow_files': []
    }
    
    if workflows_configured:
        workflow_files = [f for f in os.listdir(workflows_dir) 
                         if f.endswith('.yml') or f.endswith('.yaml')]
        results['workflow_files'] = workflow_files
        
        print(f"  ✅ GitHub Actions配置目录存在")
        print(f"  📁 发现 {len(workflow_files)} 个工作流文件:")
        for file in workflow_files:
            print(f"     - {file}")
    else:
        print("  ❌ GitHub Actions配置目录不存在")
    
    return results

def check_test_configuration():
    """检查测试配置"""
    print("\n🧪 检查测试配置...")
    
    test_configs = {
        'pytest.ini': os.path.exists('pytest.ini'),
        'pyproject.toml': os.path.exists('pyproject.toml'),
        'tox.ini': os.path.exists('tox.ini'),
        'requirements.txt': os.path.exists('requirements.txt')
    }
    
    for config_name, exists in test_configs.items():
        if exists:
            print(f"  ✅ {config_name} 存在")
        else:
            print(f"  ❌ {config_name} 不存在")
    
    tests_integrated = any(test_configs.values())
    
    return {
        'tests_integrated': tests_integrated,
        'test_configs': test_configs
    }

def check_environment_configs():
    """检查环境配置文件"""
    print("\n🌍 检查环境配置...")
    
    env_files = {
        'development': '.env',
        'acceptance': '.env.acceptance',
        'production_example': '.env.prod.example',
        'template': 'config/env_template.env'
    }
    
    results = {}
    
    for env_name, file_path in env_files.items():
        exists = os.path.exists(file_path)
        results[env_name] = exists
        
        if exists:
            print(f"  ✅ {env_name} 环境配置存在: {file_path}")
        else:
            print(f"  ❌ {env_name} 环境配置不存在: {file_path}")
    
    return results

def check_production_security():
    """检查生产环境安全配置"""
    print("\n🔒 检查生产环境安全配置...")
    
    security_files = {
        '.gitignore': os.path.exists('.gitignore'),
        '.dockerignore': os.path.exists('.dockerignore'),
        '.env.prod.example': os.path.exists('.env.prod.example')
    }
    
    for file_name, exists in security_files.items():
        if exists:
            print(f"  ✅ {file_name} 存在")
        else:
            print(f"  ❌ {file_name} 不存在")
    
    # 检查.gitignore是否包含敏感文件
    gitignore_secure = False
    if security_files['.gitignore']:
        try:
            with open('.gitignore', 'r') as f:
                gitignore_content = f.read()
                if '.env' in gitignore_content and '*.log' in gitignore_content:
                    gitignore_secure = True
                    print("  ✅ .gitignore 包含敏感文件排除规则")
                else:
                    print("  ⚠️ .gitignore 可能缺少敏感文件排除规则")
        except Exception as e:
            print(f"  ⚠️ 无法读取.gitignore: {e}")
    
    return {
        'security_files': security_files,
        'gitignore_secure': gitignore_secure,
        'security_compliant': all(security_files.values()) and gitignore_secure
    }

def run_deployment_acceptance_test():
    """运行部署验收测试"""
    print("=" * 80)
    print("StockSchool 部署验收测试")
    print("=" * 80)
    print(f"测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # 1. Docker环境检查
    test_results['docker_environment'] = check_docker_environment()
    
    # 2. Docker配置文件检查
    test_results['docker_configs'] = check_dockerfile_and_compose()
    
    # 3. CI/CD配置检查
    test_results['cicd_config'] = check_github_actions()
    
    # 4. 测试配置检查
    test_results['test_config'] = check_test_configuration()
    
    # 5. 环境配置检查
    test_results['environment_configs'] = check_environment_configs()
    
    # 6. 生产环境安全检查
    test_results['production_security'] = check_production_security()
    
    # 计算总体评分
    print("\n" + "=" * 80)
    print("部署验收测试结果汇总")
    print("=" * 80)
    
    scores = []
    
    # Docker环境评分
    docker_score = 0
    if test_results['docker_environment']['docker_available']:
        docker_score += 50
    if test_results['docker_environment']['docker_compose_available']:
        docker_score += 50
    scores.append(('Docker环境', docker_score))
    
    # Docker配置评分
    docker_config_score = sum(test_results['docker_configs'].values()) / len(test_results['docker_configs']) * 100
    scores.append(('Docker配置', docker_config_score))
    
    # CI/CD配置评分
    cicd_score = 100 if test_results['cicd_config']['workflows_configured'] else 0
    scores.append(('CI/CD配置', cicd_score))
    
    # 测试配置评分
    test_score = 100 if test_results['test_config']['tests_integrated'] else 0
    scores.append(('测试配置', test_score))
    
    # 环境配置评分
    env_score = sum(test_results['environment_configs'].values()) / len(test_results['environment_configs']) * 100
    scores.append(('环境配置', env_score))
    
    # 安全配置评分
    security_score = 100 if test_results['production_security']['security_compliant'] else 50
    scores.append(('安全配置', security_score))
    
    # 显示各项评分
    for category, score in scores:
        status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
        print(f"{status} {category}: {score:.1f}%")
    
    # 计算总体评分
    overall_score = sum(score for _, score in scores) / len(scores)
    
    print(f"\n📊 总体评分: {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("🎉 部署验收测试通过！系统已准备好部署。")
        result = True
    elif overall_score >= 60:
        print("⚠️ 部署验收测试基本通过，但有一些问题需要改进。")
        result = True
    else:
        print("❌ 部署验收测试失败，需要解决关键问题后重试。")
        result = False
    
    # 保存测试结果
    save_test_results(test_results, scores, overall_score)
    
    return result

def save_test_results(test_results, scores, overall_score):
    """保存测试结果到文件"""
    try:
        # 创建测试报告目录
        reports_dir = Path("test_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # 准备测试结果数据
        results_data = {
            'test_type': 'deployment_acceptance',
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'category_scores': dict(scores),
            'detailed_results': test_results
        }
        
        # 保存到JSON文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"deployment_acceptance_test_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 测试报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"⚠️ 保存测试报告失败: {e}")

if __name__ == "__main__":
    print("StockSchool 部署验收测试工具")
    print("=" * 50)
    
    # 运行测试
    success = run_deployment_acceptance_test()
    
    # 退出码
    sys.exit(0 if success else 1)