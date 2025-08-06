#!/usr/bin/env python3
"""
StockSchool 配置文件管理工具
用于验证、生成和管理环境配置文件
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re

class ConfigManager:
    """配置文件管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.template_file = self.config_dir / "env_template.env"
        
        # 必需的配置项
        self.required_configs = {
            'development': [
                'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
                'REDIS_HOST', 'REDIS_PORT', 'REDIS_PASSWORD',
                'TUSHARE_TOKEN', 'SECRET_KEY', 'JWT_SECRET_KEY'
            ],
            'production': [
                'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
                'REDIS_HOST', 'REDIS_PORT', 'REDIS_PASSWORD',
                'TUSHARE_TOKEN', 'SECRET_KEY', 'JWT_SECRET_KEY',
                'SMTP_HOST', 'SMTP_USER', 'SMTP_PASSWORD',
                'ALERT_WEBHOOK_URL'
            ],
            'acceptance': [
                'POSTGRES_HOST', 'POSTGRES_PORT', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
                'REDIS_HOST', 'REDIS_PORT', 'REDIS_PASSWORD',
                'TEST_DATA_PATH', 'TEST_REPORTS_PATH'
            ]
        }
        
        # 敏感配置项（需要特殊处理）
        self.sensitive_configs = [
            'POSTGRES_PASSWORD', 'REDIS_PASSWORD', 'SECRET_KEY', 'JWT_SECRET_KEY',
            'TUSHARE_TOKEN', 'SMTP_PASSWORD', 'OPENAI_API_KEY', 'GEMINI_API_KEY'
        ]
    
    def load_env_file(self, env_file: Path) -> Dict[str, str]:
        """加载环境变量文件"""
        env_vars = {}
        
        if not env_file.exists():
            print(f"警告: 配置文件 {env_file} 不存在")
            return env_vars
        
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # 跳过注释和空行
                    if not line or line.startswith('#'):
                        continue
                    
                    # 解析环境变量
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 移除引号
                        if value.startswith(('"', "'")) and value.endswith(('"', "'")):
                            value = value[1:-1]
                        
                        env_vars[key] = value
                    else:
                        print(f"警告: 第{line_num}行格式不正确: {line}")
        
        except Exception as e:
            print(f"错误: 读取配置文件失败: {e}")
        
        return env_vars
    
    def validate_config(self, env_file: Path, environment: str = 'development') -> Tuple[bool, List[str]]:
        """验证配置文件"""
        env_vars = self.load_env_file(env_file)
        issues = []
        
        # 检查必需的配置项
        required = self.required_configs.get(environment, self.required_configs['development'])
        
        for config_key in required:
            if config_key not in env_vars:
                issues.append(f"缺少必需配置: {config_key}")
            elif not env_vars[config_key] or env_vars[config_key] in ['your_password_here', 'your_token_here']:
                issues.append(f"配置项 {config_key} 需要设置实际值")
        
        # 检查敏感配置的安全性
        for sensitive_key in self.sensitive_configs:
            if sensitive_key in env_vars:
                value = env_vars[sensitive_key]
                
                # 检查密码强度
                if 'PASSWORD' in sensitive_key or 'SECRET' in sensitive_key:
                    if len(value) < 8:
                        issues.append(f"配置项 {sensitive_key} 长度应至少8个字符")
                    
                    if sensitive_key in ['SECRET_KEY', 'JWT_SECRET_KEY'] and len(value) < 32:
                        issues.append(f"配置项 {sensitive_key} 长度应至少32个字符")
        
        # 检查端口冲突
        ports = {}
        port_configs = ['POSTGRES_PORT', 'REDIS_PORT', 'APP_PORT', 'API_PORT', 'DASHBOARD_PORT']
        
        for port_config in port_configs:
            if port_config in env_vars:
                port = env_vars[port_config]
                if port in ports:
                    issues.append(f"端口冲突: {port_config} 和 {ports[port]} 都使用端口 {port}")
                else:
                    ports[port] = port_config
        
        # 检查数据库连接字符串格式
        if 'DATABASE_URL' in env_vars:
            db_url = env_vars['DATABASE_URL']
            if not re.match(r'^postgresql://[^:]+:[^@]+@[^:]+:\d+/\w+$', db_url):
                issues.append("DATABASE_URL 格式不正确")
        
        # 检查Redis连接字符串格式
        if 'REDIS_URL' in env_vars:
            redis_url = env_vars['REDIS_URL']
            if not re.match(r'^redis://:[^@]+@[^:]+:\d+/\d+$', redis_url):
                issues.append("REDIS_URL 格式不正确")
        
        return len(issues) == 0, issues
    
    def list_configs(self) -> None:
        """列出所有配置文件"""
        print("\n发现的配置文件:")
        print("=" * 40)
        
        config_files = [
            self.project_root / ".env",
            self.project_root / ".env.acceptance",
            self.project_root / ".env.prod.example",
            self.template_file
        ]
        
        for config_file in config_files:
            if config_file.exists():
                env_vars = self.load_env_file(config_file)
                environment = env_vars.get('ENVIRONMENT', '未知')
                print(f"  {config_file.name:<20} (环境: {environment}, 配置项: {len(env_vars)})")
            else:
                print(f"  {config_file.name:<20} (不存在)")
    
    def check_security(self, env_file: Path) -> None:
        """检查配置文件的安全性"""
        env_vars = self.load_env_file(env_file)
        security_issues = []
        
        print(f"\n安全检查: {env_file.name}")
        print("=" * 40)
        
        # 检查是否使用默认密码
        default_passwords = {
            'POSTGRES_PASSWORD': ['postgres', 'password', '123456', 'admin'],
            'REDIS_PASSWORD': ['redis', 'password', '123456'],
            'SECRET_KEY': ['secret', 'your-secret-key'],
            'JWT_SECRET_KEY': ['jwt-secret', 'your-jwt-secret']
        }
        
        for key, defaults in default_passwords.items():
            if key in env_vars and env_vars[key].lower() in [d.lower() for d in defaults]:
                security_issues.append(f"{key} 使用了默认或弱密码")
        
        # 检查生产环境的调试模式
        if env_vars.get('ENVIRONMENT') == 'production':
            if env_vars.get('DEBUG', '').lower() == 'true':
                security_issues.append("生产环境不应启用调试模式")
            
            if env_vars.get('LOG_LEVEL') == 'DEBUG':
                security_issues.append("生产环境不应使用DEBUG日志级别")
        
        # 检查敏感信息是否暴露
        for key in self.sensitive_configs:
            if key in env_vars:
                value = env_vars[key]
                if 'test' in value.lower() or 'example' in value.lower():
                    security_issues.append(f"{key} 可能包含测试或示例值")
        
        if security_issues:
            print("发现安全问题:")
            for issue in security_issues:
                print(f"  ⚠️  {issue}")
        else:
            print("✅ 未发现明显的安全问题")

def main():
    parser = argparse.ArgumentParser(description='StockSchool 配置文件管理工具')
    parser.add_argument('command', choices=['validate', 'list', 'security'],
                       help='要执行的命令')
    parser.add_argument('--env', '-e', default='development',
                       choices=['development', 'production', 'acceptance'],
                       help='环境类型')
    parser.add_argument('--file', '-f', type=Path,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    if args.command == 'validate':
        env_file = args.file or config_manager.project_root / '.env'
        is_valid, issues = config_manager.validate_config(env_file, args.env)
        
        print(f"\n配置验证结果: {env_file.name}")
        print("=" * 40)
        
        if is_valid:
            print("✅ 配置文件验证通过")
        else:
            print("❌ 配置文件验证失败")
            print("\n发现的问题:")
            for issue in issues:
                print(f"  • {issue}")
    
    elif args.command == 'list':
        config_manager.list_configs()
    
    elif args.command == 'security':
        env_file = args.file or config_manager.project_root / '.env'
        config_manager.check_security(env_file)

if __name__ == '__main__':
    main()