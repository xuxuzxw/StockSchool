import ast
import inspect
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import markdown
from jinja2 import Environment, FileSystemLoader, Template
from sqlalchemy import JSON, Boolean, Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# -*- coding: utf-8 -*-
"""
AI策略系统文档生成器

实现API文档、系统文档、用户手册等自动生成功能
"""


logger = logging.getLogger(__name__)


@dataclass
class APIEndpoint:
    """API端点信息"""

    path: str
    method: str
    function_name: str
    description: str
    parameters: List[Dict[str, Any]]
    responses: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    tags: List[str]
    deprecated: bool = False
    version: str = "1.0"


@dataclass
class DatabaseTable:
    """数据库表信息"""

    table_name: str
    description: str
    columns: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    created_at: datetime


@dataclass
class SystemModule:
    """系统模块信息"""

    module_name: str
    file_path: str
    description: str
    classes: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]
    dependencies: List[str]
    version: str
    last_modified: datetime


@dataclass
class DocumentTemplate:
    """文档模板"""

    template_id: str
    template_name: str
    template_type: str  # api, system, user_manual, deployment
    template_content: str
    variables: List[str]
    output_format: str  # markdown, html, pdf
    is_active: bool = True
    created_at: datetime = None


class DocumentGenerator:
    """文档生成器

    提供API文档、系统文档、用户手册等自动生成功能
    """

    def __init__(self, database_url: str = None, project_root: str = None):
        """方法描述"""
        self.engine = get_db_manager().engine
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.metadata = MetaData()

        # 创建数据库表
        self._create_tables()

        # 初始化Jinja2环境
        self.jinja_env = Environment(loader=FileSystemLoader(str(self.project_root / "templates")), autoescape=True)

        # 初始化默认模板
        self._init_default_templates()

    def _create_tables(self):
        """创建数据库表"""
        try:
            # API端点表
            api_endpoints = Table(
                "api_endpoints",
                self.metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("path", String(200), nullable=False),
                Column("method", String(10), nullable=False),
                Column("function_name", String(100), nullable=False),
                Column("description", Text, nullable=False),
                Column("parameters", JSON, nullable=True),
                Column("responses", JSON, nullable=True),
                Column("examples", JSON, nullable=True),
                Column("tags", JSON, nullable=True),
                Column("deprecated", Boolean, default=False),
                Column("version", String(20), default="1.0"),
                Column("created_at", DateTime, default=datetime.now),
                Column("updated_at", DateTime, default=datetime.now),
            )

            # 数据库表信息表
            database_tables = Table(
                "database_tables",
                self.metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("table_name", String(100), nullable=False, unique=True),
                Column("description", Text, nullable=False),
                Column("columns", JSON, nullable=False),
                Column("indexes", JSON, nullable=True),
                Column("relationships", JSON, nullable=True),
                Column("created_at", DateTime, default=datetime.now),
                Column("updated_at", DateTime, default=datetime.now),
            )

            # 系统模块表
            system_modules = Table(
                "system_modules",
                self.metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("module_name", String(100), nullable=False, unique=True),
                Column("file_path", String(500), nullable=False),
                Column("description", Text, nullable=False),
                Column("classes", JSON, nullable=True),
                Column("functions", JSON, nullable=True),
                Column("dependencies", JSON, nullable=True),
                Column("version", String(20), default="1.0"),
                Column("last_modified", DateTime, nullable=False),
                Column("created_at", DateTime, default=datetime.now),
            )

            # 文档模板表
            document_templates = Table(
                "document_templates",
                self.metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("template_id", String(100), nullable=False, unique=True),
                Column("template_name", String(200), nullable=False),
                Column("template_type", String(50), nullable=False),
                Column("template_content", Text, nullable=False),
                Column("variables", JSON, nullable=True),
                Column("output_format", String(20), default="markdown"),
                Column("is_active", Boolean, default=True),
                Column("created_at", DateTime, default=datetime.now),
                Column("updated_at", DateTime, default=datetime.now),
            )

            # 生成的文档表
            generated_documents = Table(
                "generated_documents",
                self.metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("document_id", String(100), nullable=False, unique=True),
                Column("document_name", String(200), nullable=False),
                Column("document_type", String(50), nullable=False),
                Column("template_id", String(100), nullable=False),
                Column("content", Text, nullable=False),
                Column("file_path", String(500), nullable=True),
                Column("version", String(20), default="1.0"),
                Column("status", String(20), default="generated"),  # generated, published, archived
                Column("generated_at", DateTime, default=datetime.now),
                Column("published_at", DateTime, nullable=True),
            )

            # 文档变更历史表
            document_changes = Table(
                "document_changes",
                self.metadata,
                Column("id", Integer, primary_key=True, autoincrement=True),
                Column("document_id", String(100), nullable=False),
                Column("change_type", String(50), nullable=False),  # created, updated, deleted
                Column("change_description", Text, nullable=False),
                Column("old_version", String(20), nullable=True),
                Column("new_version", String(20), nullable=False),
                Column("changed_by", String(100), nullable=True),
                Column("changed_at", DateTime, default=datetime.now),
            )

            self.metadata.create_all(self.engine)
            logger.info("文档生成器数据库表创建成功")

        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise

    def _init_default_templates(self):
        """初始化默认模板"""
        try:
            # API文档模板
            api_template = DocumentTemplate(
                template_id="api_doc_template",
                template_name="API文档模板",
                template_type="api",
                template_content=self._get_api_doc_template(),
                variables=["endpoints", "project_name", "version", "base_url"],
                output_format="markdown",
            )
            self.save_document_template(api_template)

            # 系统架构文档模板
            system_template = DocumentTemplate(
                template_id="system_doc_template",
                template_name="系统架构文档模板",
                template_type="system",
                template_content=self._get_system_doc_template(),
                variables=["modules", "database_tables", "project_name", "version"],
                output_format="markdown",
            )
            self.save_document_template(system_template)

            # 用户手册模板
            user_manual_template = DocumentTemplate(
                template_id="user_manual_template",
                template_name="用户手册模板",
                template_type="user_manual",
                template_content=self._get_user_manual_template(),
                variables=["features", "tutorials", "project_name", "version"],
                output_format="markdown",
            )
            self.save_document_template(user_manual_template)

            # 部署文档模板
            deployment_template = DocumentTemplate(
                template_id="deployment_doc_template",
                template_name="部署文档模板",
                template_type="deployment",
                template_content=self._get_deployment_doc_template(),
                variables=["requirements", "installation_steps", "configuration", "project_name"],
                output_format="markdown",
            )
            self.save_document_template(deployment_template)

        except Exception as e:
            logger.error(f"初始化默认模板失败: {e}")

    def scan_api_endpoints(self, api_file_path: str) -> List[APIEndpoint]:
        """扫描API端点

        Args:
            api_file_path: API文件路径

        Returns:
            API端点列表
        """
        try:
            endpoints = []

            # 读取API文件
            with open(api_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content)

            # 查找Flask路由装饰器
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    endpoint = self._extract_endpoint_info(node, content)
                    if endpoint:
                        endpoints.append(endpoint)

            # 保存到数据库
            for endpoint in endpoints:
                self.save_api_endpoint(endpoint)

            return endpoints

        except Exception as e:
            logger.error(f"扫描API端点失败: {e}")
            return []

    def scan_database_schema(self) -> List[DatabaseTable]:
        """扫描数据库架构

        Returns:
            数据库表列表
        """
        try:
            tables = []

            with self.engine.connect() as conn:
                # 获取所有表
                result = conn.execute(
                    text(
                        """
                    SELECT table_name, table_comment
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_type = 'BASE TABLE'
                """
                    )
                )

                for row in result:
                    table_name = row[0]
                    description = row[1] or f"{table_name}表"

                    # 获取列信息
                    columns = self._get_table_columns(table_name)

                    # 获取索引信息
                    indexes = self._get_table_indexes(table_name)

                    # 获取关系信息
                    relationships = self._get_table_relationships(table_name)

                    table_info = DatabaseTable(
                        table_name=table_name,
                        description=description,
                        columns=columns,
                        indexes=indexes,
                        relationships=relationships,
                        created_at=datetime.now(),
                    )

                    tables.append(table_info)
                    self.save_database_table(table_info)

            return tables

        except Exception as e:
            logger.error(f"扫描数据库架构失败: {e}")
            return []

    def scan_system_modules(self, source_dir: str) -> List[SystemModule]:
        """扫描系统模块

        Args:
            source_dir: 源代码目录

        Returns:
            系统模块列表
        """
        try:
            modules = []
            source_path = Path(source_dir)

            # 递归扫描Python文件
            for py_file in source_path.rglob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                module_info = self._analyze_python_module(py_file)
                if module_info:
                    modules.append(module_info)
                    self.save_system_module(module_info)

            return modules

        except Exception as e:
            logger.error(f"扫描系统模块失败: {e}")
            return []

    def generate_api_documentation(self, output_path: str = None) -> str:
        """生成API文档

        Args:
            output_path: 输出文件路径

        Returns:
            生成的文档内容
        """
        try:
            # 获取API端点
            endpoints = self.get_all_api_endpoints()

            # 获取模板
            template = self.get_document_template("api_doc_template")
            if not template:
                raise ValueError("API文档模板不存在")

            # 准备模板变量
            template_vars = {
                "endpoints": [asdict(ep) for ep in endpoints],
                "project_name": "AI策略系统",
                "version": "1.0.0",
                "base_url": "http://localhost:5000/api",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 渲染模板
            jinja_template = Template(template.template_content)
            content = jinja_template.render(**template_vars)

            # 保存文档
            doc_id = f"api_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_generated_document(doc_id, "API文档", "api", template.template_id, content, output_path)

            # 写入文件
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

            return content

        except Exception as e:
            logger.error(f"生成API文档失败: {e}")
            raise

    def generate_system_documentation(self, output_path: str = None) -> str:
        """生成系统架构文档

        Args:
            output_path: 输出文件路径

        Returns:
            生成的文档内容
        """
        try:
            # 获取系统模块
            modules = self.get_all_system_modules()

            # 获取数据库表
            tables = self.get_all_database_tables()

            # 获取模板
            template = self.get_document_template("system_doc_template")
            if not template:
                raise ValueError("系统文档模板不存在")

            # 准备模板变量
            template_vars = {
                "modules": [asdict(mod) for mod in modules],
                "database_tables": [asdict(table) for table in tables],
                "project_name": "AI策略系统",
                "version": "1.0.0",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 渲染模板
            jinja_template = Template(template.template_content)
            content = jinja_template.render(**template_vars)

            # 保存文档
            doc_id = f"system_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_generated_document(doc_id, "系统架构文档", "system", template.template_id, content, output_path)

            # 写入文件
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

            return content

        except Exception as e:
            logger.error(f"生成系统文档失败: {e}")
            raise

    def generate_user_manual(self, output_path: str = None) -> str:
        """生成用户手册

        Args:
            output_path: 输出文件路径

        Returns:
            生成的文档内容
        """
        try:
            # 获取模板
            template = self.get_document_template("user_manual_template")
            if not template:
                raise ValueError("用户手册模板不存在")

            # 准备功能列表
            features = [
                {
                    "name": "AI模型管理",
                    "description": "支持多种AI模型的训练、部署和管理",
                    "endpoints": ["/api/models", "/api/models/train", "/api/models/predict"],
                },
                {
                    "name": "策略回测",
                    "description": "提供历史数据回测功能，验证策略有效性",
                    "endpoints": ["/api/backtest", "/api/backtest/results"],
                },
                {
                    "name": "个性化定制",
                    "description": "根据用户风险偏好提供个性化策略推荐",
                    "endpoints": ["/api/strategies/recommend", "/api/risk-profile"],
                },
            ]

            # 准备教程列表
            tutorials = [
                {
                    "title": "快速开始",
                    "description": "5分钟快速上手AI策略系统",
                    "steps": ["注册账户", "配置API密钥", "创建第一个策略", "运行回测"],
                },
                {
                    "title": "高级功能",
                    "description": "深入了解系统的高级功能",
                    "steps": ["自定义因子", "模型调优", "风险管理", "性能监控"],
                },
            ]

            # 准备模板变量
            template_vars = {
                "features": features,
                "tutorials": tutorials,
                "project_name": "AI策略系统",
                "version": "1.0.0",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 渲染模板
            jinja_template = Template(template.template_content)
            content = jinja_template.render(**template_vars)

            # 保存文档
            doc_id = f"user_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_generated_document(doc_id, "用户手册", "user_manual", template.template_id, content, output_path)

            # 写入文件
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

            return content

        except Exception as e:
            logger.error(f"生成用户手册失败: {e}")
            raise

    def generate_deployment_guide(self, output_path: str = None) -> str:
        """生成部署指南

        Args:
            output_path: 输出文件路径

        Returns:
            生成的文档内容
        """
        try:
            # 获取模板
            template = self.get_document_template("deployment_doc_template")
            if not template:
                raise ValueError("部署文档模板不存在")

            # 准备部署信息
            requirements = [
                "Python 3.8+",
                "PostgreSQL 12+",
                "Redis 6.0+",
                "Docker (可选)",
                "4GB+ RAM",
                "20GB+ 磁盘空间",
            ]

            installation_steps = [
                {
                    "step": 1,
                    "title": "克隆代码库",
                    "command": "git clone https://github.com/your-org/stock-school.git",
                    "description": "从GitHub克隆项目代码",
                },
                {
                    "step": 2,
                    "title": "安装依赖",
                    "command": "pip install -r requirements.txt",
                    "description": "安装Python依赖包",
                },
                {
                    "step": 3,
                    "title": "配置数据库",
                    "command": "python manage.py init-db",
                    "description": "初始化数据库表结构",
                },
                {"step": 4, "title": "启动服务", "command": "python app.py", "description": "启动Web服务"},
            ]

            configuration = {
                "database": {
                    "url": "postgresql://user:password@localhost:5432/stockschool",
                    "pool_size": 20,
                    "max_overflow": 30,
                },
                "redis": {"url": "redis://localhost:6379/0", "max_connections": 50},
                "api": {"host": "0.0.0.0", "port": 5000, "debug": False},
            }

            # 准备模板变量
            template_vars = {
                "requirements": requirements,
                "installation_steps": installation_steps,
                "configuration": configuration,
                "project_name": "AI策略系统",
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # 渲染模板
            jinja_template = Template(template.template_content)
            content = jinja_template.render(**template_vars)

            # 保存文档
            doc_id = f"deployment_guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._save_generated_document(doc_id, "部署指南", "deployment", template.template_id, content, output_path)

            # 写入文件
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

            return content

        except Exception as e:
            logger.error(f"生成部署指南失败: {e}")
            raise

    def generate_all_documentation(self, output_dir: str) -> Dict[str, str]:
        """生成所有文档

        Args:
            output_dir: 输出目录

        Returns:
            生成的文档文件路径字典
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            results = {}

            # 生成API文档
            api_doc_path = output_path / "api_documentation.md"
            self.generate_api_documentation(str(api_doc_path))
            results["api_documentation"] = str(api_doc_path)

            # 生成系统文档
            system_doc_path = output_path / "system_architecture.md"
            self.generate_system_documentation(str(system_doc_path))
            results["system_documentation"] = str(system_doc_path)

            # 生成用户手册
            user_manual_path = output_path / "user_manual.md"
            self.generate_user_manual(str(user_manual_path))
            results["user_manual"] = str(user_manual_path)

            # 生成部署指南
            deployment_guide_path = output_path / "deployment_guide.md"
            self.generate_deployment_guide(str(deployment_guide_path))
            results["deployment_guide"] = str(deployment_guide_path)

            # 生成README
            readme_path = output_path / "README.md"
            self._generate_readme(str(readme_path))
            results["readme"] = str(readme_path)

            logger.info(f"所有文档生成完成，输出目录: {output_dir}")
            return results

        except Exception as e:
            logger.error(f"生成所有文档失败: {e}")
            raise

    def save_api_endpoint(self, endpoint: APIEndpoint):
        """保存API端点"""
        try:
            # 检查是否已存在
            existing = self._get_api_endpoint(endpoint.path, endpoint.method)
            if existing:
                # 更新现有端点
                update_sql = """
                UPDATE api_endpoints
                SET function_name = :function_name, description = :description,
                    parameters = :parameters, responses = :responses,
                    examples = :examples, tags = :tags,
                    deprecated = :deprecated, version = :version,
                    updated_at = :updated_at
                WHERE path = :path AND method = :method
                """

                with self.engine.connect() as conn:
                    conn.execute(
                        text(update_sql),
                        {
                            "path": endpoint.path,
                            "method": endpoint.method,
                            "function_name": endpoint.function_name,
                            "description": endpoint.description,
                            "parameters": json.dumps(endpoint.parameters),
                            "responses": json.dumps(endpoint.responses),
                            "examples": json.dumps(endpoint.examples),
                            "tags": json.dumps(endpoint.tags),
                            "deprecated": endpoint.deprecated,
                            "version": endpoint.version,
                            "updated_at": datetime.now(),
                        },
                    )
                    conn.commit()
            else:
                # 插入新端点
                insert_sql = """
                INSERT INTO api_endpoints (
                    path, method, function_name, description, parameters,
                    responses, examples, tags, deprecated, version
                ) VALUES (
                    :path, :method, :function_name, :description, :parameters,
                    :responses, :examples, :tags, :deprecated, :version
                )
                """

                with self.engine.connect() as conn:
                    conn.execute(
                        text(insert_sql),
                        {
                            "path": endpoint.path,
                            "method": endpoint.method,
                            "function_name": endpoint.function_name,
                            "description": endpoint.description,
                            "parameters": json.dumps(endpoint.parameters),
                            "responses": json.dumps(endpoint.responses),
                            "examples": json.dumps(endpoint.examples),
                            "tags": json.dumps(endpoint.tags),
                            "deprecated": endpoint.deprecated,
                            "version": endpoint.version,
                        },
                    )
                    conn.commit()

        except Exception as e:
            logger.error(f"保存API端点失败: {e}")
            raise

    def save_database_table(self, table: DatabaseTable):
        """保存数据库表信息"""
        try:
            # 检查是否已存在
            existing = self._get_database_table(table.table_name)
            if existing:
                # 更新现有表信息
                update_sql = """
                UPDATE database_tables
                SET description = :description, columns = :columns,
                    indexes = :indexes, relationships = :relationships,
                    updated_at = :updated_at
                WHERE table_name = :table_name
                """

                with self.engine.connect() as conn:
                    conn.execute(
                        text(update_sql),
                        {
                            "table_name": table.table_name,
                            "description": table.description,
                            "columns": json.dumps(table.columns),
                            "indexes": json.dumps(table.indexes),
                            "relationships": json.dumps(table.relationships),
                            "updated_at": datetime.now(),
                        },
                    )
                    conn.commit()
            else:
                # 插入新表信息
                insert_sql = """
                INSERT INTO database_tables (
                    table_name, description, columns, indexes, relationships
                ) VALUES (
                    :table_name, :description, :columns, :indexes, :relationships
                )
                """

                with self.engine.connect() as conn:
                    conn.execute(
                        text(insert_sql),
                        {
                            "table_name": table.table_name,
                            "description": table.description,
                            "columns": json.dumps(table.columns),
                            "indexes": json.dumps(table.indexes),
                            "relationships": json.dumps(table.relationships),
                        },
                    )
                    conn.commit()

        except Exception as e:
            logger.error(f"保存数据库表信息失败: {e}")
            raise

    def save_system_module(self, module: SystemModule):
        """保存系统模块信息"""
        try:
            # 检查是否已存在
            existing = self._get_system_module(module.module_name)
            if existing:
                # 更新现有模块信息
                update_sql = """
                UPDATE system_modules
                SET file_path = :file_path, description = :description,
                    classes = :classes, functions = :functions,
                    dependencies = :dependencies, version = :version,
                    last_modified = :last_modified
                WHERE module_name = :module_name
                """

                with self.engine.connect() as conn:
                    conn.execute(
                        text(update_sql),
                        {
                            "module_name": module.module_name,
                            "file_path": module.file_path,
                            "description": module.description,
                            "classes": json.dumps(module.classes),
                            "functions": json.dumps(module.functions),
                            "dependencies": json.dumps(module.dependencies),
                            "version": module.version,
                            "last_modified": module.last_modified,
                        },
                    )
                    conn.commit()
            else:
                # 插入新模块信息
                insert_sql = """
                INSERT INTO system_modules (
                    module_name, file_path, description, classes, functions,
                    dependencies, version, last_modified
                ) VALUES (
                    :module_name, :file_path, :description, :classes, :functions,
                    :dependencies, :version, :last_modified
                )
                """

                with self.engine.connect() as conn:
                    conn.execute(
                        text(insert_sql),
                        {
                            "module_name": module.module_name,
                            "file_path": module.file_path,
                            "description": module.description,
                            "classes": json.dumps(module.classes),
                            "functions": json.dumps(module.functions),
                            "dependencies": json.dumps(module.dependencies),
                            "version": module.version,
                            "last_modified": module.last_modified,
                        },
                    )
                    conn.commit()

        except Exception as e:
            logger.error(f"保存系统模块信息失败: {e}")
            raise

    def save_document_template(self, template: DocumentTemplate):
        """保存文档模板"""
        try:
            # 检查是否已存在
            existing = self._get_document_template(template.template_id)
            if existing:
                # 更新现有模板
                update_sql = """
                UPDATE document_templates
                SET template_name = :template_name, template_type = :template_type,
                    template_content = :template_content, variables = :variables,
                    output_format = :output_format, is_active = :is_active,
                    updated_at = :updated_at
                WHERE template_id = :template_id
                """

                with self.engine.connect() as conn:
                    conn.execute(
                        text(update_sql),
                        {
                            "template_id": template.template_id,
                            "template_name": template.template_name,
                            "template_type": template.template_type,
                            "template_content": template.template_content,
                            "variables": json.dumps(template.variables),
                            "output_format": template.output_format,
                            "is_active": template.is_active,
                            "updated_at": datetime.now(),
                        },
                    )
                    conn.commit()
            else:
                # 插入新模板
                insert_sql = """
                INSERT INTO document_templates (
                    template_id, template_name, template_type, template_content,
                    variables, output_format, is_active
                ) VALUES (
                    :template_id, :template_name, :template_type, :template_content,
                    :variables, :output_format, :is_active
                )
                """

                with self.engine.connect() as conn:
                    conn.execute(
                        text(insert_sql),
                        {
                            "template_id": template.template_id,
                            "template_name": template.template_name,
                            "template_type": template.template_type,
                            "template_content": template.template_content,
                            "variables": json.dumps(template.variables),
                            "output_format": template.output_format,
                            "is_active": template.is_active,
                        },
                    )
                    conn.commit()

        except Exception as e:
            logger.error(f"保存文档模板失败: {e}")
            raise

    # 获取方法
    def get_all_api_endpoints(self) -> List[APIEndpoint]:
        """获取所有API端点"""
        try:
            query_sql = """
            SELECT *
            FROM api_endpoints
            ORDER BY path, method
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql))
                rows = result.fetchall()

            endpoints = []
            for row in rows:
                endpoint = APIEndpoint(
                    path=row[1],
                    method=row[2],
                    function_name=row[3],
                    description=row[4],
                    parameters=json.loads(row[5]) if row[5] else [],
                    responses=json.loads(row[6]) if row[6] else [],
                    examples=json.loads(row[7]) if row[7] else [],
                    tags=json.loads(row[8]) if row[8] else [],
                    deprecated=row[9],
                    version=row[10],
                )
                endpoints.append(endpoint)

            return endpoints

        except Exception as e:
            logger.error(f"获取API端点失败: {e}")
            return []

    def get_all_database_tables(self) -> List[DatabaseTable]:
        """获取所有数据库表"""
        try:
            query_sql = """
            SELECT *
            FROM database_tables
            ORDER BY table_name
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql))
                rows = result.fetchall()

            tables = []
            for row in rows:
                table = DatabaseTable(
                    table_name=row[1],
                    description=row[2],
                    columns=json.loads(row[3]) if row[3] else [],
                    indexes=json.loads(row[4]) if row[4] else [],
                    relationships=json.loads(row[5]) if row[5] else [],
                    created_at=row[6],
                )
                tables.append(table)

            return tables

        except Exception as e:
            logger.error(f"获取数据库表失败: {e}")
            return []

    def get_all_system_modules(self) -> List[SystemModule]:
        """获取所有系统模块"""
        try:
            query_sql = """
            SELECT *
            FROM system_modules
            ORDER BY module_name
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql))
                rows = result.fetchall()

            modules = []
            for row in rows:
                module = SystemModule(
                    module_name=row[1],
                    file_path=row[2],
                    description=row[3],
                    classes=json.loads(row[4]) if row[4] else [],
                    functions=json.loads(row[5]) if row[5] else [],
                    dependencies=json.loads(row[6]) if row[6] else [],
                    version=row[7],
                    last_modified=row[8],
                )
                modules.append(module)

            return modules

        except Exception as e:
            logger.error(f"获取系统模块失败: {e}")
            return []

    def get_document_template(self, template_id: str) -> Optional[DocumentTemplate]:
        """获取文档模板"""
        try:
            query_sql = """
            SELECT *
            FROM document_templates
            WHERE template_id = :template_id AND is_active = true
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"template_id": template_id})
                row = result.fetchone()

            if row:
                return DocumentTemplate(
                    template_id=row[1],
                    template_name=row[2],
                    template_type=row[3],
                    template_content=row[4],
                    variables=json.loads(row[5]) if row[5] else [],
                    output_format=row[6],
                    is_active=row[7],
                    created_at=row[8],
                )

            return None

        except Exception as e:
            logger.error(f"获取文档模板失败: {e}")
            return None

    # 辅助方法
    def _extract_endpoint_info(self, node: ast.FunctionDef, content: str) -> Optional[APIEndpoint]:
        """从AST节点提取端点信息"""
        try:
            # 查找路由装饰器
            route_info = None
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if hasattr(decorator.func, "attr") and decorator.func.attr == "route":
                        # 提取路径和方法
                        if decorator.args:
                            path = (
                                decorator.args[0].s if hasattr(decorator.args[0], "s") else str(decorator.args[0].value)
                            )
                            methods = ["GET"]  # 默认方法

                            # 查找methods参数
                            for keyword in decorator.keywords:
                                if keyword.arg == "methods":
                                    if isinstance(keyword.value, ast.List):
                                        methods = [elt.s for elt in keyword.value.elts if hasattr(elt, "s")]

                            route_info = {"path": path, "methods": methods}
                            break

            if not route_info:
                return None

            # 提取函数文档字符串
            description = ""
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
                description = node.body[0].value.s

            # 创建端点对象（为每个HTTP方法创建一个端点）
            endpoints = []
            for method in route_info["methods"]:
                endpoint = APIEndpoint(
                    path=route_info["path"],
                    method=method,
                    function_name=node.name,
                    description=description or f"{node.name}接口",
                    parameters=[],  # 需要进一步解析
                    responses=[],  # 需要进一步解析
                    examples=[],  # 需要进一步解析
                    tags=[],
                )
                endpoints.append(endpoint)

            return endpoints[0] if endpoints else None

        except Exception as e:
            logger.error(f"提取端点信息失败: {e}")
            return None

    def _get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表列信息"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT column_name, data_type, is_nullable, column_default, column_comment
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """
                    ),
                    {"table_name": table_name},
                )

                columns = []
                for row in result:
                    column = {
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "default": row[3],
                        "comment": row[4] or "",
                    }
                    columns.append(column)

                return columns
        except:
            return []

    def _get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表索引信息"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = :table_name
                """
                    ),
                    {"table_name": table_name},
                )

                indexes = []
                for row in result:
                    index = {"name": row[0], "definition": row[1]}
                    indexes.append(index)

                return indexes
        except:
            return []

    def _get_table_relationships(self, table_name: str) -> List[Dict[str, Any]]:
        """获取表关系信息"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(
                        """
                    SELECT
                        tc.constraint_name,
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND tc.table_name = :table_name
                """
                    ),
                    {"table_name": table_name},
                )

                relationships = []
                for row in result:
                    relationship = {
                        "constraint_name": row[0],
                        "column_name": row[1],
                        "foreign_table": row[2],
                        "foreign_column": row[3],
                    }
                    relationships.append(relationship)

                return relationships
        except:
            return []

    def _analyze_python_module(self, file_path: Path) -> Optional[SystemModule]:
        """分析Python模块"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content)

            # 提取模块信息
            module_name = file_path.stem
            description = ""
            classes = []
            functions = []
            dependencies = []

            # 提取模块文档字符串
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
                description = tree.body[0].value.s

            # 遍历AST节点
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "description": ast.get_docstring(node) or "",
                        "methods": [method.name for method in node.body if isinstance(method, ast.FunctionDef)],
                    }
                    classes.append(class_info)

                elif isinstance(node, ast.FunctionDef) and not any(
                    node in cls.body for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)
                ):
                    function_info = {
                        "name": node.name,
                        "description": ast.get_docstring(node) or "",
                        "args": [arg.arg for arg in node.args.args],
                    }
                    functions.append(function_info)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)

            # 去重依赖
            dependencies = list(set(dependencies))

            return SystemModule(
                module_name=module_name,
                file_path=str(file_path),
                description=description or f"{module_name}模块",
                classes=classes,
                functions=functions,
                dependencies=dependencies,
                version="1.0",
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
            )

        except Exception as e:
            logger.error(f"分析Python模块失败: {e}")
            return None

    def _save_generated_document(
        self, doc_id: str, doc_name: str, doc_type: str, template_id: str, content: str, file_path: str = None
    ):
        """保存生成的文档"""
        try:
            insert_sql = """
            INSERT INTO generated_documents (
                document_id, document_name, document_type, template_id,
                content, file_path, version, status
            ) VALUES (
                :document_id, :document_name, :document_type, :template_id,
                :content, :file_path, :version, :status
            )
            """

            with self.engine.connect() as conn:
                conn.execute(
                    text(insert_sql),
                    {
                        "document_id": doc_id,
                        "document_name": doc_name,
                        "document_type": doc_type,
                        "template_id": template_id,
                        "content": content,
                        "file_path": file_path,
                        "version": "1.0",
                        "status": "generated",
                    },
                )
                conn.commit()

        except Exception as e:
            logger.error(f"保存生成的文档失败: {e}")

    def _generate_readme(self, output_path: str):
        """生成README文件"""
        try:
            readme_content = """
# AI策略系统

一个基于人工智能的股票投资策略系统，提供智能选股、策略回测、风险管理等功能。

## 功能特性

- **AI模型管理**: 支持多种机器学习模型的训练、部署和管理
- **策略回测**: 提供历史数据回测功能，验证策略有效性
- **个性化定制**: 根据用户风险偏好提供个性化策略推荐
- **实时监控**: 系统性能监控和告警功能
- **文档自动生成**: 自动生成API文档、系统文档等

## 快速开始

### 环境要求

- Python 3.8+
- PostgreSQL 12+
- Redis 6.0+

### 安装步骤

1. 克隆代码库
```bash
git clone https://github.com/your-org/stock-school.git
cd stock-school
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置数据库
```bash
python manage.py init-db
```

4. 启动服务
```bash
python app.py
```

## 文档

- [API文档](docs/api_documentation.md)
- [系统架构](docs/system_architecture.md)
- [用户手册](docs/user_manual.md)
- [部署指南](docs/deployment_guide.md)

## 许可证

MIT License
"""

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(readme_content)

        except Exception as e:
            logger.error(f"生成README失败: {e}")

    # 模板内容方法
    def _get_api_doc_template(self) -> str:
        """获取API文档模板"""
        return """
# {{ project_name }} API文档

版本: {{ version }}
基础URL: {{ base_url }}
生成时间: {{ generated_at }}

## 概述

{{ project_name }}提供RESTful API接口，支持AI模型管理、策略回测、个性化推荐等功能。

## 认证

所有API请求需要在请求头中包含API密钥：

```
Authorization: Bearer YOUR_API_KEY
```

## API端点

{% for endpoint in endpoints %}
### {{ endpoint.method }} {{ endpoint.path }}

**功能**: {{ endpoint.description }}

**函数**: `{{ endpoint.function_name }}`

{% if endpoint.parameters %}
**参数**:
{% for param in endpoint.parameters %}
- `{{ param.name }}` ({{ param.type }}): {{ param.description }}
{% endfor %}
{% endif %}

{% if endpoint.responses %}
**响应**:
{% for response in endpoint.responses %}
- {{ response.status_code }}: {{ response.description }}
{% endfor %}
{% endif %}

{% if endpoint.examples %}
**示例**:
{% for example in endpoint.examples %}
```{{ example.language }}
{{ example.code }}
```
{% endfor %}
{% endif %}

---

{% endfor %}

## 错误处理

API使用标准HTTP状态码表示请求结果：

- 200: 请求成功
- 400: 请求参数错误
- 401: 认证失败
- 403: 权限不足
- 404: 资源不存在
- 500: 服务器内部错误

错误响应格式：

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述",
    "details": {}
  }
}
```
"""

    def _get_system_doc_template(self) -> str:
        """获取系统文档模板"""
        return """
# {{ project_name }} 系统架构文档

版本: {{ version }}
生成时间: {{ generated_at }}

## 系统概述

{{ project_name }}是一个基于人工智能的股票投资策略系统，采用微服务架构设计，提供模块化、可扩展的解决方案。

## 技术架构

### 核心技术栈

- **后端框架**: Flask
- **数据库**: PostgreSQL
- **缓存**: Redis
- **机器学习**: scikit-learn, TensorFlow
- **数据处理**: pandas, numpy
- **任务队列**: Celery

### 系统模块

{% for module in modules %}
#### {{ module.module_name }}

**文件路径**: `{{ module.file_path }}`

**描述**: {{ module.description }}

**版本**: {{ module.version }}

**最后修改**: {{ module.last_modified }}

{% if module.classes %}
**类**:
{% for class in module.classes %}
- `{{ class.name }}`: {{ class.description }}
{% endfor %}
{% endif %}

{% if module.functions %}
**函数**:
{% for function in module.functions %}
- `{{ function.name }}`: {{ function.description }}
{% endfor %}
{% endif %}

{% if module.dependencies %}
**依赖**:
{% for dep in module.dependencies %}
- {{ dep }}
{% endfor %}
{% endif %}

---

{% endfor %}

## 数据库设计

{% for table in database_tables %}
### {{ table.table_name }}

**描述**: {{ table.description }}

**字段**:

| 字段名 | 类型 | 可空 | 默认值 | 说明 |
|--------|------|------|--------|------|
{% for column in table.columns %}
| {{ column.name }} | {{ column.type }} | {{ 'YES' if column.nullable else 'NO' }} | {{ column.default or '-' }} | {{ column.comment or '-' }} |
{% endfor %}

{% if table.indexes %}
**索引**:
{% for index in table.indexes %}
- {{ index.name }}: {{ index.definition }}
{% endfor %}
{% endif %}

{% if table.relationships %}
**关系**:
{% for rel in table.relationships %}
- {{ rel.column_name }} -> {{ rel.foreign_table }}.{{ rel.foreign_column }}
{% endfor %}
{% endif %}

---

{% endfor %}

## 部署架构

系统支持多种部署方式：

1. **单机部署**: 适用于开发和测试环境
2. **集群部署**: 适用于生产环境，支持负载均衡和高可用
3. **容器化部署**: 基于Docker的容器化部署
4. **云原生部署**: 基于Kubernetes的云原生部署

## 性能优化

- **数据库优化**: 索引优化、查询优化、连接池管理
- **缓存策略**: Redis缓存、应用层缓存
- **异步处理**: Celery任务队列、异步IO
- **负载均衡**: Nginx负载均衡、服务发现

## 监控和运维

- **性能监控**: 系统指标监控、告警机制
- **日志管理**: 结构化日志、日志聚合
- **健康检查**: 服务健康检查、自动恢复
- **备份策略**: 数据备份、灾难恢复
"""

    def _get_user_manual_template(self) -> str:
        """获取用户手册模板"""
        return """
# {{ project_name }} 用户手册

版本: {{ version }}
生成时间: {{ generated_at }}

## 欢迎使用{{ project_name }}

{{ project_name }}是一个智能化的股票投资策略系统，帮助您制定和执行投资策略。

## 主要功能

{% for feature in features %}
### {{ feature.name }}

{{ feature.description }}

**相关接口**:
{% for endpoint in feature.endpoints %}
- `{{ endpoint }}`
{% endfor %}

{% endfor %}

## 快速入门

{% for tutorial in tutorials %}
### {{ tutorial.title }}

{{ tutorial.description }}

**步骤**:
{% for step in tutorial.steps %}
{{ loop.index }}. {{ step }}
{% endfor %}

{% endfor %}

## 常见问题

### Q: 如何获取API密钥？
A: 请联系系统管理员获取API密钥，或在用户设置页面生成。

### Q: 如何设置风险偏好？
A: 在个人设置中完成风险评估问卷，系统会自动设置您的风险偏好。

### Q: 回测结果如何解读？
A: 回测结果包含收益率、最大回撤、夏普比率等指标，详细说明请参考帮助文档。

## 技术支持

如有问题，请联系技术支持团队：
- 邮箱: support@example.com
- 电话: 400-123-4567
"""

    def _get_deployment_doc_template(self) -> str:
        """获取部署文档模板"""
        return """
# {{ project_name }} 部署指南

生成时间: {{ generated_at }}

## 系统要求

{% for req in requirements %}
- {{ req }}
{% endfor %}

## 安装步骤

{% for step in installation_steps %}
### 步骤{{ step.step }}: {{ step.title }}

{{ step.description }}

```bash
{{ step.command }}
```

{% endfor %}

## 配置说明

### 数据库配置

```yaml
database:
  url: {{ configuration.database.url }}
  pool_size: {{ configuration.database.pool_size }}
  max_overflow: {{ configuration.database.max_overflow }}
```

### Redis配置

```yaml
redis:
  url: {{ configuration.redis.url }}
  max_connections: {{ configuration.redis.max_connections }}
```

### API配置

```yaml
api:
  host: {{ configuration.api.host }}
  port: {{ configuration.api.port }}
  debug: {{ configuration.api.debug }}
```

## 生产环境部署

### 使用Docker

1. 构建镜像
```bash
docker build -t {{ project_name.lower() }}:latest .
```

2. 运行容器
```bash
docker run -d -p 5000:5000 {{ project_name.lower() }}:latest
```

### 使用Docker Compose

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/stockschool
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:12
    environment:
      - POSTGRES_DB=stockschool
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## 监控和维护

### 日志管理

- 应用日志: `/var/log/{{ project_name.lower() }}/app.log`
- 错误日志: `/var/log/{{ project_name.lower() }}/error.log`
- 访问日志: `/var/log/{{ project_name.lower() }}/access.log`

### 性能监控

建议使用以下工具进行监控：
- Prometheus + Grafana
- ELK Stack (Elasticsearch + Logstash + Kibana)
- APM工具 (如New Relic、DataDog)

### 备份策略

1. 数据库备份
```bash
pg_dump stockschool > backup_$(date +%Y%m%d_%H%M%S).sql
```

2. 配置文件备份
```bash
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz config/
```

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库服务是否运行
   - 验证连接字符串配置
   - 检查网络连接

2. **Redis连接失败**
   - 检查Redis服务状态
   - 验证Redis配置
   - 检查防火墙设置

3. **API响应慢**
   - 检查数据库查询性能
   - 监控系统资源使用
   - 优化缓存策略

### 日志分析

使用以下命令查看日志：

```bash
# 查看最新日志
tail -f /var/log/{{ project_name.lower() }}/app.log

# 搜索错误日志
grep "ERROR" /var/log/{{ project_name.lower() }}/app.log

# 分析访问模式
awk '{print $1}' /var/log/{{ project_name.lower() }}/access.log | sort | uniq -c | sort -nr
```
"""

    # 辅助查询方法
    def _get_api_endpoint(self, path: str, method: str) -> bool:
        """检查API端点是否存在"""
        try:
            query_sql = """
            SELECT COUNT(*)
            FROM api_endpoints
            WHERE path = :path AND method = :method
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"path": path, "method": method})
                count = result.scalar()

            return count > 0

        except Exception as e:
            logger.error(f"检查API端点失败: {e}")
            return False

    def _get_database_table(self, table_name: str) -> bool:
        """检查数据库表是否存在"""
        try:
            query_sql = """
            SELECT COUNT(*)
            FROM database_tables
            WHERE table_name = :table_name
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"table_name": table_name})
                count = result.scalar()

            return count > 0

        except Exception as e:
            logger.error(f"检查数据库表失败: {e}")
            return False

    def _get_system_module(self, module_name: str) -> bool:
        """检查系统模块是否存在"""
        try:
            query_sql = """
            SELECT COUNT(*)
            FROM system_modules
            WHERE module_name = :module_name
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"module_name": module_name})
                count = result.scalar()

            return count > 0

        except Exception as e:
            logger.error(f"检查系统模块失败: {e}")
            return False

    def _get_document_template(self, template_id: str) -> bool:
        """检查文档模板是否存在"""
        try:
            query_sql = """
            SELECT COUNT(*)
            FROM document_templates
            WHERE template_id = :template_id
            """

            with self.engine.connect() as conn:
                result = conn.execute(text(query_sql), {"template_id": template_id})
                count = result.scalar()

            return count > 0

        except Exception as e:
            logger.error(f"检查文档模板失败: {e}")
            return False
