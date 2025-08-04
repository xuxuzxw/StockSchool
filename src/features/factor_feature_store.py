#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因子特征商店集成
将因子计算结果集成到FeatureStore，提供版本管理、元数据管理和高性能查询功能
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import json
import hashlib
import logging
from sqlalchemy import text, MetaData, Table, Column, String, Float, Date, DateTime, Integer, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid

from src.config.unified_config import config
from src.utils.db import get_db_engine
from src.features.feature_store import FeatureStore

# 配置日志
logger = logging.getLogger(__name__)

class FactorVersion:
    """因子版本信息"""
    
    def __init__(self, version_id: str, factor_name: str, version: str, 
                 created_at: datetime, metadata: Dict[str, Any]):
        self.version_id = version_id
        self.factor_name = factor_name
        self.version = version
        self.created_at = created_at
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version_id': self.version_id,
            'factor_name': self.factor_name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }

class FactorMetadata:
    """因子元数据"""
    
    def __init__(self, factor_name: str, factor_type: str, category: str,
                 description: str, formula: str, parameters: Dict[str, Any],
                 data_requirements: List[str], update_frequency: str,
                 data_schema: Dict[str, str], tags: List[str] = None):
        self.factor_name = factor_name
        self.factor_type = factor_type
        self.category = category
        self.description = description
        self.formula = formula
        self.parameters = parameters
        self.data_requirements = data_requirements
        self.update_frequency = update_frequency
        self.data_schema = data_schema
        self.tags = tags or []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'factor_type': self.factor_type,
            'category': self.category,
            'description': self.description,
            'formula': self.formula,
            'parameters': self.parameters,
            'data_requirements': self.data_requirements,
            'update_frequency': self.update_frequency,
            'data_schema': self.data_schema,
            'tags': self.tags
        }

class FactorFeatureStore:
    """因子特征商店"""
    
    def __init__(self, engine=None):
        """
        初始化因子特征商店
        
        Args:
            engine: 数据库引擎
        """
        self.engine = engine or get_db_engine()
        self.feature_store = FeatureStore(self.engine)
        self._init_factor_tables()
    
    def _init_factor_tables(self):
        """初始化因子相关表"""
        try:
            with self.engine.connect() as conn:
                # 创建因子版本表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS factor_versions (
                        version_id VARCHAR(100) PRIMARY KEY,
                        factor_name VARCHAR(100) NOT NULL,
                        version VARCHAR(50) NOT NULL,
                        algorithm_hash VARCHAR(64) NOT NULL,
                        data_hash VARCHAR(64),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{}',
                        is_active BOOLEAN DEFAULT true,
                        UNIQUE(factor_name, version)
                    )
                """))
                
                # 创建因子元数据表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS factor_metadata (
                        factor_name VARCHAR(100) PRIMARY KEY,
                        factor_type VARCHAR(50) NOT NULL,
                        category VARCHAR(100) NOT NULL,
                        description TEXT,
                        formula TEXT,
                        parameters JSONB DEFAULT '{}',
                        data_requirements TEXT[] DEFAULT ARRAY[]::TEXT[],
                        update_frequency VARCHAR(20) DEFAULT 'daily',
                        data_schema JSONB DEFAULT '{}',
                        tags TEXT[] DEFAULT ARRAY[]::TEXT[],
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # 创建因子数据血缘表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS factor_lineage (
                        lineage_id VARCHAR(100) PRIMARY KEY,
                        factor_name VARCHAR(100) NOT NULL,
                        version_id VARCHAR(100) NOT NULL,
                        source_tables TEXT[] DEFAULT ARRAY[]::TEXT[],
                        source_factors TEXT[] DEFAULT ARRAY[]::TEXT[],
                        transformation_logic TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (version_id) REFERENCES factor_versions(version_id)
                    )
                """))
                
                # 创建因子质量指标表
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS factor_quality_metrics (
                        metric_id VARCHAR(100) PRIMARY KEY,
                        factor_name VARCHAR(100) NOT NULL,
                        version_id VARCHAR(100) NOT NULL,
                        metric_date DATE NOT NULL,
                        completeness_score FLOAT,
                        consistency_score FLOAT,
                        accuracy_score FLOAT,
                        timeliness_score FLOAT,
                        uniqueness_score FLOAT,
                        validity_score FLOAT,
                        overall_score FLOAT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (version_id) REFERENCES factor_versions(version_id)
                    )
                """))
                
                # 创建索引
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_factor_versions_name ON factor_versions(factor_name)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_factor_versions_active ON factor_versions(is_active)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_factor_metadata_type ON factor_metadata(factor_type)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_factor_lineage_factor ON factor_lineage(factor_name)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_factor_quality_date ON factor_quality_metrics(metric_date)"))
                
                logger.info("因子特征商店表初始化完成")
                
        except Exception as e:
            logger.error(f"初始化因子特征商店表失败: {e}")
            raise
    
    def register_factor(self, metadata: FactorMetadata) -> str:
        """
        注册因子元数据
        
        Args:
            metadata: 因子元数据
            
        Returns:
            str: 注册成功的因子名称
        """
        try:
            with self.engine.connect() as conn:
                # 插入或更新因子元数据
                conn.execute(text("""
                    INSERT INTO factor_metadata (
                        factor_name, factor_type, category, description, formula,
                        parameters, data_requirements, update_frequency, data_schema, tags,
                        created_at, updated_at
                    ) VALUES (
                        :factor_name, :factor_type, :category, :description, :formula,
                        :parameters, :data_requirements, :update_frequency, :data_schema, :tags,
                        :created_at, :updated_at
                    )
                    ON CONFLICT (factor_name) DO UPDATE SET
                        factor_type = EXCLUDED.factor_type,
                        category = EXCLUDED.category,
                        description = EXCLUDED.description,
                        formula = EXCLUDED.formula,
                        parameters = EXCLUDED.parameters,
                        data_requirements = EXCLUDED.data_requirements,
                        update_frequency = EXCLUDED.update_frequency,
                        data_schema = EXCLUDED.data_schema,
                        tags = EXCLUDED.tags,
                        updated_at = EXCLUDED.updated_at
                """), {
                    'factor_name': metadata.factor_name,
                    'factor_type': metadata.factor_type,
                    'category': metadata.category,
                    'description': metadata.description,
                    'formula': metadata.formula,
                    'parameters': json.dumps(metadata.parameters),
                    'data_requirements': metadata.data_requirements,
                    'update_frequency': metadata.update_frequency,
                    'data_schema': json.dumps(metadata.data_schema),
                    'tags': metadata.tags,
                    'created_at': datetime.now(),
                    'updated_at': datetime.now()
                })
                
                logger.info(f"因子 {metadata.factor_name} 元数据注册成功")
                return metadata.factor_name
                
        except Exception as e:
            logger.error(f"注册因子元数据失败: {e}")
            raise
    
    def create_factor_version(self, factor_name: str, algorithm_code: str, 
                            parameters: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """
        创建因子版本
        
        Args:
            factor_name: 因子名称
            algorithm_code: 算法代码
            parameters: 算法参数
            metadata: 版本元数据
            
        Returns:
            str: 版本ID
        """
        try:
            # 生成算法哈希
            algorithm_hash = self._generate_algorithm_hash(algorithm_code, parameters)
            
            # 生成版本号
            version = self._generate_version_number(factor_name)
            
            # 生成版本ID
            version_id = f"{factor_name}_v{version}_{algorithm_hash[:8]}"
            
            with self.engine.connect() as conn:
                # 检查是否已存在相同的算法哈希
                result = conn.execute(text("""
                    SELECT version_id FROM factor_versions 
                    WHERE factor_name = :factor_name AND algorithm_hash = :algorithm_hash
                """), {
                    'factor_name': factor_name,
                    'algorithm_hash': algorithm_hash
                })
                
                existing_version = result.fetchone()
                if existing_version:
                    logger.info(f"因子 {factor_name} 已存在相同算法版本: {existing_version.version_id}")
                    return existing_version.version_id
                
                # 创建新版本
                conn.execute(text("""
                    INSERT INTO factor_versions (
                        version_id, factor_name, version, algorithm_hash, 
                        created_at, metadata, is_active
                    ) VALUES (
                        :version_id, :factor_name, :version, :algorithm_hash,
                        :created_at, :metadata, :is_active
                    )
                """), {
                    'version_id': version_id,
                    'factor_name': factor_name,
                    'version': version,
                    'algorithm_hash': algorithm_hash,
                    'created_at': datetime.now(),
                    'metadata': json.dumps(metadata or {}),
                    'is_active': True
                })
                
                logger.info(f"因子 {factor_name} 版本 {version_id} 创建成功")
                return version_id
                
        except Exception as e:
            logger.error(f"创建因子版本失败: {e}")
            raise
    
    def store_factor_data(self, factor_name: str, version_id: str, data: pd.DataFrame,
                         partition_date: date = None) -> bool:
        """
        存储因子数据到特征商店
        
        Args:
            factor_name: 因子名称
            version_id: 版本ID
            data: 因子数据
            partition_date: 分区日期
            
        Returns:
            bool: 存储是否成功
        """
        try:
            # 验证数据格式
            required_columns = ['ts_code', 'factor_date', factor_name]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"数据缺少必要列: {missing_columns}")
            
            # 添加版本信息
            data_with_version = data.copy()
            data_with_version['version_id'] = version_id
            data_with_version['created_at'] = datetime.now()
            
            # 生成数据哈希
            data_hash = self._generate_data_hash(data)
            
            # 更新版本表中的数据哈希
            with self.engine.connect() as conn:
                conn.execute(text("""
                    UPDATE factor_versions 
                    SET data_hash = :data_hash 
                    WHERE version_id = :version_id
                """), {
                    'data_hash': data_hash,
                    'version_id': version_id
                })
            
            # 存储到特征商店
            feature_name = f"{factor_name}_{version_id}"
            success = self.feature_store.store_features(
                feature_name=feature_name,
                data=data_with_version,
                partition_date=partition_date
            )
            
            if success:
                logger.info(f"因子数据 {factor_name} 版本 {version_id} 存储成功")
                
                # 记录数据血缘
                self._record_data_lineage(factor_name, version_id, data)
                
                # 计算数据质量指标
                self._calculate_quality_metrics(factor_name, version_id, data, partition_date)
            
            return success
            
        except Exception as e:
            logger.error(f"存储因子数据失败: {e}")
            return False
    
    def get_factor_data(self, factor_name: str, version_id: str = None,
                       ts_codes: List[str] = None, start_date: date = None,
                       end_date: date = None) -> pd.DataFrame:
        """
        获取因子数据
        
        Args:
            factor_name: 因子名称
            version_id: 版本ID，如果为None则获取最新版本
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 因子数据
        """
        try:
            # 如果没有指定版本，获取最新活跃版本
            if not version_id:
                version_id = self._get_latest_version(factor_name)
                if not version_id:
                    raise ValueError(f"因子 {factor_name} 没有可用版本")
            
            # 从特征商店获取数据
            feature_name = f"{factor_name}_{version_id}"
            data = self.feature_store.get_features(
                feature_name=feature_name,
                entity_ids=ts_codes,
                start_date=start_date,
                end_date=end_date
            )
            
            # 移除版本信息列
            if 'version_id' in data.columns:
                data = data.drop('version_id', axis=1)
            if 'created_at' in data.columns:
                data = data.drop('created_at', axis=1)
            
            logger.info(f"获取因子数据 {factor_name} 版本 {version_id}，共 {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"获取因子数据失败: {e}")
            return pd.DataFrame()
    
    def get_factor_metadata(self, factor_name: str = None) -> Union[FactorMetadata, List[FactorMetadata]]:
        """
        获取因子元数据
        
        Args:
            factor_name: 因子名称，如果为None则获取所有因子
            
        Returns:
            Union[FactorMetadata, List[FactorMetadata]]: 因子元数据
        """
        try:
            with self.engine.connect() as conn:
                if factor_name:
                    # 获取单个因子元数据
                    result = conn.execute(text("""
                        SELECT * FROM factor_metadata WHERE factor_name = :factor_name
                    """), {'factor_name': factor_name})
                    
                    row = result.fetchone()
                    if not row:
                        return None
                    
                    return FactorMetadata(
                        factor_name=row.factor_name,
                        factor_type=row.factor_type,
                        category=row.category,
                        description=row.description,
                        formula=row.formula,
                        parameters=json.loads(row.parameters) if row.parameters else {},
                        data_requirements=row.data_requirements,
                        update_frequency=row.update_frequency,
                        data_schema=json.loads(row.data_schema) if row.data_schema else {},
                        tags=row.tags
                    )
                else:
                    # 获取所有因子元数据
                    result = conn.execute(text("SELECT * FROM factor_metadata ORDER BY factor_name"))
                    
                    metadata_list = []
                    for row in result:
                        metadata_list.append(FactorMetadata(
                            factor_name=row.factor_name,
                            factor_type=row.factor_type,
                            category=row.category,
                            description=row.description,
                            formula=row.formula,
                            parameters=json.loads(row.parameters) if row.parameters else {},
                            data_requirements=row.data_requirements,
                            update_frequency=row.update_frequency,
                            data_schema=json.loads(row.data_schema) if row.data_schema else {},
                            tags=row.tags
                        ))
                    
                    return metadata_list
                    
        except Exception as e:
            logger.error(f"获取因子元数据失败: {e}")
            return None if factor_name else []
    
    def get_factor_versions(self, factor_name: str) -> List[FactorVersion]:
        """
        获取因子版本列表
        
        Args:
            factor_name: 因子名称
            
        Returns:
            List[FactorVersion]: 版本列表
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM factor_versions 
                    WHERE factor_name = :factor_name 
                    ORDER BY created_at DESC
                """), {'factor_name': factor_name})
                
                versions = []
                for row in result:
                    versions.append(FactorVersion(
                        version_id=row.version_id,
                        factor_name=row.factor_name,
                        version=row.version,
                        created_at=row.created_at,
                        metadata=json.loads(row.metadata) if row.metadata else {}
                    ))
                
                return versions
                
        except Exception as e:
            logger.error(f"获取因子版本失败: {e}")
            return []
    
    def activate_version(self, version_id: str) -> bool:
        """
        激活指定版本
        
        Args:
            version_id: 版本ID
            
        Returns:
            bool: 激活是否成功
        """
        try:
            with self.engine.connect() as conn:
                # 获取因子名称
                result = conn.execute(text("""
                    SELECT factor_name FROM factor_versions WHERE version_id = :version_id
                """), {'version_id': version_id})
                
                row = result.fetchone()
                if not row:
                    raise ValueError(f"版本 {version_id} 不存在")
                
                factor_name = row.factor_name
                
                # 先将该因子的所有版本设为非活跃
                conn.execute(text("""
                    UPDATE factor_versions 
                    SET is_active = false 
                    WHERE factor_name = :factor_name
                """), {'factor_name': factor_name})
                
                # 激活指定版本
                conn.execute(text("""
                    UPDATE factor_versions 
                    SET is_active = true 
                    WHERE version_id = :version_id
                """), {'version_id': version_id})
                
                logger.info(f"版本 {version_id} 激活成功")
                return True
                
        except Exception as e:
            logger.error(f"激活版本失败: {e}")
            return False
    
    def get_factor_lineage(self, factor_name: str, version_id: str = None) -> Dict[str, Any]:
        """
        获取因子数据血缘
        
        Args:
            factor_name: 因子名称
            version_id: 版本ID
            
        Returns:
            Dict[str, Any]: 血缘信息
        """
        try:
            with self.engine.connect() as conn:
                query = """
                    SELECT fl.*, fv.version, fv.created_at as version_created_at
                    FROM factor_lineage fl
                    JOIN factor_versions fv ON fl.version_id = fv.version_id
                    WHERE fl.factor_name = :factor_name
                """
                params = {'factor_name': factor_name}
                
                if version_id:
                    query += " AND fl.version_id = :version_id"
                    params['version_id'] = version_id
                
                query += " ORDER BY fv.created_at DESC"
                
                result = conn.execute(text(query), params)
                
                lineage_info = []
                for row in result:
                    lineage_info.append({
                        'lineage_id': row.lineage_id,
                        'version_id': row.version_id,
                        'version': row.version,
                        'source_tables': row.source_tables,
                        'source_factors': row.source_factors,
                        'transformation_logic': row.transformation_logic,
                        'created_at': row.created_at.isoformat(),
                        'version_created_at': row.version_created_at.isoformat()
                    })
                
                return {
                    'factor_name': factor_name,
                    'lineage': lineage_info
                }
                
        except Exception as e:
            logger.error(f"获取因子血缘失败: {e}")
            return {}
    
    def get_quality_metrics(self, factor_name: str, version_id: str = None,
                          start_date: date = None, end_date: date = None) -> pd.DataFrame:
        """
        获取因子质量指标
        
        Args:
            factor_name: 因子名称
            version_id: 版本ID
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame: 质量指标数据
        """
        try:
            with self.engine.connect() as conn:
                query = """
                    SELECT fqm.*, fv.version
                    FROM factor_quality_metrics fqm
                    JOIN factor_versions fv ON fqm.version_id = fv.version_id
                    WHERE fqm.factor_name = :factor_name
                """
                params = {'factor_name': factor_name}
                
                if version_id:
                    query += " AND fqm.version_id = :version_id"
                    params['version_id'] = version_id
                
                if start_date:
                    query += " AND fqm.metric_date >= :start_date"
                    params['start_date'] = start_date
                
                if end_date:
                    query += " AND fqm.metric_date <= :end_date"
                    params['end_date'] = end_date
                
                query += " ORDER BY fqm.metric_date DESC"
                
                result = conn.execute(text(query), params)
                
                data = []
                for row in result:
                    data.append({
                        'metric_id': row.metric_id,
                        'factor_name': row.factor_name,
                        'version_id': row.version_id,
                        'version': row.version,
                        'metric_date': row.metric_date,
                        'completeness_score': row.completeness_score,
                        'consistency_score': row.consistency_score,
                        'accuracy_score': row.accuracy_score,
                        'timeliness_score': row.timeliness_score,
                        'uniqueness_score': row.uniqueness_score,
                        'validity_score': row.validity_score,
                        'overall_score': row.overall_score,
                        'created_at': row.created_at
                    })
                
                return pd.DataFrame(data)
                
        except Exception as e:
            logger.error(f"获取质量指标失败: {e}")
            return pd.DataFrame()
    
    def search_factors(self, query: str, factor_type: str = None, 
                      category: str = None, tags: List[str] = None) -> List[FactorMetadata]:
        """
        搜索因子
        
        Args:
            query: 搜索关键词
            factor_type: 因子类型
            category: 因子分类
            tags: 标签列表
            
        Returns:
            List[FactorMetadata]: 搜索结果
        """
        try:
            with self.engine.connect() as conn:
                sql_query = """
                    SELECT * FROM factor_metadata 
                    WHERE (factor_name ILIKE :query OR description ILIKE :query)
                """
                params = {'query': f'%{query}%'}
                
                if factor_type:
                    sql_query += " AND factor_type = :factor_type"
                    params['factor_type'] = factor_type
                
                if category:
                    sql_query += " AND category = :category"
                    params['category'] = category
                
                if tags:
                    sql_query += " AND tags && :tags"
                    params['tags'] = tags
                
                sql_query += " ORDER BY factor_name"
                
                result = conn.execute(text(sql_query), params)
                
                factors = []
                for row in result:
                    factors.append(FactorMetadata(
                        factor_name=row.factor_name,
                        factor_type=row.factor_type,
                        category=row.category,
                        description=row.description,
                        formula=row.formula,
                        parameters=json.loads(row.parameters) if row.parameters else {},
                        data_requirements=row.data_requirements,
                        update_frequency=row.update_frequency,
                        data_schema=json.loads(row.data_schema) if row.data_schema else {},
                        tags=row.tags
                    ))
                
                return factors
                
        except Exception as e:
            logger.error(f"搜索因子失败: {e}")
            return []
    
    def _generate_algorithm_hash(self, algorithm_code: str, parameters: Dict[str, Any]) -> str:
        """生成算法哈希"""
        content = f"{algorithm_code}_{json.dumps(parameters, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """生成数据哈希"""
        # 使用数据的形状和部分内容生成哈希
        content = f"{data.shape}_{data.dtypes.to_dict()}_{data.head().to_string()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_version_number(self, factor_name: str) -> str:
        """生成版本号"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT MAX(CAST(SUBSTRING(version FROM '^[0-9]+') AS INTEGER)) as max_version
                    FROM factor_versions 
                    WHERE factor_name = :factor_name
                """), {'factor_name': factor_name})
                
                row = result.fetchone()
                max_version = row.max_version if row.max_version else 0
                
                return str(max_version + 1)
                
        except Exception as e:
            logger.error(f"生成版本号失败: {e}")
            return "1"
    
    def _get_latest_version(self, factor_name: str) -> Optional[str]:
        """获取最新活跃版本"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT version_id FROM factor_versions 
                    WHERE factor_name = :factor_name AND is_active = true
                    ORDER BY created_at DESC LIMIT 1
                """), {'factor_name': factor_name})
                
                row = result.fetchone()
                return row.version_id if row else None
                
        except Exception as e:
            logger.error(f"获取最新版本失败: {e}")
            return None
    
    def _record_data_lineage(self, factor_name: str, version_id: str, data: pd.DataFrame):
        """记录数据血缘"""
        try:
            lineage_id = f"{version_id}_lineage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 分析数据来源（这里简化处理，实际应该根据具体情况分析）
            source_tables = ['stock_daily']  # 默认来源表
            if 'financial' in factor_name.lower():
                source_tables.extend(['income_statement', 'balance_sheet', 'cash_flow'])
            
            source_factors = []  # 依赖的其他因子
            
            transformation_logic = f"计算因子 {factor_name}，数据记录数: {len(data)}"
            
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO factor_lineage (
                        lineage_id, factor_name, version_id, source_tables,
                        source_factors, transformation_logic, created_at
                    ) VALUES (
                        :lineage_id, :factor_name, :version_id, :source_tables,
                        :source_factors, :transformation_logic, :created_at
                    )
                """), {
                    'lineage_id': lineage_id,
                    'factor_name': factor_name,
                    'version_id': version_id,
                    'source_tables': source_tables,
                    'source_factors': source_factors,
                    'transformation_logic': transformation_logic,
                    'created_at': datetime.now()
                })
                
        except Exception as e:
            logger.error(f"记录数据血缘失败: {e}")
    
    def _calculate_quality_metrics(self, factor_name: str, version_id: str, 
                                 data: pd.DataFrame, metric_date: date = None):
        """计算数据质量指标"""
        try:
            if metric_date is None:
                metric_date = date.today()
            
            metric_id = f"{version_id}_quality_{metric_date.strftime('%Y%m%d')}"
            
            # 计算各项质量指标
            total_records = len(data)
            
            # 完整性评分（非空值比例）
            factor_column = factor_name
            if factor_column in data.columns:
                non_null_count = data[factor_column].notna().sum()
                completeness_score = (non_null_count / total_records) * 100 if total_records > 0 else 0
            else:
                completeness_score = 0
            
            # 一致性评分（数据类型一致性）
            consistency_score = 100  # 简化处理
            
            # 准确性评分（合理范围内的值比例）
            if factor_column in data.columns and data[factor_column].notna().sum() > 0:
                # 使用3σ原则检测异常值
                mean_val = data[factor_column].mean()
                std_val = data[factor_column].std()
                if std_val > 0:
                    valid_range = (mean_val - 3 * std_val, mean_val + 3 * std_val)
                    valid_count = data[
                        (data[factor_column] >= valid_range[0]) & 
                        (data[factor_column] <= valid_range[1])
                    ][factor_column].count()
                    accuracy_score = (valid_count / non_null_count) * 100
                else:
                    accuracy_score = 100
            else:
                accuracy_score = 0
            
            # 及时性评分（数据更新及时性）
            timeliness_score = 100  # 简化处理
            
            # 唯一性评分（重复记录检查）
            if 'ts_code' in data.columns and 'factor_date' in data.columns:
                unique_combinations = data[['ts_code', 'factor_date']].drop_duplicates()
                uniqueness_score = (len(unique_combinations) / total_records) * 100 if total_records > 0 else 0
            else:
                uniqueness_score = 100
            
            # 有效性评分（数据格式正确性）
            validity_score = 100  # 简化处理
            
            # 总体评分
            scores = [completeness_score, consistency_score, accuracy_score, 
                     timeliness_score, uniqueness_score, validity_score]
            overall_score = sum(scores) / len(scores)
            
            # 存储质量指标
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO factor_quality_metrics (
                        metric_id, factor_name, version_id, metric_date,
                        completeness_score, consistency_score, accuracy_score,
                        timeliness_score, uniqueness_score, validity_score,
                        overall_score, created_at
                    ) VALUES (
                        :metric_id, :factor_name, :version_id, :metric_date,
                        :completeness_score, :consistency_score, :accuracy_score,
                        :timeliness_score, :uniqueness_score, :validity_score,
                        :overall_score, :created_at
                    )
                    ON CONFLICT (metric_id) DO UPDATE SET
                        completeness_score = EXCLUDED.completeness_score,
                        consistency_score = EXCLUDED.consistency_score,
                        accuracy_score = EXCLUDED.accuracy_score,
                        timeliness_score = EXCLUDED.timeliness_score,
                        uniqueness_score = EXCLUDED.uniqueness_score,
                        validity_score = EXCLUDED.validity_score,
                        overall_score = EXCLUDED.overall_score
                """), {
                    'metric_id': metric_id,
                    'factor_name': factor_name,
                    'version_id': version_id,
                    'metric_date': metric_date,
                    'completeness_score': completeness_score,
                    'consistency_score': consistency_score,
                    'accuracy_score': accuracy_score,
                    'timeliness_score': timeliness_score,
                    'uniqueness_score': uniqueness_score,
                    'validity_score': validity_score,
                    'overall_score': overall_score,
                    'created_at': datetime.now()
                })
                
        except Exception as e:
            logger.error(f"计算质量指标失败: {e}")
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            with self.engine.connect() as conn:
                # 因子数量统计
                result = conn.execute(text("SELECT COUNT(*) as factor_count FROM factor_metadata"))
                factor_count = result.fetchone().factor_count
                
                # 版本数量统计
                result = conn.execute(text("SELECT COUNT(*) as version_count FROM factor_versions"))
                version_count = result.fetchone().version_count
                
                # 活跃版本数量
                result = conn.execute(text("SELECT COUNT(*) as active_count FROM factor_versions WHERE is_active = true"))
                active_count = result.fetchone().active_count
                
                # 按类型统计
                result = conn.execute(text("""
                    SELECT factor_type, COUNT(*) as count 
                    FROM factor_metadata 
                    GROUP BY factor_type
                """))
                type_stats = {row.factor_type: row.count for row in result}
                
                # 质量指标统计
                result = conn.execute(text("""
                    SELECT 
                        AVG(overall_score) as avg_quality,
                        MIN(overall_score) as min_quality,
                        MAX(overall_score) as max_quality
                    FROM factor_quality_metrics
                    WHERE metric_date >= CURRENT_DATE - INTERVAL '30 days'
                """))
                quality_row = result.fetchone()
                
                return {
                    'factor_count': factor_count,
                    'version_count': version_count,
                    'active_version_count': active_count,
                    'type_statistics': type_stats,
                    'quality_statistics': {
                        'average_quality': float(quality_row.avg_quality) if quality_row.avg_quality else 0,
                        'min_quality': float(quality_row.min_quality) if quality_row.min_quality else 0,
                        'max_quality': float(quality_row.max_quality) if quality_row.max_quality else 0
                    },
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}