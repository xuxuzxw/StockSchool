#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据压缩和归档管理器
实现历史因子数据的压缩存储、归档和清理策略
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from loguru import logger
from sqlalchemy import text, create_engine
import gzip
import pickle
import json
import os
import shutil
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from .factor_models import FactorType, FactorValue


class CompressionLevel(Enum):
    """压缩级别枚举"""
    NONE = 0
    LOW = 1
    MEDIUM = 6
    HIGH = 9


class ArchivePolicy(Enum):
    """归档策略枚举"""
    BY_DATE = "date"  # 按日期归档
    BY_SIZE = "size"  # 按大小归档
    BY_COUNT = "count"  # 按数量归档


class DataCompressionConfig:
    """数据压缩配置"""
    
    def __init__(self):
        # 压缩配置
        self.compression_level = CompressionLevel.MEDIUM
        self.compression_threshold_mb = 10  # 超过10MB的数据进行压缩
        
        # 归档配置
        self.archive_policy = ArchivePolicy.BY_DATE
        self.archive_after_days = 90  # 90天后归档
        self.archive_path = "./data/archive"
        
        # 清理配置
        self.cleanup_after_days = 365  # 365天后清理
        self.keep_compressed_days = 180  # 保留压缩数据180天
        
        # 备份配置
        self.backup_enabled = True
        self.backup_path = "./data/backup"
        self.backup_retention_days = 30
        
        # 性能配置
        self.batch_size = 10000  # 批处理大小
        self.max_workers = 4  # 最大工作线程数


class CompressionEngine:
    """压缩引擎"""
    
    def __init__(self, config: DataCompressionConfig):
        """初始化压缩引擎"""
        self.config = config
    
    def compress_data(self, data: Any, compression_level: CompressionLevel = None) -> bytes:
        """压缩数据"""
        if compression_level is None:
            compression_level = self.config.compression_level
        
        try:
            # 序列化数据
            serialized_data = pickle.dumps(data)
            
            # 检查是否需要压缩
            if len(serialized_data) < self.config.compression_threshold_mb * 1024 * 1024:
                return serialized_data
            
            # 压缩数据
            compressed_data = gzip.compress(
                serialized_data, 
                compresslevel=compression_level.value
            )
            
            # 添加压缩标记
            return b'GZIP:' + compressed_data
            
        except Exception as e:
            logger.error(f"数据压缩失败: {e}")
            raise
    
    def decompress_data(self, compressed_data: bytes) -> Any:
        """解压缩数据"""
        try:
            # 检查是否压缩
            if compressed_data.startswith(b'GZIP:'):
                # 移除压缩标记并解压
                compressed_data = compressed_data[5:]
                decompressed_data = gzip.decompress(compressed_data)
            else:
                decompressed_data = compressed_data
            
            # 反序列化
            return pickle.loads(decompressed_data)
            
        except Exception as e:
            logger.error(f"数据解压缩失败: {e}")
            raise
    
    def get_compression_ratio(self, original_data: Any) -> float:
        """获取压缩比"""
        try:
            original_size = len(pickle.dumps(original_data))
            compressed_size = len(self.compress_data(original_data))
            
            return compressed_size / original_size
            
        except Exception as e:
            logger.error(f"计算压缩比失败: {e}")
            return 1.0


class ArchiveManager:
    """归档管理器"""
    
    def __init__(self, engine, config: DataCompressionConfig):
        """初始化归档管理器"""
        self.engine = engine
        self.config = config
        self.compression_engine = CompressionEngine(config)
        
        # 创建归档目录
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保目录存在"""
        directories = [
            self.config.archive_path,
            self.config.backup_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_archivable_data(self, cutoff_date: date) -> List[Dict[str, Any]]:
        """获取可归档的数据"""
        try:
            query = text("""
                SELECT ts_code, factor_name, COUNT(*) as record_count,
                       MIN(factor_date) as min_date, MAX(factor_date) as max_date,
                       AVG(CASE WHEN factor_value IS NOT NULL THEN 1 ELSE 0 END) as data_quality
                FROM factor_values 
                WHERE factor_date < :cutoff_date
                GROUP BY ts_code, factor_name
                HAVING COUNT(*) > 100  -- 只归档有足够数据的因子
                ORDER BY ts_code, factor_name
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'cutoff_date': cutoff_date})
                
                archivable_data = []
                for row in result.fetchall():
                    archivable_data.append({
                        'ts_code': row.ts_code,
                        'factor_name': row.factor_name,
                        'record_count': row.record_count,
                        'min_date': row.min_date,
                        'max_date': row.max_date,
                        'data_quality': float(row.data_quality)
                    })
                
                return archivable_data
                
        except Exception as e:
            logger.error(f"获取可归档数据失败: {e}")
            return []
    
    def archive_factor_data(self, ts_code: str, factor_name: str,
                          cutoff_date: date) -> bool:
        """归档单个因子数据"""
        try:
            # 获取要归档的数据
            query = text("""
                SELECT factor_date, factor_value, created_time
                FROM factor_values 
                WHERE ts_code = :ts_code 
                AND factor_name = :factor_name
                AND factor_date < :cutoff_date
                ORDER BY factor_date
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'ts_code': ts_code,
                    'factor_name': factor_name,
                    'cutoff_date': cutoff_date
                })
                
                data = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if data.empty:
                return True
            
            # 压缩数据
            compressed_data = self.compression_engine.compress_data(data)
            
            # 生成归档文件路径
            archive_filename = f"{ts_code}_{factor_name}_{cutoff_date.isoformat()}.gz"
            archive_filepath = Path(self.config.archive_path) / archive_filename
            
            # 保存压缩数据
            with open(archive_filepath, 'wb') as f:
                f.write(compressed_data)
            
            # 记录归档信息
            self._record_archive_info(ts_code, factor_name, cutoff_date, 
                                    archive_filepath, len(data), len(compressed_data))
            
            # 从主表删除已归档的数据
            delete_query = text("""
                DELETE FROM factor_values 
                WHERE ts_code = :ts_code 
                AND factor_name = :factor_name
                AND factor_date < :cutoff_date
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(delete_query, {
                    'ts_code': ts_code,
                    'factor_name': factor_name,
                    'cutoff_date': cutoff_date
                })
                
                deleted_count = result.rowcount
            
            logger.info(f"成功归档 {ts_code} 的 {factor_name} 因子，"
                       f"删除 {deleted_count} 条记录，压缩文件: {archive_filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"归档因子数据失败: {e}")
            return False
    
    def _record_archive_info(self, ts_code: str, factor_name: str, cutoff_date: date,
                           archive_path: Path, original_count: int, compressed_size: int):
        """记录归档信息"""
        try:
            archive_info = {
                'ts_code': ts_code,
                'factor_name': factor_name,
                'cutoff_date': cutoff_date,
                'archive_path': str(archive_path),
                'original_record_count': original_count,
                'compressed_size_bytes': compressed_size,
                'archive_date': datetime.now(),
                'status': 'archived'
            }
            
            df = pd.DataFrame([archive_info])
            df.to_sql(
                'factor_archive_log',
                self.engine,
                if_exists='append',
                index=False
            )
            
        except Exception as e:
            logger.error(f"记录归档信息失败: {e}")
    
    def restore_archived_data(self, ts_code: str, factor_name: str,
                            archive_date: date) -> bool:
        """恢复归档数据"""
        try:
            # 查找归档文件
            archive_filename = f"{ts_code}_{factor_name}_{archive_date.isoformat()}.gz"
            archive_filepath = Path(self.config.archive_path) / archive_filename
            
            if not archive_filepath.exists():
                logger.error(f"归档文件不存在: {archive_filepath}")
                return False
            
            # 读取并解压数据
            with open(archive_filepath, 'rb') as f:
                compressed_data = f.read()
            
            data = self.compression_engine.decompress_data(compressed_data)
            
            # 恢复到数据库
            data.to_sql(
                'factor_values',
                self.engine,
                if_exists='append',
                index=False
            )
            
            logger.info(f"成功恢复 {ts_code} 的 {factor_name} 因子数据，"
                       f"恢复 {len(data)} 条记录")
            
            return True
            
        except Exception as e:
            logger.error(f"恢复归档数据失败: {e}")
            return False
    
    def batch_archive(self, cutoff_date: date, max_workers: int = None) -> Dict[str, Any]:
        """批量归档"""
        if max_workers is None:
            max_workers = self.config.max_workers
        
        start_time = datetime.now()
        
        # 获取可归档的数据
        archivable_data = self.get_archivable_data(cutoff_date)
        
        if not archivable_data:
            logger.info("没有需要归档的数据")
            return {'total_tasks': 0, 'successful': 0, 'failed': 0}
        
        logger.info(f"开始批量归档，共 {len(archivable_data)} 个任务")
        
        # 统计信息
        stats = {
            'total_tasks': len(archivable_data),
            'successful': 0,
            'failed': 0,
            'total_records_archived': 0,
            'total_space_saved': 0
        }
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_task = {}
            for item in archivable_data:
                future = executor.submit(
                    self.archive_factor_data,
                    item['ts_code'],
                    item['factor_name'],
                    cutoff_date
                )
                future_to_task[future] = item
            
            # 收集结果
            for future in future_to_task:
                item = future_to_task[future]
                try:
                    success = future.result(timeout=300)  # 5分钟超时
                    
                    if success:
                        stats['successful'] += 1
                        stats['total_records_archived'] += item['record_count']
                    else:
                        stats['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"归档任务失败: {e}")
                    stats['failed'] += 1
        
        # 记录执行时间
        execution_time = datetime.now() - start_time
        stats['execution_time'] = execution_time
        
        logger.info("=== 批量归档统计 ===")
        logger.info(f"总任务数: {stats['total_tasks']}")
        logger.info(f"成功归档: {stats['successful']}")
        logger.info(f"失败归档: {stats['failed']}")
        logger.info(f"归档记录数: {stats['total_records_archived']}")
        logger.info(f"执行时间: {execution_time}")
        logger.info("==================")
        
        return stats


class DataCleanupManager:
    """数据清理管理器"""
    
    def __init__(self, engine, config: DataCompressionConfig):
        """初始化数据清理管理器"""
        self.engine = engine
        self.config = config
    
    def cleanup_old_data(self, cleanup_date: date) -> Dict[str, int]:
        """清理旧数据"""
        cleanup_stats = {
            'factor_values_deleted': 0,
            'archive_files_deleted': 0,
            'log_entries_deleted': 0
        }
        
        try:
            # 清理主表中的旧数据
            query = text("""
                DELETE FROM factor_values 
                WHERE factor_date < :cleanup_date
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'cleanup_date': cleanup_date})
                cleanup_stats['factor_values_deleted'] = result.rowcount
            
            # 清理旧的归档文件
            archive_path = Path(self.config.archive_path)
            if archive_path.exists():
                for archive_file in archive_path.glob("*.gz"):
                    # 从文件名提取日期
                    try:
                        filename_parts = archive_file.stem.split('_')
                        if len(filename_parts) >= 3:
                            file_date = date.fromisoformat(filename_parts[-1])
                            if file_date < cleanup_date:
                                archive_file.unlink()
                                cleanup_stats['archive_files_deleted'] += 1
                    except (ValueError, IndexError):
                        continue
            
            # 清理归档日志
            log_query = text("""
                DELETE FROM factor_archive_log 
                WHERE cutoff_date < :cleanup_date
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(log_query, {'cleanup_date': cleanup_date})
                cleanup_stats['log_entries_deleted'] = result.rowcount
            
            logger.info(f"数据清理完成: {cleanup_stats}")
            
        except Exception as e:
            logger.error(f"数据清理失败: {e}")
        
        return cleanup_stats
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            # 数据库存储统计
            db_stats_query = text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT ts_code) as unique_stocks,
                    COUNT(DISTINCT factor_name) as unique_factors,
                    MIN(factor_date) as earliest_date,
                    MAX(factor_date) as latest_date
                FROM factor_values
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(db_stats_query)
                db_stats = dict(result.fetchone())
            
            # 归档文件统计
            archive_path = Path(self.config.archive_path)
            archive_stats = {
                'total_files': 0,
                'total_size_mb': 0
            }
            
            if archive_path.exists():
                for archive_file in archive_path.glob("*.gz"):
                    archive_stats['total_files'] += 1
                    archive_stats['total_size_mb'] += archive_file.stat().st_size / (1024 * 1024)
            
            return {
                'database': db_stats,
                'archive': archive_stats,
                'config': {
                    'archive_after_days': self.config.archive_after_days,
                    'cleanup_after_days': self.config.cleanup_after_days,
                    'compression_level': self.config.compression_level.name
                }
            }
            
        except Exception as e:
            logger.error(f"获取存储统计失败: {e}")
            return {}


class BackupManager:
    """备份管理器"""
    
    def __init__(self, engine, config: DataCompressionConfig):
        """初始化备份管理器"""
        self.engine = engine
        self.config = config
        self.compression_engine = CompressionEngine(config)
    
    def create_backup(self, backup_name: str = None) -> bool:
        """创建备份"""
        if not self.config.backup_enabled:
            logger.info("备份功能未启用")
            return False
        
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            backup_path = Path(self.config.backup_path) / f"{backup_name}.gz"
            
            # 导出数据
            query = text("""
                SELECT ts_code, factor_name, factor_date, factor_value, created_time
                FROM factor_values 
                ORDER BY ts_code, factor_name, factor_date
            """)
            
            with self.engine.connect() as conn:
                data = pd.read_sql(query, conn)
            
            if data.empty:
                logger.warning("没有数据需要备份")
                return False
            
            # 压缩并保存
            compressed_data = self.compression_engine.compress_data(data)
            
            with open(backup_path, 'wb') as f:
                f.write(compressed_data)
            
            # 记录备份信息
            backup_info = {
                'backup_name': backup_name,
                'backup_path': str(backup_path),
                'record_count': len(data),
                'compressed_size_mb': len(compressed_data) / (1024 * 1024),
                'backup_date': datetime.now()
            }
            
            backup_df = pd.DataFrame([backup_info])
            backup_df.to_sql(
                'backup_log',
                self.engine,
                if_exists='append',
                index=False
            )
            
            logger.info(f"备份创建成功: {backup_name}, "
                       f"记录数: {len(data)}, "
                       f"文件大小: {backup_info['compressed_size_mb']:.2f}MB")
            
            return True
            
        except Exception as e:
            logger.error(f"创建备份失败: {e}")
            return False
    
    def restore_backup(self, backup_name: str, replace_existing: bool = False) -> bool:
        """恢复备份"""
        try:
            backup_path = Path(self.config.backup_path) / f"{backup_name}.gz"
            
            if not backup_path.exists():
                logger.error(f"备份文件不存在: {backup_path}")
                return False
            
            # 读取并解压数据
            with open(backup_path, 'rb') as f:
                compressed_data = f.read()
            
            data = self.compression_engine.decompress_data(compressed_data)
            
            # 恢复数据
            if replace_existing:
                # 清空现有数据
                with self.engine.connect() as conn:
                    conn.execute(text("DELETE FROM factor_values"))
            
            data.to_sql(
                'factor_values',
                self.engine,
                if_exists='append',
                index=False
            )
            
            logger.info(f"备份恢复成功: {backup_name}, 恢复 {len(data)} 条记录")
            
            return True
            
        except Exception as e:
            logger.error(f"恢复备份失败: {e}")
            return False
    
    def cleanup_old_backups(self):
        """清理旧备份"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
            backup_path = Path(self.config.backup_path)
            
            deleted_count = 0
            
            if backup_path.exists():
                for backup_file in backup_path.glob("backup_*.gz"):
                    if backup_file.stat().st_mtime < cutoff_date.timestamp():
                        backup_file.unlink()
                        deleted_count += 1
            
            logger.info(f"清理了 {deleted_count} 个旧备份文件")
            
        except Exception as e:
            logger.error(f"清理旧备份失败: {e}")


class DataCompressionArchiver:
    """数据压缩归档器主类"""
    
    def __init__(self, engine, config: DataCompressionConfig = None):
        """初始化数据压缩归档器"""
        self.engine = engine
        self.config = config or DataCompressionConfig()
        
        # 初始化各个管理器
        self.archive_manager = ArchiveManager(engine, self.config)
        self.cleanup_manager = DataCleanupManager(engine, self.config)
        self.backup_manager = BackupManager(engine, self.config)
    
    def run_daily_maintenance(self) -> Dict[str, Any]:
        """运行日常维护任务"""
        logger.info("开始日常维护任务")
        
        maintenance_results = {
            'archive_stats': {},
            'cleanup_stats': {},
            'backup_success': False,
            'storage_stats': {}
        }
        
        try:
            # 1. 归档旧数据
            archive_cutoff = date.today() - timedelta(days=self.config.archive_after_days)
            maintenance_results['archive_stats'] = self.archive_manager.batch_archive(archive_cutoff)
            
            # 2. 清理过期数据
            cleanup_cutoff = date.today() - timedelta(days=self.config.cleanup_after_days)
            maintenance_results['cleanup_stats'] = self.cleanup_manager.cleanup_old_data(cleanup_cutoff)
            
            # 3. 创建备份
            if self.config.backup_enabled:
                maintenance_results['backup_success'] = self.backup_manager.create_backup()
                # 清理旧备份
                self.backup_manager.cleanup_old_backups()
            
            # 4. 获取存储统计
            maintenance_results['storage_stats'] = self.cleanup_manager.get_storage_statistics()
            
            logger.info("日常维护任务完成")
            
        except Exception as e:
            logger.error(f"日常维护任务失败: {e}")
            maintenance_results['error'] = str(e)
        
        return maintenance_results
    
    def get_maintenance_schedule(self) -> Dict[str, Any]:
        """获取维护计划"""
        today = date.today()
        
        return {
            'next_archive_date': today + timedelta(days=1),
            'next_cleanup_date': today + timedelta(days=7),
            'next_backup_date': today + timedelta(days=1),
            'archive_cutoff_date': today - timedelta(days=self.config.archive_after_days),
            'cleanup_cutoff_date': today - timedelta(days=self.config.cleanup_after_days),
            'config': {
                'archive_after_days': self.config.archive_after_days,
                'cleanup_after_days': self.config.cleanup_after_days,
                'backup_enabled': self.config.backup_enabled,
                'compression_level': self.config.compression_level.name
            }
        }