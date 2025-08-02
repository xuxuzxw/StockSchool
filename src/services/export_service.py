"""
导出服务模块

提供数据导出功能，支持多种格式（CSV、Excel、JSON）
以及报告生成和邮件发送功能

作者: StockSchool Team
创建时间: 2025-01-02
"""

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# 设置日志
logger = logging.getLogger(__name__)


class ExportConfig:
    """导出配置类"""
    
    def __init__(self, export_dir: str = "exports", temp_dir: str = "temp"):
        """
        初始化导出配置
        
        Args:
            export_dir: 导出文件存储目录
            temp_dir: 临时文件目录
        """
        self.export_dir = export_dir
        self.temp_dir = temp_dir
        self.ensure_directories()
    
    def ensure_directories(self):
        """确保存储目录存在"""
        os.makedirs(self.export_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)


class DataExporter:
    """数据导出器"""
    
    def __init__(self, config: ExportConfig):
        """
        初始化数据导出器
        
        Args:
            config: 导出配置对象
        """
        self.config = config
        logger.info("数据导出器初始化完成")
    
    async def export_to_csv(self, data: List[Dict[str, Any]], filename: str) -> str:
        """
        导出数据到CSV格式
        
        Args:
            data: 要导出的数据列表
            filename: 导出文件名（不含扩展名）
            
        Returns:
            str: 导出文件的完整路径
        """
        try:
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 生成文件路径
            filepath = os.path.join(self.config.export_dir, f"{filename}.csv")
            
            # 导出到CSV
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            logger.info(f"数据已导出到CSV: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"CSV导出失败: {e}")
            raise
    
    async def export_to_json(self, data: List[Dict[str, Any]], filename: str) -> str:
        """
        导出数据到JSON格式
        
        Args:
            data: 要导出的数据列表
            filename: 导出文件名（不含扩展名）
            
        Returns:
            str: 导出文件的完整路径
        """
        try:
            # 生成文件路径
            filepath = os.path.join(self.config.export_dir, f"{filename}.json")
            
            # 导出到JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"数据已导出到JSON: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"JSON导出失败: {e}")
            raise


class ExportService:
    """导出服务主类"""
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        初始化导出服务
        
        Args:
            config: 导出配置对象，如果为None则使用默认配置
        """
        self.config = config or ExportConfig()
        self.data_exporter = DataExporter(self.config)
        logger.info("导出服务初始化完成")
    
    async def export_data(self, data: List[Dict[str, Any]], format: str, filename: str) -> str:
        """
        导出数据
        
        Args:
            data: 要导出的数据列表
            format: 导出格式 ('csv', 'json')
            filename: 导出文件名（不含扩展名）
            
        Returns:
            str: 导出文件的完整路径
            
        Raises:
            ValueError: 不支持的导出格式
        """
        if format.lower() == 'csv':
            return await self.data_exporter.export_to_csv(data, filename)
        elif format.lower() == 'json':
            return await self.data_exporter.export_to_json(data, filename)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def validate_exported_data(self, filepath: str, expected_count: int) -> bool:
        """
        验证导出的数据完整性
        
        Args:
            filepath: 导出文件路径
            expected_count: 期望的数据条目数
            
        Returns:
            bool: 验证是否通过
        """
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                return len(df) == expected_count
            elif filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return len(data) == expected_count
            else:
                logger.warning(f"不支持的文件格式进行验证: {filepath}")
                return False
                
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return False


# 示例使用函数
async def main():
    """主函数示例"""
    # 创建导出服务
    export_service = ExportService()
    
    # 准备测试数据
    test_data = [
        {
            'timestamp': datetime.now(),
            'metric_name': 'cpu_usage',
            'metric_value': 75.5,
            'source_component': 'system'
        },
        {
            'timestamp': datetime.now(),
            'metric_name': 'memory_usage',
            'metric_value': 68.2,
            'source_component': 'system'
        }
    ]
    
    try:
        # 测试CSV导出
        csv_path = await export_service.export_data(test_data, 'csv', 'test_export')
        print(f"CSV导出成功: {csv_path}")
        
        # 测试JSON导出
        json_path = await export_service.export_data(test_data, 'json', 'test_export')
        print(f"JSON导出成功: {json_path}")
        
        # 验证数据完整性
        csv_valid = export_service.validate_exported_data(csv_path, len(test_data))
        json_valid = export_service.validate_exported_data(json_path, len(test_data))
        
        print(f"CSV数据验证: {'通过' if csv_valid else '失败'}")
        print(f"JSON数据验证: {'通过' if json_valid else '失败'}")
        
    except Exception as e:
        print(f"导出过程出现错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())