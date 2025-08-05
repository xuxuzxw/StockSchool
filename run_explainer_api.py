#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型解释器API服务启动脚本

启动FastAPI服务，提供模型解释功能的RESTful API接口
"""

import uvicorn
import argparse
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.explainer_api import router
from src.config.unified_config import config
from src.monitoring.logger import get_logger

logger = get_logger(__name__)

def create_app():
    """创建FastAPI应用"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(
        title="StockSchool 模型解释器API",
        description="提供机器学习模型解释功能的RESTful API服务",
        version="1.0.0"
    )
    
    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 包含路由
    app.include_router(router)
    
    @app.get("/")
    async def root():
        """根路径欢迎信息"""
        return {
            "message": "欢迎使用StockSchool模型解释器API服务",
            "version": "1.0.0",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    
    @app.get("/health")
    async def health_check():
        """健康检查端点"""
        return {"status": "healthy", "timestamp": "2025-07-31T18:00:00Z"}
    
    return app

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动模型解释器API服务")
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    
    args = parser.parse_args()
    
    try:
        logger.info(f"启动模型解释器API服务: http://{args.host}:{args.port}")
        logger.info("文档地址: http://localhost:8000/docs")
        
        # 创建应用
        app = create_app()
        
        # 启动服务
        uvicorn.run(
            "run_explainer_api:create_app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            factory=True
        )
        
    except KeyboardInterrupt:
        logger.info("服务被用户中断")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
