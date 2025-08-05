import glob
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

#!/usr/bin/env python3
"""
StockSchool项目清理脚本
用于清理临时文件、过时文档和整理项目结构
"""


class ProjectCleaner:
    """类描述"""
        self.project_root = Path(project_root)
        self.cleanup_log = []

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def log_cleanup(self, action, path, description=""):
        """记录清理操作"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "path": str(path),
            "description": description
        }
        self.cleanup_log.append(entry)
        self.logger.info(f"{action}: {path} - {description}")

    def find_temp_files(self):
        """查找临时文件"""
        temp_patterns = [
            "**/*.tmp",
            "**/*.log",
            "**/*.pyc",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            "**/.mypy_cache/**",
            "**/.coverage",
            "**/htmlcov/**",
            "**/.tox/**",
            "**/dist/**",
            "**/build/**",
            "**/*.egg-info/**",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.bak",
            "**/*.swp",
            "**/*.swo",
            "**/*~",
        ]

        temp_files = []
        for pattern in temp_patterns:
            files = self.project_root.glob(pattern)
            temp_files.extend(files)

        return temp_files

    def find_outdated_docs(self):
        """查找过时的文档文件"""
        outdated_docs = []

        # 需要清理的过时文档
        outdated_patterns = [
            "docs/1comprehensive_review.md",
            "docs/2refactoring_plan.md",
            "docs/3.md",
            "docs/REFACTORING_PLAN.md",  # 与重构计划重复
            "docs/DEPLOYMENT_SUMMARY.md",  # 与部署指南重复
            "docs/README_STAGE3.md",  # 过时的README
            "docs/data_sync_refactoring_log.md",  # 与重构报告重复
            "docs/factor_calculation_engine_review.md",  # 与技术指南重复
            "docs/factor_engine_refactoring_summary.md",  # 与重构报告重复
        ]

        for pattern in outdated_patterns:
            files = self.project_root.glob(pattern)
            outdated_docs.extend(files)

        return outdated_docs

    def merge_similar_docs(self):
        """合并相似的文档内容"""
        merges = []

        # 检查可以合并的文档
        doc_pairs = [
            ("docs/data_sync_refactoring.md", "docs/data_sync_refactoring_log.md"),
            ("docs/factor_calculation_technical_guide.md", "docs/factor_calculation_engine_review.md"),
        ]

        for main_doc, secondary_doc in doc_pairs:
            main_path = self.project_root / main_doc
            secondary_path = self.project_root / secondary_doc

            if main_path.exists() and secondary_path.exists():
                merges.append((main_path, secondary_path))

        return merges

    def update_readme(self):
        """更新README.md，整合最新内容"""
        readme_path = self.project_root / "README.md"

        if not readme_path.exists():
            return False

        # 读取当前内容
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 添加第二阶段优化内容
        stage2_content = """
## 🚀 第二阶段优化 (v2.0.0)

### 性能优化
- **并行计算引擎**: 多进程并行因子计算，支持CPU/GPU加速
- **智能缓存系统**: Redis+内存多级缓存，LRU淘汰策略
- **负载均衡**: Nginx反向代理，支持多实例部署
- **批处理优化**: 智能批量大小计算，内存自适应管理

### 监控增强
- **实时监控**: Prometheus+Grafana监控栈
- **数据质量监控**: 实时数据质量检查和告警
- **性能仪表板**: 可视化系统性能指标
- **智能告警**: 多渠道告警通知系统

### 架构优化
- **设计模式**: 依赖注入、观察者模式、工厂模式
- **模块化重构**: 单一职责原则，降低耦合度
- **容错机制**: 重试策略、熔断降级、健康检查
- **配置管理**: 集中式配置管理，支持热更新

### 部署优化
- **容器化部署**: Docker+Docker Compose完整方案
- **负载均衡**: 多实例负载均衡配置
- **自动扩展**: 基于CPU/内存的自动扩展
- **健康检查**: 多层次健康检查和自愈机制

### 快速开始 (第二阶段)
```bash
# 使用Docker快速部署
docker-compose -f docker-compose.stage2.yml up -d

# 运行性能测试
python scripts/performance_test_runner.py

# 查看监控仪表板
open http://localhost:3000
```

### 配置文件
- `stage2_optimization_config.yml` - 优化配置
- `docker-compose.stage2.yml` - 容器编排
- `Dockerfile.stage2` - 生产镜像
- `requirements-stage2.txt` - 优化依赖
"""

        # 在适当位置插入新内容
        if "## 技术栈" in content:
            insert_point = content.find("## 技术栈")
            new_content = content[:insert_point] + stage2_content + "\n" + content[insert_point:]

            # 备份原文件
            backup_path = readme_path.with_suffix('.md.backup')
            shutil.copy2(readme_path, backup_path)

            # 写入更新内容
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            self.log_cleanup("update", readme_path, "添加第二阶段优化内容")
            return True

        return False

    def clean_temp_files(self):
        """清理临时文件"""
        temp_files = self.find_temp_files()

        for file_path in temp_files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    self.log_cleanup("delete", file_path, "临时文件")
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    self.log_cleanup("delete", file_path, "临时目录")
            except Exception as e:
                self.logger.error(f"无法删除 {file_path}: {e}")

    def clean_outdated_docs(self):
        """清理过时文档"""
        outdated_docs = self.find_outdated_docs()

        for doc_path in outdated_docs:
            if doc_path.exists():
                try:
                    # 移动到备份目录
                    backup_dir = self.project_root / "docs" / "archived"
                    backup_dir.mkdir(exist_ok=True)

                    backup_path = backup_dir / f"{doc_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{doc_path.suffix}"
                    shutil.move(str(doc_path), str(backup_path))

                    self.log_cleanup("archive", doc_path, f"移动到 {backup_path}")
                except Exception as e:
                    self.logger.error(f"无法归档 {doc_path}: {e}")

    def generate_cleanup_report(self):
        """生成清理报告"""
        report_path = self.project_root / "cleanup_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# StockSchool 项目清理报告\n\n")
            f.write(f"清理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 清理操作记录\n\n")

            for entry in self.cleanup_log:
                f.write(f"- **{entry['action']}**: `{entry['path']}` - {entry['description']}\n")

            f.write("\n## 建议\n\n")
            f.write("- 定期检查临时文件和过时文档\n")
            f.write("- 使用版本控制管理重要文档变更\n")
            f.write("- 建立文档归档策略\n")

        self.log_cleanup("create", report_path, "清理报告")

    def run_cleanup(self):
        """执行完整清理流程"""
        print("🧹 开始清理StockSchool项目...")

        # 1. 清理临时文件
        print("📁 清理临时文件...")
        self.clean_temp_files()

        # 2. 清理过时文档
        print("📄 归档过时文档...")
        self.clean_outdated_docs()

        # 3. 更新README
        print("📝 更新README.md...")
        self.update_readme()

        # 4. 生成清理报告
        print("📊 生成清理报告...")
        self.generate_cleanup_report()

        print("✅ 清理完成！")
        print(f"📋 共执行 {len(self.cleanup_log)} 项清理操作")
        print("📄 查看详细报告: cleanup_report.md")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    cleaner = ProjectCleaner(project_root)
    cleaner.run_cleanup()