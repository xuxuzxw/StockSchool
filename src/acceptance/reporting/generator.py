"""
测试报告生成器
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from ..core.models import AcceptanceReport, TestStatus


class ReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_json_report(self, report: AcceptanceReport) -> str:
        """生成JSON格式报告"""
        file_path = self.output_dir / f"acceptance_report_{report.test_session_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        return str(file_path)
    
    def generate_html_report(self, report: AcceptanceReport) -> str:
        """生成HTML格式报告"""
        file_path = self.output_dir / f"acceptance_report_{report.test_session_id}.html"
        
        html_content = self._build_html_content(report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(file_path)
    
    def generate_markdown_report(self, report: AcceptanceReport) -> str:
        """生成Markdown格式报告"""
        file_path = self.output_dir / f"acceptance_report_{report.test_session_id}.md"
        
        md_content = self._build_markdown_content(report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(file_path)
    
    def _build_html_content(self, report: AcceptanceReport) -> str:
        """构建HTML报告内容"""
        # 按阶段分组结果
        phases = self._group_results_by_phase(report.phase_results)
        
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StockSchool 验收测试报告</title>
    <style>
        {css_styles}
    </style>
</head>
<body>
    <div class="header">
        <h1>StockSchool 验收测试报告</h1>
        <div class="meta-info">
            <p><strong>测试会话ID:</strong> {test_session_id}</p>
            <p><strong>开始时间:</strong> {start_time}</p>
            <p><strong>结束时间:</strong> {end_time}</p>
            <p><strong>整体结果:</strong> <span class="{result_class}">{overall_result}</span></p>
        </div>
    </div>
    
    <div class="summary">
        <div class="metric passed">
            <h3>{passed_tests}</h3>
            <p>通过测试</p>
        </div>
        <div class="metric failed">
            <h3>{failed_tests}</h3>
            <p>失败测试</p>
        </div>
        <div class="metric skipped">
            <h3>{skipped_tests}</h3>
            <p>跳过测试</p>
        </div>
        <div class="metric">
            <h3>{total_tests}</h3>
            <p>总测试数</p>
        </div>
    </div>
    
    <div class="phase-results">
        <h2>测试阶段结果</h2>
        {phase_content}
    </div>
    
    {performance_section}
    {recommendations_section}
</body>
</html>
"""
        
        return html_template.format(
            css_styles=self._get_css_styles(),
            test_session_id=report.test_session_id,
            start_time=report.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time=report.end_time.strftime('%Y-%m-%d %H:%M:%S') if report.end_time else '进行中',
            result_class='passed' if report.overall_result else 'failed',
            overall_result='通过' if report.overall_result else '失败',
            passed_tests=report.passed_tests,
            failed_tests=report.failed_tests,
            skipped_tests=report.skipped_tests,
            total_tests=report.total_tests,
            phase_content=self._build_phase_content(phases),
            performance_section=self._build_performance_section(report.performance_metrics),
            recommendations_section=self._build_recommendations_section(report.recommendations)
        )
    
    def _build_markdown_content(self, report: AcceptanceReport) -> str:
        """构建Markdown报告内容"""
        phases = self._group_results_by_phase(report.phase_results)
        
        md_content = f"""# StockSchool 验收测试报告

## 基本信息

- **测试会话ID**: {report.test_session_id}
- **开始时间**: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **结束时间**: {report.end_time.strftime('%Y-%m-%d %H:%M:%S') if report.end_time else '进行中'}
- **整体结果**: {'✅ 通过' if report.overall_result else '❌ 失败'}

## 测试统计

| 指标 | 数量 |
|------|------|
| 总测试数 | {report.total_tests} |
| 通过测试 | {report.passed_tests} |
| 失败测试 | {report.failed_tests} |
| 跳过测试 | {report.skipped_tests} |

## 测试阶段结果

"""
        
        for phase_name, results in phases.items():
            passed_count = sum(1 for r in results if r.status == TestStatus.PASSED)
            total_count = len(results)
            
            md_content += f"""### {phase_name} ({passed_count}/{total_count} 通过)

"""
            
            for result in results:
                status_emoji = {
                    TestStatus.PASSED: '✅',
                    TestStatus.FAILED: '❌',
                    TestStatus.SKIPPED: '⏭️'
                }.get(result.status, '❓')
                
                md_content += f"- {status_emoji} **{result.test_name}** ({result.execution_time:.2f}s)"
                
                if result.error_message:
                    md_content += f"\n  - 错误: {result.error_message}"
                
                md_content += "\n"
            
            md_content += "\n"
        
        # 添加性能指标
        if report.performance_metrics:
            md_content += "## 性能指标\n\n"
            for key, value in report.performance_metrics.items():
                if isinstance(value, float):
                    md_content += f"- **{key}**: {value:.2f}\n"
                else:
                    md_content += f"- **{key}**: {value}\n"
            md_content += "\n"
        
        # 添加建议
        if report.recommendations:
            md_content += "## 改进建议\n\n"
            for i, rec in enumerate(report.recommendations, 1):
                md_content += f"{i}. {rec}\n"
        
        return md_content
    
    def _group_results_by_phase(self, results):
        """按阶段分组测试结果"""
        phases = {}
        for result in results:
            if result.phase not in phases:
                phases[result.phase] = []
            phases[result.phase].append(result)
        return phases
    
    def _get_css_styles(self) -> str:
        """获取CSS样式"""
        return """
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .meta-info p { margin: 5px 0; }
        .summary { display: flex; justify-content: space-around; margin: 20px 0; gap: 10px; }
        .metric { text-align: center; padding: 15px; border: 1px solid #ddd; border-radius: 8px; flex: 1; }
        .metric h3 { margin: 0; font-size: 2em; }
        .metric p { margin: 5px 0 0 0; color: #666; }
        .passed { background-color: #d4edda; border-color: #c3e6cb; }
        .failed { background-color: #f8d7da; border-color: #f5c6cb; }
        .skipped { background-color: #fff3cd; border-color: #ffeaa7; }
        .phase-results { margin: 20px 0; }
        .phase { margin: 15px 0; padding: 15px; border-left: 4px solid #007bff; background-color: #f8f9fa; }
        .test-result { margin: 8px 0; padding: 8px; border-radius: 4px; }
        .test-result.passed { background-color: #d4edda; }
        .test-result.failed { background-color: #f8d7da; }
        .test-result.skipped { background-color: #fff3cd; }
        .recommendations { background-color: #e9ecef; padding: 15px; border-radius: 8px; margin-top: 20px; }
        .performance-metrics { background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin-top: 20px; }
        """
    
    def _build_phase_content(self, phases) -> str:
        """构建阶段内容HTML"""
        content = ""
        for phase_name, results in phases.items():
            passed_count = sum(1 for r in results if r.status == TestStatus.PASSED)
            total_count = len(results)
            
            content += f"""
        <div class="phase">
            <h3>{phase_name} ({passed_count}/{total_count} 通过)</h3>
"""
            
            for result in results:
                status_class = result.status.value
                status_text = {
                    'passed': '通过',
                    'failed': '失败', 
                    'skipped': '跳过'
                }.get(result.status.value, result.status.value)
                
                content += f"""
            <div class="test-result {status_class}">
                <strong>{result.test_name}</strong> - {status_text} ({result.execution_time:.2f}s)
"""
                if result.error_message:
                    content += f"<br><small style='color: #dc3545;'>错误: {result.error_message}</small>"
                
                content += "</div>"
            
            content += "</div>"
        
        return content
    
    def _build_performance_section(self, metrics) -> str:
        """构建性能指标部分"""
        if not metrics:
            return ""
        
        content = """
    <div class="performance-metrics">
        <h2>性能指标</h2>
        <ul>
"""
        for key, value in metrics.items():
            if isinstance(value, float):
                content += f"<li><strong>{key}:</strong> {value:.2f}</li>"
            else:
                content += f"<li><strong>{key}:</strong> {value}</li>"
        
        content += "</ul></div>"
        return content
    
    def _build_recommendations_section(self, recommendations) -> str:
        """构建建议部分"""
        if not recommendations:
            return ""
        
        content = """
    <div class="recommendations">
        <h2>改进建议</h2>
        <ol>
"""
        for rec in recommendations:
            content += f"<li>{rec}</li>"
        
        content += "</ol></div>"
        return content