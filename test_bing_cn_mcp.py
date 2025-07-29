#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 bing-cn-mcp 工具的示例脚本
"""

import json
import subprocess
import sys

def test_bing_search():
    """测试必应搜索功能"""
    print("=== 测试 bing-cn-mcp 搜索功能 ===")
    
    # 搜索示例
    search_query = "中国股市最新动态"
    print(f"搜索关键词: {search_query}")
    
    # 这里演示如何使用 MCP 工具调用
    # 在实际的 MCP 环境中，您可以直接调用:
    # bing_search(query="中国股市最新动态", num_results=5)
    
    print("搜索结果示例:")
    print("1. [标题] 中国股市今日行情分析 - 来源网站")
    print("2. [标题] A股最新趋势预测 - 来源网站") 
    print("3. [标题] 上证指数近期走势解读 - 来源网站")
    print("\n注意: 在支持 MCP 的环境中，可以直接调用 bing_search 和 fetch_webpage 工具")
    
def main():
    print("bing-cn-mcp 测试脚本")
    print("=" * 50)
    
    test_bing_search()
    
    print("\n=== 使用说明 ===")
    print("1. 确保已安装 bing-cn-mcp: npm install -g bing-cn-mcp")
    print("2. 启动服务器: npx bing-cn-mcp")
    print("3. 在 MCP 环境中配置服务器:")
    print("""   {
     "bingcn": {
       "command": "cmd",
       "args": ["/c", "npx", "bing-cn-mcp"]
     }
   }""")
    print("4. 使用工具:")
    print("   - bing_search: 进行中文必应搜索")
    print("   - fetch_webpage: 获取搜索结果页面内容")

if __name__ == "__main__":
    main()
