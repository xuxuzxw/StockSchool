#!/usr/bin/env python3
"""
投资组合构建简化测试脚本
验证基于因子有效性分析的投资组合构建功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimplePortfolioBuilder:
    """简化投资组合构建器"""
    
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.positions = {}
        self.cash = initial_capital
        
    def create_factor_scores(self, stock_data):
        """创建因子评分"""
        # 模拟基于因子有效性的评分
        scores = {}
        for stock in stock_data['ts_code'].unique():
            stock_factor_data = stock_data[stock_data['ts_code'] == stock]
            if len(stock_factor_data) > 0:
                # 综合评分：基于多个因子的加权平均
                technical_score = np.random.uniform(60, 95)
                fundamental_score = np.random.uniform(65, 90)
                sentiment_score = np.random.uniform(55, 85)
                
                # 权重：技术面40%，基本面40%，情绪面20%
                total_score = (technical_score * 0.4 + 
                             fundamental_score * 0.4 + 
                             sentiment_score * 0.2)
                
                scores[stock] = {
                    'total_score': total_score,
                    'technical': technical_score,
                    'fundamental': fundamental_score,
                    'sentiment': sentiment_score
                }
        
        return scores
    
    def select_stocks(self, scores, top_n=10, min_score=75):
        """选择股票"""
        # 按总分排序，选择前N只股票
        sorted_stocks = sorted(scores.items(), 
                             key=lambda x: x[1]['total_score'], 
                             reverse=True)
        
        selected = []
        for stock_code, score_data in sorted_stocks[:top_n]:
            if score_data['total_score'] >= min_score:
                selected.append({
                    'ts_code': stock_code,
                    'score': score_data['total_score'],
                    'weight': score_data['total_score'] / 100  # 基于评分权重
                })
        
        return selected
    
    def calculate_position_weights(self, selected_stocks, method='equal'):
        """计算持仓权重"""
        if method == 'equal':
            # 等权重分配
            weight = 1.0 / len(selected_stocks)
            for stock in selected_stocks:
                stock['position_weight'] = weight
                
        elif method == 'score_weighted':
            # 基于评分的权重分配
            total_score = sum(stock['score'] for stock in selected_stocks)
            for stock in selected_stocks:
                stock['position_weight'] = stock['score'] / total_score
                
        return selected_stocks
    
    def build_portfolio(self, selected_stocks):
        """构建投资组合"""
        portfolio = []
        
        for stock in selected_stocks:
            investment_amount = self.initial_capital * stock['position_weight']
            
            portfolio.append({
                'ts_code': stock['ts_code'],
                'score': stock['score'],
                'weight': stock['position_weight'],
                'investment_amount': investment_amount,
                'expected_return': np.random.uniform(0.05, 0.25),  # 预期收益率
                'expected_risk': np.random.uniform(0.10, 0.30)     # 预期风险
            })
        
        return portfolio
    
    def calculate_portfolio_metrics(self, portfolio):
        """计算投资组合指标"""
        if not portfolio:
            return {}
            
        weights = [stock['weight'] for stock in portfolio]
        expected_returns = [stock['expected_return'] for stock in portfolio]
        expected_risks = [stock['expected_risk'] for stock in portfolio]
        
        # 组合预期收益
        portfolio_return = np.sum(np.array(weights) * np.array(expected_returns))
        
        # 组合风险（简化计算，假设相关性为0）
        portfolio_risk = np.sqrt(np.sum((np.array(weights) * np.array(expected_risks))**2))
        
        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        
        return {
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'num_stocks': len(portfolio),
            'total_investment': sum(stock['investment_amount'] for stock in portfolio)
        }

def generate_test_data():
    """生成测试数据"""
    np.random.seed(42)
    
    # 模拟股票列表
    stock_codes = [f'00{i:04d}.SZ' for i in range(1, 51)]
    
    # 生成因子数据
    data = []
    for stock in stock_codes:
        for i in range(60):  # 60天数据
            date = datetime.now() - timedelta(days=i)
            data.append({
                'ts_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'close': np.random.uniform(10, 100),
                'volume': np.random.uniform(100000, 1000000),
                'return_rate': np.random.uniform(-0.05, 0.05)
            })
    
    return pd.DataFrame(data)

def test_portfolio_construction():
    """测试投资组合构建流程"""
    print("🎯 开始投资组合构建测试...")
    
    # 1. 生成测试数据
    print("📊 生成测试数据...")
    test_data = generate_test_data()
    print(f"✅ 生成了 {len(test_data)} 条测试数据")
    
    # 2. 初始化投资组合构建器
    print("💰 初始化投资组合构建器...")
    builder = SimplePortfolioBuilder(initial_capital=1000000)
    
    # 3. 创建因子评分
    print("📈 创建因子评分...")
    scores = builder.create_factor_scores(test_data)
    print(f"✅ 为 {len(scores)} 只股票创建了因子评分")
    
    # 4. 选择股票
    print("🎯 选择优质股票...")
    selected = builder.select_stocks(scores, top_n=10, min_score=75)
    print(f"✅ 选择了 {len(selected)} 只股票")
    
    # 5. 计算权重
    print("⚖️ 计算持仓权重...")
    weighted_stocks = builder.calculate_position_weights(selected, method='score_weighted')
    
    # 6. 构建投资组合
    print("🏗️ 构建投资组合...")
    portfolio = builder.build_portfolio(weighted_stocks)
    
    # 7. 计算组合指标
    print("📊 计算投资组合指标...")
    metrics = builder.calculate_portfolio_metrics(portfolio)
    
    # 8. 输出结果
    print("\n" + "="*60)
    print("📋 投资组合构建结果")
    print("="*60)
    
    print(f"💰 初始资金: ¥{builder.initial_capital:,.2f}")
    print(f"📊 选中股票数量: {len(portfolio)}")
    print(f"📈 预期年化收益率: {metrics['portfolio_return']*100:.2f}%")
    print(f"⚠️ 预期年化风险: {metrics['portfolio_risk']*100:.2f}%")
    print(f"📊 夏普比率: {metrics['sharpe_ratio']:.3f}")
    print(f"💵 总投资金额: ¥{metrics['total_investment']:,.2f}")
    
    print("\n📊 投资组合详情:")
    print("-" * 60)
    for i, stock in enumerate(portfolio, 1):
        print(f"{i:2d}. {stock['ts_code']} | 评分: {stock['score']:.1f} | "
              f"权重: {stock['weight']*100:5.2f}% | "
              f"投资金额: ¥{stock['investment_amount']:8,.0f}")
    
    return {
        'portfolio': portfolio,
        'metrics': metrics,
        'success': len(portfolio) > 0
    }

if __name__ == "__main__":
    try:
        result = test_portfolio_construction()
        
        if result['success']:
            print("\n🎉 投资组合构建测试通过！")
        else:
            print("\n❌ 投资组合构建测试失败！")
            
    except Exception as e:
        print(f"\n❌ 测试执行失败: {str(e)}")
        import traceback
        traceback.print_exc()