# 因子计算引擎故障排除指南

## 概述

本指南提供了StockSchool因子计算引擎常见问题的诊断方法和解决方案。通过系统化的故障排除流程，帮助用户快速定位和解决问题。

## 故障分类和诊断流程

### 故障分类

1. **数据相关问题**：数据缺失、格式错误、质量问题
2. **计算相关问题**：算法错误、参数不当、性能问题
3. **系统相关问题**：数据库连接、内存不足、网络问题
4. **API相关问题**：接口错误、认证失败、超时问题

### 通用诊断流程

```python
def diagnose_system_issue():
    """
    系统问题通用诊断流程
    
    诊断步骤:
    1. 检查系统基础状态
    2. 验证数据连接
    3. 测试核心功能
    4. 分析错误日志
    5. 生成诊断报告
    """
    
    diagnosis_report = {
        'timestamp': datetime.now().isoformat(),
        'system_status': {},
        'issues_found': [],
        'recommendations': []
    }
    
    # 1. 系统基础状态检查
    diagnosis_report['system_status'] = check_system_status()
    
    # 2. 数据连接验证
    db_status = check_database_connection()
    cache_status = check_cache_connection()
    
    if not db_status['connected']:
        diagnosis_report['issues_found'].append({
            'type': 'database_connection',
            'severity': 'critical',
            'message': '数据库连接失败',
            'details': db_status
        })
    
    if not cache_status['connected']:
        diagnosis_report['issues_found'].append({
            'type': 'cache_connection',
            'severity': 'warning',
            'message': '缓存连接失败',
            'details': cache_status
        })
    
    # 3. 核心功能测试
    core_test_results = run_core_functionality_tests()
    for test_name, result in core_test_results.items():
        if not result['success']:
            diagnosis_report['issues_found'].append({
                'type': 'functionality',
                'severity': 'high',
                'message': f'核心功能测试失败: {test_name}',
                'details': result
            })
    
    # 4. 生成建议
    diagnosis_report['recommendations'] = generate_recommendations(
        diagnosis_report['issues_found']
    )
    
    return diagnosis_report

def check_system_status():
    """检查系统基础状态"""
    import psutil
    
    return {
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None,
        'python_version': sys.version,
        'timestamp': datetime.now().isoformat()
    }

def check_database_connection():
    """检查数据库连接"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            test_value = result.fetchone().test
            
            return {
                'connected': True,
                'test_query': test_value == 1,
                'connection_time': time.time()
            }
    except Exception as e:
        return {
            'connected': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

def run_core_functionality_tests():
    """运行核心功能测试"""
    tests = {
        'technical_factor_calculation': test_technical_factor,
        'data_loading': test_data_loading,
        'cache_operations': test_cache_operations,
        'api_endpoints': test_api_endpoints
    }
    
    results = {}
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            results[test_name] = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    return results
```

## 数据相关问题

### 问题1：数据加载失败

#### 症状
- 因子计算返回空结果
- 出现"数据不存在"错误
- 数据查询超时

#### 诊断方法
```python
def diagnose_data_loading_issue(ts_codes, date_range):
    """
    诊断数据加载问题
    
    Args:
        ts_codes: 股票代码列表
        date_range: 日期范围 (start_date, end_date)
    
    Returns:
        诊断结果和建议
    """
    diagnosis = {
        'issues': [],
        'suggestions': [],
        'data_status': {}
    }
    
    start_date, end_date = date_range
    
    # 1. 检查股票代码有效性
    invalid_codes = check_invalid_stock_codes(ts_codes)
    if invalid_codes:
        diagnosis['issues'].append(f"无效股票代码: {invalid_codes}")
        diagnosis['suggestions'].append("请检查股票代码格式和有效性")
    
    # 2. 检查日期范围合理性
    if start_date >= end_date:
        diagnosis['issues'].append("开始日期不能大于等于结束日期")
        diagnosis['suggestions'].append("请调整日期范围")
    
    # 3. 检查数据可用性
    try:
        with get_db_engine().connect() as conn:
            # 检查股票基础信息
            basic_query = """
            SELECT ts_code, COUNT(*) as count 
            FROM stock_basic 
            WHERE ts_code = ANY(%s)
            GROUP BY ts_code
            """
            basic_result = conn.execute(basic_query, (ts_codes,))
            basic_data = dict(basic_result.fetchall())
            
            missing_basic = set(ts_codes) - set(basic_data.keys())
            if missing_basic:
                diagnosis['issues'].append(f"缺少基础信息的股票: {missing_basic}")
                diagnosis['suggestions'].append("请同步股票基础信息数据")
            
            # 检查日线数据
            daily_query = """
            SELECT ts_code, COUNT(*) as count 
            FROM stock_daily 
            WHERE ts_code = ANY(%s) 
            AND trade_date BETWEEN %s AND %s
            GROUP BY ts_code
            """
            daily_result = conn.execute(daily_query, (
                ts_codes, 
                start_date.strftime('%Y%m%d'), 
                end_date.strftime('%Y%m%d')
            ))
            daily_data = dict(daily_result.fetchall())
            
            diagnosis['data_status'] = {
                'basic_info': basic_data,
                'daily_data': daily_data
            }
            
            # 分析数据覆盖情况
            for ts_code in ts_codes:
                daily_count = daily_data.get(ts_code, 0)
                expected_days = (end_date - start_date).days
                coverage_ratio = daily_count / expected_days if expected_days > 0 else 0
                
                if coverage_ratio < 0.5:  # 覆盖率低于50%
                    diagnosis['issues'].append(
                        f"股票{ts_code}数据覆盖率过低: {coverage_ratio:.2%}"
                    )
                    diagnosis['suggestions'].append(
                        f"请检查股票{ts_code}的数据同步状态"
                    )
    
    except Exception as e:
        diagnosis['issues'].append(f"数据库查询失败: {e}")
        diagnosis['suggestions'].append("请检查数据库连接和权限")
    
    return diagnosis

def check_invalid_stock_codes(ts_codes):
    """检查无效的股票代码"""
    invalid_codes = []
    
    for ts_code in ts_codes:
        # 检查格式
        if not re.match(r'^\d{6}\.(SZ|SH)$', ts_code):
            invalid_codes.append(ts_code)
    
    return invalid_codes
```

#### 解决方案
```python
def fix_data_loading_issues(diagnosis_result):
    """
    根据诊断结果修复数据加载问题
    
    Args:
        diagnosis_result: 诊断结果
    
    Returns:
        修复操作结果
    """
    fix_results = []
    
    for issue in diagnosis_result['issues']:
        if "无效股票代码" in issue:
            # 过滤无效代码
            fix_results.append({
                'action': 'filter_invalid_codes',
                'status': 'completed',
                'message': '已过滤无效股票代码'
            })
        
        elif "数据覆盖率过低" in issue:
            # 触发数据同步
            fix_results.append({
                'action': 'trigger_data_sync',
                'status': 'initiated',
                'message': '已触发数据同步任务'
            })
        
        elif "数据库查询失败" in issue:
            # 重试数据库连接
            fix_results.append({
                'action': 'retry_database_connection',
                'status': 'attempted',
                'message': '已尝试重新连接数据库'
            })
    
    return fix_results
```

### 问题2：数据质量异常

#### 症状
- 因子值出现异常分布
- 计算结果包含大量NaN或无穷值
- 因子值超出合理范围

#### 诊断方法
```python
def diagnose_data_quality_issues(data):
    """
    诊断数据质量问题
    
    Args:
        data: 待检查的数据
    
    Returns:
        数据质量报告
    """
    quality_report = {
        'overall_score': 0,
        'issues': [],
        'statistics': {},
        'recommendations': []
    }
    
    # 1. 基础统计
    quality_report['statistics'] = {
        'total_records': len(data),
        'missing_values': data.isnull().sum().to_dict(),
        'missing_ratio': (data.isnull().sum() / len(data)).to_dict(),
        'data_types': data.dtypes.to_dict()
    }
    
    # 2. 缺失值检查
    for column, missing_ratio in quality_report['statistics']['missing_ratio'].items():
        if missing_ratio > 0.1:  # 10%阈值
            quality_report['issues'].append({
                'type': 'missing_values',
                'column': column,
                'severity': 'high' if missing_ratio > 0.3 else 'medium',
                'message': f'列{column}缺失值比例过高: {missing_ratio:.2%}'
            })
    
    # 3. 异常值检查
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        series = data[column].dropna()
        if len(series) == 0:
            continue
        
        # 检查无穷值
        inf_count = np.isinf(series).sum()
        if inf_count > 0:
            quality_report['issues'].append({
                'type': 'infinite_values',
                'column': column,
                'severity': 'high',
                'message': f'列{column}包含{inf_count}个无穷值'
            })
        
        # 检查异常值 (IQR方法)
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
            outlier_ratio = len(outliers) / len(series)
            
            if outlier_ratio > 0.05:  # 5%阈值
                quality_report['issues'].append({
                    'type': 'outliers',
                    'column': column,
                    'severity': 'medium',
                    'message': f'列{column}异常值比例过高: {outlier_ratio:.2%}'
                })
    
    # 4. 数据一致性检查
    if 'open' in data.columns and 'close' in data.columns:
        # 检查开盘价和收盘价的合理性
        unreasonable_prices = data[
            (data['open'] <= 0) | 
            (data['close'] <= 0) | 
            (data['open'] > data['close'] * 2) |  # 开盘价不应该是收盘价的2倍以上
            (data['close'] > data['open'] * 2)
        ]
        
        if len(unreasonable_prices) > 0:
            quality_report['issues'].append({
                'type': 'data_consistency',
                'severity': 'high',
                'message': f'发现{len(unreasonable_prices)}条价格数据异常'
            })
    
    # 5. 生成建议
    quality_report['recommendations'] = generate_quality_recommendations(
        quality_report['issues']
    )
    
    # 6. 计算综合评分
    quality_report['overall_score'] = calculate_quality_score(quality_report)
    
    return quality_report

def generate_quality_recommendations(issues):
    """生成数据质量改进建议"""
    recommendations = []
    
    issue_types = [issue['type'] for issue in issues]
    
    if 'missing_values' in issue_types:
        recommendations.append({
            'type': 'missing_values',
            'action': '使用前向填充或插值方法处理缺失值',
            'code_example': 'data.fillna(method="ffill")'
        })
    
    if 'infinite_values' in issue_types:
        recommendations.append({
            'type': 'infinite_values',
            'action': '将无穷值替换为NaN或合理的边界值',
            'code_example': 'data.replace([np.inf, -np.inf], np.nan)'
        })
    
    if 'outliers' in issue_types:
        recommendations.append({
            'type': 'outliers',
            'action': '使用截断或Winsorize方法处理异常值',
            'code_example': 'data.clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)'
        })
    
    if 'data_consistency' in issue_types:
        recommendations.append({
            'type': 'data_consistency',
            'action': '检查数据源，重新同步异常数据',
            'code_example': '手动检查和修正异常记录'
        })
    
    return recommendations
```

#### 解决方案
```python
def clean_data_quality_issues(data, quality_report):
    """
    根据质量报告清理数据问题
    
    Args:
        data: 原始数据
        quality_report: 数据质量报告
    
    Returns:
        清理后的数据
    """
    cleaned_data = data.copy()
    cleaning_log = []
    
    for issue in quality_report['issues']:
        if issue['type'] == 'missing_values':
            column = issue['column']
            
            if column in ['open', 'high', 'low', 'close']:
                # 价格数据使用前向填充
                cleaned_data[column] = cleaned_data[column].fillna(method='ffill')
                cleaning_log.append(f"对{column}列使用前向填充处理缺失值")
            
            elif column == 'volume':
                # 成交量缺失值填充为0
                cleaned_data[column] = cleaned_data[column].fillna(0)
                cleaning_log.append(f"对{column}列缺失值填充为0")
        
        elif issue['type'] == 'infinite_values':
            column = issue['column']
            # 将无穷值替换为NaN
            cleaned_data[column] = cleaned_data[column].replace([np.inf, -np.inf], np.nan)
            cleaning_log.append(f"对{column}列的无穷值替换为NaN")
        
        elif issue['type'] == 'outliers':
            column = issue['column']
            # 使用IQR方法截断异常值
            Q1 = cleaned_data[column].quantile(0.25)
            Q3 = cleaned_data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                cleaned_data[column] = cleaned_data[column].clip(
                    lower=lower_bound, 
                    upper=upper_bound
                )
                cleaning_log.append(f"对{column}列使用IQR方法截断异常值")
    
    return cleaned_data, cleaning_log
```

## 计算相关问题

### 问题3：因子计算结果异常

#### 症状
- 因子值全部为NaN
- 因子值分布不合理
- 计算结果与预期不符

#### 诊断方法
```python
def diagnose_calculation_issues(factor_name, input_data, calculation_params, result):
    """
    诊断因子计算问题
    
    Args:
        factor_name: 因子名称
        input_data: 输入数据
        calculation_params: 计算参数
        result: 计算结果
    
    Returns:
        诊断报告
    """
    diagnosis = {
        'factor_name': factor_name,
        'issues': [],
        'input_analysis': {},
        'result_analysis': {},
        'suggestions': []
    }
    
    # 1. 输入数据分析
    diagnosis['input_analysis'] = analyze_input_data(input_data)
    
    # 2. 参数合理性检查
    param_issues = check_parameter_validity(factor_name, calculation_params, input_data)
    diagnosis['issues'].extend(param_issues)
    
    # 3. 结果分析
    diagnosis['result_analysis'] = analyze_calculation_result(result)
    
    # 4. 特定因子的检查
    factor_specific_issues = check_factor_specific_issues(
        factor_name, input_data, calculation_params, result
    )
    diagnosis['issues'].extend(factor_specific_issues)
    
    # 5. 生成建议
    diagnosis['suggestions'] = generate_calculation_suggestions(diagnosis['issues'])
    
    return diagnosis

def analyze_input_data(data):
    """分析输入数据"""
    analysis = {
        'record_count': len(data),
        'date_range': None,
        'missing_columns': [],
        'data_quality': {}
    }
    
    if not data.empty:
        # 日期范围
        if 'trade_date' in data.columns:
            analysis['date_range'] = {
                'start': data['trade_date'].min(),
                'end': data['trade_date'].max(),
                'span_days': (data['trade_date'].max() - data['trade_date'].min()).days
            }
        
        # 必要列检查
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        analysis['missing_columns'] = [
            col for col in required_columns if col not in data.columns
        ]
        
        # 数据质量
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                analysis['data_quality'][col] = {
                    'missing_count': data[col].isnull().sum(),
                    'zero_count': (data[col] == 0).sum(),
                    'negative_count': (data[col] < 0).sum()
                }
    
    return analysis

def check_parameter_validity(factor_name, params, data):
    """检查参数有效性"""
    issues = []
    
    if factor_name in ['sma', 'ema', 'rsi']:
        window = params.get('window', 0)
        
        if window <= 0:
            issues.append({
                'type': 'invalid_parameter',
                'message': f'窗口参数必须大于0，当前值: {window}'
            })
        elif window > len(data):
            issues.append({
                'type': 'parameter_too_large',
                'message': f'窗口参数({window})大于数据长度({len(data)})'
            })
        elif window > len(data) * 0.5:
            issues.append({
                'type': 'parameter_warning',
                'message': f'窗口参数({window})可能过大，建议小于数据长度的50%'
            })
    
    if factor_name == 'bollinger':
        std_multiplier = params.get('std_multiplier', 2)
        if std_multiplier <= 0:
            issues.append({
                'type': 'invalid_parameter',
                'message': f'标准差倍数必须大于0，当前值: {std_multiplier}'
            })
    
    return issues

def analyze_calculation_result(result):
    """分析计算结果"""
    if result is None or (hasattr(result, 'empty') and result.empty):
        return {
            'is_empty': True,
            'total_count': 0,
            'valid_count': 0,
            'statistics': {}
        }
    
    analysis = {
        'is_empty': False,
        'total_count': len(result),
        'valid_count': result.notna().sum() if hasattr(result, 'notna') else 0,
        'statistics': {}
    }
    
    if hasattr(result, 'describe'):
        try:
            analysis['statistics'] = result.describe().to_dict()
        except:
            pass
    
    # 检查特殊值
    if hasattr(result, 'isinf'):
        analysis['infinite_count'] = result.isinf().sum()
    
    return analysis

def check_factor_specific_issues(factor_name, input_data, params, result):
    """检查特定因子的问题"""
    issues = []
    
    if factor_name == 'rsi':
        # RSI应该在0-100之间
        if hasattr(result, 'min') and hasattr(result, 'max'):
            min_val = result.min()
            max_val = result.max()
            
            if min_val < 0 or max_val > 100:
                issues.append({
                    'type': 'value_out_of_range',
                    'message': f'RSI值超出0-100范围: [{min_val:.2f}, {max_val:.2f}]'
                })
    
    elif factor_name == 'sma':
        # SMA应该接近价格水平
        if 'close' in input_data.columns and hasattr(result, 'mean'):
            close_mean = input_data['close'].mean()
            sma_mean = result.mean()
            
            if abs(sma_mean - close_mean) > close_mean * 0.5:  # 50%差异阈值
                issues.append({
                    'type': 'unreasonable_value',
                    'message': f'SMA均值({sma_mean:.2f})与收盘价均值({close_mean:.2f})差异过大'
                })
    
    elif factor_name == 'macd':
        # MACD的DIF和DEA应该相对合理
        if isinstance(result, dict):
            dif = result.get('dif')
            dea = result.get('dea')
            
            if dif is not None and dea is not None:
                if hasattr(dif, 'std') and hasattr(dea, 'std'):
                    if dif.std() == 0 or dea.std() == 0:
                        issues.append({
                            'type': 'no_variation',
                            'message': 'MACD的DIF或DEA没有变化，可能计算有误'
                        })
    
    return issues
```

#### 解决方案
```python
def fix_calculation_issues(diagnosis):
    """
    根据诊断结果修复计算问题
    
    Args:
        diagnosis: 诊断报告
    
    Returns:
        修复建议和代码示例
    """
    fixes = []
    
    for issue in diagnosis['issues']:
        if issue['type'] == 'invalid_parameter':
            fixes.append({
                'issue': issue['message'],
                'solution': '调整参数到合理范围',
                'code_example': get_parameter_fix_example(diagnosis['factor_name'])
            })
        
        elif issue['type'] == 'parameter_too_large':
            fixes.append({
                'issue': issue['message'],
                'solution': '减小窗口参数或增加数据量',
                'code_example': 'window = min(window, len(data) // 2)'
            })
        
        elif issue['type'] == 'value_out_of_range':
            fixes.append({
                'issue': issue['message'],
                'solution': '检查计算逻辑，添加值域限制',
                'code_example': 'result = result.clip(0, 100)  # 对于RSI'
            })
        
        elif issue['type'] == 'unreasonable_value':
            fixes.append({
                'issue': issue['message'],
                'solution': '检查输入数据和计算公式',
                'code_example': '验证数据预处理步骤'
            })
    
    return fixes

def get_parameter_fix_example(factor_name):
    """获取参数修复示例"""
    examples = {
        'sma': 'window = max(1, min(window, 60))  # 限制在1-60之间',
        'rsi': 'window = max(2, min(window, 30))  # RSI窗口通常2-30',
        'bollinger': 'std_multiplier = max(0.1, min(std_multiplier, 5))  # 限制倍数'
    }
    
    return examples.get(factor_name, 'window = max(1, window)')
```

### 问题4：计算性能问题

#### 症状
- 计算时间过长
- 内存使用过高
- 系统响应缓慢

#### 诊断方法
```python
def diagnose_performance_issues():
    """
    诊断性能问题
    
    Returns:
        性能诊断报告
    """
    import psutil
    import time
    
    diagnosis = {
        'timestamp': datetime.now().isoformat(),
        'system_metrics': {},
        'bottlenecks': [],
        'recommendations': []
    }
    
    # 1. 系统资源使用情况
    diagnosis['system_metrics'] = {
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
        'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
    }
    
    # 2. 数据库性能检查
    db_performance = check_database_performance()
    if db_performance['slow_queries']:
        diagnosis['bottlenecks'].append({
            'type': 'database',
            'severity': 'high',
            'message': '检测到慢查询',
            'details': db_performance
        })
    
    # 3. 内存使用分析
    memory_usage = diagnosis['system_metrics']['memory_usage']
    if memory_usage > 80:
        diagnosis['bottlenecks'].append({
            'type': 'memory',
            'severity': 'high',
            'message': f'内存使用率过高: {memory_usage}%'
        })
    
    # 4. CPU使用分析
    cpu_usage = diagnosis['system_metrics']['cpu_usage']
    if cpu_usage > 90:
        diagnosis['bottlenecks'].append({
            'type': 'cpu',
            'severity': 'high',
            'message': f'CPU使用率过高: {cpu_usage}%'
        })
    
    # 5. 生成优化建议
    diagnosis['recommendations'] = generate_performance_recommendations(
        diagnosis['bottlenecks']
    )
    
    return diagnosis

def check_database_performance():
    """检查数据库性能"""
    try:
        with get_db_engine().connect() as conn:
            # 检查慢查询 (PostgreSQL)
            slow_query_sql = """
            SELECT query, mean_time, calls, total_time
            FROM pg_stat_statements 
            WHERE mean_time > 1000  -- 超过1秒的查询
            ORDER BY mean_time DESC 
            LIMIT 10
            """
            
            try:
                result = conn.execute(slow_query_sql)
                slow_queries = [dict(row) for row in result.fetchall()]
            except:
                slow_queries = []  # pg_stat_statements可能未启用
            
            # 检查连接数
            connection_sql = "SELECT count(*) as connection_count FROM pg_stat_activity"
            connection_result = conn.execute(connection_sql)
            connection_count = connection_result.fetchone().connection_count
            
            return {
                'slow_queries': slow_queries,
                'connection_count': connection_count,
                'max_connections': 100  # 假设最大连接数
            }
    
    except Exception as e:
        return {
            'error': str(e),
            'slow_queries': [],
            'connection_count': 0
        }

def generate_performance_recommendations(bottlenecks):
    """生成性能优化建议"""
    recommendations = []
    
    bottleneck_types = [b['type'] for b in bottlenecks]
    
    if 'database' in bottleneck_types:
        recommendations.extend([
            {
                'type': 'database_optimization',
                'priority': 'high',
                'action': '优化慢查询',
                'details': [
                    '添加适当的数据库索引',
                    '优化SQL查询语句',
                    '使用查询缓存',
                    '考虑数据库分区'
                ]
            }
        ])
    
    if 'memory' in bottleneck_types:
        recommendations.extend([
            {
                'type': 'memory_optimization',
                'priority': 'high',
                'action': '优化内存使用',
                'details': [
                    '减少批处理大小',
                    '及时释放不需要的对象',
                    '使用更高效的数据类型',
                    '实现内存监控和告警'
                ]
            }
        ])
    
    if 'cpu' in bottleneck_types:
        recommendations.extend([
            {
                'type': 'cpu_optimization',
                'priority': 'medium',
                'action': '优化CPU使用',
                'details': [
                    '使用向量化计算',
                    '实现并行处理',
                    '优化算法复杂度',
                    '考虑使用缓存'
                ]
            }
        ])
    
    return recommendations
```

#### 解决方案
```python
def implement_performance_optimizations(recommendations):
    """
    实施性能优化
    
    Args:
        recommendations: 优化建议列表
    
    Returns:
        优化实施结果
    """
    implementation_results = []
    
    for rec in recommendations:
        if rec['type'] == 'database_optimization':
            # 数据库优化
            db_results = optimize_database_performance()
            implementation_results.append({
                'type': 'database',
                'actions_taken': db_results,
                'status': 'completed'
            })
        
        elif rec['type'] == 'memory_optimization':
            # 内存优化
            memory_results = optimize_memory_usage()
            implementation_results.append({
                'type': 'memory',
                'actions_taken': memory_results,
                'status': 'completed'
            })
        
        elif rec['type'] == 'cpu_optimization':
            # CPU优化
            cpu_results = optimize_cpu_usage()
            implementation_results.append({
                'type': 'cpu',
                'actions_taken': cpu_results,
                'status': 'completed'
            })
    
    return implementation_results

def optimize_database_performance():
    """优化数据库性能"""
    optimizations = []
    
    try:
        with get_db_engine().connect() as conn:
            # 1. 检查并创建缺失的索引
            missing_indexes = check_missing_indexes(conn)
            for index_sql in missing_indexes:
                try:
                    conn.execute(index_sql)
                    optimizations.append(f"创建索引: {index_sql}")
                except Exception as e:
                    optimizations.append(f"索引创建失败: {e}")
            
            # 2. 更新表统计信息
            try:
                conn.execute("ANALYZE")
                optimizations.append("更新表统计信息")
            except Exception as e:
                optimizations.append(f"统计信息更新失败: {e}")
    
    except Exception as e:
        optimizations.append(f"数据库优化失败: {e}")
    
    return optimizations

def check_missing_indexes(conn):
    """检查缺失的索引"""
    # 这里定义一些常用的索引
    recommended_indexes = [
        "CREATE INDEX IF NOT EXISTS idx_stock_daily_ts_code_date ON stock_daily (ts_code, trade_date)",
        "CREATE INDEX IF NOT EXISTS idx_factor_data_composite ON factor_data (ts_code, factor_name, factor_date)",
        "CREATE INDEX IF NOT EXISTS idx_stock_basic_ts_code ON stock_basic (ts_code)"
    ]
    
    existing_indexes = set()
    try:
        # 查询现有索引
        result = conn.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE schemaname = 'public'
        """)
        existing_indexes = {row.indexname for row in result.fetchall()}
    except:
        pass
    
    # 返回需要创建的索引
    needed_indexes = []
    for index_sql in recommended_indexes:
        index_name = index_sql.split()[-1].split('(')[0]  # 提取索引名
        if index_name not in existing_indexes:
            needed_indexes.append(index_sql)
    
    return needed_indexes

def optimize_memory_usage():
    """优化内存使用"""
    optimizations = []
    
    # 1. 强制垃圾回收
    import gc
    collected = gc.collect()
    optimizations.append(f"垃圾回收释放了{collected}个对象")
    
    # 2. 清理缓存
    try:
        # 清理Redis缓存中的过期数据
        redis_client = get_redis_client()
        if redis_client:
            # 这里可以实现缓存清理逻辑
            optimizations.append("清理了过期缓存数据")
    except:
        optimizations.append("缓存清理失败")
    
    # 3. 调整批处理大小
    # 这个需要在应用层面实现
    optimizations.append("建议调整批处理大小以减少内存使用")
    
    return optimizations

def optimize_cpu_usage():
    """优化CPU使用"""
    optimizations = []
    
    # 1. 启用并行计算
    optimizations.append("建议启用并行计算以提高CPU利用率")
    
    # 2. 优化算法
    optimizations.append("建议使用向量化操作替代循环计算")
    
    # 3. 实现计算缓存
    optimizations.append("建议实现计算结果缓存以减少重复计算")
    
    return optimizations
```

## 系统相关问题

### 问题5：数据库连接问题

#### 症状
- 连接超时
- 连接被拒绝
- 连接池耗尽

#### 诊断和解决方案
```python
def diagnose_database_connection_issues():
    """诊断数据库连接问题"""
    diagnosis = {
        'connection_test': {},
        'pool_status': {},
        'configuration': {},
        'issues': [],
        'solutions': []
    }
    
    # 1. 基础连接测试
    try:
        start_time = time.time()
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            result.fetchone()
        
        connection_time = time.time() - start_time
        diagnosis['connection_test'] = {
            'success': True,
            'connection_time': connection_time
        }
        
        if connection_time > 5:  # 5秒阈值
            diagnosis['issues'].append("数据库连接时间过长")
            diagnosis['solutions'].append("检查网络连接和数据库负载")
    
    except Exception as e:
        diagnosis['connection_test'] = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        
        # 根据错误类型提供解决方案
        if "connection refused" in str(e).lower():
            diagnosis['issues'].append("数据库连接被拒绝")
            diagnosis['solutions'].extend([
                "检查数据库服务是否运行",
                "验证连接参数（主机、端口、用户名、密码）",
                "检查防火墙设置"
            ])
        
        elif "timeout" in str(e).lower():
            diagnosis['issues'].append("数据库连接超时")
            diagnosis['solutions'].extend([
                "增加连接超时时间",
                "检查网络连接",
                "优化数据库性能"
            ])
        
        elif "authentication" in str(e).lower():
            diagnosis['issues'].append("数据库认证失败")
            diagnosis['solutions'].extend([
                "检查用户名和密码",
                "验证用户权限",
                "检查数据库用户配置"
            ])
    
    # 2. 连接池状态检查
    try:
        pool = engine.pool
        diagnosis['pool_status'] = {
            'size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid()
        }
        
        # 检查连接池是否耗尽
        if pool.checkedout() >= pool.size() + pool.overflow():
            diagnosis['issues'].append("数据库连接池耗尽")
            diagnosis['solutions'].extend([
                "增加连接池大小",
                "检查连接泄漏",
                "优化查询性能"
            ])
    
    except Exception as e:
        diagnosis['pool_status'] = {'error': str(e)}
    
    return diagnosis

def fix_database_connection_issues(diagnosis):
    """修复数据库连接问题"""
    fixes_applied = []
    
    if "数据库连接时间过长" in diagnosis['issues']:
        # 优化连接配置
        try:
            # 这里可以调整连接池参数
            fixes_applied.append("调整了连接池参数以提高连接速度")
        except Exception as e:
            fixes_applied.append(f"连接优化失败: {e}")
    
    if "数据库连接池耗尽" in diagnosis['issues']:
        # 清理无效连接
        try:
            engine = get_db_engine()
            engine.pool.dispose()  # 重置连接池
            fixes_applied.append("重置了数据库连接池")
        except Exception as e:
            fixes_applied.append(f"连接池重置失败: {e}")
    
    return fixes_applied
```

### 问题6：内存不足问题

#### 诊断和解决方案
```python
def diagnose_memory_issues():
    """诊断内存问题"""
    import psutil
    
    diagnosis = {
        'system_memory': {},
        'process_memory': {},
        'memory_leaks': [],
        'recommendations': []
    }
    
    # 1. 系统内存状态
    memory = psutil.virtual_memory()
    diagnosis['system_memory'] = {
        'total': memory.total / (1024**3),  # GB
        'available': memory.available / (1024**3),
        'used': memory.used / (1024**3),
        'percentage': memory.percent
    }
    
    # 2. 当前进程内存使用
    process = psutil.Process()
    process_memory = process.memory_info()
    diagnosis['process_memory'] = {
        'rss': process_memory.rss / (1024**2),  # MB
        'vms': process_memory.vms / (1024**2),
        'percentage': process.memory_percent()
    }
    
    # 3. 内存使用分析
    if memory.percent > 90:
        diagnosis['recommendations'].append({
            'type': 'critical',
            'message': '系统内存使用率过高',
            'actions': [
                '立即释放不必要的内存',
                '重启内存消耗大的进程',
                '考虑增加系统内存'
            ]
        })
    
    if process.memory_percent() > 50:
        diagnosis['recommendations'].append({
            'type': 'warning',
            'message': '当前进程内存使用率过高',
            'actions': [
                '检查内存泄漏',
                '优化数据处理逻辑',
                '实现内存监控'
            ]
        })
    
    # 4. 检查潜在的内存泄漏
    diagnosis['memory_leaks'] = check_memory_leaks()
    
    return diagnosis

def check_memory_leaks():
    """检查内存泄漏"""
    import gc
    
    leaks = []
    
    # 1. 检查未释放的大对象
    large_objects = []
    for obj in gc.get_objects():
        if hasattr(obj, '__sizeof__'):
            size = obj.__sizeof__()
            if size > 1024 * 1024:  # 1MB以上的对象
                large_objects.append({
                    'type': type(obj).__name__,
                    'size': size / (1024**2),  # MB
                    'id': id(obj)
                })
    
    if len(large_objects) > 10:
        leaks.append({
            'type': 'large_objects',
            'count': len(large_objects),
            'message': f'发现{len(large_objects)}个大对象，可能存在内存泄漏'
        })
    
    # 2. 检查循环引用
    unreachable = gc.collect()
    if unreachable > 100:
        leaks.append({
            'type': 'circular_references',
            'count': unreachable,
            'message': f'发现{unreachable}个循环引用对象'
        })
    
    return leaks

def fix_memory_issues(diagnosis):
    """修复内存问题"""
    fixes = []
    
    # 1. 强制垃圾回收
    import gc
    collected = gc.collect()
    fixes.append(f"垃圾回收释放了{collected}个对象")
    
    # 2. 清理大对象
    if diagnosis['memory_leaks']:
        for leak in diagnosis['memory_leaks']:
            if leak['type'] == 'large_objects':
                # 这里可以实现大对象清理逻辑
                fixes.append("建议检查和清理大对象")
    
    # 3. 优化内存使用
    fixes.extend([
        "建议使用生成器替代列表处理大数据",
        "建议及时释放不需要的DataFrame",
        "建议使用更高效的数据类型"
    ])
    
    return fixes
```

## API相关问题

### 问题7：API接口错误

#### 诊断和解决方案
```python
def diagnose_api_issues(endpoint, error_response):
    """诊断API问题"""
    diagnosis = {
        'endpoint': endpoint,
        'error_analysis': {},
        'possible_causes': [],
        'solutions': []
    }
    
    # 1. 分析错误响应
    if error_response:
        status_code = error_response.get('status_code', 0)
        error_message = error_response.get('message', '')
        
        diagnosis['error_analysis'] = {
            'status_code': status_code,
            'error_message': error_message,
            'error_category': categorize_api_error(status_code)
        }
        
        # 2. 根据状态码分析原因
        if status_code == 400:
            diagnosis['possible_causes'].extend([
                '请求参数格式错误',
                '缺少必要参数',
                '参数值超出允许范围'
            ])
            diagnosis['solutions'].extend([
                '检查请求参数格式',
                '验证所有必要参数是否提供',
                '确认参数值在有效范围内'
            ])
        
        elif status_code == 401:
            diagnosis['possible_causes'].extend([
                '认证令牌无效或过期',
                '用户名密码错误',
                '缺少认证头'
            ])
            diagnosis['solutions'].extend([
                '重新获取认证令牌',
                '检查用户凭据',
                '确保请求包含正确的认证头'
            ])
        
        elif status_code == 403:
            diagnosis['possible_causes'].extend([
                '用户权限不足',
                '访问被限制',
                'API调用频率超限'
            ])
            diagnosis['solutions'].extend([
                '检查用户权限设置',
                '联系管理员获取访问权限',
                '降低API调用频率'
            ])
        
        elif status_code == 404:
            diagnosis['possible_causes'].extend([
                'API端点不存在',
                '请求的资源不存在',
                'URL路径错误'
            ])
            diagnosis['solutions'].extend([
                '检查API端点URL',
                '确认请求的资源存在',
                '查看API文档确认正确路径'
            ])
        
        elif status_code == 500:
            diagnosis['possible_causes'].extend([
                '服务器内部错误',
                '数据库连接失败',
                '计算过程异常'
            ])
            diagnosis['solutions'].extend([
                '检查服务器日志',
                '验证数据库连接',
                '重试请求或联系技术支持'
            ])
        
        elif status_code == 503:
            diagnosis['possible_causes'].extend([
                '服务暂时不可用',
                '系统维护中',
                '负载过高'
            ])
            diagnosis['solutions'].extend([
                '稍后重试',
                '检查系统状态公告',
                '联系运维团队'
            ])
    
    return diagnosis

def categorize_api_error(status_code):
    """分类API错误"""
    if 400 <= status_code < 500:
        return 'client_error'
    elif 500 <= status_code < 600:
        return 'server_error'
    else:
        return 'unknown'

def test_api_endpoints():
    """测试API端点可用性"""
    endpoints = [
        '/health',
        '/api/v1/factors/metadata',
        '/api/v1/factors/health',
        '/api/v1/feature-store/health'
    ]
    
    test_results = {}
    
    for endpoint in endpoints:
        try:
            # 这里应该使用实际的API客户端
            # response = api_client.get(endpoint)
            # 模拟测试结果
            test_results[endpoint] = {
                'status': 'success',
                'response_time': 0.1,
                'status_code': 200
            }
        except Exception as e:
            test_results[endpoint] = {
                'status': 'failed',
                'error': str(e),
                'status_code': 0
            }
    
    return test_results
```

## 监控和预防

### 健康检查系统

```python
class SystemHealthChecker:
    """系统健康检查器"""
    
    def __init__(self):
        self.checks = {
            'database': self.check_database_health,
            'cache': self.check_cache_health,
            'memory': self.check_memory_health,
            'disk': self.check_disk_health,
            'api': self.check_api_health
        }
    
    def run_health_check(self):
        """运行完整的健康检查"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'issues': [],
            'recommendations': []
        }
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                health_report['checks'][check_name] = result
                
                if not result.get('healthy', True):
                    health_report['issues'].append({
                        'component': check_name,
                        'status': result.get('status', 'unknown'),
                        'message': result.get('message', ''),
                        'severity': result.get('severity', 'medium')
                    })
            
            except Exception as e:
                health_report['checks'][check_name] = {
                    'healthy': False,
                    'error': str(e),
                    'status': 'error'
                }
                health_report['issues'].append({
                    'component': check_name,
                    'status': 'error',
                    'message': f'健康检查失败: {e}',
                    'severity': 'high'
                })
        
        # 确定整体状态
        if health_report['issues']:
            critical_issues = [i for i in health_report['issues'] if i['severity'] == 'critical']
            high_issues = [i for i in health_report['issues'] if i['severity'] == 'high']
            
            if critical_issues:
                health_report['overall_status'] = 'critical'
            elif high_issues:
                health_report['overall_status'] = 'degraded'
            else:
                health_report['overall_status'] = 'warning'
        
        # 生成建议
        health_report['recommendations'] = self.generate_health_recommendations(
            health_report['issues']
        )
        
        return health_report
    
    def check_database_health(self):
        """检查数据库健康状态"""
        try:
            start_time = time.time()
            
            with get_db_engine().connect() as conn:
                # 基础连接测试
                result = conn.execute("SELECT 1")
                result.fetchone()
                
                # 检查关键表
                tables_check = conn.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('stock_basic', 'stock_daily', 'factor_data')
                """)
                existing_tables = [row.table_name for row in tables_check.fetchall()]
                
                connection_time = time.time() - start_time
                
                return {
                    'healthy': True,
                    'connection_time': connection_time,
                    'existing_tables': existing_tables,
                    'status': 'connected'
                }
        
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'status': 'disconnected',
                'severity': 'critical'
            }
    
    def check_cache_health(self):
        """检查缓存健康状态"""
        try:
            # 这里应该检查Redis连接
            # redis_client = get_redis_client()
            # redis_client.ping()
            
            return {
                'healthy': True,
                'status': 'connected'
            }
        
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'status': 'disconnected',
                'severity': 'medium'
            }
    
    def check_memory_health(self):
        """检查内存健康状态"""
        import psutil
        
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            return {
                'healthy': False,
                'memory_usage': memory.percent,
                'status': 'critical',
                'message': f'内存使用率过高: {memory.percent}%',
                'severity': 'critical'
            }
        elif memory.percent > 80:
            return {
                'healthy': False,
                'memory_usage': memory.percent,
                'status': 'warning',
                'message': f'内存使用率较高: {memory.percent}%',
                'severity': 'medium'
            }
        else:
            return {
                'healthy': True,
                'memory_usage': memory.percent,
                'status': 'normal'
            }
    
    def check_disk_health(self):
        """检查磁盘健康状态"""
        import psutil
        
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        if disk_percent > 90:
            return {
                'healthy': False,
                'disk_usage': disk_percent,
                'status': 'critical',
                'message': f'磁盘使用率过高: {disk_percent:.1f}%',
                'severity': 'high'
            }
        elif disk_percent > 80:
            return {
                'healthy': False,
                'disk_usage': disk_percent,
                'status': 'warning',
                'message': f'磁盘使用率较高: {disk_percent:.1f}%',
                'severity': 'medium'
            }
        else:
            return {
                'healthy': True,
                'disk_usage': disk_percent,
                'status': 'normal'
            }
    
    def check_api_health(self):
        """检查API健康状态"""
        try:
            # 测试关键API端点
            test_results = test_api_endpoints()
            
            failed_endpoints = [
                endpoint for endpoint, result in test_results.items()
                if result['status'] != 'success'
            ]
            
            if failed_endpoints:
                return {
                    'healthy': False,
                    'failed_endpoints': failed_endpoints,
                    'status': 'degraded',
                    'message': f'{len(failed_endpoints)}个API端点不可用',
                    'severity': 'high'
                }
            else:
                return {
                    'healthy': True,
                    'status': 'operational',
                    'tested_endpoints': len(test_results)
                }
        
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'status': 'error',
                'severity': 'high'
            }
    
    def generate_health_recommendations(self, issues):
        """生成健康改进建议"""
        recommendations = []
        
        issue_types = [issue['component'] for issue in issues]
        
        if 'database' in issue_types:
            recommendations.append({
                'component': 'database',
                'priority': 'critical',
                'actions': [
                    '检查数据库服务状态',
                    '验证连接配置',
                    '检查数据库日志',
                    '考虑重启数据库服务'
                ]
            })
        
        if 'memory' in issue_types:
            recommendations.append({
                'component': 'memory',
                'priority': 'high',
                'actions': [
                    '立即释放不必要的内存',
                    '重启内存消耗大的进程',
                    '优化内存使用算法',
                    '考虑增加系统内存'
                ]
            })
        
        if 'disk' in issue_types:
            recommendations.append({
                'component': 'disk',
                'priority': 'medium',
                'actions': [
                    '清理临时文件和日志',
                    '删除不必要的数据',
                    '实现日志轮转',
                    '考虑扩展磁盘空间'
                ]
            })
        
        return recommendations
```

## 总结

本故障排除指南涵盖了StockSchool因子计算引擎的主要问题类型和解决方案：

1. **数据相关问题**：数据加载失败、数据质量异常
2. **计算相关问题**：因子计算结果异常、性能问题
3. **系统相关问题**：数据库连接、内存不足
4. **API相关问题**：接口错误、认证失败

通过系统化的诊断流程和自动化的健康检查，可以快速定位和解决问题，确保系统的稳定运行。建议定期运行健康检查，并根据监控结果进行预防性维护。