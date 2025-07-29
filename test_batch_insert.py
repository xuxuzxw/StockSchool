import pandas as pd
from sqlalchemy import create_engine, text

# 创建测试数据库
engine = create_engine('sqlite:///test.db')

# 创建测试数据
data = [
    {'ts_code': '000001.SZ', 'end_date': '20231231', 'report_type': '1', 'total_revenue': 1000000},
    {'ts_code': '000002.SZ', 'end_date': '20231231', 'report_type': '1', 'total_revenue': 2000000},
    {'ts_code': '000003.SZ', 'end_date': '20231231', 'report_type': '1', 'total_revenue': 3000000}
]

# 批量插入测试
df = pd.DataFrame(data)
df.to_sql('financial_reports', engine, if_exists='replace', index=False, method='multi')
print('批量插入测试成功')

# 验证数据
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM financial_reports')).fetchone()
    print(f'插入记录数: {result[0]}')

# 测试逐条插入
engine2 = create_engine('sqlite:///test2.db')
for record in data:
    df_single = pd.DataFrame([record])
    df_single.to_sql('financial_reports', engine2, if_exists='append', index=False)

print('逐条插入测试成功')

with engine2.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM financial_reports')).fetchone()
    print(f'逐条插入记录数: {result[0]}')