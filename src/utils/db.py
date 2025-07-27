import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 加载环境变量，指定编码格式
load_dotenv(encoding='utf-8')

def get_db_engine():
    db_user = "stockschool"
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_name = "stockschool"
    db_host = "localhost"  # 使用localhost连接本地数据库
    db_port = os.getenv('POSTGRES_PORT', '15432')
    
    # 确保密码是字符串并且正确编码
    if db_password:
        db_password = str(db_password).strip()
    
    db_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    print(f"数据库连接字符串: {db_uri}")
    
    # 添加连接参数以处理编码问题
    engine = create_engine(db_uri, connect_args={"client_encoding": "utf8"})
    return engine