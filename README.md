# StockSchool 量化投研系统

一个基于Python的量化投资研究平台，集成数据获取、因子计算、策略评估、性能监控和告警等功能。

## 🚀 项目特色

## 核心特性

### 📊 数据层
- **多数据源集成**: 支持Tushare、Akshare等主流金融数据源
- **时序数据库**: 基于TimescaleDB的高性能时序数据存储
- **数据质量监控**: 自动化数据质量检查和异常检测
- **增量更新**: 支持数据的增量更新和历史数据回填

### 🧮 因子计算层
- **多维度因子**: 技术面、基本面、情绪面等多维度因子计算
- **实时计算**: 支持实时因子计算和历史因子回测
- **因子评分**: 智能因子评分和排名系统
- **特征工程**: 自动化特征提取和特征选择

### 🤖 AI校长系统
- **智能推荐**: 基于机器学习的股票推荐系统
- **策略优化**: 自动化策略参数优化
- **风险控制**: 智能风险评估和控制
- **自我学习**: 持续学习和模型优化

### 📈 可视化与监控
- **实时监控**: Grafana + Prometheus监控体系
- **交互式图表**: 丰富的数据可视化组件
- **性能分析**: 系统性能和策略表现分析
- **告警系统**: 多渠道告警通知

## 技术架构

### 后端技术栈
- **Python 3.11**: 主要开发语言
- **FastAPI**: 高性能Web框架
- **PostgreSQL + TimescaleDB**: 时序数据库
- **Redis**: 缓存和消息队列
- **Celery**: 异步任务处理
- **SQLAlchemy**: ORM框架
- **Pandas**: 数据处理
- **NumPy**: 数值计算
- **Scikit-learn**: 机器学习

### 前端技术栈
- **React**: 前端框架
- **TypeScript**: 类型安全
- **Ant Design**: UI组件库
- **ECharts**: 图表库
- **Redux**: 状态管理

### 基础设施
- **Docker**: 容器化部署
- **Docker Compose**: 服务编排
- **Nginx**: 反向代理
- **Grafana**: 监控面板
- **Prometheus**: 指标收集

## 快速开始

### 环境要求
- Docker 20.0+
- Docker Compose 2.0+
- Python 3.11+ (开发环境)
- Node.js 18+ (前端开发)

### 1. 克隆项目
```bash
git clone https://github.com/your-username/stockschool.git
cd stockschool
```

### 2. 配置环境变量
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量文件
vim .env
```

**重要**: 请在`.env`文件中配置您的Tushare API Token:
```bash
TUSHARE_TOKEN=your_actual_tushare_token_here
```

### 3. 启动服务
```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f stockschool_app
```

### 4. 使用 `run.py` 启动和管理应用
为了更方便地启动和管理StockSchool应用，我们提供了一个 `run.py` 脚本，它提供了一个交互式菜单。

```bash
python run.py
```

运行 `run.py` 后，您将看到一个菜单，可以选择启动API服务器或执行数据同步。



### 4. 数据同步
```bash
# 启动数据库服务
docker-compose up -d

# 等待数据库启动完成后，执行数据同步
# 完整同步（推荐首次使用）
python src/data/tushare_sync.py full

# 或分步同步
python src/data/tushare_sync.py basic     # 同步股票基本信息
python src/data/tushare_sync.py calendar  # 同步交易日历
python src/data/tushare_sync.py daily     # 同步日线数据
```

### 5. 访问数据库
- **PostgreSQL数据库**: localhost:15432
- **Redis缓存**: localhost:16379
- **数据库用户**: stockschool / stockschool123
- **数据库名称**: stockschool

可使用数据库客户端工具（如DBeaver、pgAdmin等）连接查看数据。

## 开发指南

### 本地开发环境

1. **安装Python依赖**
```bash
pip install -r requirements.txt
```

2. **启动数据库服务**
```bash
docker-compose up -d postgres redis
```

3. **使用 `run.py` 运行开发服务器和数据同步**
在本地开发环境中，您可以使用 `run.py` 脚本来启动API服务和执行数据同步，而无需手动输入复杂的命令。

```bash
python run.py
```

运行 `run.py` 后，您将看到一个菜单，可以选择启动API服务器或执行数据同步。对于Celery Worker和Celery Beat，您仍然需要单独启动它们：

```bash
# 启动Celery Worker
celery -A src.tasks.celery_app worker --loglevel=info

# 启动Celery Beat
celery -A src.tasks.celery_app beat --loglevel=info
```

### 项目结构
```
StockSchool/
├── src/                    # 源代码目录
│   ├── data/              # 数据获取模块
│   │   ├── __init__.py
│   │   └── tushare_sync.py # Tushare数据同步器
│   ├── compute/           # 因子计算模块
│   ├── strategy/          # 策略模块
│   ├── database/          # 数据库模块
│   ├── utils/             # 工具模块
│   │   ├── db.py         # 数据库连接工具
│   │   └── retry.py      # 重试装饰器
│   ├── monitoring/        # 监控模块
│   ├── features/          # 特征工程
│   └── tests/             # 测试代码
├── input/                 # 输入文档目录
├── docker-compose.yml     # Docker编排文件
├── Dockerfile            # Docker镜像文件
├── requirements.txt      # Python依赖
├── database_schema.sql   # 数据库Schema
├── log.md                # 开发日志
├── .env                  # 环境变量配置
└── README.md            # 项目说明
```

### 代码规范

1. **Python代码规范**
   - 遵循PEP 8代码风格
   - 使用Black进行代码格式化
   - 使用isort进行导入排序
   - 使用mypy进行类型检查

2. **提交规范**
   - 使用Conventional Commits规范
   - 格式: `type(scope): description`
   - 类型: feat, fix, docs, style, refactor, test, chore

3. **测试规范**
   - 单元测试覆盖率 > 80%
   - 使用pytest进行测试
   - 测试文件命名: `test_*.py`

## 数据源配置

### Tushare配置
1. 注册Tushare账号: https://tushare.pro/
2. 获取API Token
3. 在`.env`文件中配置`TUSHARE_TOKEN`

### Akshare配置
1. Akshare为免费数据源，无需Token
2. 部分接口有频率限制，请合理使用

## 部署指南

### 生产环境部署

1. **服务器要求**
   - CPU: 4核心以上
   - 内存: 8GB以上
   - 存储: 100GB以上SSD
   - 操作系统: Ubuntu 20.04+ / CentOS 8+

2. **安全配置**
```bash
# 修改默认密码
POSTGRES_PASSWORD=your-strong-password
REDIS_PASSWORD=your-strong-password
SECRET_KEY=your-very-strong-secret-key

# 设置生产环境
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
```

3. **SSL证书配置**
```bash
# 将SSL证书放置到config/nginx/ssl/目录
cp your-cert.pem config/nginx/ssl/
cp your-key.pem config/nginx/ssl/
```

4. **备份策略**
```bash
# 数据库备份
docker-compose exec postgres pg_dump -U stockschool stockschool > backup.sql

# 数据文件备份
tar -czf data_backup.tar.gz data/ logs/
```

### 监控与告警

1. **Grafana面板**
   - 系统性能监控
   - 数据质量监控
   - 业务指标监控

2. **告警配置**
   - 邮件告警
   - 钉钉/企业微信告警
   - 短信告警(可选)

## 常见问题

### Q1: 数据库连接失败
A: 检查PostgreSQL服务是否正常启动，确认连接参数正确。

### Q2: Tushare数据获取失败
A: 检查Token是否正确，是否超出API调用限制。

### Q3: 内存使用过高
A: 调整数据处理批次大小，优化SQL查询，增加服务器内存。

### Q4: 因子计算速度慢
A: 使用并行计算，优化算法逻辑，考虑使用GPU加速。

## 贡献指南

1. Fork项目
2. 创建特性分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交Pull Request

## 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

## 联系我们

- 项目主页: https://github.com/your-username/stockschool
- 问题反馈: https://github.com/your-username/stockschool/issues
- 邮箱: your-email@example.com

## 更新日志

### v1.0.0 (2025-07-27)
- 初始版本发布
- 基础数据获取功能
- 因子计算框架
- Docker部署支持

---

**免责声明**: 本系统仅供学习和研究使用，不构成投资建议。投资有风险，入市需谨慎。