# StockSchool 验收测试报告

## 基本信息

- **测试会话ID**: c585c930-6263-4f84-9280-55ca320ecca0
- **开始时间**: 2025-08-06 04:47:59
- **结束时间**: 2025-08-06 04:48:12
- **整体结果**: ❌ 失败

## 测试统计

| 指标 | 数量 |
|------|------|
| 总测试数 | 8 |
| 通过测试 | 1 |
| 失败测试 | 7 |
| 跳过测试 | 0 |

## 测试阶段结果

### 基础设施验收 (1/8 通过)

- ✅ **docker_daemon_check** (0.05s)
- ❌ **postgres_container_check** (0.01s)
  - 错误: PostgreSQL容器不存在，请先启动Docker Compose
- ❌ **redis_container_check** (0.00s)
  - 错误: Redis容器不存在，请先启动Docker Compose
- ❌ **network_connectivity_check** (4.52s)
  - 错误: 网络连通性检查失败: 部分服务端口不可访问: postgres(localhost:5433), redis(localhost:6380)
- ❌ **postgres_connection_check** (4.08s)
  - 错误: PostgreSQL连接失败: PostgreSQL连接失败: connection to server at "localhost" (::1), port 5433 failed: Connection refused (0x0000274D/10061)
	Is the server running on that host and accepting TCP/IP connections?
connection to server at "localhost" (127.0.0.1), port 5433 failed: Connection refused (0x0000274D/10061)
	Is the server running on that host and accepting TCP/IP connections?

- ❌ **redis_connection_check** (4.09s)
  - 错误: Redis连接失败: Redis连接失败: Error 10061 connecting to localhost:6380. 由于目标计算机积极拒绝，无法连接。.
- ❌ **environment_variables_check** (0.00s)
  - 错误: 缺少必需的环境变量: DATABASE_URL (数据库连接URL), REDIS_URL (Redis连接URL)
- ❌ **python_dependencies_check** (0.17s)
  - 错误: 缺少关键Python包: psycopg2

## 改进建议

1. 建议检查Docker服务状态和网络连接配置
2. 整体通过率较低，建议进行全面的系统检查和优化
