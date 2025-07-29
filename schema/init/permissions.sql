-- 数据库权限设置

-- 创建只读用户
CREATE USER stock_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE stockschool TO stock_readonly;
GRANT USAGE ON SCHEMA public TO stock_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO stock_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO stock_readonly;

-- 创建应用用户
CREATE USER stock_app WITH PASSWORD 'app_password';
GRANT CONNECT ON DATABASE stockschool TO stock_app;
GRANT USAGE ON SCHEMA public TO stock_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO stock_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO stock_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO stock_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE ON SEQUENCES TO stock_app;

-- 创建管理员用户
CREATE USER stock_admin WITH PASSWORD 'admin_password' SUPERUSER;
GRANT ALL PRIVILEGES ON DATABASE stockschool TO stock_admin;

-- 角色分配
CREATE ROLE data_analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO data_analyst;
GRANT data_analyst TO stock_readonly;

CREATE ROLE data_maintainer;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO data_maintainer;
GRANT data_maintainer TO stock_app;

CREATE ROLE system_admin;
GRANT ALL PRIVILEGES ON SCHEMA public TO system_admin;
GRANT system_admin TO stock_admin;