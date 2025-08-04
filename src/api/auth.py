#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API认证和权限控制模块
提供JWT token验证、用户权限管理、API访问控制等功能
"""

import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import redis
import logging

from src.utils.config_loader import config
from src.utils.db import get_db_engine

# 配置日志
logger = logging.getLogger(__name__)

# JWT配置
JWT_SECRET_KEY = config.get('api.jwt_secret_key', 'your-secret-key-change-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = config.get('api.jwt_expiration_hours', 24)

# Redis连接（用于token黑名单）
try:
    redis_client = redis.Redis(
        host=config.get('redis.host', 'localhost'),
        port=config.get('redis.port', 6379),
        password=config.get('redis.password'),
        db=config.get('redis.auth_db', 1),
        decode_responses=True
    )
except Exception as e:
    logger.warning(f"Redis连接失败，将使用内存存储: {e}")
    redis_client = None

# 内存存储（Redis不可用时的备选方案）
memory_blacklist = set()
memory_sessions = {}

# 安全认证
security = HTTPBearer()

# 数据模型
class UserRole(BaseModel):
    """用户角色"""
    role_name: str
    permissions: List[str]
    description: str

class User(BaseModel):
    """用户信息"""
    user_id: str
    username: str
    email: str
    roles: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

class TokenData(BaseModel):
    """Token数据"""
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    exp: datetime
    iat: datetime

class LoginRequest(BaseModel):
    """登录请求"""
    username: str
    password: str

class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]

# 权限定义
PERMISSIONS = {
    # 因子查询权限
    'factor:read': '查询因子数据',
    'factor:write': '修改因子数据',
    'factor:calculate': '触发因子计算',
    'factor:delete': '删除因子数据',
    
    # 因子管理权限
    'factor_management:read': '查看因子管理信息',
    'factor_management:write': '修改因子管理配置',
    'factor_management:admin': '因子管理员权限',
    
    # 系统管理权限
    'system:monitor': '系统监控',
    'system:config': '系统配置',
    'system:admin': '系统管理员',
    
    # 用户管理权限
    'user:read': '查看用户信息',
    'user:write': '修改用户信息',
    'user:admin': '用户管理员',
}

# 角色定义
ROLES = {
    'viewer': {
        'name': '查看者',
        'permissions': ['factor:read']
    },
    'analyst': {
        'name': '分析师',
        'permissions': ['factor:read', 'factor:calculate', 'system:monitor']
    },
    'developer': {
        'name': '开发者',
        'permissions': [
            'factor:read', 'factor:write', 'factor:calculate',
            'factor_management:read', 'system:monitor'
        ]
    },
    'admin': {
        'name': '管理员',
        'permissions': list(PERMISSIONS.keys())
    }
}

class AuthManager:
    """认证管理器"""
    
    def __init__(self):
        self.engine = get_db_engine()
        self._init_default_users()
    
    def _init_default_users(self):
        """初始化默认用户"""
        try:
            with self.engine.connect() as conn:
                # 检查是否已有管理员用户
                result = conn.execute(
                    "SELECT COUNT(*) as count FROM users WHERE username = 'admin'"
                )
                
                if result.fetchone().count == 0:
                    # 创建默认管理员用户
                    admin_password = self._hash_password('admin123')
                    conn.execute("""
                        INSERT INTO users (user_id, username, email, password_hash, roles, is_active, created_at)
                        VALUES (:user_id, :username, :email, :password_hash, :roles, :is_active, :created_at)
                    """, {
                        'user_id': 'admin_001',
                        'username': 'admin',
                        'email': 'admin@stockschool.com',
                        'password_hash': admin_password,
                        'roles': ['admin'],
                        'is_active': True,
                        'created_at': datetime.now()
                    })
                    
                    logger.info("默认管理员用户创建成功")
                    
        except Exception as e:
            logger.error(f"初始化默认用户失败: {e}")
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        try:
            salt, hash_value = password_hash.split(':')
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hash_value == password_hash_check.hex()
        except Exception:
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """用户认证"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute("""
                    SELECT user_id, username, email, password_hash, roles, is_active, created_at, last_login
                    FROM users 
                    WHERE username = :username AND is_active = true
                """, {'username': username})
                
                user_data = result.fetchone()
                if not user_data:
                    return None
                
                # 验证密码
                if not self._verify_password(password, user_data.password_hash):
                    return None
                
                # 更新最后登录时间
                conn.execute("""
                    UPDATE users SET last_login = :last_login WHERE user_id = :user_id
                """, {
                    'user_id': user_data.user_id,
                    'last_login': datetime.now()
                })
                
                return User(
                    user_id=user_data.user_id,
                    username=user_data.username,
                    email=user_data.email,
                    roles=user_data.roles,
                    is_active=user_data.is_active,
                    created_at=user_data.created_at,
                    last_login=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"用户认证失败: {e}")
            return None
    
    def get_user_permissions(self, roles: List[str]) -> List[str]:
        """获取用户权限"""
        permissions = set()
        for role in roles:
            if role in ROLES:
                permissions.update(ROLES[role]['permissions'])
        return list(permissions)
    
    def create_access_token(self, user: User) -> str:
        """创建访问令牌"""
        permissions = self.get_user_permissions(user.roles)
        
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        
        # 存储会话信息
        self._store_session(user.user_id, token, payload['exp'])
        
        return token
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """验证访问令牌"""
        try:
            # 检查token是否在黑名单中
            if self._is_token_blacklisted(token):
                return None
            
            # 解码token
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # 验证token是否过期
            exp_timestamp = payload.get('exp')
            if exp_timestamp and datetime.utcnow().timestamp() > exp_timestamp:
                return None
            
            return TokenData(
                user_id=payload['user_id'],
                username=payload['username'],
                roles=payload['roles'],
                permissions=payload['permissions'],
                exp=datetime.fromtimestamp(exp_timestamp),
                iat=datetime.fromtimestamp(payload['iat'])
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token已过期")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"无效的token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token验证失败: {e}")
            return None
    
    def revoke_token(self, token: str):
        """撤销令牌"""
        try:
            # 将token添加到黑名单
            if redis_client:
                # 使用Redis存储黑名单
                redis_client.sadd('token_blacklist', token)
                # 设置过期时间（与token过期时间一致）
                redis_client.expire('token_blacklist', JWT_EXPIRATION_HOURS * 3600)
            else:
                # 使用内存存储
                memory_blacklist.add(token)
                
        except Exception as e:
            logger.error(f"撤销token失败: {e}")
    
    def _is_token_blacklisted(self, token: str) -> bool:
        """检查token是否在黑名单中"""
        try:
            if redis_client:
                return redis_client.sismember('token_blacklist', token)
            else:
                return token in memory_blacklist
        except Exception:
            return False
    
    def _store_session(self, user_id: str, token: str, exp_time: datetime):
        """存储会话信息"""
        try:
            session_data = {
                'user_id': user_id,
                'token': token,
                'exp_time': exp_time.isoformat(),
                'created_at': datetime.now().isoformat()
            }
            
            if redis_client:
                redis_client.hset(f'session:{user_id}', mapping=session_data)
                redis_client.expire(f'session:{user_id}', JWT_EXPIRATION_HOURS * 3600)
            else:
                memory_sessions[user_id] = session_data
                
        except Exception as e:
            logger.error(f"存储会话信息失败: {e}")

# 全局认证管理器实例
auth_manager = AuthManager()

# 依赖注入函数
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """验证访问令牌"""
    token = credentials.credentials
    token_data = auth_manager.verify_token(token)
    
    if not token_data:
        raise HTTPException(
            status_code=401,
            detail="无效的访问令牌",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return token_data

def require_permissions(required_permissions: List[str]):
    """权限检查装饰器"""
    def permission_checker(token_data: TokenData = Depends(verify_token)) -> TokenData:
        user_permissions = set(token_data.permissions)
        required_permissions_set = set(required_permissions)
        
        if not required_permissions_set.issubset(user_permissions):
            missing_permissions = required_permissions_set - user_permissions
            raise HTTPException(
                status_code=403,
                detail=f"权限不足，缺少权限: {', '.join(missing_permissions)}"
            )
        
        return token_data
    
    return permission_checker

def require_roles(required_roles: List[str]):
    """角色检查装饰器"""
    def role_checker(token_data: TokenData = Depends(verify_token)) -> TokenData:
        user_roles = set(token_data.roles)
        required_roles_set = set(required_roles)
        
        if not required_roles_set.intersection(user_roles):
            raise HTTPException(
                status_code=403,
                detail=f"角色权限不足，需要角色: {', '.join(required_roles)}"
            )
        
        return token_data
    
    return role_checker

# 便捷的权限检查函数
def require_factor_read():
    """需要因子读取权限"""
    return require_permissions(['factor:read'])

def require_factor_write():
    """需要因子写入权限"""
    return require_permissions(['factor:write'])

def require_factor_admin():
    """需要因子管理权限"""
    return require_permissions(['factor_management:admin'])

def require_system_admin():
    """需要系统管理员权限"""
    return require_roles(['admin'])

# API端点
from fastapi import APIRouter

auth_router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

@auth_router.post("/login", response_model=LoginResponse, summary="用户登录")
async def login(request: LoginRequest):
    """用户登录"""
    user = auth_manager.authenticate_user(request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误"
        )
    
    access_token = auth_manager.create_access_token(user)
    
    return LoginResponse(
        access_token=access_token,
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        user_info={
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'roles': user.roles,
            'permissions': auth_manager.get_user_permissions(user.roles)
        }
    )

@auth_router.post("/logout", summary="用户登出")
async def logout(token_data: TokenData = Depends(verify_token)):
    """用户登出"""
    # 这里需要从请求中获取原始token
    # 在实际实现中，可能需要修改verify_token函数来返回原始token
    # auth_manager.revoke_token(original_token)
    
    return {"message": "登出成功"}

@auth_router.get("/me", summary="获取当前用户信息")
async def get_current_user(token_data: TokenData = Depends(verify_token)):
    """获取当前用户信息"""
    return {
        'user_id': token_data.user_id,
        'username': token_data.username,
        'roles': token_data.roles,
        'permissions': token_data.permissions,
        'token_exp': token_data.exp.isoformat()
    }

@auth_router.get("/permissions", summary="获取权限列表")
async def get_permissions(token_data: TokenData = Depends(require_system_admin())):
    """获取系统权限列表"""
    return {
        'permissions': PERMISSIONS,
        'roles': ROLES
    }