# Security Architecture Plan for Keisei Web UI

**Domain**: Security, Authentication, Authorization, and Resource Protection
**Planning Agent**: Security Architect
**Date**: 2025-08-23

## Security Architecture Overview

### Threat Model

**Assets to Protect**:
1. **Training Models** - Intellectual property and trained weights
2. **Training Data** - Game histories and learning experiences  
3. **System Resources** - GPU/CPU compute resources
4. **Configuration** - Hyperparameters and training strategies
5. **Logs and Metrics** - Training performance and debugging data

**Threat Actors**:
1. **Unauthorized Users** - External actors attempting access
2. **Malicious Insiders** - Legitimate users exceeding permissions
3. **Resource Abuse** - Users consuming excessive compute resources
4. **Data Exfiltration** - Attempting to steal models or training data

**Attack Vectors**:
1. **Web Application** - API exploitation, XSS, CSRF
2. **Authentication Bypass** - Token manipulation, session hijacking
3. **Resource Exhaustion** - DoS through training resource monopolization
4. **Configuration Injection** - Malicious configuration parameters
5. **File System Access** - Unauthorized access to model files and logs

### Authentication Architecture

#### Primary Authentication Strategy: JWT with Session Management

**JWT Token Structure**:
```json
{
  "header": {
    "alg": "HS256",
    "typ": "JWT"
  },
  "payload": {
    "user_id": "user123",
    "username": "researcher1", 
    "role": "trainer",
    "permissions": ["training.start", "training.monitor", "models.read"],
    "exp": 1724428800,
    "iat": 1724425200,
    "session_id": "sess_abc123"
  }
}
```

**Authentication Flow**:
1. **Login Request** → POST `/api/auth/login` with credentials
2. **Credential Validation** → Against local database or LDAP/OAuth
3. **JWT Generation** → Signed token with user permissions
4. **Session Storage** → Server-side session tracking for revocation
5. **Token Response** → JWT returned to client for API access

#### Multi-Factor Authentication (MFA) Support

**TOTP Integration**:
```python
class AuthenticationManager:
    def verify_mfa_token(self, user_id: str, token: str) -> bool:
        """Verify TOTP token for user."""
        user_secret = self.get_user_mfa_secret(user_id)
        return pyotp.TOTP(user_secret).verify(token, valid_window=1)
        
    def require_mfa_for_actions(self, actions: List[str]):
        """Configure which actions require MFA verification."""
        self.mfa_required_actions = actions
```

### Authorization System

#### Role-Based Access Control (RBAC)

**Role Definitions**:
```python
class UserRole(Enum):
    ADMIN = "admin"           # Full system access
    TRAINER = "trainer"       # Training control and monitoring
    RESEARCHER = "researcher" # Read-only access to results  
    VIEWER = "viewer"        # Basic monitoring only

class Permission(Enum):
    # Training permissions
    TRAINING_START = "training.start"
    TRAINING_STOP = "training.stop"  
    TRAINING_CONFIG = "training.config"
    TRAINING_MONITOR = "training.monitor"
    
    # Model permissions
    MODELS_READ = "models.read"
    MODELS_WRITE = "models.write"
    MODELS_DELETE = "models.delete"
    MODELS_EXPORT = "models.export"
    
    # System permissions
    SYSTEM_CONFIG = "system.config"
    SYSTEM_LOGS = "system.logs"
    USERS_MANAGE = "users.manage"
```

**Permission Matrix**:
| Role | Training Control | Model Management | System Config | User Management |
|------|------------------|------------------|---------------|-----------------|
| Admin | Full | Full | Full | Full |
| Trainer | Start/Stop/Monitor | Read/Write | Read | None |
| Researcher | Monitor Only | Read/Export | Read | None |
| Viewer | Monitor Only | Read Only | None | None |

#### Resource-Based Authorization

**Training Session Access Control**:
```python
class ResourceAuthz:
    def can_access_session(self, user: User, session_id: str) -> bool:
        """Check if user can access specific training session."""
        session = self.get_session(session_id)
        
        # Admin can access all sessions
        if user.role == UserRole.ADMIN:
            return True
            
        # Users can access their own sessions
        if session.owner_id == user.id:
            return True
            
        # Check shared session permissions
        return session_id in user.shared_sessions
        
    def can_control_training(self, user: User, session_id: str) -> bool:
        """Check if user can control training for session."""
        return (self.can_access_session(user, session_id) and 
                Permission.TRAINING_START in user.permissions)
```

### API Security Implementation

#### Request Authentication Middleware
```python
from functools import wraps
from flask import request, jsonify
import jwt

def require_auth(permissions: List[str] = None):
    """Decorator for API endpoints requiring authentication."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            token = extract_jwt_from_header(request)
            
            if not token:
                return jsonify({'error': 'Authentication required'}), 401
                
            try:
                payload = jwt.decode(token, app.config['JWT_SECRET'], algorithms=['HS256'])
                current_user = get_user_by_id(payload['user_id'])
                
                # Check session is still valid
                if not is_session_active(payload['session_id']):
                    return jsonify({'error': 'Session expired'}), 401
                    
                # Check permissions if specified
                if permissions:
                    if not has_permissions(current_user, permissions):
                        return jsonify({'error': 'Insufficient permissions'}), 403
                        
                return f(current_user=current_user, *args, **kwargs)
                
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid token'}), 401
                
        return decorated_function
    return decorator

# Usage example
@app.route('/api/training/start', methods=['POST'])
@require_auth([Permission.TRAINING_START])
def start_training(current_user):
    # Training start logic
    pass
```

#### Input Validation and Sanitization

**Configuration Validation**:
```python
from pydantic import BaseModel, validator
import bleach

class ConfigurationRequest(BaseModel):
    training_config: TrainingConfig
    model_config: ModelConfig
    
    @validator('*', pre=True)
    def sanitize_strings(cls, v):
        """Sanitize all string inputs."""
        if isinstance(v, str):
            return bleach.clean(v, tags=[], strip=True)
        return v
        
    @validator('training_config')
    def validate_training_bounds(cls, v):
        """Ensure training parameters are within safe bounds."""
        if v.learning_rate <= 0 or v.learning_rate > 1.0:
            raise ValueError("Learning rate must be between 0 and 1")
        if v.total_timesteps > 10_000_000:
            raise ValueError("Total timesteps exceeds maximum limit")
        return v
```

### WebSocket Security

#### Connection Authentication
```python
import socketio
from functools import wraps

def authenticated_only(f):
    """Decorator for SocketIO events requiring authentication."""
    @wraps(f)
    def decorated_function(sid, *args, **kwargs):
        # Extract JWT from connection auth data
        session_data = socketio_server.get_session(sid)
        token = session_data.get('auth_token')
        
        if not token or not validate_jwt(token):
            socketio_server.emit('auth_error', {'message': 'Authentication required'}, to=sid)
            return
            
        user = get_user_from_jwt(token)
        return f(sid, user, *args, **kwargs)
    return decorated_function

@socketio_server.on('subscribe_training_updates')
@authenticated_only
def handle_training_subscription(sid, user, data):
    """Handle subscription to training updates.""" 
    session_id = data.get('session_id')
    
    if not can_access_session(user, session_id):
        socketio_server.emit('error', {'message': 'Access denied'}, to=sid)
        return
        
    # Add to subscription group
    socketio_server.enter_room(sid, f'training_{session_id}')
```

### Resource Protection

#### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Endpoint-specific rate limits
@app.route('/api/training/start', methods=['POST'])
@limiter.limit("5 per minute")  # Prevent training spam
@require_auth([Permission.TRAINING_START])
def start_training():
    pass

@app.route('/api/metrics/current', methods=['GET'])  
@limiter.limit("60 per minute")  # Higher limit for monitoring
@require_auth([Permission.TRAINING_MONITOR])
def get_current_metrics():
    pass
```

#### Resource Quotas
```python
class ResourceManager:
    def __init__(self):
        self.active_trainings = {}  # user_id -> training_session
        self.resource_limits = {
            UserRole.TRAINER: {'max_concurrent_trainings': 2, 'gpu_hours_per_day': 24},
            UserRole.RESEARCHER: {'max_concurrent_trainings': 1, 'gpu_hours_per_day': 8},
            UserRole.VIEWER: {'max_concurrent_trainings': 0, 'gpu_hours_per_day': 0}
        }
        
    def can_start_training(self, user: User) -> Tuple[bool, str]:
        """Check if user can start new training session."""
        limits = self.resource_limits[user.role]
        
        # Check concurrent training limit
        user_trainings = [t for t in self.active_trainings.values() if t.user_id == user.id]
        if len(user_trainings) >= limits['max_concurrent_trainings']:
            return False, "Maximum concurrent trainings exceeded"
            
        # Check daily GPU hour limit
        daily_usage = self.get_daily_gpu_usage(user.id)
        if daily_usage >= limits['gpu_hours_per_day']:
            return False, "Daily GPU hour limit exceeded"
            
        return True, "OK"
```

### Data Protection

#### Model File Security
```python
import os
from pathlib import Path

class ModelFileManager:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def get_user_model_path(self, user_id: str, model_name: str) -> Path:
        """Get secure path for user's model files."""
        # Prevent path traversal attacks
        safe_model_name = secure_filename(model_name)
        user_dir = self.base_path / f"user_{user_id}"
        
        # Ensure path is within user directory
        model_path = (user_dir / safe_model_name).resolve()
        if not str(model_path).startswith(str(user_dir.resolve())):
            raise ValueError("Invalid model path")
            
        return model_path
        
    def can_access_model(self, user: User, model_path: Path) -> bool:
        """Check if user can access model file."""
        # Admin can access all models
        if user.role == UserRole.ADMIN:
            return True
            
        # Users can only access their own models
        user_dir = self.base_path / f"user_{user.id}"
        return str(model_path).startswith(str(user_dir))
```

### Logging and Monitoring

#### Security Event Logging
```python
import logging
from enum import Enum

class SecurityEvent(Enum):
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

class SecurityLogger:
    def __init__(self):
        self.logger = logging.getLogger('security')
        
    def log_security_event(self, event: SecurityEvent, user_id: str = None, 
                          details: Dict[str, Any] = None):
        """Log security-related events."""
        log_data = {
            'event': event.value,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'ip_address': request.remote_addr if request else None,
            'user_agent': request.headers.get('User-Agent') if request else None,
            'details': details or {}
        }
        
        self.logger.warning(f"Security Event: {json.dumps(log_data)}")
```

## PLAN_UNCERTAINTY Tags

**PLAN_UNCERTAINTY**: OAuth integration strategy - whether to support external OAuth providers (Google, GitHub) or use local authentication only

**PLAN_UNCERTAINTY**: Certificate management for HTTPS - whether to use self-signed certificates, Let's Encrypt, or corporate CA

**PLAN_UNCERTAINTY**: Session storage backend - Redis vs in-memory vs database for session management scalability

**PLAN_UNCERTAINTY**: Audit log retention policy - how long to keep security logs and where to store them

**PLAN_UNCERTAINTY**: Multi-tenancy requirements - whether single installation supports multiple organizations or just multiple users

## Dependencies on Other Plans

- **API Plan**: Authentication middleware integration points
- **Integration Plan**: Manager access patterns that need authorization  
- **Performance Plan**: Rate limiting and resource quota enforcement
- **Frontend Plan**: Authentication state management and secure token handling