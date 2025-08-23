# Unified Architecture Blueprint: Keisei Web UI Integration

**Synthesis Agent**: Plan Integration Specialist
**Date**: 2025-08-23
**Status**: SYNTHESIZED

## Executive Summary

This blueprint synthesizes five domain-specific plans into a cohesive architecture for adding a comprehensive web-based user interface to the Keisei Deep Reinforcement Learning system. The solution preserves the existing manager-based architecture while adding a minimal, non-disruptive web layer.

## High-Level Architecture

### Three-Tier Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                 WEB FRONTEND                        │
│  React + TypeScript + Redux + Socket.IO            │
│  - Dashboard, Training Monitor, Game Visualization  │
│  - Configuration Editor, Model Management          │
└─────────────────────────────────────────────────────┘
                            │
                            │ HTTP/WebSocket
                            ▼
┌─────────────────────────────────────────────────────┐
│                 API GATEWAY                         │
│  FastAPI + WebSocket + JWT Auth + Rate Limiting    │
│  - REST API, Real-time Updates, Security           │
└─────────────────────────────────────────────────────┘
                            │
                            │ Manager Integration
                            ▼
┌─────────────────────────────────────────────────────┐
│              TRAINING SYSTEM                        │
│  Existing 9 Managers + New WebInterfaceManager     │
│  - Minimal Integration, Event-Driven Updates       │
└─────────────────────────────────────────────────────┘
```

## Integrated Component Architecture

### New Components Required

#### 1. WebInterfaceManager (Primary Integration Point)
**Location**: `keisei/web/web_interface_manager.py`
**Responsibilities**:
- Coordinate between web requests and existing managers
- Aggregate training state from multiple managers
- Handle web-specific callbacks and events
- Manage resource access control

#### 2. API Server (FastAPI-based)
**Location**: `keisei/web/api_server.py`
**Components**:
- REST endpoint handlers
- WebSocket connection management
- Authentication middleware
- Request validation and rate limiting

#### 3. StatePublisher (Real-time Updates)
**Location**: `keisei/web/state_publisher.py`
**Responsibilities**:
- WebSocket connection pool management
- Differential state updates
- Message queuing and delivery
- Connection health monitoring

#### 4. WebAuthManager (Security Layer)
**Location**: `keisei/web/auth_manager.py`
**Components**:
- JWT token management
- Role-based access control
- Session management
- MFA support (optional)

#### 5. ResourceController (Multi-user Support)
**Location**: `keisei/web/resource_controller.py`
**Responsibilities**:
- Training resource allocation
- User quota management
- Concurrent session handling
- Resource conflict resolution

### Manager Integration Points

#### Existing Manager Extensions (Minimal Changes)

**MetricsManager Extensions**:
```python
def get_web_metrics(self) -> Dict[str, Any]:
    """Return web-optimized metrics format."""
    
def subscribe_web_updates(self, callback: Callable):
    """Subscribe to metric updates for web broadcasting."""
```

**SessionManager Extensions**:
```python
def get_session_info(self) -> Dict[str, Any]:
    """Return session metadata for web display."""
    
def get_available_sessions(self) -> List[Dict]:
    """List available training sessions."""
```

**CallbackManager Extensions**:
```python
def register_web_callbacks(self, web_handlers: WebCallbackHandlers):
    """Register web-specific event handlers."""
```

## Unified API Specification

### REST Endpoints (Resolved from API Plan)

**Training Control**:
- `POST /api/training/start` → WebInterfaceManager.start_training()
- `POST /api/training/stop` → WebInterfaceManager.stop_training()
- `GET /api/training/status` → Aggregate from multiple managers

**Configuration Management**:
- `GET /api/config` → SessionManager.get_effective_config()
- `PUT /api/config` → Validate + SessionManager.update_config()

**Metrics and Monitoring**:
- `GET /api/metrics/current` → MetricsManager.get_web_metrics()
- `GET /api/metrics/history` → From performance-optimized storage

### WebSocket Real-Time Updates (Unified Format)

**Message Structure** (Standardized across all updates):
```json
{
  "type": "training_update|game_state|system_status",
  "timestamp": "ISO-8601",
  "session_id": "uuid",
  "data": { /* type-specific payload */ }
}
```

**Update Frequency Control** (From Performance Plan):
- Training metrics: 500ms maximum frequency
- Game state: 250ms minimum interval  
- System status: 1s interval
- Differential updates to minimize bandwidth

## Security Architecture Integration

### Authentication Flow (Resolved Dependencies)
1. **JWT-based authentication** with 1-hour token expiration
2. **Role-based authorization** (Admin, Trainer, Researcher, Viewer)
3. **Resource-based access control** for training sessions
4. **Rate limiting** per user role and endpoint type

### Resource Protection Strategy
- **Process isolation** for training sessions
- **Resource quotas** based on user roles
- **API rate limiting** with role-specific limits
- **File system access control** with user-scoped directories

## Performance Architecture Integration

### Scalability Targets (Validated Across Plans)
- **20 concurrent web users**
- **50 WebSocket connections** 
- **5 concurrent training sessions**
- **100 API requests/second peak**

### Optimization Strategies
- **Multi-level caching**: Memory + Redis
- **Differential state updates**: Minimize data transfer
- **Connection pooling**: WebSocket resource management
- **Database optimization**: SQLite with indexing for metrics storage

## Data Flow Architecture

### Training State Synchronization
```
Training Step → MetricsManager → WebCallback → StatePublisher → WebSocket → Frontend
     ↓              ↓               ↓             ↓             ↓          ↓
   Updates      Aggregates      Formats       Broadcasts    Delivers   Updates UI
   Internal       Data          for Web        to All        to Each    Components
   Metrics                                    Connections    Client
```

### User Action Flow
```
Frontend → API Request → WebInterfaceManager → Manager Method → Training System
    ↓          ↓              ↓                      ↓               ↓
  User        HTTP         Validates              Executes        State
  Action      Call         + Authorizes           Action          Changes
```

## Configuration Integration

### Web Configuration Schema (Added to AppConfig)
```yaml
web:
  enabled: false                    # Enable web interface
  host: "localhost"                 # Server host
  port: 8080                       # Server port
  cors_origins: ["http://localhost:3000"]
  max_connections: 50              # WebSocket limit
  update_interval_ms: 500          # Broadcast frequency
  auth:
    jwt_secret: "secure-secret"    # JWT signing key
    token_expiration: 3600         # 1 hour tokens
    require_mfa: false             # MFA requirement
  resources:
    max_concurrent_trainings: 5    # System limit
    user_quotas_enabled: true      # Enable user quotas
```

## Implementation Phase Strategy

### Phase 1: Core Infrastructure (4-6 weeks)
**Sprint 1-2**: 
- WebInterfaceManager skeleton
- Basic API server with authentication
- Simple training status endpoints

**Sprint 3**: 
- StatePublisher with WebSocket support
- Basic frontend dashboard shell
- Training start/stop functionality

### Phase 2: Real-Time Features (3-4 weeks)
**Sprint 4**:
- Live metrics visualization
- Game state real-time updates
- WebSocket optimization

**Sprint 5**:
- Configuration management UI
- Model browser and management
- Performance monitoring

### Phase 3: Advanced Features (2-3 weeks)
**Sprint 6**:
- Multi-user support and authorization
- Resource management and quotas
- Evaluation result visualization

**Sprint 7**:
- Security hardening
- Performance optimization
- Production deployment preparation

## Risk Analysis and Mitigation

### High-Risk Items (Require Careful Implementation)

**Training Loop Performance Impact**:
- **Risk**: Web integration adds overhead to training
- **Mitigation**: Async updates, minimal manager modifications, performance testing

**Multi-User Resource Conflicts**:
- **Risk**: Users interfering with each other's training
- **Mitigation**: Process isolation, resource quotas, robust authorization

**WebSocket Connection Management**:
- **Risk**: Connection leaks and scaling issues  
- **Mitigation**: Connection pooling, health checks, graceful degradation

### Medium-Risk Items

**Configuration Hot-Reloading**:
- **Uncertainty**: Whether config changes require training restart
- **Approach**: Start with restart-required, add hot-reloading incrementally

**State Consistency**:
- **Risk**: Web API seeing inconsistent state during rapid updates
- **Mitigation**: Manager state snapshots, consistent read patterns

## Success Criteria

### Functional Requirements
- ✅ Web UI can monitor training in real-time
- ✅ Users can start/stop training sessions via web
- ✅ Configuration management through web interface
- ✅ Multi-user support with proper authorization
- ✅ Game visualization with move history

### Performance Requirements  
- ✅ < 100ms API response times for status endpoints
- ✅ < 5% training performance degradation
- ✅ Support 20 concurrent users
- ✅ Real-time updates within 500ms

### Security Requirements
- ✅ Secure authentication and authorization
- ✅ Resource isolation between users
- ✅ Protection of training data and models
- ✅ Audit logging for security events

## Resolved Cross-Plan Dependencies

**API ↔ Frontend**: WebSocket message formats standardized
**Security ↔ Integration**: Manager access patterns secured with RBAC
**Performance ↔ API**: Response time targets drive caching strategy
**Integration ↔ Performance**: Manager extension patterns minimize overhead

## Technology Stack (Finalized)

**Frontend**: React 18 + TypeScript + Redux Toolkit + Socket.IO
**Backend**: FastAPI + Python 3.9+ + Pydantic
**Database**: SQLite (metrics) + Redis (caching/sessions)  
**WebSocket**: FastAPI WebSocket + Socket.IO
**Authentication**: JWT + PBKDF2 password hashing
**Deployment**: Docker containers + docker-compose

This unified blueprint provides a comprehensive, implementable architecture that preserves Keisei's existing strengths while adding powerful web-based capabilities for training monitoring, management, and collaboration.