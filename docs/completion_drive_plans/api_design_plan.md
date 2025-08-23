# API Design Plan for Keisei Web UI Integration

**Domain**: API Architecture and Endpoint Design
**Planning Agent**: API Design Specialist
**Date**: 2025-08-23

## REST API Endpoints Design

### Core Training Control Endpoints

**Training Management**:
- `POST /api/training/start` - Start new training session
- `POST /api/training/stop` - Stop current training
- `POST /api/training/pause` - Pause training (if supported)
- `GET /api/training/status` - Get current training status
- `POST /api/training/resume` - Resume from checkpoint

**Configuration Management**:
- `GET /api/config` - Get current configuration
- `PUT /api/config` - Update configuration  
- `POST /api/config/validate` - Validate configuration
- `GET /api/config/templates` - Get configuration templates
- `POST /api/config/save` - Save configuration to file

**Model Management**:
- `GET /api/models` - List available models
- `GET /api/models/{model_id}` - Get model details
- `POST /api/models/{model_id}/load` - Load specific model
- `DELETE /api/models/{model_id}` - Delete model
- `POST /api/models/export` - Export model

### Evaluation Endpoints

**Evaluation Control**:
- `POST /api/evaluation/start` - Start evaluation
- `GET /api/evaluation/status` - Get evaluation status  
- `GET /api/evaluation/history` - Get evaluation history
- `GET /api/evaluation/results/{eval_id}` - Get specific evaluation results

**Game Replay**:
- `GET /api/games` - List saved games
- `GET /api/games/{game_id}` - Get game details
- `GET /api/games/{game_id}/moves` - Get game move sequence
- `POST /api/games/{game_id}/replay` - Start game replay

### Monitoring and Metrics Endpoints

**Real-time Metrics**:
- `GET /api/metrics/current` - Current training metrics
- `GET /api/metrics/history` - Historical metrics
- `GET /api/metrics/performance` - System performance metrics

**Session Management**:
- `GET /api/sessions` - List training sessions
- `GET /api/sessions/{session_id}` - Get session details
- `POST /api/sessions/{session_id}/artifacts` - Get session artifacts

## WebSocket Real-Time Communication

### Connection Management
- **Connection URL**: `ws://host:port/ws/training/{session_id}`
- **Authentication**: JWT token in connection headers
- **Heartbeat**: 30-second ping/pong for connection health

### Message Types

**Training Updates**:
```json
{
  "type": "training_update",
  "timestamp": "2025-08-23T10:30:00Z",
  "data": {
    "global_timestep": 125000,
    "episode_count": 450,
    "reward_mean": 0.75,
    "policy_loss": 0.023,
    "value_loss": 0.089,
    "gradient_norm": 0.45
  }
}
```

**Game State Updates**:
```json
{
  "type": "game_state",
  "timestamp": "2025-08-23T10:30:00Z", 
  "data": {
    "board_state": "...",
    "current_player": "black",
    "move_count": 45,
    "game_status": "ongoing"
  }
}
```

**System Status**:
```json
{
  "type": "system_status",
  "timestamp": "2025-08-23T10:30:00Z",
  "data": {
    "cpu_usage": 65.2,
    "memory_usage": 78.1,
    "gpu_usage": 89.5,
    "training_speed": 847.3
  }
}
```

## Data Contracts and Schemas

### Training Status Response
```json
{
  "status": "running|paused|stopped",
  "session_id": "uuid",
  "start_time": "2025-08-23T10:00:00Z",
  "current_timestep": 125000,
  "total_timesteps": 500000,
  "estimated_completion": "2025-08-23T14:30:00Z",
  "current_metrics": {
    "episode_reward_mean": 0.75,
    "policy_loss": 0.023,
    "value_loss": 0.089
  }
}
```

### Configuration Schema
```json
{
  "env": {
    "device": "cpu|cuda",
    "seed": 42,
    "input_channels": 46,
    "num_actions_total": 13527
  },
  "training": {
    "learning_rate": 0.0003,
    "total_timesteps": 500000,
    "steps_per_epoch": 2048
  }
}
```

## Integration with Existing Managers

### State Access Patterns
- **MetricsManager**: Direct access to formatted metrics via new `get_web_metrics()` method
- **SessionManager**: Expose session metadata via `get_session_info()` method  
- **ModelManager**: Add `get_model_metadata()` for web display
- **EvaluationManager**: Extend with `get_evaluation_status()` for web monitoring

### Event System Integration
- **CallbackManager**: Add web callback handlers for real-time updates
- **DisplayManager**: Create `WebDisplayAdapter` to bridge Rich UI to web data

## PLAN_UNCERTAINTY Tags

**PLAN_UNCERTAINTY**: Authentication integration with existing WandB auth - need to determine if we leverage WandB OAuth or implement separate auth system

**PLAN_UNCERTAINTY**: Real-time game visualization data format - board state serialization needs optimization for frequent updates

**PLAN_UNCERTAINTY**: Multi-user resource contention - how to handle multiple users trying to start training simultaneously

**PLAN_UNCERTAINTY**: WebSocket scaling - connection management for multiple concurrent users and sessions

**PLAN_UNCERTAINTY**: API rate limiting strategy - need to balance responsiveness with system protection

## Dependencies on Other Plans

- **Frontend Plan**: WebSocket message format compatibility
- **Security Plan**: Authentication and authorization requirements  
- **Integration Plan**: Manager modification patterns
- **Performance Plan**: API response time requirements and caching strategy