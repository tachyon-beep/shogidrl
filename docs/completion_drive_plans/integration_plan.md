# Integration Plan for Keisei Web UI with Existing Manager System

**Domain**: System Integration and Manager Modification Patterns
**Planning Agent**: Integration Specialist  
**Date**: 2025-08-23

## Manager Integration Architecture

### New WebInterfaceManager Component

**Purpose**: Central coordinator for all web-related functionality
**Location**: `keisei/web/web_interface_manager.py`

```python
class WebInterfaceManager:
    """Manages web interface integration with existing training system."""
    
    def __init__(self, config: AppConfig, trainer_reference: 'Trainer'):
        self.config = config
        self.trainer = trainer_reference
        self.state_publisher = StatePublisher()
        self.api_server = APIServer(config.web)
        self.web_callbacks = WebCallbackHandlers()
        
    def setup(self):
        """Initialize web interface components."""
        # Register web-specific callbacks with CallbackManager
        # Setup state publishing from existing managers
        # Initialize API server with manager references
        
    def get_training_status(self) -> Dict[str, Any]:
        """Aggregate training status from multiple managers."""
        return {
            'session': self.trainer.session_manager.get_session_info(),
            'metrics': self.trainer.metrics_manager.get_web_metrics(),
            'model': self.trainer.model_manager.get_model_status(),
            'evaluation': self.trainer.evaluation_manager.get_status()
        }
```

### Manager Modification Patterns

#### 1. MetricsManager Extensions
**New Methods to Add**:
```python
def get_web_metrics(self) -> Dict[str, Any]:
    """Return web-optimized metrics format."""
    return {
        'training': {
            'global_timestep': self.global_timestep,
            'episode_count': self.total_episodes_completed,
            'progress_percentage': (self.global_timestep / self.total_timesteps) * 100
        },
        'performance': {
            'reward_mean': self.get_recent_reward_mean(),
            'policy_loss': self.get_latest_policy_loss(),
            'value_loss': self.get_latest_value_loss(),
            'gradient_norm': self.get_latest_gradient_norm()
        },
        'system': {
            'training_speed': self.get_training_speed(),
            'memory_usage': self.get_memory_usage()
        }
    }

def subscribe_web_updates(self, callback: Callable[[Dict], None]):
    """Allow web system to subscribe to metric updates."""
    self._web_callbacks.append(callback)
    
def _notify_web_subscribers(self, metrics: Dict[str, Any]):
    """Notify web subscribers of metric changes.""" 
    for callback in self._web_callbacks:
        callback(metrics)
```

#### 2. SessionManager Web Extensions  
**New Methods to Add**:
```python
def get_session_info(self) -> Dict[str, Any]:
    """Return session information for web display."""
    return {
        'run_name': self.run_name,
        'start_time': self.session_start_time,
        'run_artifact_dir': self.run_artifact_dir,
        'wandb_active': self.is_wandb_active,
        'config_summary': self._get_config_summary()
    }

def get_available_sessions(self) -> List[Dict[str, Any]]:
    """Return list of available training sessions."""
    # Scan model directories for previous sessions
    # Return metadata about each session

def load_session_artifacts(self, session_id: str) -> Dict[str, Any]:
    """Load artifacts and metadata for a specific session."""
    # Load session logs, checkpoints, evaluation results
```

#### 3. CallbackManager Web Integration
**Web Callback Handler**:
```python
class WebCallbackHandlers:
    """Web-specific callback handlers for training events."""
    
    def __init__(self, state_publisher: StatePublisher):
        self.state_publisher = state_publisher
        
    def on_training_step(self, metrics: Dict[str, Any]):
        """Handle training step completion."""
        self.state_publisher.broadcast_training_update(metrics)
        
    def on_evaluation_start(self, eval_config: Dict[str, Any]):
        """Handle evaluation start."""
        self.state_publisher.broadcast_evaluation_status('started', eval_config)
        
    def on_evaluation_complete(self, results: Dict[str, Any]):
        """Handle evaluation completion."""
        self.state_publisher.broadcast_evaluation_results(results)
        
    def on_checkpoint_save(self, checkpoint_path: str):
        """Handle checkpoint save completion."""
        self.state_publisher.broadcast_checkpoint_saved(checkpoint_path)

# Integration with existing CallbackManager
def setup_web_callbacks(callback_manager: CallbackManager, web_handlers: WebCallbackHandlers):
    """Register web handlers with existing callback system."""
    callback_manager.add_callback('on_training_step', web_handlers.on_training_step)
    callback_manager.add_callback('on_evaluation_start', web_handlers.on_evaluation_start)
    callback_manager.add_callback('on_evaluation_complete', web_handlers.on_evaluation_complete)
    callback_manager.add_callback('on_checkpoint_save', web_handlers.on_checkpoint_save)
```

### State Publishing Architecture

**StatePublisher Component**:
```python
import asyncio
from typing import Dict, List, Callable
import websockets

class StatePublisher:
    """Publishes training state changes to connected web clients."""
    
    def __init__(self):
        self.connections: List[websockets.WebSocketServerProtocol] = []
        self.message_queue = asyncio.Queue()
        
    async def add_connection(self, websocket):
        """Add new WebSocket connection."""
        self.connections.append(websocket)
        # Send current state snapshot
        await self.send_initial_state(websocket)
        
    async def remove_connection(self, websocket):
        """Remove WebSocket connection.""" 
        if websocket in self.connections:
            self.connections.remove(websocket)
            
    def broadcast_training_update(self, metrics: Dict[str, Any]):
        """Broadcast training metrics update."""
        message = {
            'type': 'training_update',
            'timestamp': datetime.now().isoformat(),
            'data': metrics
        }
        asyncio.create_task(self._broadcast_message(message))
        
    def broadcast_game_state(self, game_state: Dict[str, Any]):
        """Broadcast current game state."""
        message = {
            'type': 'game_state', 
            'timestamp': datetime.now().isoformat(),
            'data': game_state
        }
        asyncio.create_task(self._broadcast_message(message))
```

### Training Loop Integration Points

#### Minimal Trainer Modifications
**Add WebInterfaceManager to Trainer**:
```python
class Trainer:
    def __init__(self, config: AppConfig, args: Any):
        # ... existing initialization ...
        
        # Add web interface manager if web UI is enabled
        if hasattr(config, 'web') and config.web.enabled:
            self.web_interface_manager = WebInterfaceManager(config, self)
            self.web_interface_manager.setup()
            
        # ... rest of initialization ...
        
    def _initialize_components(self):
        """Initialize all training components using SetupManager."""
        # ... existing component initialization ...
        
        # Setup web integration after all other components
        if hasattr(self, 'web_interface_manager'):
            self.web_interface_manager.integrate_with_managers(
                session_manager=self.session_manager,
                metrics_manager=self.metrics_manager,
                model_manager=self.model_manager,
                callback_manager=self.callback_manager,
                evaluation_manager=self.evaluation_manager
            )
```

#### Training Loop Hooks
**Minimal changes to TrainingLoopManager**:
```python
def run(self):
    """Main training loop with web integration."""
    while self.should_continue_training():
        # ... existing training step logic ...
        
        # Web integration point - publish state after each step
        if hasattr(self.trainer, 'web_interface_manager'):
            self.trainer.web_interface_manager.publish_training_state()
            
        # ... rest of training loop ...
```

### Configuration Integration

**Web Configuration Schema Addition**:
```python
class WebConfig(BaseModel):
    """Web interface configuration."""
    enabled: bool = Field(False, description="Enable web interface")
    host: str = Field("localhost", description="Web server host")
    port: int = Field(8080, description="Web server port")
    api_prefix: str = Field("/api", description="API endpoint prefix")
    websocket_path: str = Field("/ws", description="WebSocket endpoint path")
    cors_origins: List[str] = Field(["http://localhost:3000"], description="Allowed CORS origins")
    max_connections: int = Field(10, description="Maximum concurrent WebSocket connections")
    update_interval_ms: int = Field(500, description="State update broadcast interval")
    
# Add to main AppConfig
class AppConfig(BaseModel):
    # ... existing config sections ...
    web: WebConfig = WebConfig()
```

### Data Flow Architecture

**Training State → Web Client Flow**:
1. **Training Step Completion** → MetricsManager updates internal state
2. **MetricsManager** → Calls web subscriber callbacks
3. **WebCallbackHandler** → Formats data for web consumption  
4. **StatePublisher** → Broadcasts via WebSocket to connected clients
5. **Frontend** → Receives update and updates UI components

**User Action → Training System Flow**:  
1. **Frontend** → Sends API request (e.g., start training)
2. **APIServer** → Validates request and calls WebInterfaceManager
3. **WebInterfaceManager** → Calls appropriate manager method
4. **Manager** → Executes action and triggers callbacks
5. **WebCallbackHandler** → Publishes state change
6. **Frontend** → Receives confirmation via WebSocket

## PLAN_UNCERTAINTY Tags

**PLAN_UNCERTAINTY**: Thread safety for web callbacks - existing managers not designed for multi-threaded access from web requests

**PLAN_UNCERTAINTY**: Training loop performance impact - need to measure overhead of web state publishing

**PLAN_UNCERTAINTY**: Manager state consistency - ensuring web API sees consistent state across managers during rapid updates

**PLAN_UNCERTAINTY**: Configuration hot-reloading - whether configuration changes require training restart or can be applied dynamically

**PLAN_UNCERTAINTY**: Memory management for web connections - handling connection cleanup and preventing memory leaks

## Dependencies on Other Plans

- **API Plan**: Endpoint specifications that drive integration points
- **Security Plan**: Authentication requirements for manager access
- **Performance Plan**: Update frequency limits and resource constraints  
- **Frontend Plan**: Data format requirements and real-time update patterns