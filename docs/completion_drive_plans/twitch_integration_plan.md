# Twitch Integration Technical Plan - Integration Specialist

## Domain Expertise: Real-time Data Integration & API Orchestration

**Planning Phase**: Integration Specialist Domain Plan
**Target**: Complete technical integration specifications for Keisei-to-Twitch data pipeline
**Integration Points**: Keisei training managers, WebSocket infrastructure, Twitch APIs, Frontend systems

## Executive Summary

The integration layer must seamlessly connect Keisei's training system with real-time streaming infrastructure while maintaining training performance and providing reliable, educational data streams. This requires careful API design, robust error handling, and performance monitoring.

## Core Integration Architecture

### Data Flow Pipeline Design

**Primary Integration Pattern**: Async Publisher-Subscriber with Circuit Breakers

```
[Training System] → [Event Bus] → [Stream Aggregator] → [WebSocket Gateway] → [Client Apps]
        ↓               ↓              ↓                    ↓                    ↓
   [Instrumentation] [Buffering]   [Processing]        [Broadcasting]     [Visualization]
```

**Key Design Principles**:
- **Non-invasive**: Zero impact on training performance
- **Resilient**: Graceful degradation when streaming fails
- **Scalable**: Support multiple concurrent viewers
- **Real-time**: <100ms latency for critical updates

**PLAN_UNCERTAINTY**: Need to validate performance impact of instrumentation hooks in production training scenarios. Mock data may not reveal actual overhead.

## Training System Integration

### Manager Instrumentation Strategy

**Minimal Modification Approach**:
```python
# Enhanced MetricsManager with streaming hooks
class MetricsManager:
    def __init__(self, ..., stream_publisher=None):
        # Existing initialization
        self.stream_publisher = stream_publisher
        self.streaming_enabled = stream_publisher is not None
    
    def format_ppo_metrics(self, learn_metrics: Dict[str, float]) -> str:
        formatted = super().format_ppo_metrics(learn_metrics)
        
        # Non-blocking streaming hook
        if self.streaming_enabled:
            asyncio.create_task(
                self.stream_publisher.publish_ppo_metrics(learn_metrics)
            )
        
        return formatted
```

**Performance Safeguards**:
- Async task creation to prevent blocking
- Circuit breakers to disable streaming on failure
- Memory bounded queues to prevent overflow
- Performance monitoring with automatic fallback

### Data Extraction Points

**Critical Training Data Sources**:

1. **PPO Metrics** (from PPOAgent.learn()):
   ```python
   # Integration point in PPOAgent
   def learn(self, experience_buffer) -> Dict[str, float]:
       metrics = self._compute_ppo_metrics(experience_buffer)
       
       # Streaming hook (async, non-blocking)
       if hasattr(self, 'stream_hook') and self.stream_hook:
           self.stream_hook.emit_ppo_metrics(metrics, self.global_step)
       
       return metrics
   ```

2. **SE Block Attention Weights** (from ResNet model):
   ```python
   # Modified SqueezeExcitation for attention extraction
   class SqueezeExcitation(nn.Module):
       def forward(self, x):
           s = F.adaptive_avg_pool2d(x, 1)
           s = F.relu(self.fc1(s))
           attention_weights = torch.sigmoid(self.fc2(s))
           
           # Store for streaming (no performance impact)
           if hasattr(self, '_stream_hook'):
               self._stream_hook.cache_attention(attention_weights.detach())
           
           return x * attention_weights
   ```

3. **Move Predictions** (from StepManager):
   ```python
   # Integration in StepManager.step()
   def step(self, action) -> EpisodeState:
       # Get predictions before action execution
       if self.streaming_enabled:
           top_moves = self.agent.get_top_predictions(n=5)
           self.stream_hook.emit_predictions(top_moves, self.game.position)
       
       # Continue with normal step execution
       return self._execute_step(action)
   ```

**PLAN_UNCERTAINTY**: SE Block attention extraction timing unclear. Need to validate whether we extract during forward pass or store activations for later retrieval.

## Real-time Event System

### Message Bus Architecture

**High-Performance Event Bus**:
```python
import asyncio
from typing import Any, Callable, Dict, List
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    PPO_METRICS = "ppo_metrics"
    SE_ATTENTION = "se_attention"
    MOVE_PREDICTION = "move_prediction"
    GAME_STATE = "game_state"
    MILESTONE = "milestone"

@dataclass
class StreamEvent:
    type: EventType
    timestamp: float
    data: Dict[str, Any]
    priority: int = 1  # 1=high, 2=medium, 3=low

class StreamEventBus:
    def __init__(self, max_queue_size=10000):
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_queue = asyncio.Queue(maxsize=max_queue_size)
        self.circuit_breakers: Dict[str, bool] = {}
        
    async def publish(self, event: StreamEvent):
        try:
            await self.event_queue.put(event)
        except asyncio.QueueFull:
            # Prioritized dropping - remove low priority events
            await self._handle_queue_overflow(event)
    
    async def _handle_queue_overflow(self, new_event: StreamEvent):
        """Drop low-priority events to make room for high-priority ones"""
        if new_event.priority == 1:  # High priority
            # Drop oldest low-priority event
            temp_events = []
            while not self.event_queue.empty():
                event = await self.event_queue.get()
                if event.priority > 1:
                    break  # Drop this event
                temp_events.append(event)
            
            # Put back remaining events and add new one
            for event in temp_events:
                await self.event_queue.put(event)
            await self.event_queue.put(new_event)
```

### WebSocket Gateway Implementation

**Multi-channel WebSocket Server**:
```python
import websockets
import json
from typing import Dict, Set

class StreamingWebSocketServer:
    def __init__(self, event_bus: StreamEventBus):
        self.event_bus = event_bus
        self.clients: Dict[str, Set[websockets.WebSocketServerProtocol]] = {
            'all': set(),
            'metrics': set(),
            'predictions': set(),
            'attention': set(),
            'chat': set()
        }
        
    async def register_client(self, websocket, channels: List[str]):
        """Register client for specific data channels"""
        for channel in channels:
            if channel in self.clients:
                self.clients[channel].add(websocket)
        self.clients['all'].add(websocket)
        
        await websocket.send(json.dumps({
            'type': 'connection_success',
            'channels': channels,
            'server_time': time.time()
        }))
    
    async def broadcast_event(self, event: StreamEvent):
        """Broadcast event to relevant subscribers"""
        channel_mapping = {
            EventType.PPO_METRICS: 'metrics',
            EventType.SE_ATTENTION: 'attention', 
            EventType.MOVE_PREDICTION: 'predictions',
            EventType.GAME_STATE: 'all',
            EventType.MILESTONE: 'all'
        }
        
        target_channel = channel_mapping.get(event.type, 'all')
        message = json.dumps({
            'type': event.type.value,
            'timestamp': event.timestamp,
            'data': event.data
        })
        
        # Broadcast to channel subscribers
        disconnected = set()
        for client in self.clients[target_channel].copy():
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Clean up disconnected clients
        for client in disconnected:
            await self._cleanup_client(client)
```

## Twitch Platform Integration

### Chat API Integration

**Twitch IRC Integration with Commands**:
```python
import asyncio
import ssl
from typing import Dict, Callable

class TwitchChatIntegration:
    def __init__(self, channel: str, oauth_token: str, client_id: str):
        self.channel = channel.lower()
        self.oauth_token = oauth_token
        self.client_id = client_id
        self.commands: Dict[str, Callable] = {}
        self.reader = None
        self.writer = None
        
    async def connect(self):
        """Connect to Twitch IRC"""
        context = ssl.create_default_context()
        self.reader, self.writer = await asyncio.open_connection(
            'irc.chat.twitch.tv', 6697, ssl=context
        )
        
        # Authenticate
        self.writer.write(f"PASS {self.oauth_token}\r\n".encode())
        self.writer.write(f"NICK {self.client_id}\r\n".encode())
        self.writer.write(f"JOIN #{self.channel}\r\n".encode())
        await self.writer.drain()
        
    def register_command(self, command: str, handler: Callable):
        """Register chat command handler"""
        self.commands[command.lower()] = handler
        
    async def listen(self):
        """Listen for chat messages and process commands"""
        while True:
            try:
                line = await self.reader.readline()
                message = line.decode('utf-8').strip()
                
                if message.startswith('PING'):
                    await self._send_pong(message)
                elif 'PRIVMSG' in message:
                    await self._process_message(message)
                    
            except Exception as e:
                logging.error(f"Chat error: {e}")
                await self._reconnect()
    
    async def _process_message(self, message: str):
        """Process chat message for commands"""
        # Parse Twitch IRC format
        # :username!username@username.tmi.twitch.tv PRIVMSG #channel :message
        parts = message.split(':', 2)
        if len(parts) < 3:
            return
            
        username = parts[1].split('!')[0]
        content = parts[2].strip()
        
        if content.startswith('!'):
            command_parts = content[1:].split(' ', 1)
            command = command_parts[0].lower()
            args = command_parts[1] if len(command_parts) > 1 else ""
            
            if command in self.commands:
                await self.commands[command](username, args)
```

**Interactive Command Handlers**:
```python
class InteractiveCommandHandlers:
    def __init__(self, stream_server: StreamingWebSocketServer):
        self.stream_server = stream_server
        self.prediction_sessions: Dict[str, PredictionSession] = {}
        
    async def handle_predict(self, username: str, move_notation: str):
        """Handle !predict command"""
        # Validate move notation
        try:
            move = parse_shogi_notation(move_notation)
            
            # Store prediction
            session_id = self._get_active_prediction_session()
            if session_id:
                await self._record_prediction(session_id, username, move)
                
                # Broadcast update
                await self.stream_server.broadcast_event(StreamEvent(
                    type=EventType.MOVE_PREDICTION,
                    timestamp=time.time(),
                    data={
                        'type': 'viewer_prediction',
                        'username': username,
                        'move': move_notation,
                        'session_id': session_id
                    }
                ))
                
        except InvalidMoveError:
            # Send error response (if bot has mod permissions)
            pass
    
    async def handle_explain(self, username: str, args: str):
        """Handle !explain command for educational content"""
        # Trigger educational overlay
        await self.stream_server.broadcast_event(StreamEvent(
            type=EventType.GAME_STATE,
            timestamp=time.time(),
            data={
                'type': 'explain_request',
                'requester': username,
                'context': args or 'current_position'
            }
        ))
```

### Streaming Overlay Integration

**OBS WebSocket Integration**:
```python
import websockets
import json

class OBSIntegration:
    def __init__(self, obs_websocket_url="ws://localhost:4455", password=None):
        self.url = obs_websocket_url
        self.password = password
        self.websocket = None
        
    async def connect(self):
        self.websocket = await websockets.connect(self.url)
        
        # Authenticate if password provided
        if self.password:
            await self._authenticate()
    
    async def trigger_milestone_transition(self, milestone_type: str):
        """Trigger scene transition for milestone celebration"""
        await self._send_request("TriggerHotkeyByName", {
            "hotkeyName": f"milestone_{milestone_type}"
        })
    
    async def set_overlay_text(self, source_name: str, text: str):
        """Update text overlay for current AI insights"""
        await self._send_request("SetInputSettings", {
            "inputName": source_name,
            "inputSettings": {
                "text": text
            }
        })
    
    async def _send_request(self, request_type: str, request_data: dict):
        message = {
            "op": 6,  # Request
            "d": {
                "requestType": request_type,
                "requestId": f"req_{time.time()}",
                "requestData": request_data
            }
        }
        await self.websocket.send(json.dumps(message))
```

## Performance Monitoring and Optimization

### System Health Monitoring

**Performance Metrics Collection**:
```python
@dataclass
class PerformanceMetrics:
    training_fps: float
    streaming_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    messages_per_second: float
    error_rate: float

class PerformanceMonitor:
    def __init__(self, alert_thresholds: Dict[str, float]):
        self.thresholds = alert_thresholds
        self.metrics_history = deque(maxlen=1000)
        
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        return PerformanceMetrics(
            training_fps=await self._measure_training_fps(),
            streaming_latency_ms=await self._measure_streaming_latency(),
            memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
            cpu_usage_percent=psutil.cpu_percent(),
            active_connections=len(self.websocket_server.clients['all']),
            messages_per_second=self._calculate_message_rate(),
            error_rate=self._calculate_error_rate()
        )
    
    async def check_health(self) -> Dict[str, str]:
        """Health check with actionable recommendations"""
        metrics = await self.collect_metrics()
        health_status = {}
        
        if metrics.training_fps < self.thresholds['min_training_fps']:
            health_status['training'] = 'degraded'
            
        if metrics.streaming_latency_ms > self.thresholds['max_latency_ms']:
            health_status['streaming'] = 'slow'
            
        if metrics.memory_usage_mb > self.thresholds['max_memory_mb']:
            health_status['memory'] = 'high'
            
        return health_status or {'overall': 'healthy'}
```

### Circuit Breaker Implementation

**Failure Recovery System**:
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        
    async def call(self, func, *args, **kwargs):
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'half-open'
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - reset failure count
            if self.state == 'half-open':
                self.state = 'closed'
            self.failure_count = 0
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
```

## Error Handling and Resilience

### Graceful Degradation Strategy

**Fallback Modes**:
1. **Training Protected**: Streaming fails, training continues normally
2. **Reduced Quality**: Lower resolution/frequency data when bandwidth limited
3. **Offline Mode**: Local recording for later replay when connection fails
4. **Emergency Stop**: Complete streaming shutdown if training performance degrades

**Error Recovery Patterns**:
```python
class StreamingService:
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.fallback_modes = ['full', 'reduced', 'minimal', 'disabled']
        self.current_mode = 'full'
        
    async def stream_data(self, data: StreamEvent):
        try:
            await self.circuit_breaker.call(self._send_data, data)
        except CircuitBreakerOpenError:
            # Degrade service quality
            await self._degrade_service()
        except NetworkError:
            # Temporary storage for retry
            await self._store_for_retry(data)
        except Exception as e:
            # Log and continue
            logging.error(f"Unexpected streaming error: {e}")
    
    async def _degrade_service(self):
        current_index = self.fallback_modes.index(self.current_mode)
        if current_index < len(self.fallback_modes) - 1:
            self.current_mode = self.fallback_modes[current_index + 1]
            logging.warning(f"Degraded to mode: {self.current_mode}")
```

**PLAN_UNCERTAINTY**: Network partition scenarios need testing. WebSocket reconnection logic may need fine-tuning for various failure modes.

## Testing and Validation Strategy

### Integration Testing Framework

**End-to-End Testing**:
```python
import pytest
import asyncio
from unittest.mock import Mock, patch

class TestIntegrationPipeline:
    async def test_ppo_metrics_flow(self):
        # Setup mock training system
        mock_agent = Mock()
        event_bus = StreamEventBus()
        websocket_server = StreamingWebSocketServer(event_bus)
        
        # Simulate PPO learning event
        test_metrics = {
            'ppo/kl_divergence': 0.02,
            'ppo/clip_fraction': 0.1,
            'ppo/policy_loss': -0.005
        }
        
        # Inject streaming hook
        stream_hook = StreamingHook(event_bus)
        mock_agent.stream_hook = stream_hook
        
        # Trigger metrics emission
        await stream_hook.emit_ppo_metrics(test_metrics, global_step=1000)
        
        # Verify event propagation
        event = await event_bus.event_queue.get()
        assert event.type == EventType.PPO_METRICS
        assert event.data['metrics'] == test_metrics
    
    async def test_performance_under_load(self):
        # Stress test with high-frequency events
        event_bus = StreamEventBus()
        
        # Generate 1000 events rapidly
        tasks = []
        for i in range(1000):
            event = StreamEvent(
                type=EventType.PPO_METRICS,
                timestamp=time.time(),
                data={'test': i}
            )
            tasks.append(event_bus.publish(event))
        
        # Measure processing time
        start_time = time.time()
        await asyncio.gather(*tasks)
        processing_time = time.time() - start_time
        
        assert processing_time < 1.0  # Should handle 1000 events in <1s
```

**Performance Regression Testing**:
- Automated benchmarking after each change
- Training performance impact measurement
- Memory leak detection in long-running tests
- Network condition simulation (latency, packet loss)

## Deployment Architecture

### Production Infrastructure

**Containerized Services**:
```yaml
# docker-compose.yml
version: '3.8'
services:
  keisei-training:
    build: ./keisei
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - STREAMING_ENABLED=true
      - STREAM_ENDPOINT=ws://stream-gateway:8080
  
  stream-gateway:
    build: ./streaming
    ports:
      - "8080:8080"  # WebSocket
      - "8081:8081"  # HTTP API
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - WEBSOCKET_URL=ws://stream-gateway:8080
```

**Scalability Considerations**:
- Horizontal scaling of WebSocket gateways
- Redis for session state sharing
- Load balancing for multiple concurrent streams
- CDN integration for static assets

This integration plan provides a robust, scalable foundation for connecting Keisei's AI training system with Twitch streaming infrastructure while maintaining the performance and reliability required for both educational value and entertainment.