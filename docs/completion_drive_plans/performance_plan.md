# Performance and Scalability Plan for Keisei Web UI

**Domain**: Performance Engineering, Scalability, and Resource Management
**Planning Agent**: Performance Engineer  
**Date**: 2025-08-23

## Performance Requirements and Targets

### Response Time Targets

**API Endpoints**:
- **Training Status**: < 100ms (high-frequency polling)
- **Configuration Read**: < 200ms (complex data aggregation)
- **Training Control**: < 500ms (start/stop operations) 
- **Model Management**: < 1s (file system operations)
- **Historical Data**: < 2s (database queries with pagination)

**WebSocket Updates**:
- **Training Metrics**: 500ms interval maximum (2 updates/second)
- **Game State**: 250ms minimum interval (4 updates/second max)
- **System Status**: 1s interval (CPU/memory monitoring)
- **Connection Latency**: < 50ms (real-time responsiveness)

**Frontend Rendering**:
- **Dashboard Load**: < 2s initial load
- **Chart Updates**: < 16ms (60fps smooth rendering)
- **Navigation**: < 200ms between sections
- **Configuration Form**: < 100ms input response

### Concurrent User Targets
- **Maximum Concurrent Users**: 20 simultaneous web sessions
- **WebSocket Connections**: 50 concurrent connections
- **Training Sessions**: 5 concurrent training processes
- **API Requests**: 100 requests/second peak load

## Resource Management Architecture

### Training Process Resource Isolation

**Process Separation Strategy**:
```python
class TrainingResourceManager:
    """Manages resource allocation for training processes."""
    
    def __init__(self):
        self.active_trainings = {}  # session_id -> ResourceAllocation
        self.resource_pools = {
            'cpu': ResourcePool('cpu', total_cores=16),
            'memory': ResourcePool('memory', total_gb=64), 
            'gpu': ResourcePool('gpu', total_devices=2)
        }
        
    def allocate_training_resources(self, user_id: str, config: TrainingConfig) -> ResourceAllocation:
        """Allocate resources for new training session."""
        required_resources = self.estimate_resource_needs(config)
        
        # Check resource availability
        if not self.can_allocate(required_resources):
            raise ResourceExhaustionError("Insufficient resources available")
            
        # Create resource allocation
        allocation = ResourceAllocation(
            cpu_cores=required_resources['cpu'],
            memory_gb=required_resources['memory'], 
            gpu_devices=required_resources['gpu'],
            priority=self.get_user_priority(user_id)
        )
        
        # Reserve resources
        for resource_type, amount in required_resources.items():
            self.resource_pools[resource_type].reserve(amount, allocation.id)
            
        return allocation
        
    def estimate_resource_needs(self, config: TrainingConfig) -> Dict[str, int]:
        """Estimate resource requirements based on configuration."""
        base_memory = 8  # GB base memory requirement
        
        # Scale memory with model size
        model_memory = (config.tower_width * config.tower_depth) / 1000  # Rough estimate
        
        # Scale with batch size
        batch_memory = config.minibatch_size * 0.1  # GB per batch item
        
        return {
            'cpu': 4,  # 4 cores per training
            'memory': int(base_memory + model_memory + batch_memory),
            'gpu': 1 if config.device == 'cuda' else 0
        }
```

### WebSocket Connection Management

**Connection Pooling and Scaling**:
```python
import asyncio
from typing import Dict, Set
import weakref

class WebSocketConnectionManager:
    """Manages WebSocket connections with performance optimizations."""
    
    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.connections: Dict[str, Set[websockets.WebSocketServerProtocol]] = {}
        self.connection_stats = ConnectionStats()
        self.message_buffer = MessageBuffer(max_size=1000)
        
    async def add_connection(self, session_id: str, websocket):
        """Add WebSocket connection with resource limits."""
        if self.get_total_connections() >= self.max_connections:
            await websocket.close(code=1008, reason="Connection limit exceeded")
            return False
            
        if session_id not in self.connections:
            self.connections[session_id] = set()
            
        self.connections[session_id].add(websocket)
        self.connection_stats.record_connection(session_id)
        
        # Send buffered messages for this session
        await self.send_buffered_messages(session_id, websocket)
        return True
        
    async def broadcast_to_session(self, session_id: str, message: Dict, 
                                  buffer_if_no_connections: bool = True):
        """Broadcast message to all connections for a session."""
        if session_id not in self.connections:
            if buffer_if_no_connections:
                self.message_buffer.add_message(session_id, message)
            return
            
        # Remove dead connections while broadcasting
        active_connections = set()
        
        for websocket in self.connections[session_id].copy():
            try:
                await websocket.send(json.dumps(message))
                active_connections.add(websocket)
            except websockets.exceptions.ConnectionClosed:
                # Connection is dead, don't add to active set
                pass
                
        self.connections[session_id] = active_connections
        
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics for monitoring."""
        return {
            'total_connections': self.get_total_connections(),
            'connections_by_session': {
                session: len(conns) for session, conns in self.connections.items()
            },
            'max_connections': self.max_connections,
            'connection_utilization': self.get_total_connections() / self.max_connections
        }
```

### Data Update Optimization

**Differential Updates and Caching**:
```python
class StateUpdateOptimizer:
    """Optimizes state updates to minimize data transfer."""
    
    def __init__(self):
        self.last_state = {}  # session_id -> last_sent_state
        self.update_throttle = {}  # session_id -> last_update_time
        
    def should_send_update(self, session_id: str, current_state: Dict) -> bool:
        """Determine if state update should be sent."""
        now = time.time()
        
        # Throttle updates to maximum frequency
        last_update = self.update_throttle.get(session_id, 0)
        if now - last_update < 0.5:  # 500ms minimum interval
            return False
            
        # Check if state has meaningfully changed
        if session_id in self.last_state:
            if not self.has_significant_change(self.last_state[session_id], current_state):
                return False
                
        return True
        
    def create_differential_update(self, session_id: str, current_state: Dict) -> Dict:
        """Create differential update containing only changes."""
        if session_id not in self.last_state:
            # First update - send full state
            self.last_state[session_id] = current_state.copy()
            return current_state
            
        diff_update = {}
        last_state = self.last_state[session_id]
        
        # Compare and build differential
        for key, value in current_state.items():
            if key not in last_state or last_state[key] != value:
                diff_update[key] = value
                
        # Update cached state
        last_state.update(current_state)
        return diff_update
        
    def has_significant_change(self, old_state: Dict, new_state: Dict) -> bool:
        """Check if changes are significant enough to warrant update."""
        # Define significance thresholds
        thresholds = {
            'global_timestep': 10,      # Update every 10 steps
            'episode_count': 1,         # Update every episode
            'reward_mean': 0.01,        # 1% change in reward
            'policy_loss': 0.001,       # 0.1% change in loss
            'training_speed': 10.0      # 10 steps/sec change
        }
        
        for key, threshold in thresholds.items():
            if key in old_state and key in new_state:
                if isinstance(new_state[key], (int, float)):
                    change = abs(new_state[key] - old_state[key])
                    if change >= threshold:
                        return True
                        
        return False
```

### Database and Storage Optimization

**Training History Storage**:
```python
import sqlite3
from typing import List, Optional
import json

class TrainingHistoryStore:
    """Optimized storage for training metrics and history."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
        self.write_batch = []
        self.batch_size = 100
        
    def init_database(self):
        """Initialize database with optimized schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    session_id TEXT NOT NULL,
                    timestep INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, timestep)
                )
            ''')
            
            # Create index for efficient queries
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_timestep 
                ON training_metrics(session_id, timestep DESC)
            ''')
            
    def add_metric(self, session_id: str, timestep: int, metrics: Dict):
        """Add metric to batch for efficient writing."""
        self.write_batch.append({
            'session_id': session_id,
            'timestep': timestep,
            'timestamp': time.time(),
            'metrics_json': json.dumps(metrics)
        })
        
        if len(self.write_batch) >= self.batch_size:
            self.flush_batch()
            
    def flush_batch(self):
        """Write batched metrics to database."""
        if not self.write_batch:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany('''
                INSERT OR REPLACE INTO training_metrics 
                (session_id, timestep, timestamp, metrics_json)
                VALUES (:session_id, :timestep, :timestamp, :metrics_json)
            ''', self.write_batch)
            
        self.write_batch.clear()
        
    def get_metrics_range(self, session_id: str, start_timestep: int, 
                         end_timestep: int, max_points: int = 1000) -> List[Dict]:
        """Get metrics with automatic downsampling for performance."""
        with sqlite3.connect(self.db_path) as conn:
            # Calculate sampling interval
            total_points = end_timestep - start_timestep
            sample_interval = max(1, total_points // max_points)
            
            cursor = conn.execute('''
                SELECT timestep, metrics_json FROM training_metrics
                WHERE session_id = ? AND timestep >= ? AND timestep <= ?
                AND timestep % ? = 0
                ORDER BY timestep
            ''', (session_id, start_timestep, end_timestep, sample_interval))
            
            results = []
            for row in cursor:
                metrics = json.loads(row[1])
                metrics['timestep'] = row[0]
                results.append(metrics)
                
            return results
```

### Caching Strategy

**Multi-Level Caching Architecture**:
```python
from functools import lru_cache
from typing import Any
import redis
import pickle

class CacheManager:
    """Multi-level caching for API responses and computed data."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.memory_cache = {}  # L1 cache - in-memory
        self.redis_client = redis.from_url(redis_url) if redis_url else None  # L2 cache
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first, then L2)."""
        # Check L1 cache
        if key in self.memory_cache:
            return self.memory_cache[key]
            
        # Check L2 cache (Redis)
        if self.redis_client:
            cached_value = self.redis_client.get(key)
            if cached_value:
                value = pickle.loads(cached_value)
                # Populate L1 cache
                self.memory_cache[key] = value
                return value
                
        return None
        
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in both cache levels."""
        # Set in L1 cache
        self.memory_cache[key] = value
        
        # Set in L2 cache with TTL
        if self.redis_client:
            self.redis_client.setex(key, ttl, pickle.dumps(value))
            
    @lru_cache(maxsize=128)
    def get_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """Cached model metadata retrieval."""
        # Expensive operation - load and analyze model file
        return self._load_model_metadata(model_path)
```

## Monitoring and Performance Metrics

### Performance Monitoring Dashboard

**Key Performance Indicators**:
```python
class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        
    def record_api_response_time(self, endpoint: str, duration_ms: float):
        """Record API endpoint response time."""
        self.metrics[f'api_response_time_{endpoint}'].append(duration_ms)
        
    def record_websocket_latency(self, session_id: str, latency_ms: float):
        """Record WebSocket message latency."""
        self.metrics[f'websocket_latency_{session_id}'].append(latency_ms)
        
    def record_training_performance(self, session_id: str, steps_per_second: float):
        """Record training performance impact."""
        self.metrics[f'training_performance_{session_id}'].append(steps_per_second)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary report."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'p95': self._percentile(values, 0.95),
                    'p99': self._percentile(values, 0.99)
                }
                
        return summary
```

## PLAN_UNCERTAINTY Tags

**PLAN_UNCERTAINTY**: WebSocket connection scaling strategy - whether to use Redis pub/sub for horizontal scaling across multiple server instances

**PLAN_UNCERTAINTY**: Database choice for metrics storage - SQLite vs PostgreSQL vs InfluxDB for time-series data optimization

**PLAN_UNCERTAINTY**: Caching eviction policy - LRU vs TTL vs size-based eviction for optimal memory usage

**PLAN_UNCERTAINTY**: Training process containerization - Docker isolation vs direct process management for resource control

**PLAN_UNCERTAINTY**: CDN requirements - whether to serve static assets through CDN for better global performance

**PLAN_UNCERTAINTY**: Load testing requirements - need to define specific load testing scenarios and acceptance criteria

## Dependencies on Other Plans

- **Integration Plan**: Manager extension patterns that affect performance overhead
- **API Plan**: Endpoint response time requirements and caching strategies
- **Security Plan**: Authentication overhead and rate limiting implementation
- **Frontend Plan**: Data update frequency requirements and client-side optimization needs