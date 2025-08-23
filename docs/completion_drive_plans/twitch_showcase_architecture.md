# Twitch Showcase Architecture Plan - System Architect

## Domain Expertise: Real-time AI Visualization System Architecture

**Planning Phase**: System Architect Domain Plan
**Target**: Complete architecture for Keisei Shogi AI Twitch showcase interface
**Integration Points**: Frontend UI, Backend services, Training system, Streaming infrastructure

## Executive Summary

The Twitch showcase requires a zero-disruption architecture that extracts rich AI training data in real-time while maintaining <100ms latency for viewer engagement. The system must balance educational depth with streaming entertainment value.

## Core Architectural Patterns

### Real-time Data Pipeline Architecture

**Primary Pattern**: Event-Driven Data Flow with Buffering Layers
```
[Training Managers] → [Data Extraction Service] → [Aggregation Layer] → [WebSocket Gateway] → [Frontend UI]
                                ↓
                           [Historical Store] → [Replay Service]
```

**PLAN_UNCERTAINTY**: How to handle data extraction from training managers without impacting performance? Need to validate impact of instrumentation code.

**Key Components**:

1. **Training Data Extractor**: Non-invasive hooks into existing managers
   - Metrics Manager hook for PPO metrics and game stats
   - Model Manager hook for SE block attention weights
   - Step Manager hook for move predictions and position evaluations

2. **Real-time Aggregation Service**: 
   - 50ms aggregation windows for smooth visualization
   - Data compression and delta encoding for WebSocket efficiency
   - Educational context injection (beginner/advanced explanations)

3. **WebSocket Gateway**:
   - Separate channels for different visualization types
   - Client subscription management
   - Frame rate optimization (30fps for smooth animations)

### Interactive UI Architecture

**Primary Pattern**: Multi-layered Information Architecture with Progressive Disclosure

**Component Hierarchy**:
```
StreamingApp
├── ConfigurationPanel (Streamer Controls)
├── MainVisualization
│   ├── ShogiBoard (Canvas/SVG hybrid)
│   │   ├── BoardRenderer
│   │   ├── AttentionHeatmapOverlay
│   │   ├── MoveHighlighter
│   │   └── PositionEvaluationLayer
│   ├── MetricsPanel
│   │   ├── PPOMetricsChart (real-time)
│   │   ├── LearningProgressIndicator
│   │   └── PerformanceDashboard
│   └── PredictionPanel
│       ├── TopMovesRanking
│       ├── ConfidenceBars
│       └── StrategicAnalysis
├── ChatIntegration
│   ├── PauseAndPredictManager
│   ├── CommandProcessor
│   └── ViewerResponseCollector
└── EducationalLayer
    ├── ConceptExplainer
    ├── MilestoneDetector
    └── ReplayManager
```

**PLAN_UNCERTAINTY**: Canvas vs WebGL performance tradeoffs for 60fps Shogi board animations with complex heatmap overlays need benchmarking.

### Performance Optimization Architecture

**Primary Pattern**: Layered Caching with Intelligent Prefetching

**Performance Layers**:

1. **Training Performance Protection**:
   - Async data extraction with circular buffers
   - Copy-on-write semantics for shared data structures
   - Background thread pool for data processing
   - Maximum 1% overhead target for training throughput

2. **Streaming Performance**:
   - CDN-ready asset delivery
   - WebSocket message batching and compression
   - Client-side prediction for smooth animations
   - Adaptive quality based on connection strength

3. **Memory Management**:
   - Time-based data retention policies
   - Efficient replay data storage (compressed deltas)
   - Browser memory leak prevention
   - Mobile-responsive layouts for multi-device viewing

## Technical Specifications

### Backend Service Architecture

**PLAN_UNCERTAINTY**: Need to validate integration points with existing Keisei manager architecture. Current managers may need minimal modification.

**Core Services**:

1. **Training Integration Service** (`keisei-twitch-bridge`)
   - Lightweight instrumentation layer
   - Async data extraction from manager components
   - Zero-copy serialization where possible
   - Health monitoring and failover

2. **Real-time Aggregation Service** (`stream-aggregator`)
   - High-frequency data processing (20Hz internal, 10Hz external)
   - Educational content enrichment
   - Multi-client state synchronization
   - Replay data generation

3. **WebSocket Gateway** (`ws-streaming-gateway`)
   - Channel-based message routing
   - Client capability negotiation
   - Bandwidth-adaptive streaming
   - Chat integration APIs

4. **Historical Data Service** (`replay-service`)
   - Milestone detection and classification
   - Compressed replay storage
   - Search and retrieval APIs
   - Breakthrough moment identification

### Data Schemas

**WebSocket Message Types**:

```typescript
// SE Block Attention Data
interface AttentionUpdate {
  type: 'se_attention';
  timestamp: number;
  board_position: string; // FEN-like for Shogi
  attention_weights: number[][]; // 9x9 grid
  layer_index: number;
  confidence: number;
}

// PPO Metrics Update
interface PPOUpdate {
  type: 'ppo_metrics';
  timestamp: number;
  metrics: {
    kl_divergence: number;
    clip_fraction: number;
    policy_loss: number;
    value_loss: number;
    entropy: number;
  };
  educational_context: {
    level: 'beginner' | 'intermediate' | 'advanced';
    explanation: string;
    color_coding: string;
  };
}

// Move Prediction Data
interface MovePrediction {
  type: 'move_prediction';
  timestamp: number;
  position: string;
  top_moves: Array<{
    move: string;
    confidence: number;
    value_estimate: number;
    strategic_reasoning: string;
  }>;
  current_player: 'black' | 'white';
}

// Learning Milestone
interface Milestone {
  type: 'milestone';
  timestamp: number;
  milestone_type: 'breakthrough' | 'plateau' | 'regression' | 'strategy_shift';
  description: string;
  confidence: number;
  replay_data: ReplaySnapshot[];
}
```

### Frontend Technology Stack

**PLAN_UNCERTAINTY**: React vs Vue vs Svelte performance comparison needed for real-time updates at 60fps with complex visualizations.

**Recommended Stack**:
- **Framework**: React 18 with Concurrent Features
- **State Management**: Zustand for performance-critical updates
- **Visualization**: D3.js + Canvas for board rendering, Chart.js for metrics
- **Styling**: Tailwind CSS with streaming-optimized color schemes
- **Real-time**: Native WebSockets with reconnection logic
- **Animations**: Framer Motion for smooth transitions

**Key Libraries**:
- `react-canvas-draw` for efficient board rendering
- `chart.js` with streaming plugin for real-time metrics
- `react-hotkeys-hook` for streamer keyboard shortcuts
- `react-window` for efficient chat message virtualization

## UI/UX Design Framework

### Visual Hierarchy for Streaming

**PLAN_UNCERTAINTY**: Color schemes need testing for stream compression and colorblind accessibility. Twitch's encoding may affect visualization quality.

**Primary Layout** (1920x1080 optimized):
```
┌─────────────────────┬─────────────────────────────────┐
│   Shogi Board       │        PPO Metrics              │
│   (with overlays)   │   ┌─────────────────────────────┤
│                     │   │ KL Divergence (color coded) │
│       9x9 grid      │   │ Clip Fraction (trend)       │
│    + attention      │   │ Entropy (educational)       │
│    + evaluations    │   └─────────────────────────────┤
│                     │        Move Predictions         │
│                     │   ┌─────────────────────────────┤
│                     │   │ 1. P-7f (95%) Strategic... │
│                     │   │ 2. P-2f (87%) Opening...   │
│                     │   │ 3. N-7g (76%) Defensive... │
│                     │   │ 4. S-6h (65%) Castle...    │
│                     │   │ 5. G-6i (54%) Solid...     │
└─────────────────────┴─────────────────────────────────┤
│                Chat Integration & Controls              │
│  [Pause & Predict] [Explain Current] [Show Milestone]  │
└─────────────────────────────────────────────────────────┘
```

**Educational Layer Design**:
- **Beginner Mode**: Simplified metrics with explanatory text
- **Intermediate Mode**: Full metrics with contextual tooltips  
- **Advanced Mode**: Raw data with mathematical formulations
- **Auto-Switch**: Contextual mode switching based on detected breakthroughs

### Interactive Features Specification

**Pause and Predict System**:
1. Streamer triggers pause during interesting positions
2. UI freezes current position and predictions
3. Chat commands: `!predict [move]` `!confidence [percentage]`
4. Real-time viewer prediction aggregation
5. Reveal and comparison with AI predictions
6. Educational scoring and explanation phase

**Milestone Replay System**:
- Automatic detection of learning breakthroughs
- Instant replay with before/after comparisons
- Side-by-side old vs new decision making
- Community voting on most impressive milestones

## Implementation Roadmap

### Phase 1: MVP (4-6 weeks)
**Goal**: Basic streaming functionality with core visualizations

**Deliverables**:
- Training data extraction service (minimal performance impact)
- WebSocket gateway with basic message types
- React frontend with Shogi board and basic PPO metrics
- Simple chat integration for pause/predict functionality

**Success Criteria**:
- <100ms latency for data updates
- <1% impact on training performance
- Stable streaming for 2+ hours
- Basic educational value demonstrated

### Phase 2: Enhanced Visualizations (3-4 weeks)
**Goal**: Rich SE block attention and position evaluation features

**Deliverables**:
- SE block attention heatmap overlays
- Position evaluation visualization system
- Advanced move prediction interface with strategic reasoning
- Milestone detection and replay functionality

**Success Criteria**:
- Smooth 60fps board animations
- Educational value validated by test viewers
- Milestone detection accuracy >80%
- Replay system working reliably

### Phase 3: Production Polish (2-3 weeks)
**Goal**: Streaming-ready system with full interactivity

**Deliverables**:
- Complete chat integration with all planned commands
- Streamer control panel with scene management
- Performance optimizations and mobile responsiveness
- Comprehensive error handling and failover systems

**Success Criteria**:
- Multi-hour stability testing passed
- Streamer workflow validation completed
- Performance benchmarks met across all components
- Community feature testing successful

## Risk Assessment and Mitigation

### High Risk Items

**Training Performance Impact**:
- **Risk**: Data extraction slows training significantly
- **Mitigation**: Async extraction with performance monitoring, killswitch for degradation
- **Validation**: Continuous benchmarking during development

**Streaming Stability**:
- **Risk**: WebSocket disconnections during long streams  
- **Mitigation**: Automatic reconnection, state synchronization, offline mode
- **Validation**: Extended stress testing with network interruptions

### Medium Risk Items

**Educational Value**:
- **Risk**: Complex AI concepts too difficult for general audience
- **Mitigation**: Multi-tier explanation system, user testing with diverse audiences
- **Validation**: A/B testing different educational approaches

**PLAN_UNCERTAINTY**: User attention span for AI training content unknown. May need gamification elements to maintain engagement.

## Integration Points with Keisei Architecture

### Required Manager Modifications

**Minimal Invasive Changes**:

1. **MetricsManager**: Add optional streaming hooks
   ```python
   def update_progress_metrics(self, key: str, value: Any, stream_hook=None) -> None:
       self.pending_progress_updates[key] = value
       if stream_hook and self.enable_streaming:
           stream_hook.emit('metric_update', {'key': key, 'value': value})
   ```

2. **PPOAgent**: Add attention weight extraction
   ```python  
   def get_attention_weights(self) -> Optional[Dict[str, torch.Tensor]]:
       if hasattr(self.model, 'get_attention_weights'):
           return self.model.get_attention_weights()
       return None
   ```

3. **StepManager**: Add move prediction hooks
   ```python
   def get_top_move_predictions(self, n=5) -> List[Dict]:
       # Extract top-N moves with confidence scores
       # Add strategic reasoning if available
   ```

**PLAN_UNCERTAINTY**: Need to validate that these additions don't break existing functionality or add significant overhead.

## Technology Integration Specifications

### Twitch-Specific Optimizations

**Stream Overlay Compatibility**:
- CSS transforms for different overlay positions
- Transparent backgrounds for chroma key compatibility  
- High contrast modes for various stream backgrounds
- Scene transition animations that work with OBS

**Chat API Integration**:
```javascript
// Twitch Chat Integration
class TwitchChatIntegration {
  constructor(channel, clientId) {
    this.tmi = new tmi.Client({
      connection: { reconnect: true },
      channels: [channel]
    });
    this.commands = new Map();
    this.setupCommands();
  }

  setupCommands() {
    this.commands.set('predict', this.handlePredict);
    this.commands.set('explain', this.handleExplain); 
    this.commands.set('milestone', this.handleMilestone);
  }

  handlePredict(channel, user, message) {
    // Parse move prediction from chat
    // Aggregate with other predictions
    // Update UI with community predictions
  }
}
```

### Performance Benchmarking Framework

**Key Metrics to Track**:
- Training throughput impact (target: <1%)
- WebSocket message latency (target: <100ms)
- Frontend rendering performance (target: 60fps)
- Memory usage growth over time (target: stable)
- Network bandwidth utilization (target: <2Mbps)

**Automated Testing**:
- Continuous performance monitoring during development
- Load testing with simulated viewer counts
- Long-running stability tests (8+ hours)
- Network condition simulation (poor connections)

This architectural plan provides the foundation for a sophisticated yet performant Twitch showcase system that balances educational value with entertainment while protecting the core training system's performance.