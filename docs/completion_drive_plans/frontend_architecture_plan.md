# Frontend Architecture Plan for Keisei Web UI

**Domain**: Frontend Architecture and User Interface Design  
**Planning Agent**: Frontend Architecture Specialist
**Date**: 2025-08-23

## Technology Stack Selection

### Primary Framework
**React 18** with TypeScript for component-based architecture
- **Rationale**: Mature ecosystem, excellent real-time update handling, strong TypeScript support
- **Alternative considered**: Vue.js (simpler but less ecosystem support for complex data visualization)

### State Management  
**Redux Toolkit** for global application state
- **Training state**: Current session, metrics, configuration
- **UI state**: Dashboard layout, user preferences, modal states
- **Real-time state**: WebSocket connection status, live data feeds

### Real-Time Communication
**Socket.IO Client** for WebSocket management
- Automatic reconnection handling
- Message queue for offline scenarios  
- Event-based data updates

### Data Visualization
**Plotly.js/React-Plotly** for training metrics visualization
- **Rationale**: Handles real-time updates efficiently, supports complex multi-dimensional plots
- **Alternative**: D3.js (more flexible but requires more development time)

## Component Architecture

### Layout Structure
```
App
├── HeaderBar (navigation, user info, session status)
├── SidebarNavigation (main sections)
└── MainContent
    ├── Dashboard (overview)
    ├── TrainingMonitor (live training view)
    ├── GameVisualization (Shogi board display)  
    ├── ModelManagement (model browser)
    ├── Configuration (config editor)
    └── EvaluationResults (evaluation history)
```

### Core Components

#### 1. Dashboard Overview
**Components**:
- `TrainingStatusCard` - Current training session summary
- `MetricsGrid` - Key performance indicators
- `RecentActivityFeed` - Latest training/evaluation events
- `SystemResourceMonitor` - CPU/GPU/Memory usage
- `QuickActions` - Start/stop/pause controls

**State Requirements**:
- Real-time metrics updates via WebSocket
- Session metadata from REST API
- System performance data

#### 2. Training Monitor
**Components**:
- `LiveMetricsChart` - Real-time training metrics plotting
- `TrainingProgressBar` - Timestep/episode progress
- `PPOMetricsPanel` - PPO algorithm-specific metrics
- `TrainingLog` - Filtered log message display
- `PerformanceChart` - Training speed and efficiency

**Real-Time Data Flow**:
```typescript
// WebSocket message handling
const handleTrainingUpdate = (update: TrainingUpdate) => {
  dispatch(updateMetrics(update.data));
  // Trigger chart re-render with new data point
  setChartData(prev => [...prev.slice(-1000), update.data]);
};
```

#### 3. Shogi Game Visualization
**Components**:
- `ShogiBoard` - Interactive board display
- `MoveHistory` - Move sequence with notation
- `GameControls` - Play/pause/step through moves
- `PieceCaptures` - Captured pieces display
- `GameInfo` - Game metadata and status

**Board Representation**:
```typescript
interface BoardState {
  pieces: (Piece | null)[][];  // 9x9 grid
  hands: { [color: string]: { [pieceType: string]: number } };
  currentPlayer: 'black' | 'white';
  moveCount: number;
  gameStatus: 'ongoing' | 'checkmate' | 'draw';
}
```

#### 4. Configuration Editor
**Components**:
- `ConfigurationForm` - Tabbed configuration editor
- `ConfigValidation` - Real-time validation feedback
- `ConfigTemplates` - Predefined configuration sets
- `ConfigDiff` - Visual diff of configuration changes
- `ConfigSave` - Save/load configuration management

**Form Structure**:
- Environment settings (device, seed, channels)
- Training parameters (learning rate, PPO settings)
- Model architecture (ResNet depth, width, SE blocks)
- Evaluation configuration (strategies, intervals)

## State Management Architecture

### Redux Store Structure
```typescript
interface AppState {
  auth: AuthState;
  training: {
    currentSession: TrainingSession | null;
    metrics: MetricsHistory;
    status: 'idle' | 'running' | 'paused' | 'stopped';
  };
  ui: {
    selectedTab: string;
    dashboardLayout: LayoutConfig;
    notifications: Notification[];
  };
  websocket: {
    connected: boolean;
    reconnectAttempts: number;
    messageQueue: QueuedMessage[];
  };
  configuration: {
    current: KiseiConfig;
    templates: ConfigTemplate[];
    validation: ValidationResult[];
  };
  models: {
    available: ModelInfo[];
    selected: ModelInfo | null;
  };
  evaluation: {
    history: EvaluationResult[];
    currentEvaluation: EvaluationStatus | null;
  };
}
```

### Real-Time Data Integration
```typescript
// WebSocket middleware for Redux
const websocketMiddleware: Middleware = store => next => action => {
  if (action.type === 'websocket/connect') {
    // Establish WebSocket connection
    const ws = new io('/ws/training');
    
    ws.on('training_update', (data) => {
      store.dispatch(updateTrainingMetrics(data));
    });
    
    ws.on('game_state', (data) => {
      store.dispatch(updateGameState(data));
    });
  }
  
  return next(action);
};
```

## User Experience Design

### Responsive Design Principles
- **Desktop First**: Primary use case is development workstations
- **Tablet Compatible**: Support for tablet-based monitoring  
- **Mobile Aware**: Basic monitoring capability on mobile devices

### Dashboard Customization
- **Widget System**: Drag-and-drop dashboard widgets
- **Layout Persistence**: Save user layout preferences
- **Data Filtering**: Customizable metric filtering and grouping

### Real-Time Performance
- **Virtual Scrolling**: For large log/history displays
- **Data Decimation**: Reduce chart data points for smooth rendering
- **Update Throttling**: Limit UI updates to maintain 60fps

## Integration Points

### API Integration Patterns
```typescript
// Custom hooks for API integration
export const useTrainingStatus = () => {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  
  useEffect(() => {
    const fetchStatus = async () => {
      const response = await fetch('/api/training/status');
      setStatus(await response.json());
    };
    
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);
  
  return status;
};
```

### Configuration Management
- **Schema Validation**: Client-side validation using JSON Schema
- **Diff Visualization**: Show configuration changes before apply
- **Template System**: Predefined configurations for common scenarios

## PLAN_UNCERTAINTY Tags

**PLAN_UNCERTAINTY**: Shogi board rendering performance with frequent updates - may need canvas-based rendering instead of DOM elements

**PLAN_UNCERTAINTY**: Chart performance with high-frequency data updates - need to determine optimal data retention and rendering strategies

**PLAN_UNCERTAINTY**: Offline capability requirements - should the UI work when training system is unavailable?

**PLAN_UNCERTAINTY**: Multi-session handling - UI design for managing multiple concurrent training sessions

**PLAN_UNCERTAINTY**: Accessibility requirements - need clarification on WCAG compliance level needed

**PLAN_UNCERTAINTY**: Browser compatibility matrix - modern browsers only vs broader support

## Dependencies on Other Plans

- **API Plan**: REST endpoint specifications, WebSocket message formats
- **Security Plan**: Authentication flow, session management
- **Performance Plan**: Data update frequency limits, caching requirements
- **Integration Plan**: State synchronization patterns with backend managers