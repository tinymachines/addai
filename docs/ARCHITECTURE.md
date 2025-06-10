# EEG Monitor Architecture

## Overview
Real-time EEG data visualization system that reads NeuroSky Brain Library CSV data from serial connection and displays it in a web application.

## Data Format Analysis
The sample data shows CSV format with variable columns:
- **Signal Quality** (0-200): First column, 0=good connection, 200=no signal
- **Attention** (0-100): Second column, eSense attention value  
- **Meditation** (0-100): Third column, eSense meditation value
- **EEG Power Bands** (8 values): Delta, Theta, Low Alpha, High Alpha, Low Beta, High Beta, Low Gamma, Mid Gamma

Lines can have 4 fields (basic) or 11 fields (with power bands).

## System Architecture

```
[EEG Device] → [Serial /dev/ttyACM0] → [FastAPI Server] → [WebSocket] → [WASM Frontend]
```

### Components

#### 1. FastAPI Server (`./api/`)
- **Serial Monitor**: Reads from `/dev/ttyACM0` continuously
- **Data Parser**: Parses CSV format, validates data integrity
- **WebSocket Server**: Broadcasts real-time data to connected clients
- **Data Storage**: Optional buffering for historical data

**Tech Stack:**
- FastAPI with WebSocket support
- pySerial for serial communication
- Pydantic for data validation
- Python 3.11+

#### 2. WASM Frontend (`./frontend/`)
- **Real-time Visualization**: Line charts for EEG bands, gauges for attention/meditation
- **WebSocket Client**: Connects to FastAPI server
- **Data Processing**: Client-side filtering, smoothing, calculations

**Tech Stack:**
- Rust + wasm-pack for WASM modules
- TypeScript/JavaScript for web components
- Chart.js or D3.js for visualization
- WebSocket API for real-time data
- Vite for build tooling

## Project Structure

```
addai/
├── api/
│   ├── pyproject.toml          # Poetry/pip configuration
│   ├── requirements.txt        # Dependencies
│   ├── src/
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── serial_monitor.py   # Serial port reader
│   │   ├── data_parser.py     # CSV parsing logic
│   │   └── websocket_handler.py # WebSocket management
│   └── tests/
├── frontend/
│   ├── package.json           # npm/pnpm dependencies
│   ├── Cargo.toml            # Rust WASM dependencies
│   ├── src/
│   │   ├── lib.rs            # Rust WASM entry point
│   │   ├── main.ts           # TypeScript entry point
│   │   ├── components/       # Web components
│   │   └── charts/           # Visualization modules
│   ├── www/
│   │   └── index.html        # Main HTML page
│   └── pkg/                  # Generated WASM output
├── ARCHITECTURE.md
└── CLAUDE.md
```

## Data Flow

1. **Serial Input**: EEG device writes CSV data to `/dev/ttyACM0`
2. **Server Processing**: FastAPI reads serial port, parses CSV, validates format
3. **WebSocket Broadcast**: Server sends JSON payload to all connected clients
4. **Client Rendering**: WASM modules process data and update visualizations

## Development Workflow

### API Development
```bash
cd api
pip install -e .
python src/main.py --dev
```

### Frontend Development  
```bash
cd frontend
npm install
npm run dev
```

## Message Protocol

WebSocket messages use JSON format:
```json
{
  "timestamp": "2025-01-06T12:34:56Z",
  "signal_quality": 0,
  "attention": 78,
  "meditation": 81,
  "eeg_power": {
    "delta": 910390,
    "theta": 33747,
    "low_alpha": 19027,
    "high_alpha": 9087,
    "low_beta": 3565,
    "high_beta": 2152,
    "low_gamma": 947,
    "mid_gamma": 597
  }
}
```

## V1 Features

- Real-time serial data reading
- WebSocket data streaming
- Basic line chart visualization for EEG bands
- Attention/meditation gauges
- Connection status indicators
- Error handling for serial disconnections