# EEG Monitor

Real-time EEG data visualization system that reads NeuroSky Brain Library CSV data from serial connection and displays it in a web application.

## Project Structure

```
addai/
├── api/                    # FastAPI backend
├── frontend/               # WASM + TypeScript frontend  
├── Brain/                  # Original Arduino Brain Library
├── ARCHITECTURE.md         # System design documentation
└── CLAUDE.md              # Development guidance
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Rust (for WASM compilation)
- EEG device connected to `/dev/ttyACM0`

### Ubuntu x86_64 Setup

Install all required dependencies:

```bash
# Update package list
sudo apt update

# Python 3.11+ and pip
sudo apt install python3.11 python3.11-dev python3-pip python3.11-venv

# Node.js 18+ (via NodeSource repository)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install nodejs

# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# wasm-pack for building WASM modules
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build tools and dependencies
sudo apt install build-essential pkg-config libssl-dev

# Serial port permissions (for EEG device access)
sudo usermod -a -G dialout $USER
# Note: Log out and back in for group changes to take effect
```

### Backend Setup

```bash
cd api
pip install -e .
python src/main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Build WASM module
wasm-pack build --target web --out-dir pkg

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Data Format

The system expects CSV data from NeuroSky devices in two formats:

**Basic (3 fields):**
```
signal_quality,attention,meditation
```

**Full (11 fields):**
```
signal_quality,attention,meditation,delta,theta,low_alpha,high_alpha,low_beta,high_beta,low_gamma,mid_gamma
```

## Features

- **Real-time Data Streaming**: WebSocket connection between API and frontend
- **Interactive Visualization**: Line charts for attention/meditation, bar chart for EEG bands
- **WASM Processing**: Client-side data processing using Rust-compiled WebAssembly
- **Connection Monitoring**: Automatic reconnection and error handling
- **Signal Quality Indicators**: Visual feedback on connection status

## Development

### API Development
```bash
cd api
pip install -e ".[dev]"
python src/main.py --reload
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### Testing Sample Data

Use the included `sample-captures-1s.csv` to test the system:

```bash
# Simulate serial data (in another terminal)
cat sample-captures-1s.csv > /dev/ttyACM0
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design and data flow information.