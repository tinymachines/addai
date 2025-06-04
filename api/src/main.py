#!/usr/bin/env python3
"""
FastAPI server for real-time EEG data monitoring.
Reads data from serial port and broadcasts via WebSocket.
"""
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from serial_monitor import SerialMonitor
from websocket_handler import ConnectionManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Start serial monitoring
    serial_monitor = SerialMonitor(device="/dev/ttyACM0", baud_rate=9600)
    connection_manager = ConnectionManager()
    
    # Store in app state
    app.state.serial_monitor = serial_monitor
    app.state.connection_manager = connection_manager
    
    # Start monitoring task
    monitor_task = asyncio.create_task(
        serial_monitor.start_monitoring(connection_manager.broadcast)
    )
    
    yield
    
    # Cleanup
    monitor_task.cancel()
    await serial_monitor.stop()


app = FastAPI(
    title="EEG Monitor API",
    description="Real-time EEG data streaming via WebSocket",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="../frontend/www"), name="static")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "running", "message": "EEG Monitor API"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time EEG data."""
    await app.state.connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        app.state.connection_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )