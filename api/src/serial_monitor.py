"""
Serial port monitoring for EEG data.
Reads from NeuroSky device and parses CSV data.
"""
import asyncio
from typing import Callable, Optional

import serial
import serial.tools.list_ports

from data_parser import parse_csv_line


class SerialMonitor:
    """Monitors serial port for EEG data."""
    
    def __init__(self, device: str = "/dev/ttyACM0", baud_rate: int = 9600):
        self.device = device
        self.baud_rate = baud_rate
        self.serial_connection: Optional[serial.Serial] = None
        self.running = False
    
    async def start_monitoring(self, callback: Callable):
        """Start monitoring serial port and call callback with parsed data."""
        self.running = True
        
        while self.running:
            try:
                if not self.serial_connection or not self.serial_connection.is_open:
                    await self._connect()
                
                if self.serial_connection and self.serial_connection.is_open:
                    await self._read_data(callback)
                else:
                    # No connection, wait before retrying
                    await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"Serial monitoring error: {e}")
                await self._disconnect()
                await asyncio.sleep(2)
    
    async def _connect(self):
        """Establish serial connection."""
        try:
            # Check if device exists
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            if self.device not in available_ports:
                print(f"Device {self.device} not found. Available: {available_ports}")
                return
            
            self.serial_connection = serial.Serial(
                port=self.device,
                baudrate=self.baud_rate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            print(f"Connected to {self.device} at {self.baud_rate} baud")
            
        except Exception as e:
            print(f"Failed to connect to {self.device}: {e}")
            self.serial_connection = None
    
    async def _disconnect(self):
        """Close serial connection."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print(f"Disconnected from {self.device}")
        self.serial_connection = None
    
    async def _read_data(self, callback: Callable):
        """Read and parse data from serial port."""
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                return
            
            # Read line (blocking with timeout)
            line = await asyncio.get_event_loop().run_in_executor(
                None, self.serial_connection.readline
            )
            
            if line:
                line_str = line.decode('utf-8', errors='ignore').strip()
                if line_str:
                    # Parse the CSV data
                    eeg_data = parse_csv_line(line_str)
                    if eeg_data:
                        # Send to callback (WebSocket broadcast)
                        await callback(eeg_data.to_dict())
                    
        except Exception as e:
            print(f"Error reading serial data: {e}")
            raise
    
    async def stop(self):
        """Stop monitoring and close connection."""
        self.running = False
        await self._disconnect()