# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains an Arduino Brain Library for parsing EEG data from NeuroSky-based headsets (Star Wars Force Trainer, Mattel MindFlex). The library processes brain wave data and provides both CSV output and direct access to individual metrics.

## Library Architecture

**Core Components:**
- `Brain/Brain.h` - Main library header with class definition and public API
- `Brain/Brain.cpp` - Implementation of packet parsing, data extraction, and output formatting
- `Brain/examples/` - Three example sketches demonstrating different use cases

**Key Classes:**
- `Brain` class - Main interface that takes a Stream object (Serial/SoftwareSerial) and parses incoming NeuroSky data packets

**Data Flow:**
1. Raw serial data from NeuroSky headset → `update()` method
2. Packet parsing and checksum validation → internal data storage
3. Processed data available via individual getters or CSV format

## Installation for Arduino IDE

The Brain library is designed for Arduino IDE installation:
- Manual: Place `Brain/` folder in Arduino's `libraries/` directory
- IDE: Use "Sketch → Import Library" with downloaded ZIP

## Key Data Types

The library provides access to:
- **Signal Quality** (0-200): 0 = good connection, 200 = no signal
- **eSense Values** (0-100): attention and meditation levels
- **EEG Power Bands**: 8 frequency bands (delta, theta, alpha, beta, gamma) as 32-bit values
- **CSV Output**: All values in comma-delimited format for easy data logging

## Example Usage Patterns

Refer to the three examples:
- `BrainBlinker.ino` - LED control based on attention
- `BrainSerialTest.ino` - CSV data output over serial
- `BrainSoftSerialTest.ino` - Software serial input, hardware serial output