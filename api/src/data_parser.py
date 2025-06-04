"""
Data parsing module for EEG CSV data.
Handles validation and conversion of NeuroSky Brain Library format.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, validator


class EEGPowerBands(BaseModel):
    """EEG frequency power bands."""
    delta: int
    theta: int
    low_alpha: int
    high_alpha: int
    low_beta: int
    high_beta: int
    low_gamma: int
    mid_gamma: int


class EEGData(BaseModel):
    """Complete EEG data packet."""
    timestamp: datetime = Field(default_factory=datetime.now)
    signal_quality: int = Field(ge=0, le=200)
    attention: int = Field(ge=0, le=100)
    meditation: int = Field(ge=0, le=100)
    eeg_power: Optional[EEGPowerBands] = None

    @validator('signal_quality')
    def validate_signal_quality(cls, v):
        """Signal quality: 0=good, 200=no signal."""
        return v

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "signal_quality": self.signal_quality,
            "attention": self.attention,
            "meditation": self.meditation,
        }
        if self.eeg_power:
            data["eeg_power"] = {
                "delta": self.eeg_power.delta,
                "theta": self.eeg_power.theta,
                "low_alpha": self.eeg_power.low_alpha,
                "high_alpha": self.eeg_power.high_alpha,
                "low_beta": self.eeg_power.low_beta,
                "high_beta": self.eeg_power.high_beta,
                "low_gamma": self.eeg_power.low_gamma,
                "mid_gamma": self.eeg_power.mid_gamma,
            }
        return data


def parse_csv_line(line: str) -> Optional[EEGData]:
    """
    Parse a single CSV line into EEGData.
    
    Expected formats:
    - Basic: signal_quality,attention,meditation
    - Full: signal_quality,attention,meditation,delta,theta,low_alpha,high_alpha,low_beta,high_beta,low_gamma,mid_gamma
    """
    try:
        line = line.strip()
        if not line:
            return None
            
        parts = line.split(',')
        
        if len(parts) < 3:
            return None
            
        # Parse basic values with validation
        signal_quality = int(parts[0])
        attention = int(parts[1])
        meditation = int(parts[2])
        
        # Validate ranges - if invalid, skip this line
        if not (0 <= signal_quality <= 200):
            print(f"Invalid signal_quality: {signal_quality}")
            return None
        if not (0 <= attention <= 100):
            print(f"Invalid attention: {attention}")
            return None
        if not (0 <= meditation <= 100):
            print(f"Invalid meditation: {meditation}")
            return None
        
        # Parse EEG power bands if available
        eeg_power = None
        if len(parts) == 11:
            try:
                eeg_power = EEGPowerBands(
                    delta=int(parts[3]),
                    theta=int(parts[4]),
                    low_alpha=int(parts[5]),
                    high_alpha=int(parts[6]),
                    low_beta=int(parts[7]),
                    high_beta=int(parts[8]),
                    low_gamma=int(parts[9]),
                    mid_gamma=int(parts[10]),
                )
            except (ValueError, IndexError):
                # If EEG power parsing fails, just skip it
                eeg_power = None
        
        return EEGData(
            signal_quality=signal_quality,
            attention=attention,
            meditation=meditation,
            eeg_power=eeg_power,
        )
        
    except (ValueError, IndexError) as e:
        # Log but don't crash on parse errors
        print(f"Error parsing CSV line '{line}': {e}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected error parsing CSV line '{line}': {e}")
        return None