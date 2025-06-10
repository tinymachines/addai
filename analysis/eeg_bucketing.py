#!/usr/bin/env python3
"""
EEG Data Bucketing and Tokenization System

This module provides functionality to:
1. Bucket EEG data streams into meaningful regions
2. Detect statistically significant patterns
3. Tokenize data for predictive model training
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EEGSample:
    """Single EEG data sample"""

    timestamp: str
    signal_quality: int
    attention: float
    meditation: float
    eeg_power: Dict[str, int]

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for analysis"""
        return np.array(
            [
                self.signal_quality,
                self.attention,
                self.meditation,
                self.eeg_power.get("delta", 0),
                self.eeg_power.get("theta", 0),
                self.eeg_power.get("low_alpha", 0),
                self.eeg_power.get("high_alpha", 0),
                self.eeg_power.get("low_beta", 0),
                self.eeg_power.get("high_beta", 0),
                self.eeg_power.get("low_gamma", 0),
                self.eeg_power.get("mid_gamma", 0),
            ]
        )


@dataclass
class Region:
    """Represents a bucketed region of EEG data"""

    start_idx: int
    end_idx: int
    samples: List[EEGSample]
    features: Dict[str, Any]
    tags: List[str]

    @property
    def duration(self) -> int:
        """Region duration in samples"""
        return self.end_idx - self.start_idx + 1

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistical features for the region"""
        vectors = np.array([s.to_vector() for s in self.samples])

        stats_dict = {}
        feature_names = [
            "signal_quality",
            "attention",
            "meditation",
            "delta",
            "theta",
            "low_alpha",
            "high_alpha",
            "low_beta",
            "high_beta",
            "low_gamma",
            "mid_gamma",
        ]

        for i, name in enumerate(feature_names):
            if vectors[:, i].size > 0:
                stats_dict[name] = {
                    "mean": np.mean(vectors[:, i]),
                    "std": np.std(vectors[:, i]),
                    "min": np.min(vectors[:, i]),
                    "max": np.max(vectors[:, i]),
                    "median": np.median(vectors[:, i]),
                }

        return stats_dict


class EEGBucketing:
    """Main class for EEG data bucketing and analysis"""

    def __init__(self, window_size: int = 10, overlap: int = 2):
        self.window_size = window_size
        self.overlap = overlap
        self.scaler = StandardScaler()

    def load_data(self, file_paths: List[str]) -> List[EEGSample]:
        """Load and parse EEG data from JSON files"""
        samples = []

        for file_path in file_paths:
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        sample = EEGSample(
                            timestamp=data["timestamp"],
                            signal_quality=data["signal_quality"],
                            attention=data["attention"],
                            meditation=data["meditation"],
                            eeg_power=data["eeg_power"],
                        )
                        samples.append(sample)
                    except Exception as e:
                        logger.error(f"Error parsing line: {e}")

        return samples

    def detect_change_points(
        self, samples: List[EEGSample], threshold: float = 2.0
    ) -> List[int]:
        """Detect significant change points in the data stream"""
        vectors = np.array([s.to_vector() for s in samples])

        # Skip signal quality column for change detection
        features = vectors[:, 1:]  # Exclude signal_quality

        # Calculate rolling statistics
        change_points = []

        for i in range(self.window_size, len(features) - self.window_size):
            prev_window = features[i - self.window_size : i]
            next_window = features[i : i + self.window_size]

            # Calculate statistical distance between windows
            prev_mean = np.mean(prev_window, axis=0)
            next_mean = np.mean(next_window, axis=0)

            # Normalize by standard deviation
            std = np.std(features, axis=0)
            std[std == 0] = 1  # Avoid division by zero

            distance = np.linalg.norm((next_mean - prev_mean) / std)

            if distance > threshold:
                change_points.append(i)

        return change_points

    def create_regions(
        self, samples: List[EEGSample], change_points: List[int]
    ) -> List[Region]:
        """Create regions based on change points"""
        regions = []

        # Add boundaries
        boundaries = [0] + change_points + [len(samples) - 1]
        boundaries = sorted(list(set(boundaries)))

        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]

            region_samples = samples[start_idx : end_idx + 1]

            # Skip regions with poor signal quality
            avg_signal_quality = np.mean([s.signal_quality for s in region_samples])

            region = Region(
                start_idx=start_idx,
                end_idx=end_idx,
                samples=region_samples,
                features={},
                tags=[],
            )

            # Tag regions based on characteristics
            region.tags = self._tag_region(region, avg_signal_quality)

            # Store statistical features
            region.features = region.get_statistics()

            regions.append(region)

        return regions

    def _tag_region(self, region: Region, avg_signal_quality: float) -> List[str]:
        """Tag regions based on their characteristics"""
        tags = []

        # Signal quality tags
        if avg_signal_quality == 0:
            tags.append("good_signal")
        elif avg_signal_quality < 50:
            tags.append("moderate_signal")
        elif avg_signal_quality < 200:
            tags.append("poor_signal")
        else:
            tags.append("no_signal")

        # Only analyze regions with reasonable signal
        if avg_signal_quality < 100:
            stats = region.get_statistics()

            # Attention/meditation tags
            if "attention" in stats and stats["attention"]["mean"] > 50:
                tags.append("high_attention")
            if "meditation" in stats and stats["meditation"]["mean"] > 50:
                tags.append("high_meditation")

            # Dominant brainwave tags
            brainwaves = [
                "delta",
                "theta",
                "low_alpha",
                "high_alpha",
                "low_beta",
                "high_beta",
                "low_gamma",
                "mid_gamma",
            ]

            wave_means = {
                wave: stats.get(wave, {}).get("mean", 0) for wave in brainwaves
            }

            if any(wave_means.values()):
                dominant = max(wave_means, key=wave_means.get)
                tags.append(f"dominant_{dominant}")

        # Duration tags
        if region.duration < 5:
            tags.append("transient")
        elif region.duration < 20:
            tags.append("short")
        elif region.duration < 100:
            tags.append("medium")
        else:
            tags.append("long")

        return tags

    def tokenize_regions(
        self, regions: List[Region], vocab_size: Optional[int] = None
    ) -> List[str]:
        """Convert regions to tokens for model training"""
        tokens = []

        for region in regions:
            # Create token based on tags and dominant features
            token_parts = []

            # Add signal quality token
            sig_tags = [t for t in region.tags if "signal" in t]
            if sig_tags:
                token_parts.append(sig_tags[0])

            # Add dominant brainwave token
            dom_tags = [t for t in region.tags if "dominant" in t]
            if dom_tags:
                token_parts.append(dom_tags[0])

            # Add attention/meditation state
            state_tags = [
                t for t in region.tags if "attention" in t or "meditation" in t
            ]
            if state_tags:
                token_parts.extend(state_tags)

            # Add duration
            dur_tags = [
                t for t in region.tags if t in ["transient", "short", "medium", "long"]
            ]
            if dur_tags:
                token_parts.append(dur_tags[0])

            # Create composite token
            if token_parts:
                token = "_".join(token_parts)
                tokens.append(token)

        return tokens

    def find_patterns(
        self, regions: List[Region], min_support: int = 3
    ) -> Dict[str, List[Region]]:
        """Find recurring patterns in regions"""
        patterns = {}

        # Group regions by their tag combinations
        for region in regions:
            tag_key = "_".join(sorted(region.tags))
            if tag_key not in patterns:
                patterns[tag_key] = []
            patterns[tag_key].append(region)

        # Filter by minimum support
        patterns = {k: v for k, v in patterns.items() if len(v) >= min_support}

        return patterns

    def export_regions(self, regions: List[Region], output_file: str) -> None:
        """Export regions to JSON for further analysis"""
        export_data = []

        for region in regions:
            # Convert numpy types to native Python types for JSON serialization
            features = {}
            for key, value in region.features.items():
                if isinstance(value, dict):
                    features[key] = {k: float(v) for k, v in value.items()}
                else:
                    features[key] = float(value) if hasattr(value, "item") else value

            export_data.append(
                {
                    "start_idx": int(region.start_idx),
                    "end_idx": int(region.end_idx),
                    "duration": int(region.duration),
                    "tags": region.tags,
                    "features": features,
                    "start_timestamp": region.samples[0].timestamp,
                    "end_timestamp": region.samples[-1].timestamp,
                }
            )

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(regions)} regions to {output_file}")


def main():
    """EEG bucketing system with command line support"""
    import sys
    
    # Initialize bucketing system
    bucketing = EEGBucketing(window_size=10, overlap=2)

    # Handle command line arguments
    if len(sys.argv) > 1:
        file_paths = [sys.argv[1]]  # Single file from command line
    else:
        # Default files for batch processing
        file_paths = [
            "/home/bisenbek/projects/tinymachines/addai/analysis/logs/tmp/xaa.json",
            "/home/bisenbek/projects/tinymachines/addai/analysis/logs/tmp/xab.json", 
            "/home/bisenbek/projects/tinymachines/addai/analysis/logs/tmp/xac.json",
            "/home/bisenbek/projects/tinymachines/addai/analysis/logs/tmp/xad.json",
            "/home/bisenbek/projects/tinymachines/addai/analysis/logs/tmp/xae.json",
        ]

    logger.info("Loading EEG data...")
    samples = bucketing.load_data(file_paths)
    logger.info(f"Loaded {len(samples)} samples")

    # Detect change points
    logger.info("Detecting change points...")
    change_points = bucketing.detect_change_points(samples, threshold=2.0)
    logger.info(f"Found {len(change_points)} change points")

    # Create regions
    logger.info("Creating regions...")
    regions = bucketing.create_regions(samples, change_points)
    logger.info(f"Created {len(regions)} regions")

    # Find patterns
    logger.info("Finding patterns...")
    patterns = bucketing.find_patterns(regions, min_support=3)
    logger.info(f"Found {len(patterns)} recurring patterns")

    for pattern_key, pattern_regions in patterns.items():
        logger.info(f"Pattern '{pattern_key}': {len(pattern_regions)} occurrences")

    # Tokenize regions
    logger.info("Tokenizing regions...")
    tokens = bucketing.tokenize_regions(regions)
    logger.info(f"Generated {len(tokens)} tokens")
    logger.info(f"Unique tokens: {len(set(tokens))}")

    # Export results (to current directory)
    bucketing.export_regions(regions, "eeg_regions.json")

    # Export tokens
    with open("eeg_tokens.json", "w") as f:
        json.dump(
            {
                "tokens": tokens,
                "unique_tokens": list(set(tokens)),
                "token_counts": {token: tokens.count(token) for token in set(tokens)},
            },
            f,
            indent=2,
        )

    logger.info("Bucketing complete!")


if __name__ == "__main__":
    main()
