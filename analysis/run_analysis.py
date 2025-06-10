#!/usr/bin/env python3
"""
Unified EEG Analysis Runner
Runs all analysis tools in sequence: report generation, bucketing, and visualization
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path to import analysis modules
sys.path.append(str(Path(__file__).parent.parent))

from analysis.report_generator import ReportGenerator
from analysis.eeg_bucketing import main as run_bucketing
from analysis.eeg_analyzer import main as run_analyzer


def run_full_analysis(log_file: str, report_name: str = None):
    """Run complete analysis pipeline on a log file"""
    print(f"\n{'='*60}")
    print(f"Running Full EEG Analysis Pipeline")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")
    
    # Determine report name
    if not report_name:
        report_name = Path(log_file).stem
    
    # Create report directory in reports folder
    reports_base = Path(__file__).parent.parent / "reports"
    reports_base.mkdir(exist_ok=True)
    report_dir = reports_base / report_name
    report_dir.mkdir(exist_ok=True)
    
    print(f"Report directory: {report_dir}")
    
    # Step 1: Generate base report
    print("\n" + "-"*40)
    print("Step 1: Generating EEG Report")
    print("-"*40)
    
    # Initialize report generator
    report_gen = ReportGenerator(str(reports_base))
    print(f"Report generator initialized for directory: {reports_base}")
    
    # Step 2: Run bucketing analysis
    print("\n" + "-"*40)
    print("Step 2: Running Bucketing Analysis")
    print("-"*40)
    
    sys.argv = ['eeg_bucketing.py', log_file]
    run_bucketing()
    
    # Move bucketing output files to report directory
    analysis_dir = Path(__file__).parent
    tokens_file = analysis_dir / "eeg_tokens.json"
    regions_file = analysis_dir / "eeg_regions.json"
    
    if tokens_file.exists():
        tokens_file.rename(report_dir / tokens_file.name)
        print(f"Moved {tokens_file.name} to report directory")
    
    if regions_file.exists():
        regions_file.rename(report_dir / regions_file.name)
        print(f"Moved {regions_file.name} to report directory")
    
    # Step 3: Run detailed analyzer
    print("\n" + "-"*40)
    print("Step 3: Running Detailed EEG Analysis")
    print("-"*40)
    
    # Pass output directory to analyzer
    sys.argv = ['eeg_analyzer.py', log_file, '--output-dir', str(report_dir)]
    run_analyzer()
    
    # Check for any stray files in the analysis directory
    for file in analysis_dir.glob("*.png"):
        # Move any PNG files to the report directory
        file.rename(report_dir / file.name)
        print(f"Moved {file.name} to report directory")
    
    for file in analysis_dir.glob("*.json"):
        # Move any JSON files to the report directory (except logs and existing files)
        if file.name not in ['run_analysis.py'] and not file.is_relative_to(analysis_dir / "logs"):
            file.rename(report_dir / file.name)
            print(f"Moved {file.name} to report directory")
    
    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"All results saved to: {report_dir}")
    print(f"{'='*60}\n")
    
    # List all generated files
    print("Generated files:")
    for file in sorted(report_dir.iterdir()):
        if file.is_file():
            print(f"  - {file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete EEG analysis pipeline"
    )
    parser.add_argument(
        "log_file",
        help="Path to EEG log file (JSON format)"
    )
    parser.add_argument(
        "--name",
        help="Name for the report directory (default: log file name)"
    )
    
    args = parser.parse_args()
    
    # Verify log file exists
    if not Path(args.log_file).exists():
        print(f"Error: Log file '{args.log_file}' not found")
        sys.exit(1)
    
    run_full_analysis(args.log_file, args.name)


if __name__ == "__main__":
    main()