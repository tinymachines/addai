# EEG Analysis Reports

This directory contains the unified analysis runner and generated reports.

## Usage

Run the complete EEG analysis pipeline with:

```bash
./run_analysis.py <log_file> [--name <report_name>]
```

### Examples

```bash
# Analyze a log file with auto-generated report name
./run_analysis.py ../analysis/logs/1749331786.json

# Analyze with custom report name
./run_analysis.py ../analysis/logs/1749331786.json --name my_session

# From project root
python reports/run_analysis.py analysis/logs/1749331786.json --name test_run
```

## What it does

The script runs all three analysis tools in sequence:

1. **Report Generator** - Creates markdown, HTML, and JSON reports with visualizations
2. **EEG Bucketing** - Performs temporal segmentation and pattern recognition 
3. **EEG Analyzer** - Comprehensive statistical analysis and ML readiness assessment

## Output Files

Each analysis creates a report directory containing:

- `report.md` - Markdown report with analysis summary
- `report.html` - Interactive HTML report with embedded charts
- `report.json` - Machine-readable analysis data
- `eeg_analysis_plots.png` - Comprehensive statistical plots
- `eeg_tokens.json` - Temporal bucketing tokens
- `eeg_regions.json` - Segmented time regions
- `*.png` - Individual visualization plots

## File Organization

All output files are automatically organized into the correct report directory. No manual file management needed!

## Requirements

- Python 3.x
- All dependencies from analysis/ tools
- Matplotlib for visualizations
- Valid EEG log file in JSON format