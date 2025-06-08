import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from funkybob import RandomNameGenerator
import base64
from io import BytesIO


class ReportGenerator:
    def __init__(self, base_path: str = "./reports"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.name_generator = RandomNameGenerator()
        
    def create_run_folder(self) -> Tuple[str, Path]:
        """Create a new run folder with random name."""
        run_name = next(iter(self.name_generator))
        run_path = self.base_path / run_name
        run_path.mkdir(exist_ok=True)
        return run_name, run_path
    
    def save_plot(self, fig: plt.Figure, run_path: Path, filename: str) -> str:
        """Save plot to run folder and return filename."""
        filepath = run_path / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        return filename
    
    def plot_to_base64(self, fig: plt.Figure) -> str:
        """Convert plot to base64 for HTML embedding."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        buffer.close()
        return f"data:image/png;base64,{img_str}"
    
    def generate_report(self, analysis_data: Dict[str, Any]) -> Tuple[str, Path]:
        """Generate all report formats."""
        run_name, run_path = self.create_run_folder()
        
        # Generate plots
        plots = self._generate_plots(analysis_data, run_path)
        
        # Generate reports
        self._generate_markdown_report(analysis_data, plots, run_path)
        self._generate_html_report(analysis_data, plots, run_path)
        self._generate_json_report(analysis_data, plots, run_path)
        
        return run_name, run_path
    
    def _generate_plots(self, data: Dict[str, Any], run_path: Path) -> Dict[str, str]:
        """Generate all plots and return filenames."""
        plots = {}
        
        # 1. Signal Quality Over Time
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['timestamps'], data['signal_quality'], 'b-', alpha=0.7)
        ax.fill_between(data['timestamps'], data['signal_quality'], alpha=0.3)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Signal Quality')
        ax.set_title('Signal Quality Over Time')
        ax.invert_yaxis()  # Lower is better
        ax.grid(True, alpha=0.3)
        plots['signal_quality'] = self.save_plot(fig, run_path, 'signal_quality.png')
        plots['signal_quality_base64'] = self.plot_to_base64(fig)
        plt.close(fig)
        
        # 2. Attention and Meditation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(data['timestamps'], data['attention'], 'g-', label='Attention', linewidth=2)
        ax1.fill_between(data['timestamps'], data['attention'], alpha=0.3, color='green')
        ax1.set_ylabel('Attention Level')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(data['timestamps'], data['meditation'], 'b-', label='Meditation', linewidth=2)
        ax2.fill_between(data['timestamps'], data['meditation'], alpha=0.3, color='blue')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Meditation Level')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle('Attention and Meditation Levels Over Time')
        plots['attention_meditation'] = self.save_plot(fig, run_path, 'attention_meditation.png')
        plots['attention_meditation_base64'] = self.plot_to_base64(fig)
        plt.close(fig)
        
        # 3. Brainwave Distribution
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wave_types = list(data['brainwaves'].keys())
        avg_values = [np.mean(values) for values in data['brainwaves'].values()]
        
        bars = ax.bar(wave_types, avg_values, color=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'purple'])
        ax.set_xlabel('Brainwave Type')
        ax.set_ylabel('Average Power')
        ax.set_title('Average Brainwave Power Distribution')
        ax.set_yscale('log')
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plots['brainwave_distribution'] = self.save_plot(fig, run_path, 'brainwave_distribution.png')
        plots['brainwave_distribution_base64'] = self.plot_to_base64(fig)
        plt.close(fig)
        
        # 4. Brainwave Heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create matrix for heatmap
        wave_matrix = []
        for wave_type in wave_types:
            wave_matrix.append(data['brainwaves'][wave_type])
        
        # Downsample if too many time points
        if len(data['timestamps']) > 100:
            step = len(data['timestamps']) // 100
            wave_matrix = [wave[::step] for wave in wave_matrix]
            time_labels = data['timestamps'][::step]
        else:
            time_labels = data['timestamps']
        
        # Create heatmap
        sns.heatmap(wave_matrix, 
                   xticklabels=False,
                   yticklabels=wave_types,
                   cmap='viridis',
                   cbar_kws={'label': 'Power'},
                   ax=ax)
        
        ax.set_xlabel('Time')
        ax.set_title('Brainwave Activity Heatmap')
        
        plots['brainwave_heatmap'] = self.save_plot(fig, run_path, 'brainwave_heatmap.png')
        plots['brainwave_heatmap_base64'] = self.plot_to_base64(fig)
        plt.close(fig)
        
        # 5. Statistical Summary Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Attention distribution
        ax1.hist(data['attention'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax1.axvline(np.mean(data['attention']), color='red', linestyle='--', label=f"Mean: {np.mean(data['attention']):.1f}")
        ax1.set_xlabel('Attention Level')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Attention Distribution')
        ax1.legend()
        
        # Meditation distribution
        ax2.hist(data['meditation'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(np.mean(data['meditation']), color='red', linestyle='--', label=f"Mean: {np.mean(data['meditation']):.1f}")
        ax2.set_xlabel('Meditation Level')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Meditation Distribution')
        ax2.legend()
        
        # Correlation scatter
        ax3.scatter(data['attention'], data['meditation'], alpha=0.5)
        ax3.set_xlabel('Attention')
        ax3.set_ylabel('Meditation')
        ax3.set_title(f"Attention vs Meditation (r={np.corrcoef(data['attention'], data['meditation'])[0,1]:.2f})")
        
        # Box plots
        box_data = [data['attention'], data['meditation']]
        ax4.boxplot(box_data, labels=['Attention', 'Meditation'])
        ax4.set_ylabel('Level')
        ax4.set_title('Statistical Summary')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plots['statistical_summary'] = self.save_plot(fig, run_path, 'statistical_summary.png')
        plots['statistical_summary_base64'] = self.plot_to_base64(fig)
        plt.close(fig)
        
        return plots
    
    def _generate_markdown_report(self, data: Dict[str, Any], plots: Dict[str, str], run_path: Path):
        """Generate Markdown report."""
        stats = data['statistics']
        
        markdown = f"""# EEG Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Duration**: {data['duration']:.1f} seconds  
**Total Samples**: {data['total_samples']}  
**Run ID**: {run_path.name}

## Signal Quality

Average signal quality: {stats['signal_quality']['mean']:.1f} (0=best, 200=worst)

![Signal Quality]({plots['signal_quality']})

## Attention and Meditation

### Attention
- Mean: {stats['attention']['mean']:.1f}
- Std Dev: {stats['attention']['std']:.1f}
- Min: {stats['attention']['min']:.1f}
- Max: {stats['attention']['max']:.1f}

### Meditation
- Mean: {stats['meditation']['mean']:.1f}
- Std Dev: {stats['meditation']['std']:.1f}
- Min: {stats['meditation']['min']:.1f}
- Max: {stats['meditation']['max']:.1f}

![Attention and Meditation]({plots['attention_meditation']})

## Brainwave Analysis

### Average Power Distribution

![Brainwave Distribution]({plots['brainwave_distribution']})

### Brainwave Activity Over Time

![Brainwave Heatmap]({plots['brainwave_heatmap']})

### Brainwave Statistics

| Wave Type | Mean | Std Dev | Min | Max |
|-----------|------|---------|-----|-----|
"""
        
        for wave_type, wave_stats in stats['brainwaves'].items():
            markdown += f"| {wave_type.capitalize()} | {wave_stats['mean']:.0f} | {wave_stats['std']:.0f} | {wave_stats['min']:.0f} | {wave_stats['max']:.0f} |\n"
        
        markdown += f"""

## Statistical Analysis

![Statistical Summary]({plots['statistical_summary']})

### Key Findings

1. **Signal Quality**: {'Good' if stats['signal_quality']['mean'] < 50 else 'Poor' if stats['signal_quality']['mean'] > 100 else 'Moderate'} average signal quality
2. **Focus State**: {'High' if stats['attention']['mean'] > 60 else 'Low' if stats['attention']['mean'] < 40 else 'Moderate'} average attention levels
3. **Relaxation State**: {'High' if stats['meditation']['mean'] > 60 else 'Low' if stats['meditation']['mean'] < 40 else 'Moderate'} average meditation levels
4. **Dominant Brainwave**: {max(stats['brainwaves'].items(), key=lambda x: x[1]['mean'])[0].capitalize()}
5. **Attention-Meditation Correlation**: {np.corrcoef(data['attention'], data['meditation'])[0,1]:.2f}

## Session Summary

- Recording started: {data.get('start_time', 'N/A')}
- Recording ended: {data.get('end_time', 'N/A')}
- Valid samples: {data.get('valid_samples', data['total_samples'])}
- Invalid samples: {data.get('invalid_samples', 0)}
- Average sample rate: {data['total_samples'] / data['duration']:.1f} Hz

---
*Report generated by AddAI EEG Analysis Tool*
"""
        
        with open(run_path / 'report.md', 'w') as f:
            f.write(markdown)
    
    def _generate_html_report(self, data: Dict[str, Any], plots: Dict[str, str], run_path: Path):
        """Generate standalone HTML report with embedded assets."""
        stats = data['statistics']
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EEG Analysis Report - {run_path.name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        h1 {{
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .meta-info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }}
        .findings {{
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .finding-item {{
            margin: 10px 0;
            padding-left: 20px;
            position: relative;
        }}
        .finding-item:before {{
            content: "▸";
            position: absolute;
            left: 0;
            color: #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>EEG Analysis Report</h1>
        
        <div class="meta-info">
            <strong>Run ID:</strong> {run_path.name}<br>
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Duration:</strong> {data['duration']:.1f} seconds<br>
            <strong>Total Samples:</strong> {data['total_samples']}
        </div>
        
        <h2>Signal Quality</h2>
        <p>Average signal quality: <strong>{stats['signal_quality']['mean']:.1f}</strong> (0=best, 200=worst)</p>
        <div class="plot-container">
            <img src="{plots['signal_quality_base64']}" alt="Signal Quality">
        </div>
        
        <h2>Attention and Meditation</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Attention</div>
                <div class="stat-value">{stats['attention']['mean']:.1f}</div>
                <div>Std Dev: {stats['attention']['std']:.1f}</div>
                <div>Range: {stats['attention']['min']:.0f} - {stats['attention']['max']:.0f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Meditation</div>
                <div class="stat-value">{stats['meditation']['mean']:.1f}</div>
                <div>Std Dev: {stats['meditation']['std']:.1f}</div>
                <div>Range: {stats['meditation']['min']:.0f} - {stats['meditation']['max']:.0f}</div>
            </div>
        </div>
        
        <div class="plot-container">
            <img src="{plots['attention_meditation_base64']}" alt="Attention and Meditation">
        </div>
        
        <h2>Brainwave Analysis</h2>
        
        <h3>Average Power Distribution</h3>
        <div class="plot-container">
            <img src="{plots['brainwave_distribution_base64']}" alt="Brainwave Distribution">
        </div>
        
        <h3>Brainwave Activity Over Time</h3>
        <div class="plot-container">
            <img src="{plots['brainwave_heatmap_base64']}" alt="Brainwave Heatmap">
        </div>
        
        <h3>Brainwave Statistics</h3>
        <table>
            <tr>
                <th>Wave Type</th>
                <th>Mean</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
            </tr>
"""
        
        for wave_type, wave_stats in stats['brainwaves'].items():
            html += f"""
            <tr>
                <td>{wave_type.capitalize()}</td>
                <td>{wave_stats['mean']:.0f}</td>
                <td>{wave_stats['std']:.0f}</td>
                <td>{wave_stats['min']:.0f}</td>
                <td>{wave_stats['max']:.0f}</td>
            </tr>
"""
        
        html += f"""
        </table>
        
        <h2>Statistical Analysis</h2>
        <div class="plot-container">
            <img src="{plots['statistical_summary_base64']}" alt="Statistical Summary">
        </div>
        
        <div class="findings">
            <h3>Key Findings</h3>
            <div class="finding-item">
                <strong>Signal Quality:</strong> {'Good' if stats['signal_quality']['mean'] < 50 else 'Poor' if stats['signal_quality']['mean'] > 100 else 'Moderate'} average signal quality
            </div>
            <div class="finding-item">
                <strong>Focus State:</strong> {'High' if stats['attention']['mean'] > 60 else 'Low' if stats['attention']['mean'] < 40 else 'Moderate'} average attention levels
            </div>
            <div class="finding-item">
                <strong>Relaxation State:</strong> {'High' if stats['meditation']['mean'] > 60 else 'Low' if stats['meditation']['mean'] < 40 else 'Moderate'} average meditation levels
            </div>
            <div class="finding-item">
                <strong>Dominant Brainwave:</strong> {max(stats['brainwaves'].items(), key=lambda x: x[1]['mean'])[0].capitalize()}
            </div>
            <div class="finding-item">
                <strong>Attention-Meditation Correlation:</strong> {np.corrcoef(data['attention'], data['meditation'])[0,1]:.2f}
            </div>
        </div>
        
        <h2>Session Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Recording Started</div>
                <div>{data.get('start_time', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Recording Ended</div>
                <div>{data.get('end_time', 'N/A')}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Valid Samples</div>
                <div class="stat-value">{data.get('valid_samples', data['total_samples'])}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sample Rate</div>
                <div class="stat-value">{data['total_samples'] / data['duration']:.1f} Hz</div>
            </div>
        </div>
        
        <hr style="margin-top: 50px;">
        <p style="text-align: center; color: #7f8c8d;">
            <em>Report generated by AddAI EEG Analysis Tool</em>
        </p>
    </div>
</body>
</html>
"""
        
        with open(run_path / 'report.html', 'w') as f:
            f.write(html)
    
    def _generate_json_report(self, data: Dict[str, Any], plots: Dict[str, str], run_path: Path):
        """Generate JSON report for programmatic consumption."""
        report_data = {
            'run_id': run_path.name,
            'generated_at': datetime.now().isoformat(),
            'duration_seconds': data['duration'],
            'total_samples': data['total_samples'],
            'valid_samples': data.get('valid_samples', data['total_samples']),
            'invalid_samples': data.get('invalid_samples', 0),
            'sample_rate_hz': data['total_samples'] / data['duration'],
            'start_time': data.get('start_time', None),
            'end_time': data.get('end_time', None),
            'statistics': data['statistics'],
            'plots': {k: v for k, v in plots.items() if not k.endswith('_base64')},
            'analysis': {
                'signal_quality_assessment': 'good' if data['statistics']['signal_quality']['mean'] < 50 else 'poor' if data['statistics']['signal_quality']['mean'] > 100 else 'moderate',
                'attention_level': 'high' if data['statistics']['attention']['mean'] > 60 else 'low' if data['statistics']['attention']['mean'] < 40 else 'moderate',
                'meditation_level': 'high' if data['statistics']['meditation']['mean'] > 60 else 'low' if data['statistics']['meditation']['mean'] < 40 else 'moderate',
                'dominant_brainwave': max(data['statistics']['brainwaves'].items(), key=lambda x: x[1]['mean'])[0],
                'attention_meditation_correlation': float(np.corrcoef(data['attention'], data['meditation'])[0,1])
            },
            'raw_data_summary': {
                'attention': {
                    'samples': len(data['attention']),
                    'first_10': data['attention'][:10] if len(data['attention']) >= 10 else data['attention'],
                    'last_10': data['attention'][-10:] if len(data['attention']) >= 10 else data['attention']
                },
                'meditation': {
                    'samples': len(data['meditation']),
                    'first_10': data['meditation'][:10] if len(data['meditation']) >= 10 else data['meditation'],
                    'last_10': data['meditation'][-10:] if len(data['meditation']) >= 10 else data['meditation']
                },
                'brainwaves': {
                    wave_type: {
                        'samples': len(values),
                        'first_10': values[:10] if len(values) >= 10 else values,
                        'last_10': values[-10:] if len(values) >= 10 else values
                    }
                    for wave_type, values in data['brainwaves'].items()
                }
            }
        }
        
        with open(run_path / 'report.json', 'w') as f:
            json.dump(report_data, f, indent=2)