import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.signal import welch, coherence
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
import sys
from report_generator import ReportGenerator

warnings.filterwarnings("ignore")


class EEGDataAnalyzer:
    """
    Comprehensive EEG data analyzer for ML preprocessing and quality assessment.
    Designed for attention detection model training.
    """

    def __init__(self, data_files=None):
        self.data_files = data_files or [
            f"xa{chr(ord('a') + i)}.json" for i in range(5)
        ]
        self.raw_data = []
        self.df = None
        self.eeg_bands = [
            "delta",
            "theta",
            "low_alpha",
            "high_alpha",
            "low_beta",
            "high_beta",
            "low_gamma",
            "mid_gamma",
        ]
        self.stats_summary = {}
        self.report_generator = ReportGenerator()
        self.analysis_data = None

    def load_data(self):
        """Load and parse JSON data files."""
        print("Loading EEG data files...")

        for file_path in self.data_files:
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data_point = json.loads(line.strip())
                                self.raw_data.append(data_point)
                            except json.JSONDecodeError:
                                continue
                print(f"✓ Loaded {file_path}")
            except FileNotFoundError:
                print(f"✗ File {file_path} not found")

        if not self.raw_data:
            raise ValueError("No data loaded. Check file paths.")

        print(f"Total data points loaded: {len(self.raw_data)}")
        self._create_dataframe()

    def _create_dataframe(self):
        """Convert raw data to structured DataFrame."""
        rows = []

        for point in self.raw_data:
            row = {
                "timestamp": pd.to_datetime(point["timestamp"]),
                "signal_quality": point["signal_quality"],
            }

            # Extract EEG power bands (ignoring attention/meditation)
            if "eeg_power" in point:
                for band in self.eeg_bands:
                    if band in point["eeg_power"]:
                        row[f"power_{band}"] = point["eeg_power"][band]

            rows.append(row)

        self.df = pd.DataFrame(rows)
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)

        # Calculate time differences for temporal analysis
        self.df["time_delta"] = self.df["timestamp"].diff().dt.total_seconds()

        print(
            f"DataFrame created: {self.df.shape[0]} samples, {self.df.shape[1]} features"
        )

    def basic_statistics(self):
        """Compute comprehensive statistical summary."""
        print("\n" + "=" * 60)
        print("BASIC STATISTICAL ANALYSIS")
        print("=" * 60)

        power_cols = [col for col in self.df.columns if col.startswith("power_")]

        # Basic descriptive statistics
        desc_stats = self.df[power_cols].describe()
        print("\nDescriptive Statistics (EEG Power Bands):")
        print(desc_stats)

        # Data quality metrics
        missing_data = self.df[power_cols].isnull().sum()
        zero_values = (self.df[power_cols] == 0).sum()

        print(f"\nData Quality Assessment:")
        print(f"Missing values: {missing_data.sum()}")
        print(f"Zero values: {zero_values.sum()}")
        print(f"Total samples: {len(self.df)}")

        # Signal quality analysis
        sq_stats = self.df["signal_quality"].describe()
        print(f"\nSignal Quality Statistics:")
        print(f"Mean: {sq_stats['mean']:.2f}")
        print(f"Std:  {sq_stats['std']:.2f}")
        print(f"Min:  {sq_stats['min']:.2f}")
        print(f"Max:  {sq_stats['max']:.2f}")

        # Temporal sampling analysis
        time_deltas = self.df["time_delta"].dropna()
        print(f"\nTemporal Sampling Analysis:")
        print(f"Mean sampling interval: {time_deltas.mean():.3f}s")
        print(f"Std sampling interval:  {time_deltas.std():.3f}s")
        print(f"Effective sampling rate: ~{1/time_deltas.mean():.1f} Hz")

        self.stats_summary["basic"] = {
            "descriptive": desc_stats,
            "missing_data": missing_data,
            "zero_values": zero_values,
            "signal_quality": sq_stats,
            "sampling_rate": 1 / time_deltas.mean(),
        }

    def randomness_analysis(self):
        """Assess if data shows non-random patterns suitable for ML."""
        print("\n" + "=" * 60)
        print("RANDOMNESS & ML-READINESS ANALYSIS")
        print("=" * 60)

        power_cols = [col for col in self.df.columns if col.startswith("power_")]

        # Autocorrelation analysis
        print("\nAutocorrelation Analysis (lag=1):")
        autocorr_results = {}

        for col in power_cols:
            series = self.df[col].dropna()
            if len(series) > 1:
                autocorr = series.autocorr(lag=1)
                autocorr_results[col] = autocorr
                print(f"{col:15}: {autocorr:.4f}")

        # Runs test for randomness
        print("\nRuns Test for Randomness (p-values):")
        runs_test_results = {}

        for col in power_cols:
            series = self.df[col].dropna()
            if len(series) > 10:
                median_val = series.median()
                runs = []
                current_run = []

                for val in series:
                    above_median = val > median_val
                    if not current_run or (
                        len(current_run) > 0
                        and (current_run[-1] > median_val) == above_median
                    ):
                        current_run.append(val)
                    else:
                        runs.append(len(current_run))
                        current_run = [val]

                if current_run:
                    runs.append(len(current_run))

                n_runs = len(runs)
                n_pos = sum(series > median_val)
                n_neg = len(series) - n_pos

                if n_pos > 0 and n_neg > 0:
                    expected_runs = (2 * n_pos * n_neg) / (n_pos + n_neg) + 1
                    var_runs = (
                        2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)
                    ) / ((n_pos + n_neg) ** 2 * (n_pos + n_neg - 1))

                    if var_runs > 0:
                        z_score = (n_runs - expected_runs) / np.sqrt(var_runs)
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                        runs_test_results[col] = p_value
                        print(
                            f"{col:15}: {p_value:.4f} {'(Non-random)' if p_value < 0.05 else '(Random)'}"
                        )

        # Variance analysis across time windows
        print("\nTemporal Variance Analysis:")
        window_size = min(100, len(self.df) // 10)  # Adaptive window size

        for col in power_cols:
            series = self.df[col].dropna()
            if len(series) >= window_size * 2:
                windows = [
                    series[i : i + window_size]
                    for i in range(0, len(series) - window_size, window_size)
                ]
                window_vars = [
                    window.var() for window in windows if len(window) == window_size
                ]

                if len(window_vars) > 1:
                    var_stability = np.std(window_vars) / np.mean(window_vars)
                    print(f"{col:15}: Variance stability = {var_stability:.4f}")

        self.stats_summary["randomness"] = {
            "autocorrelations": autocorr_results,
            "runs_test": runs_test_results,
        }

    def correlation_analysis(self):
        """Analyze correlations between EEG power bands."""
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)

        power_cols = [col for col in self.df.columns if col.startswith("power_")]
        power_data = self.df[power_cols]

        # Pearson correlation matrix
        corr_matrix = power_data.corr()
        print("\nPearson Correlation Matrix:")
        print(corr_matrix.round(3))

        # Find highly correlated pairs
        print("\nHighly Correlated Pairs (|r| > 0.7):")
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                    high_corr_pairs.append(pair)
                    print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

        if not high_corr_pairs:
            print("No highly correlated pairs found.")

        # Rolling correlation analysis
        print(f"\nTemporal Correlation Stability (30-sample windows):")
        window_size = min(30, len(self.df) // 10)

        for col1, col2, _ in high_corr_pairs[:3]:  # Top 3 pairs
            rolling_corr = power_data[col1].rolling(window_size).corr(power_data[col2])
            rolling_corr = rolling_corr.dropna()

            if len(rolling_corr) > 0:
                stability = rolling_corr.std()
                print(f"{col1} <-> {col2}: σ = {stability:.4f}")

        self.stats_summary["correlations"] = {
            "matrix": corr_matrix,
            "high_correlations": high_corr_pairs,
        }

    def normalization_analysis(self):
        """Test different normalization strategies."""
        print("\n" + "=" * 60)
        print("NORMALIZATION ANALYSIS")
        print("=" * 60)

        power_cols = [col for col in self.df.columns if col.startswith("power_")]
        power_data = self.df[power_cols].dropna()

        if len(power_data) == 0:
            print("No data available for normalization analysis.")
            return

        # Test different normalization methods
        scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
        }

        normalization_results = {}

        for name, scaler in scalers.items():
            try:
                scaled_data = scaler.fit_transform(power_data)
                scaled_df = pd.DataFrame(scaled_data, columns=power_cols)

                # Analyze normalized distribution
                means = scaled_df.mean()
                stds = scaled_df.std()
                skewness = scaled_df.skew()

                print(f"\n{name} Results:")
                print(f"Mean range: [{means.min():.3f}, {means.max():.3f}]")
                print(f"Std range:  [{stds.min():.3f}, {stds.max():.3f}]")
                print(f"Skewness range: [{skewness.min():.3f}, {skewness.max():.3f}]")

                normalization_results[name] = {
                    "means": means,
                    "stds": stds,
                    "skewness": skewness,
                    "data": scaled_df,
                }

            except Exception as e:
                print(f"Error with {name}: {e}")

        self.stats_summary["normalization"] = normalization_results

    def sliding_window_analysis(self, window_sizes=[10, 30, 60]):
        """Analyze data using sliding windows for tokenization insights."""
        print("\n" + "=" * 60)
        print("SLIDING WINDOW ANALYSIS")
        print("=" * 60)

        power_cols = [col for col in self.df.columns if col.startswith("power_")]

        for window_size in window_sizes:
            if window_size >= len(self.df):
                continue

            print(f"\nWindow Size: {window_size} samples")

            # Create sliding windows
            windows = []
            for i in range(len(self.df) - window_size + 1):
                window = self.df[power_cols].iloc[i : i + window_size]
                windows.append(window)

            # Analyze window statistics
            window_means = [w.mean() for w in windows]
            window_stds = [w.std() for w in windows]

            # Convert to arrays for analysis
            means_array = np.array(window_means)
            stds_array = np.array(window_stds)

            print(f"Number of windows: {len(windows)}")
            print(
                f"Mean stability (std of window means): {means_array.std(axis=0).mean():.4f}"
            )
            print(
                f"Variance stability (std of window stds): {stds_array.std(axis=0).mean():.4f}"
            )

            # Analyze window-to-window transitions
            if len(windows) > 1:
                transitions = []
                for i in range(len(windows) - 1):
                    # Euclidean distance between consecutive windows
                    dist = np.linalg.norm(window_means[i + 1] - window_means[i])
                    transitions.append(dist)

                print(f"Mean transition distance: {np.mean(transitions):.4f}")
                print(f"Transition variability: {np.std(transitions):.4f}")

    def prepare_analysis_data(self):
        """Prepare data structure for report generation."""
        power_cols = [col for col in self.df.columns if col.startswith("power_")]
        
        # Extract time series data
        timestamps = (self.df.index * self.stats_summary.get('basic', {}).get('sampling_rate', 1)).tolist()
        
        # Calculate attention and meditation from brainwave patterns
        # Using standard neurofeedback algorithms
        attention = []
        meditation = []
        
        for idx in range(len(self.df)):
            # Attention: based on beta/theta ratio
            if 'power_low_beta' in self.df.columns and 'power_theta' in self.df.columns:
                beta = self.df['power_low_beta'].iloc[idx]
                theta = self.df['power_theta'].iloc[idx]
                attention_val = min(100, max(0, 50 + 10 * np.log10((beta + 1) / (theta + 1))))
            else:
                attention_val = 50
            attention.append(attention_val)
            
            # Meditation: based on alpha power
            if 'power_low_alpha' in self.df.columns and 'power_high_alpha' in self.df.columns:
                alpha = self.df['power_low_alpha'].iloc[idx] + self.df['power_high_alpha'].iloc[idx]
                meditation_val = min(100, max(0, 30 + 0.01 * np.log10(alpha + 1)))
            else:
                meditation_val = 50
            meditation.append(meditation_val)
        
        # Organize brainwave data
        brainwaves = {}
        for band in self.eeg_bands:
            col_name = f"power_{band}"
            if col_name in self.df.columns:
                brainwaves[band] = self.df[col_name].tolist()
        
        # Calculate statistics
        statistics = {
            'signal_quality': {
                'mean': float(self.df['signal_quality'].mean()),
                'std': float(self.df['signal_quality'].std()),
                'min': float(self.df['signal_quality'].min()),
                'max': float(self.df['signal_quality'].max())
            },
            'attention': {
                'mean': float(np.mean(attention)),
                'std': float(np.std(attention)),
                'min': float(np.min(attention)),
                'max': float(np.max(attention))
            },
            'meditation': {
                'mean': float(np.mean(meditation)),
                'std': float(np.std(meditation)),
                'min': float(np.min(meditation)),
                'max': float(np.max(meditation))
            },
            'brainwaves': {}
        }
        
        for band, values in brainwaves.items():
            statistics['brainwaves'][band] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        self.analysis_data = {
            'timestamps': timestamps,
            'signal_quality': self.df['signal_quality'].tolist(),
            'attention': attention,
            'meditation': meditation,
            'brainwaves': brainwaves,
            'statistics': statistics,
            'duration': timestamps[-1] if timestamps else 0,
            'total_samples': len(self.df),
            'valid_samples': len(self.df[self.df['signal_quality'] < 200]),
            'invalid_samples': len(self.df[self.df['signal_quality'] >= 200]),
            'start_time': self.df['timestamp'].iloc[0].isoformat() if len(self.df) > 0 else None,
            'end_time': self.df['timestamp'].iloc[-1].isoformat() if len(self.df) > 0 else None
        }
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        power_cols = [col for col in self.df.columns if col.startswith("power_")]

        # Set up the plotting style
        plt.style.use("default")
        fig = plt.figure(figsize=(20, 15))

        # 1. Time series plot
        plt.subplot(3, 3, 1)
        for col in power_cols[:4]:  # First 4 bands
            plt.plot(
                self.df.index, self.df[col], alpha=0.7, label=col.replace("power_", "")
            )
        plt.title("EEG Power Bands Over Time (First 4)")
        plt.xlabel("Sample Index")
        plt.ylabel("Power (relative units)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Distribution plots
        plt.subplot(3, 3, 2)
        self.df[power_cols].boxplot(ax=plt.gca())
        plt.title("Power Band Distributions")
        plt.xticks(rotation=45)
        plt.ylabel("Power (relative units)")

        # 3. Correlation heatmap
        plt.subplot(3, 3, 3)
        corr_matrix = self.df[power_cols].corr()
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", center=0, square=True, fmt=".2f"
        )
        plt.title("Power Band Correlations")

        # 4. Signal quality over time
        plt.subplot(3, 3, 4)
        plt.plot(self.df.index, self.df["signal_quality"], color="red", alpha=0.7)
        plt.title("Signal Quality Over Time")
        plt.xlabel("Sample Index")
        plt.ylabel("Signal Quality")
        plt.grid(True, alpha=0.3)

        # 5. Power ratios (important for attention detection)
        plt.subplot(3, 3, 5)
        if "power_theta" in self.df.columns and "power_low_beta" in self.df.columns:
            theta_beta_ratio = self.df["power_theta"] / (
                self.df["power_low_beta"] + 1e-8
            )
            plt.plot(self.df.index, theta_beta_ratio, color="purple", alpha=0.7)
            plt.title("Theta/Beta Ratio (Attention Marker)")
            plt.xlabel("Sample Index")
            plt.ylabel("Theta/Beta Ratio")
            plt.grid(True, alpha=0.3)

        # 6. Alpha suppression analysis
        plt.subplot(3, 3, 6)
        if (
            "power_low_alpha" in self.df.columns
            and "power_high_alpha" in self.df.columns
        ):
            total_alpha = self.df["power_low_alpha"] + self.df["power_high_alpha"]
            plt.plot(self.df.index, total_alpha, color="green", alpha=0.7)
            plt.title("Total Alpha Power (8-12Hz)")
            plt.xlabel("Sample Index")
            plt.ylabel("Alpha Power")
            plt.grid(True, alpha=0.3)

        # 7. Sampling interval analysis
        plt.subplot(3, 3, 7)
        time_deltas = self.df["time_delta"].dropna()
        plt.hist(time_deltas, bins=50, alpha=0.7, color="orange")
        plt.title("Sampling Interval Distribution")
        plt.xlabel("Time Delta (seconds)")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # 8. PCA analysis
        plt.subplot(3, 3, 8)
        try:
            power_data = self.df[power_cols].dropna()
            if len(power_data) > 0:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(power_data)

                pca = PCA()
                pca_result = pca.fit_transform(scaled_data)

                plt.plot(
                    range(1, len(pca.explained_variance_ratio_) + 1),
                    np.cumsum(pca.explained_variance_ratio_),
                    "bo-",
                )
                plt.title("PCA Explained Variance")
                plt.xlabel("Number of Components")
                plt.ylabel("Cumulative Explained Variance")
                plt.grid(True, alpha=0.3)
        except Exception as e:
            plt.text(0.5, 0.5, f"PCA Error: {str(e)}", ha="center", va="center")
            plt.title("PCA Analysis (Error)")

        # 9. Band power relationships
        plt.subplot(3, 3, 9)
        if len(power_cols) >= 2:
            plt.scatter(self.df[power_cols[0]], self.df[power_cols[1]], alpha=0.5)
            plt.xlabel(power_cols[0])
            plt.ylabel(power_cols[1])
            plt.title(f"{power_cols[0]} vs {power_cols[1]}")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("eeg_analysis_plots.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("✓ Visualizations saved as 'eeg_analysis_plots.png'")
        
        # Generate reports if analysis data is prepared
        if self.analysis_data:
            run_name, run_path = self.report_generator.generate_report(self.analysis_data)
            print(f"✓ Reports generated in: {run_path}")
            print(f"  - Markdown: {run_path}/report.md")
            print(f"  - HTML: {run_path}/report.html")
            print(f"  - JSON: {run_path}/report.json")

    def ml_readiness_report(self):
        """Generate ML readiness assessment."""
        print("\n" + "=" * 80)
        print("ML READINESS ASSESSMENT REPORT")
        print("=" * 80)

        # Data volume assessment
        print(f"\n📊 Data Volume:")
        print(f"   Total samples: {len(self.df)}")
        print(
            f"   Duration: ~{len(self.df) * self.stats_summary.get('basic', {}).get('sampling_rate', 1):.1f} seconds"
        )

        # Signal quality assessment
        sq_mean = self.df["signal_quality"].mean()
        sq_good_pct = (self.df["signal_quality"] > 25).mean() * 100

        print(f"\n📡 Signal Quality:")
        print(f"   Mean quality: {sq_mean:.1f}/100")
        print(f"   Good quality samples (>25): {sq_good_pct:.1f}%")

        # Non-randomness assessment
        autocorr_results = self.stats_summary.get("randomness", {}).get(
            "autocorrelations", {}
        )
        strong_autocorr = sum(1 for r in autocorr_results.values() if abs(r) > 0.3)

        print(f"\n🎯 Pattern Detection:")
        print(
            f"   Bands with strong autocorrelation (>0.3): {strong_autocorr}/{len(autocorr_results)}"
        )

        runs_results = self.stats_summary.get("randomness", {}).get("runs_test", {})
        non_random = sum(1 for p in runs_results.values() if p < 0.05)

        print(f"   Bands showing non-random patterns: {non_random}/{len(runs_results)}")

        # Feature correlation assessment
        high_corr = len(
            self.stats_summary.get("correlations", {}).get("high_correlations", [])
        )
        print(f"   Highly correlated band pairs: {high_corr}")

        # Overall ML readiness score
        scores = []

        # Volume score
        if len(self.df) > 1000:
            scores.append(1.0)
        elif len(self.df) > 500:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # Quality score
        if sq_good_pct > 80:
            scores.append(1.0)
        elif sq_good_pct > 60:
            scores.append(0.7)
        else:
            scores.append(0.4)

        # Pattern score
        pattern_ratio = (strong_autocorr + non_random) / (
            len(autocorr_results) + len(runs_results) + 1e-8
        )
        if pattern_ratio > 0.5:
            scores.append(1.0)
        elif pattern_ratio > 0.3:
            scores.append(0.7)
        else:
            scores.append(0.4)

        overall_score = np.mean(scores) * 100

        print(f"\n🎖️  Overall ML Readiness Score: {overall_score:.1f}/100")

        if overall_score >= 80:
            print("   ✅ EXCELLENT - Data is ready for ML training")
        elif overall_score >= 60:
            print("   ⚠️  GOOD - Data is suitable with some preprocessing")
        else:
            print("   ❌ POOR - Consider collecting more/better data")

        # Recommendations
        print(f"\n💡 Recommendations:")

        if len(self.df) < 1000:
            print("   • Collect more data samples for robust training")

        if sq_good_pct < 70:
            print("   • Improve signal quality by reducing artifacts")

        if strong_autocorr < len(autocorr_results) * 0.5:
            print("   • Check for temporal structure in your experimental setup")

        if high_corr > len(self.eeg_bands) * 0.6:
            print("   • Consider dimensionality reduction due to high correlations")

        print("   • Use sliding windows (30-60 samples) for feature extraction")
        print("   • Apply robust normalization before training")
        print("   • Consider theta/beta ratio and alpha suppression as key features")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("🧠 EEG DATA STATISTICAL ANALYSIS TOOL")
        print("🎯 Optimized for Attention Detection ML Training")
        print("=" * 80)

        try:
            self.load_data()
            self.basic_statistics()
            self.randomness_analysis()
            self.correlation_analysis()
            self.normalization_analysis()
            self.sliding_window_analysis()
            self.prepare_analysis_data()  # Prepare data for reports
            self.generate_visualizations()
            self.ml_readiness_report()

            print(f"\n✅ Analysis complete! Results saved in analyzer.stats_summary")

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise


# Usage example
if __name__ == "__main__":

    datadir="./logs"

    # Initialize analyzer with your data files
    analyzer = EEGDataAnalyzer(
        [   f"{datadir}/xaa.json",
            f"{datadir}/xab.json",
            f"{datadir}/xac.json",
            f"{datadir}/xad.json",
            f"{datadir}/xae.json",
        ]
    )

    # Run complete analysis
    analyzer.run_complete_analysis()

    # Access specific results
    # print(analyzer.stats_summary['basic'])
    # print(analyzer.stats_summary['correlations']['high_correlations'])
