#!/usr/bin/env python3
"""
AddAI CLI - Command Line Interface for EEG Analysis Tools
Provides a unified interface for running analysis scripts and managing reports
"""

import os
import sys
import json
import random
import subprocess
from pathlib import Path
from datetime import datetime

# Funkybob name generator - lists of adjectives and scientists
ADJECTIVES = [
    "agile", "bold", "clever", "daring", "eager", "fierce", "gentle", "happy",
    "jolly", "keen", "lively", "merry", "noble", "proud", "quiet", "serene",
    "swift", "trusty", "vibrant", "wise", "zealous", "bright", "cosmic", "dynamic",
    "electric", "fluid", "graceful", "harmonic", "infinite", "jovial", "kinetic",
    "luminous", "mystical", "nimble", "organic", "peaceful", "quantum", "radiant",
    "stellar", "tranquil", "unified", "vivid", "whimsical", "xenial", "youthful"
]

SCIENTISTS = [
    "curie", "einstein", "newton", "darwin", "galilei", "tesla", "pasteur", "mendel",
    "faraday", "maxwell", "bohr", "planck", "schrodinger", "heisenberg", "hawking",
    "turing", "lovelace", "franklin", "watson", "crick", "goodall", "carson",
    "mcclintock", "hopper", "johnson", "jemison", "ride", "tharp", "blackwell",
    "leavitt", "meitner", "noether", "hypatia", "agnesi", "germain", "somerville",
    "kovalevskaya", "nightingale", "hamilton", "babbage", "volta", "ampere", "ohm",
    "joule", "kelvin", "celsius", "fahrenheit", "pascal", "watt", "hertz", "bell",
    "morse", "edison", "marconi", "fermi", "oppenheimer", "feynman", "dirac",
    "rutherford", "boyle", "lavoisier", "dalton", "avogadro", "mendeleev", "nobel",
    "bunsen", "arrhenius", "pauling", "watson", "wilkins", "rosalind", "barbara",
    "dorothy", "marie", "pierre", "irene", "joliot", "becquerel", "roentgen",
    "bragg", "compton", "millikan", "michelson", "morley", "fizeau", "foucault",
    "doppler", "mach", "wien", "stefan", "boltzmann", "carnot", "clausius", "gibbs",
    "maxwell", "lorentz", "poincare", "minkowski", "schwarzschild", "chandrasekhar",
    "hubble", "lemaitre", "gamow", "hoyle", "sagan", "drake", "shoemaker", "hale",
    "yerkes", "palomar", "keck", "hubble", "spitzer", "kepler", "galileo", "cassini",
    "voyager", "pioneer", "mariner", "viking", "pathfinder", "opportunity", "spirit",
    "curiosity", "perseverance", "ingenuity", "juno", "new", "horizons", "rosetta",
    "philae", "hayabusa", "osiris", "rex", "parker", "solar", "probe", "james",
    "webb", "space", "telescope", "allen", "heyrovsky", "kossel", "raman"
]


def generate_funkybob_name():
    """Generate a funkybob name using adjective + scientist pattern"""
    adjective = random.choice(ADJECTIVES)
    scientist = random.choice(SCIENTISTS)
    return f"{adjective}_{scientist}"


class AddAICLI:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.analysis_dir = self.base_dir / "analysis"
        self.reports_dir = self.base_dir / "reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
    
    def list_logs(self):
        """List available log files"""
        logs_dir = self.analysis_dir / "logs"
        if not logs_dir.exists():
            print("No logs directory found")
            return []
        
        log_files = []
        for root, dirs, files in os.walk(logs_dir):
            for file in files:
                if file.endswith('.json'):
                    rel_path = Path(root) / file
                    log_files.append(str(rel_path))
        
        return sorted(log_files)
    
    def list_reports(self):
        """List existing reports"""
        if not self.reports_dir.exists():
            return []
        
        reports = []
        for item in self.reports_dir.iterdir():
            if item.is_dir():
                reports.append(item.name)
        
        return sorted(reports)
    
    def run_analysis(self, log_file, report_name=None):
        """Run full analysis pipeline"""
        if not report_name:
            report_name = generate_funkybob_name()
        
        # Ensure log file path is absolute
        if not Path(log_file).is_absolute():
            log_file = str(self.analysis_dir / "logs" / log_file)
        
        if not Path(log_file).exists():
            print(f"Error: Log file '{log_file}' not found")
            return False
        
        print(f"Running analysis with report name: {report_name}")
        
        # Run the analysis script
        run_analysis_script = self.analysis_dir / "run_analysis.py"
        cmd = [sys.executable, str(run_analysis_script), log_file, "--name", report_name]
        
        try:
            # Update the run_analysis.py to output to reports directory
            result = subprocess.run(cmd, cwd=str(self.base_dir), check=True, 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Analysis failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False
    
    def run_bucketing(self, log_file, output_name=None):
        """Run just the bucketing analysis"""
        if not output_name:
            output_name = generate_funkybob_name()
        
        if not Path(log_file).is_absolute():
            log_file = str(self.analysis_dir / "logs" / log_file)
        
        if not Path(log_file).exists():
            print(f"Error: Log file '{log_file}' not found")
            return False
        
        print(f"Running bucketing analysis with output name: {output_name}")
        
        bucketing_script = self.analysis_dir / "eeg_bucketing.py"
        cmd = [sys.executable, str(bucketing_script), log_file]
        
        try:
            result = subprocess.run(cmd, cwd=str(self.analysis_dir), check=True,
                                  capture_output=True, text=True)
            
            # Move output files to reports directory
            output_dir = self.reports_dir / output_name
            output_dir.mkdir(exist_ok=True)
            
            for file in ["eeg_tokens.json", "eeg_regions.json"]:
                src = self.analysis_dir / file
                if src.exists():
                    src.rename(output_dir / file)
                    print(f"Moved {file} to {output_dir}")
            
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Bucketing analysis failed: {e}")
            return False
    
    def clean_reports(self):
        """Clean up empty or invalid report directories"""
        if not self.reports_dir.exists():
            return
        
        cleaned = 0
        for item in self.reports_dir.iterdir():
            if item.is_dir():
                # Check if directory is empty or only contains README
                files = list(item.iterdir())
                if not files or (len(files) == 1 and files[0].name == "README.md"):
                    print(f"Removing empty report directory: {item.name}")
                    if files:  # Remove README if it exists
                        files[0].unlink()
                    item.rmdir()
                    cleaned += 1
        
        print(f"Cleaned {cleaned} empty report directories")
    
    def show_help(self):
        """Show help information"""
        help_text = """
AddAI CLI - EEG Analysis Tools

Available commands:
  list-logs          List available log files
  list-reports       List existing reports
  analyze <logfile>  Run full analysis pipeline
  bucket <logfile>   Run bucketing analysis only
  clean              Clean up empty report directories
  help               Show this help message
  exit               Exit the CLI

Log files can be specified as:
  - Full path: /path/to/file.json
  - Relative to logs: brad/1749331786.json
  - Just filename: 1749331786.json (searches in logs/)

Report names are automatically generated using funkybob naming.
"""
        print(help_text)
    
    def run_interactive(self):
        """Run interactive CLI loop"""
        print("Welcome to AddAI CLI")
        print("Type 'help' for available commands, 'exit' to quit")
        
        while True:
            try:
                user_input = input("addai> ").strip()
                if not user_input:
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                elif command in ['help', 'h']:
                    self.show_help()
                elif command in ['list-logs', 'll']:
                    logs = self.list_logs()
                    if logs:
                        print("Available log files:")
                        for log in logs:
                            print(f"  {log}")
                    else:
                        print("No log files found")
                elif command in ['list-reports', 'lr']:
                    reports = self.list_reports()
                    if reports:
                        print("Existing reports:")
                        for report in reports:
                            print(f"  {report}")
                    else:
                        print("No reports found")
                elif command in ['analyze', 'a']:
                    if len(parts) < 2:
                        print("Usage: analyze <logfile>")
                        continue
                    self.run_analysis(parts[1])
                elif command in ['bucket', 'b']:
                    if len(parts) < 2:
                        print("Usage: bucket <logfile>")
                        continue
                    self.run_bucketing(parts[1])
                elif command in ['clean', 'c']:
                    self.clean_reports()
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break


def main():
    """Main entry point"""
    cli = AddAICLI()
    
    if len(sys.argv) > 1:
        # Non-interactive mode
        command = sys.argv[1].lower()
        if command == 'help':
            cli.show_help()
        elif command == 'list-logs':
            logs = cli.list_logs()
            for log in logs:
                print(log)
        elif command == 'list-reports':
            reports = cli.list_reports()
            for report in reports:
                print(report)
        elif command == 'analyze' and len(sys.argv) > 2:
            cli.run_analysis(sys.argv[2])
        elif command == 'bucket' and len(sys.argv) > 2:
            cli.run_bucketing(sys.argv[2])
        elif command == 'clean':
            cli.clean_reports()
        else:
            print("Invalid command or missing arguments")
            cli.show_help()
    else:
        # Interactive mode
        cli.run_interactive()


if __name__ == "__main__":
    main()