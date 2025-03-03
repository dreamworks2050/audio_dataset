#!/usr/bin/env python3
"""
Example script demonstrating how to use the enhanced visualization capabilities
to aggregate results across multiple batches of transcription analysis.
"""

import os
import argparse
import sys
from ai_optimize.visualize_results import (
    load_multiple_batch_results,
    generate_aggregate_visualizations,
    generate_comprehensive_report
)

def visualize_multiple_batches(base_dir, batch_pattern="*batch*"):
    """
    Load results from multiple batch directories and generate aggregate visualizations.
    
    Args:
        base_dir: Base directory containing batch folders
        batch_pattern: Glob pattern to match batch directories
    """
    print(f"Loading results from multiple batches in {base_dir}")
    print(f"Using batch pattern: {batch_pattern}")
    
    # Load results from all matching batch directories
    results, summary = load_multiple_batch_results(base_dir, batch_pattern)
    
    if not results:
        print("No results found. Please check your directory structure and batch pattern.")
        return False
    
    # Generate aggregate visualizations
    generate_aggregate_visualizations(results, base_dir)
    
    # Generate comprehensive HTML report
    generate_comprehensive_report(results, summary, base_dir)
    
    print("\nVisualization complete!")
    print(f"Aggregate visualizations saved to: {os.path.join(base_dir, 'aggregate_visualizations')}")
    print(f"Comprehensive report: {os.path.join(base_dir, 'aggregate_visualizations', 'comprehensive_report.html')}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Visualize and aggregate results from multiple transcription analysis batches")
    parser.add_argument("base_dir", help="Base directory containing batch results directories")
    parser.add_argument("--pattern", default="*batch*", help="Glob pattern to match batch directories (default: '*batch*')")
    args = parser.parse_args()
    
    # Verify the base directory exists
    if not os.path.isdir(args.base_dir):
        print(f"Error: Directory not found: {args.base_dir}")
        return 1
    
    success = visualize_multiple_batches(args.base_dir, args.pattern)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 