#!/usr/bin/env python3
"""
Example script showing how to use the enhanced visualization tools
to generate charts and reports from transcription analysis results.
"""

import sys
import os
from ai_optimize.visualize_results import load_all_grading_results, generate_comparison_charts

def visualize_results(run_dir=None):
    """
    Visualize results from a specified run directory
    
    Args:
        run_dir: Path to the run directory (default: example_output)
    """
    # Use example_output if no directory specified
    if run_dir is None:
        run_dir = "example_output"
    
    if not os.path.exists(run_dir):
        print(f"Error: Directory '{run_dir}' does not exist")
        return
    
    print(f"Loading results from: {run_dir}")
    results = load_all_grading_results(run_dir)
    
    if not results:
        print("No results found to visualize")
        return
    
    # Create visualizations output directory
    output_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating charts in: {output_dir}")
    generate_comparison_charts(results, output_dir)
    
    print("\nVisualization completed!")
    print(f"HTML report: {os.path.join(output_dir, 'report.html')}")
    print(f"Charts generated: {len(os.listdir(output_dir))} files")
    
    # List the charts created
    print("\nGenerated files:")
    for file in sorted(os.listdir(output_dir)):
        print(f"- {file}")

def main():
    """Main function to handle command-line arguments"""
    # Get directory from command line if provided
    run_dir = None
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    
    print("=== TranscriptionAnalyzer Visualization Tool ===")
    visualize_results(run_dir)
    print("=== Visualization Complete ===")

if __name__ == "__main__":
    main() 