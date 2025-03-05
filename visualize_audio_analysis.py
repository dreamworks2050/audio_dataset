import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Output directories
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_processing_times():
    """Load the processing times data"""
    with open("audio_ai_optimized/processing_times.json", "r") as f:
        return json.load(f)

def load_summary_files():
    """Load all summary files from the analysis directories"""
    summary_files = glob.glob("audio_ai_optimized/analysis/*/summary.json")
    summaries = []
    for file_path in summary_files:
        try:
            with open(file_path, "r") as f:
                summary = json.load(f)
                # Extract combination from the filepath if not in the file
                if "combination" not in summary:
                    combination = re.search(r"analysis/(chunk\d+s_overlap\d+s)", file_path)
                    if combination:
                        summary["combination"] = combination.group(1)
                summaries.append(summary)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return summaries

def calculate_average_scores(summaries):
    """Calculate average scores for each parameter combination"""
    scores_data = []
    for summary in summaries:
        combination = summary.get("combination", "")
        if not combination:
            continue
            
        match = re.match(r"chunk(\d+)s_overlap(\d+)s", combination)
        if not match:
            continue
            
        chunk_length = int(match.group(1))
        overlap_length = int(match.group(2))
        
        # Extract scores from detailed results if available
        avg_score = summary.get("overall_average_score", 0)
        
        # Extract metrics that are present
        metrics = {}
        if "results" in summary:
            for chunk_result in summary["results"]:
                if "analysis_result" in chunk_result:
                    analysis = chunk_result["analysis_result"]
                    for key, value in analysis.items():
                        if isinstance(value, (int, float)) and key != "step_number":
                            metrics[key] = metrics.get(key, 0) + value
            
            # Calculate averages
            chunk_count = len(summary["results"])
            if chunk_count > 0:
                for key in metrics:
                    metrics[key] = metrics[key] / chunk_count
        
        scores_data.append({
            "chunk_length": chunk_length,
            "overlap_length": overlap_length, 
            "combination": combination,
            "average_score": avg_score,
            **metrics
        })
    
    return pd.DataFrame(scores_data)

def plot_processing_times(processing_times):
    """Generate plots for processing times by chunk length and overlap length"""
    # Extract data for plotting
    chunk_lengths = []
    chunk_avg_times = []
    for length, data in processing_times["chunk_length"].items():
        chunk_lengths.append(int(length))
        chunk_avg_times.append(data["average"])
    
    overlap_lengths = []
    overlap_avg_times = []
    for length, data in processing_times["overlap_length"].items():
        overlap_lengths.append(int(length))
        overlap_avg_times.append(data["average"])
    
    # Sort by length
    chunk_data = sorted(zip(chunk_lengths, chunk_avg_times))
    chunk_lengths, chunk_avg_times = zip(*chunk_data)
    
    overlap_data = sorted(zip(overlap_lengths, overlap_avg_times))
    overlap_lengths, overlap_avg_times = zip(*overlap_data)
    
    # Create figure with two subplots
    plt.figure(figsize=(16, 7))
    
    # Plot chunk length processing times
    plt.subplot(1, 2, 1)
    plt.bar(chunk_lengths, chunk_avg_times, color='skyblue')
    plt.xlabel('Chunk Length (s)')
    plt.ylabel('Average Processing Time (s)')
    plt.title('Processing Time by Chunk Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot overlap length processing times
    plt.subplot(1, 2, 2)
    plt.bar(overlap_lengths, overlap_avg_times, color='lightgreen')
    plt.xlabel('Overlap Length (s)')
    plt.ylabel('Average Processing Time (s)')
    plt.title('Processing Time by Overlap Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/processing_times.png", dpi=300)
    plt.close()

def plot_parameter_heatmap(scores_df):
    """Create a heatmap of average scores for different parameter combinations"""
    if scores_df.empty:
        print("No score data available for heatmap")
        return
    
    # Pivot the data for the heatmap
    pivot_data = scores_df.pivot_table(
        index='chunk_length', 
        columns='overlap_length',
        values='average_score',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, cmap="viridis", 
                fmt=".2f", linewidths=.5, cbar_kws={'label': 'Average Score'})
    plt.title('Average Scores by Chunk Length and Overlap')
    plt.ylabel('Chunk Length (s)')
    plt.xlabel('Overlap Length (s)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/parameter_heatmap.png", dpi=300)
    plt.close()

def plot_metric_comparison(scores_df):
    """Create visualization for various metrics comparison"""
    if scores_df.empty:
        print("No score data available for metrics comparison")
        return
    
    # Identify metrics columns (skip non-metric columns)
    non_metrics = ['chunk_length', 'overlap_length', 'combination', 'average_score']
    metric_columns = [col for col in scores_df.columns if col not in non_metrics]
    
    if not metric_columns:
        print("No metric columns found for analysis")
        return
    
    # Calculate average for each metric
    metrics_avg = {}
    for metric in metric_columns:
        metrics_avg[metric] = scores_df[metric].mean()
    
    # Plot metrics comparison
    plt.figure(figsize=(14, 8))
    metrics = list(metrics_avg.keys())
    values = list(metrics_avg.values())
    
    # Sort for better visualization
    sorted_data = sorted(zip(metrics, values), key=lambda x: x[1])
    metrics, values = zip(*sorted_data)
    
    # Create more readable metric names
    readable_metrics = [m.replace('_score', '').replace('_', ' ').title() for m in metrics]
    
    bars = plt.barh(readable_metrics, values, color=plt.cm.viridis(np.linspace(0, 0.8, len(metrics))))
    plt.xlabel('Average Score')
    plt.title('Comparison of Evaluation Metrics')
    plt.xlim(0, 10)  # Assuming scores are on a 0-10 scale
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metrics_comparison.png", dpi=300)
    plt.close()

def plot_best_combinations(scores_df, n=5):
    """Plot the top N parameter combinations by average score"""
    if scores_df.empty:
        print("No score data available for best combinations")
        return
    
    # Get top combinations
    top_combinations = scores_df.sort_values('average_score', ascending=False).head(n)
    
    plt.figure(figsize=(14, 7))
    
    # Create a bar chart
    bars = plt.bar(
        top_combinations['combination'], 
        top_combinations['average_score'],
        color=plt.cm.viridis(np.linspace(0, 0.8, len(top_combinations)))
    )
    
    plt.ylabel('Average Score')
    plt.xlabel('Parameter Combination')
    plt.title(f'Top {n} Parameter Combinations by Average Score')
    plt.ylim(0, 10)  # Assuming scores are on a 0-10 scale
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                 f'{bar.get_height():.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_combinations.png", dpi=300)
    plt.close()

def plot_chunk_vs_overlap_scatter(scores_df):
    """Create a scatter plot showing the relationship between chunk length, overlap, and scores"""
    if scores_df.empty:
        print("No score data available for scatter plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Calculate overlap percentage for color mapping
    scores_df['overlap_percent'] = (scores_df['overlap_length'] / scores_df['chunk_length']) * 100
    
    scatter = plt.scatter(
        scores_df['chunk_length'], 
        scores_df['average_score'],
        c=scores_df['overlap_percent'], 
        s=100,  # Size of points
        cmap='viridis',
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Overlap Percentage (%)')
    
    # Add text labels
    for idx, row in scores_df.iterrows():
        plt.text(row['chunk_length'], row['average_score'], 
                 f"{row['overlap_length']}s", 
                 fontsize=8)
    
    plt.xlabel('Chunk Length (s)')
    plt.ylabel('Average Score')
    plt.title('Scores by Chunk Length and Overlap')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/chunk_overlap_scatter.png", dpi=300)
    plt.close()

def plot_category_scores_by_parameter_set(scores_df):
    """
    Create a line plot comparing scores across all categories for each parameter set
    Each line represents a parameter set (combination of chunk length and overlap)
    """
    if scores_df.empty:
        print("No score data available for category scores comparison")
        return
    
    # Identify metrics columns (skip non-metric columns)
    non_metrics = ['chunk_length', 'overlap_length', 'combination', 'average_score']
    metric_columns = [col for col in scores_df.columns if col not in non_metrics]
    
    if not metric_columns:
        print("No metric columns found for analysis")
        return
    
    # Create more readable metric names for x-axis
    readable_metrics = [m.replace('_score', '').replace('_', ' ').title() for m in metric_columns]
    
    # Create the figure
    plt.figure(figsize=(14, 8))
    
    # Plot a line for each parameter set
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(scores_df)))
    
    legend_items = []
    
    for idx, (_, row) in enumerate(scores_df.iterrows()):
        combination = row['combination']
        values = [row[metric] for metric in metric_columns]
        line, = plt.plot(readable_metrics, values, marker=markers[idx % len(markers)], 
                         color=colors[idx], linewidth=2, markersize=8)
        legend_items.append(line)
    
    plt.legend(legend_items, scores_df['combination'], title="Parameter Sets", 
               loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    plt.ylabel('Score (0-10)')
    plt.title('Comparison of Category Scores by Parameter Set')
    plt.ylim(0, 10)  # Set y-axis from 0 to 10
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Make x-axis labels more readable
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/category_scores_comparison.png", dpi=300)
    plt.close()

def generate_html_report(processing_times, scores_df):
    """Generate a comprehensive HTML report with all visualizations"""
    # Create report data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate overall statistics
    overall_avg_score = scores_df['average_score'].mean() if not scores_df.empty else 0
    best_combination = scores_df.loc[scores_df['average_score'].idxmax()]['combination'] if not scores_df.empty else "N/A"
    best_score = scores_df['average_score'].max() if not scores_df.empty else 0
    
    # Get top 3 combinations
    if not scores_df.empty:
        top_combinations = scores_df.sort_values('average_score', ascending=False).head(3)
        top_combinations_html = ""
        for idx, row in top_combinations.iterrows():
            top_combinations_html += f"""
            <div class="score-box score-high">
                <h4>{row['combination']}</h4>
                <div style="font-size: 18px;">{row['average_score']:.2f}</div>
            </div>
            """
    else:
        top_combinations_html = "<p>No data available</p>"
    
    # HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Analysis Optimization Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart-container {{ margin: 20px 0; text-align: center; }}
            .chart-container img {{ max-width: 100%; height: auto; }}
            .summary-card {{ 
                border: 1px solid #ddd; 
                border-radius: 5px; 
                padding: 15px; 
                margin-bottom: 20px;
                background-color: #f9f9f9;
            }}
            .score-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .score-box {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                min-width: 120px;
                text-align: center;
            }}
            .score-high {{ background-color: #d4edda; }}
            .score-medium {{ background-color: #fff3cd; }}
            .score-low {{ background-color: #f8d7da; }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-card {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                background-color: #f9f9f9;
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
                color: #333;
            }}
            .stat-title {{
                font-size: 14px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Audio Analysis Optimization Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="summary-card">
                <h2>Summary</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-title">Average Score Across All Tests</div>
                        <div class="stat-value">{overall_avg_score:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-title">Best Parameter Combination</div>
                        <div class="stat-value">{best_combination}</div>
                        <div>{best_score:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-title">Total Parameter Combinations</div>
                        <div class="stat-value">{len(scores_df) if not scores_df.empty else 0}</div>
                    </div>
                </div>
                
                <h3>Top Performing Combinations</h3>
                <div class="score-container">
                    {top_combinations_html}
                </div>
            </div>
            
            <h2>AI Analysis Results</h2>
            <div class="chart-container">
                <img src="category_scores_comparison.png" alt="Category Scores by Parameter Set">
            </div>
            
            <h2>Processing Time Analysis</h2>
            <div class="chart-container">
                <img src="processing_times.png" alt="Processing Times">
            </div>
            
            <h2>Parameter Performance Analysis</h2>
            <div class="chart-container">
                <img src="parameter_heatmap.png" alt="Parameter Performance Heatmap">
            </div>
            
            <h2>Top Parameter Combinations</h2>
            <div class="chart-container">
                <img src="top_combinations.png" alt="Top Parameter Combinations">
            </div>
            
            <h2>Chunk vs Overlap Analysis</h2>
            <div class="chart-container">
                <img src="chunk_overlap_scatter.png" alt="Chunk vs Overlap Analysis">
            </div>
            
            <h2>Metrics Comparison</h2>
            <div class="chart-container">
                <img src="metrics_comparison.png" alt="Metrics Comparison">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML report
    with open(f"{OUTPUT_DIR}/comprehensive_report.html", "w") as f:
        f.write(html_content)

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    processing_times = load_processing_times()
    summaries = load_summary_files()
    scores_df = calculate_average_scores(summaries)
    
    # Generate visualizations
    plot_processing_times(processing_times)
    plot_parameter_heatmap(scores_df)
    plot_metric_comparison(scores_df)
    plot_best_combinations(scores_df)
    plot_chunk_vs_overlap_scatter(scores_df)
    plot_category_scores_by_parameter_set(scores_df)
    
    # Generate HTML report
    generate_html_report(processing_times, scores_df)
    
    print(f"Visualizations generated in {OUTPUT_DIR}/ directory")

if __name__ == "__main__":
    main() 