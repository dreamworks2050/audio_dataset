import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import glob

def load_all_grading_results(run_dir: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load all grading results from a run directory
    
    Args:
        run_dir: Path to the run directory containing analysis results
        
    Returns:
        Tuple of (results, summary) where:
            results: Dictionary mapping parameter combinations to DataFrames
            summary: Summary DataFrame with metrics by combination
    """
    print(f"Processing single parameter directory: {os.path.basename(run_dir)}")
    
    # Check if we have an analysis directory within the run_dir
    analysis_dir = os.path.join(run_dir, "analysis")
    if os.path.exists(analysis_dir):
        # Look for raw_responses/example_parameters/all_grading_results.json first
        example_results_path = os.path.join(analysis_dir, "raw_responses", "example_parameters", "all_grading_results.json")
        if os.path.exists(example_results_path):
            print(f"Found grading results at: {example_results_path}")
            try:
                with open(example_results_path, 'r') as f:
                    grading_data = json.load(f)
                
                all_data = []
                for item in grading_data:
                    all_data.append(item)
                
                if all_data:
                    df = pd.DataFrame(all_data)
                    results = {"example_parameters": df}
                    
                    # Create a summary DataFrame
                    metrics = ["verbatim_match_score", "sentence_preservation_score", "content_duplication_score", 
                              "content_loss_score", "join_transition_score", "contextual_flow_score", 
                              "average_score"]
                    
                    # Only include metrics that exist in the DataFrame
                    available_metrics = [m for m in metrics if m in df.columns]
                    
                    summary_data = []
                    summary_data.append({
                        "parameter_set": "example_parameters",
                        "file_count": len(df),
                        **{metric: df[metric].mean() for metric in available_metrics if metric in df.columns}
                    })
                    
                    summary_df = pd.DataFrame(summary_data)
                    
                    # Print summary with readable names
                    print(f"Loaded {len(df)} results from {os.path.basename(run_dir)}")
                    print("Average metrics:")
                    for metric in available_metrics:
                        print(f"  - {get_readable_metric_name(metric)}: {df[metric].mean():.2f}")
                    
                    return results, summary_df
            except Exception as e:
                print(f"Error loading grading results: {e}")
        
        # Fall back to other locations
        raw_responses_dir = os.path.join(analysis_dir, "raw_responses")
        if os.path.exists(raw_responses_dir):
            all_results_path = os.path.join(raw_responses_dir, "all_grading_results.json")
            if os.path.exists(all_results_path):
                print(f"Found grading results at: {all_results_path}")
                try:
                    with open(all_results_path, 'r') as f:
                        grading_data = json.load(f)
                    
                    all_data = []
                    for item in grading_data:
                        all_data.append(item)
                    
                    if all_data:
                        df = pd.DataFrame(all_data)
                        results = {"default": df}
                        
                        # Create a summary DataFrame
                        metrics = ["verbatim_match_score", "sentence_preservation_score", "content_duplication_score", 
                                  "content_loss_score", "join_transition_score", "contextual_flow_score", 
                                  "average_score"]
                        
                        # Only include metrics that exist in the DataFrame
                        available_metrics = [m for m in metrics if m in df.columns]
                        
                        summary_data = []
                        summary_data.append({
                            "parameter_set": "default",
                            "file_count": len(df),
                            **{metric: df[metric].mean() for metric in available_metrics if metric in df.columns}
                        })
                        
                        summary_df = pd.DataFrame(summary_data)
                        
                        # Print summary with readable names
                        print(f"Loaded {len(df)} results from {os.path.basename(run_dir)}")
                        print("Average metrics:")
                        for metric in available_metrics:
                            print(f"  - {get_readable_metric_name(metric)}: {df[metric].mean():.2f}")
                            
                        return results, summary_df
                except Exception as e:
                    print(f"Error loading grading results: {e}")
    
    # Look for combination directories
    combo_dirs = []
    parameters_dir = os.path.join(run_dir, "analysis", "parameters")
    if os.path.exists(parameters_dir):
        combo_dirs = [d for d in os.listdir(parameters_dir) if os.path.isdir(os.path.join(parameters_dir, d))]
    
    results = {}
    summary_data = []
    
    if combo_dirs:
        print(f"Found {len(combo_dirs)} parameter combinations")
        
        for combo in combo_dirs:
            combo_dir = os.path.join(parameters_dir, combo)
            summary_path = os.path.join(combo_dir, "summary.json")
            
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        summary_json = json.load(f)
                    
                    results_path = os.path.join(combo_dir, "results.json")
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as f:
                            results_json = json.load(f)
                            
                        if "results" in results_json and results_json["results"]:
                            df = pd.DataFrame(results_json["results"])
                            results[combo] = df
                            
                            # Add to summary data
                            metrics = ["verbatim_match_score", "sentence_preservation_score", "content_duplication_score", 
                                      "content_loss_score", "join_transition_score", "contextual_flow_score", 
                                      "average_score"]
                            
                            # Only include metrics that exist in the DataFrame
                            available_metrics = [m for m in metrics if m in df.columns]
                            
                            summary_data.append({
                                "parameter_set": combo,
                                "file_count": len(df),
                                **{metric: df[metric].mean() for metric in available_metrics if metric in df.columns}
                            })
                            
                            # Print summary with readable names
                            print(f"Loaded {len(df)} results for combination: {combo}")
                            print("Average metrics:")
                            for metric in available_metrics:
                                print(f"  - {get_readable_metric_name(metric)}: {df[metric].mean():.2f}")
                                
                except Exception as e:
                    print(f"Error loading results for combination {combo}: {e}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
    
    return results, summary_df

def load_multiple_batch_results(base_dir: str, batch_pattern: str = "*batch*") -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load all grading results from multiple batch directories
    
    Args:
        base_dir: Base directory containing batch folders
        batch_pattern: Glob pattern to match batch directories
        
    Returns:
        results: Dictionary mapping batch+parameter combinations to DataFrames
        summary: Summary DataFrame with metrics for all batches
    """
    print(f"Loading results from all batch directories in {base_dir} matching pattern: {batch_pattern}")
    
    # Find all batch directories
    batch_dirs = glob.glob(os.path.join(base_dir, batch_pattern))
    print(f"Found {len(batch_dirs)} batch directories: {batch_dirs}")
    
    all_results = {}
    all_summaries = []
    
    for batch_dir in batch_dirs:
        batch_name = os.path.basename(batch_dir)
        print(f"\nProcessing batch: {batch_name}")
        
        # Load results from this batch
        batch_results, batch_summary = load_all_grading_results(batch_dir)
        
        # Add batch name to the combination names
        batch_results_renamed = {}
        for combo_name, df in batch_results.items():
            df = df.copy()
            df["batch"] = batch_name
            batch_results_renamed[f"{batch_name}_{combo_name}"] = df
        
        all_results.update(batch_results_renamed)
        
        # Add batch name to summary
        if not batch_summary.empty:
            batch_summary["batch"] = batch_name
            all_summaries.append(batch_summary)
    
    # Combine summaries
    combined_summary = pd.concat(all_summaries) if all_summaries else pd.DataFrame()
    
    # Print overall summary with readable names
    if not combined_summary.empty:
        metrics = ["verbatim_match_score", "sentence_preservation_score", "content_duplication_score", 
                   "content_loss_score", "join_transition_score", "contextual_flow_score", 
                   "average_score"]
                   
        available_metrics = [m for m in metrics if m in combined_summary.columns]
        
        print("\nOverall average metrics across all batches:")
        for metric in available_metrics:
            if metric in combined_summary.columns:
                print(f"  - {get_readable_metric_name(metric)}: {combined_summary[metric].mean():.2f}")
    
    print(f"\nLoaded {len(all_results)} parameter combinations across {len(batch_dirs)} batches")
    return all_results, combined_summary

def generate_comparison_charts(results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Generate comparison charts for the different parameter combinations
    
    Args:
        results: Dictionary mapping parameter combinations to DataFrames
        output_dir: Directory to save the charts
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not results:
        print("No results to visualize")
        return
    
    # Create a combined DataFrame with a 'combination' column
    combined_data = []
    for combo_name, df in results.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy["combination"] = combo_name
            combined_data.append(df_copy)
    
    if not combined_data:
        print("No valid data to visualize")
        return
    
    all_data = pd.concat(combined_data)
    
    # Create a summary DataFrame with average scores for each metric by combination
    metrics = ["verbatim_match_score", "sentence_preservation_score", "content_duplication_score", 
               "content_loss_score", "join_transition_score", "contextual_flow_score", 
               "average_score"]
    
    summary = all_data.groupby("combination")[metrics].mean().reset_index()
    
    # Save summary table as CSV
    summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    
    # 1. Overall comparison bar chart
    plt.figure(figsize=(12, 8))
    sns.barplot(x="combination", y="average_score", data=summary)
    plt.title(f"{get_readable_metric_name('average_score')} by Combination")
    plt.xticks(rotation=45)
    plt.ylabel(get_readable_metric_name("average_score"))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_comparison.png"))
    plt.close()
    
    # 2. Detailed metrics comparison
    melted = pd.melt(summary, id_vars=["combination"], 
                     value_vars=[m for m in metrics if m != "average_score"],
                     var_name="Metric", value_name="Score")
    
    # Add readable metric names
    melted["Readable_Metric"] = melted["Metric"].apply(get_readable_metric_name)
    
    plt.figure(figsize=(15, 10))
    sns.barplot(x="combination", y="Score", hue="Readable_Metric", data=melted)
    plt.title("Detailed Metrics by Combination")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detailed_metrics.png"))
    plt.close()
    
    # 3. Heatmap of metrics
    plt.figure(figsize=(12, 10))
    pivot = melted.pivot(index="Readable_Metric", columns="combination", values="Score")
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title("Metrics Heatmap by Combination")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_heatmap.png"))
    plt.close()
    
    # 4. Boxplots for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="combination", y=metric, data=all_data)
        plt.title(f"{get_readable_metric_name(metric)} Distribution by Combination")
        plt.xticks(rotation=45)
        plt.ylabel(get_readable_metric_name(metric))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_boxplot.png"))
        plt.close()
    
    # 5. If audio_file is available, create a distribution plot by audio file
    if "audio_file" in all_data.columns:
        # Get top 10 audio files by count
        audio_counts = all_data["audio_file"].value_counts().head(10)
        top_audio_files = audio_counts.index.tolist()
        
        # Filter to only include top audio files
        filtered_data = all_data[all_data["audio_file"].isin(top_audio_files)]
        
        if not filtered_data.empty:
            plt.figure(figsize=(15, 8))
            sns.barplot(x="audio_file", y="average_score", hue="combination", data=filtered_data)
            plt.title(f"{get_readable_metric_name('average_score')} by Audio File and Combination")
            plt.xticks(rotation=45)
            plt.ylabel(get_readable_metric_name("average_score"))
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "audio_file_comparison.png"))
            plt.close()
    
    # Generate HTML report
    generate_html_report(results, summary, output_dir)
    
    print(f"Generated visualization charts in {output_dir}")

def generate_aggregate_visualizations(results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Generate visualizations that aggregate data across all batches and parameters
    
    Args:
        results: Dictionary mapping batch+parameter combinations to DataFrames
        output_dir: Directory to save the charts
    """
    # Create output directory if it doesn't exist
    agg_dir = os.path.join(output_dir, "aggregate_visualizations")
    os.makedirs(agg_dir, exist_ok=True)
    
    # Combine all dataframes with batch and parameter info
    combined_df = pd.DataFrame()
    for combo_name, df in results.items():
        if df.empty:
            continue
            
        df_copy = df.copy()
        
        # Extract batch and parameter set from combo_name
        if "_" in combo_name:
            batch, param_set = combo_name.split("_", 1)
            df_copy["batch"] = batch
            df_copy["parameter_set"] = param_set
        else:
            df_copy["parameter_set"] = combo_name
            
        combined_df = pd.concat([combined_df, df_copy])
    
    if combined_df.empty:
        print("No data available for visualization")
        return
    
    # Define metrics for visualization
    metrics = ["verbatim_match_score", "sentence_preservation_score", "content_duplication_score", 
               "content_loss_score", "join_transition_score", "contextual_flow_score", 
               "average_score"]
    
    # Only keep metrics that exist in the DataFrame and are numeric
    numeric_df = combined_df.select_dtypes(include=['number'])
    available_metrics = [m for m in metrics if m in numeric_df.columns]
    
    if not available_metrics:
        print("No numeric metrics available for visualization")
        return
    
    # 1. Overall performance by batch
    if "average_score" in numeric_df.columns and "batch" in combined_df.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=combined_df, x="batch", y="average_score")
        plt.title(f"{get_readable_metric_name('average_score')} by Batch")
        plt.xlabel("Batch")
        plt.ylabel(get_readable_metric_name("average_score"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(agg_dir, "batch_comparison.png"))
        plt.close()
    
    # 2. Parameter set comparison across all batches
    if "average_score" in numeric_df.columns and "parameter_set" in combined_df.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=combined_df, x="parameter_set", y="average_score")
        plt.title(f"{get_readable_metric_name('average_score')} by Parameter Set (All Batches)")
        plt.xlabel("Parameter Set")
        plt.ylabel(get_readable_metric_name("average_score"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(agg_dir, "parameter_comparison.png"))
        plt.close()
    
    # 3. Batch and parameter combination comparison
    if "average_score" in numeric_df.columns and "batch" in combined_df.columns and "parameter_set" in combined_df.columns:
        plt.figure(figsize=(15, 10))
        sns.boxplot(data=combined_df, x="parameter_set", y="average_score", hue="batch")
        plt.title(f"{get_readable_metric_name('average_score')} by Parameter Set and Batch")
        plt.xlabel("Parameter Set")
        plt.ylabel(get_readable_metric_name("average_score"))
        plt.xticks(rotation=45)
        plt.legend(title="Batch")
        plt.tight_layout()
        plt.savefig(os.path.join(agg_dir, "batch_parameter_comparison.png"))
        plt.close()
    
    # 4. Metrics comparison across all data
    if len(available_metrics) > 1:  # Need at least 2 metrics
        metrics_df = numeric_df[available_metrics].melt()
        metrics_df["readable_variable"] = metrics_df["variable"].apply(get_readable_metric_name)
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=metrics_df, x="readable_variable", y="value")
        plt.title("Metrics Comparison (All Batches)")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(agg_dir, "metrics_comparison.png"))
        plt.close()
    
    # 5. Heatmap of average metrics by batch
    if len(available_metrics) > 1 and "batch" in combined_df.columns:  # Need at least 2 metrics
        try:
            # Explicitly use numeric_only=True
            batch_metrics = combined_df.groupby("batch")[available_metrics].mean(numeric_only=True).reset_index()
            batch_metrics_melted = batch_metrics.melt(id_vars=["batch"], value_vars=available_metrics)
            batch_metrics_melted["readable_variable"] = batch_metrics_melted["variable"].apply(get_readable_metric_name)
            
            pivot_df = batch_metrics_melted.pivot(index="batch", columns="readable_variable", values="value")
            plt.figure(figsize=(14, 10))
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Average Metrics by Batch")
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, "batch_metrics_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Error generating batch metrics heatmap: {e}")
    
    # 6. Heatmap of average metrics by parameter set
    if len(available_metrics) > 1 and "parameter_set" in combined_df.columns:  # Need at least 2 metrics
        try:
            # Explicitly use numeric_only=True
            param_metrics = combined_df.groupby("parameter_set")[available_metrics].mean(numeric_only=True).reset_index()
            param_metrics_melted = param_metrics.melt(id_vars=["parameter_set"], value_vars=available_metrics)
            param_metrics_melted["readable_variable"] = param_metrics_melted["variable"].apply(get_readable_metric_name)
            
            pivot_df = param_metrics_melted.pivot(index="parameter_set", columns="readable_variable", values="value")
            plt.figure(figsize=(14, 10))
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Average Metrics by Parameter Set")
            plt.tight_layout()
            plt.savefig(os.path.join(agg_dir, "parameter_metrics_heatmap.png"))
            plt.close()
        except Exception as e:
            print(f"Error generating parameter metrics heatmap: {e}")
    
    print(f"Generated aggregate visualization charts in {agg_dir}")

def generate_comprehensive_report(results: Dict[str, pd.DataFrame], summary: pd.DataFrame, output_dir: str) -> None:
    """
    Generate a comprehensive HTML report for multiple batches
    
    Args:
        results: Dictionary mapping batch+parameter combinations to DataFrames
        summary: Summary DataFrame with metrics for all batches
        output_dir: Directory to save the report
    """
    # Create output directory if it doesn't exist
    agg_dir = os.path.join(output_dir, "aggregate_visualizations")
    os.makedirs(agg_dir, exist_ok=True)
    
    metrics = ["verbatim_match_score", "sentence_preservation_score", "content_duplication_score", 
              "content_loss_score", "join_transition_score", "contextual_flow_score", 
              "average_score"]
    
    with open(os.path.join(agg_dir, "comprehensive_report.html"), "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Transcription Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .chart-container { margin: 20px 0; text-align: center; }
                .chart-container img { max-width: 100%; height: auto; }
                .summary-card { 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin-bottom: 20px;
                    background-color: #f9f9f9;
                }
                .score-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                .score-box {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    min-width: 120px;
                    text-align: center;
                }
                .score-high { background-color: #d4edda; }
                .score-medium { background-color: #fff3cd; }
                .score-low { background-color: #f8d7da; }
                .batch-section {
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 2px solid #eee;
                }
                .parameter-section {
                    margin-left: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Comprehensive Transcription Analysis Report</h1>
                <p>Generated on: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                
                <h2>Aggregate Visualizations</h2>
                
                <h3>Batch Comparison</h3>
                <div class="chart-container">
                    <img src="batch_comparison.png" alt="Batch Comparison">
                </div>
                
                <h3>Parameter Set Comparison</h3>
                <div class="chart-container">
                    <img src="parameter_comparison.png" alt="Parameter Set Comparison">
                </div>
                
                <h3>Parameter Set Performance by Batch</h3>
                <div class="chart-container">
                    <img src="batch_parameter_comparison.png" alt="Batch & Parameter Comparison">
                </div>
                
                <h3>Metrics Comparison</h3>
                <div class="chart-container">
                    <img src="metrics_comparison.png" alt="Metrics Comparison">
                </div>
                
                <h3>Metrics by Batch</h3>
                <div class="chart-container">
                    <img src="batch_metrics_heatmap.png" alt="Batch Metrics Heatmap">
                </div>
                
                <h3>Metrics by Parameter Set</h3>
                <div class="chart-container">
                    <img src="parameter_metrics_heatmap.png" alt="Parameter Metrics Heatmap">
                </div>
        """)
        
        # Summarize metrics by batch
        if not summary.empty and "batch" in summary.columns:
            f.write("<h2>Summary by Batch</h2>")
            
            # Get numeric columns only
            numeric_cols = summary.select_dtypes(include=['number']).columns.tolist()
            if "batch" in numeric_cols:
                numeric_cols.remove("batch")
                
            # Keep only batch column and numeric columns for groupby
            summary_cols = ["batch"] + numeric_cols
            summary_for_groupby = summary[summary_cols]
            
            batch_summary = summary_for_groupby.groupby("batch")[numeric_cols].mean().reset_index() if numeric_cols else pd.DataFrame({"batch": summary["batch"].unique()})
            
            # Rename columns to more readable format
            readable_batch_summary = batch_summary.copy()
            for col in readable_batch_summary.columns:
                if col in metrics:
                    readable_batch_summary.rename(columns={col: get_readable_metric_name(col)}, inplace=True)
                    
            f.write(readable_batch_summary.to_html(index=False, classes="dataframe"))
        
        # Detailed results by batch and parameter
        f.write("<h2>Detailed Results by Batch and Parameter</h2>")
        
        # Get unique batches
        batches = set()
        for combo_name in results.keys():
            if "_" in combo_name:
                batch = combo_name.split("_", 1)[0]
                batches.add(batch)
        
        # Group results by batch
        for batch in sorted(batches):
            batch_combos = [c for c in results.keys() if c.startswith(f"{batch}_")]
            
            if not batch_combos:
                continue
                
            f.write(f'<div class="batch-section"><h3>Batch: {batch}</h3>')
            
            # Add parameters within this batch
            for combo_name in sorted(batch_combos):
                df = results[combo_name]
                if df.empty:
                    continue
                    
                param_name = combo_name.split("_", 1)[1] if "_" in combo_name else combo_name
                avg_score = df["average_score"].mean() if "average_score" in df.columns else 0
                score_class = "score-high" if avg_score >= 7 else "score-medium" if avg_score >= 5 else "score-low"
                
                f.write(f'<div class="parameter-section"><h4>Parameter: {param_name}</h4>')
                f.write(f'<div class="score-box {score_class}"><h4>{get_readable_metric_name("average_score")}</h4><div style="font-size: 24px;">{avg_score:.1f}</div></div>')
                
                # Add specific metrics for this parameter
                metric_scores = {}
                for metric in metrics:
                    if metric in df.columns and metric != "average_score":
                        metric_scores[metric] = df[metric].mean()
                
                if metric_scores:
                    f.write("<h4>Metrics</h4><div class='score-container'>")
                    for metric, score in metric_scores.items():
                        metric_class = "score-high" if score >= 7 else "score-medium" if score >= 5 else "score-low"
                        metric_name = get_readable_metric_name(metric)
                        f.write(f"""
                        <div class="score-box {metric_class}">
                            <div>{metric_name}</div>
                            <div style="font-size: 18px;">{score:.1f}</div>
                        </div>
                        """)
                    f.write("</div>")
                    
                f.write("</div>")  # Close parameter-section
            
            f.write("</div>")  # Close batch-section
            
        f.write("""
            </div>
        </body>
        </html>
        """)
    
    print(f"Generated comprehensive report at {os.path.join(agg_dir, 'comprehensive_report.html')}")

def generate_html_report(results: Dict[str, pd.DataFrame], 
                        summary: pd.DataFrame,
                        output_dir: str) -> None:
    """
    Generate an HTML report with interactive tables
    
    Args:
        results: Dictionary mapping parameter combinations to DataFrames
        summary: Summary DataFrame with metrics by combination
        output_dir: Directory to save the report
    """
    # Define metrics list for reporting
    metrics = ["verbatim_match_score", "sentence_preservation_score", "content_duplication_score", 
               "content_loss_score", "join_transition_score", "contextual_flow_score", 
               "average_score"]
               
    with open(os.path.join(output_dir, "report.html"), "w") as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Transcription Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .container { max-width: 1200px; margin: 0 auto; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .chart-container { margin: 20px 0; text-align: center; }
                .chart-container img { max-width: 100%; height: auto; }
                .metric-good { color: green; }
                .metric-medium { color: orange; }
                .metric-poor { color: red; }
                .summary-card { 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin-bottom: 20px;
                    background-color: #f9f9f9;
                }
                .score-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                }
                .score-box {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 10px;
                    min-width: 120px;
                    text-align: center;
                }
                .score-high { background-color: #d4edda; }
                .score-medium { background-color: #fff3cd; }
                .score-low { background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Transcription Analysis Report</h1>
                <p>Generated on: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                
                <h2>Summary of Results</h2>
        """)
        
        # Add summary table
        f.write(summary.to_html(index=False, classes="dataframe"))
        
        # Add parameter-specific summaries
        f.write("<h2>Parameter Set Summaries</h2>")
        
        for combo_name, df in results.items():
            avg_score = df["average_score"].mean() if "average_score" in df.columns else 0
            total_files = len(df)
            successful = df["success"].sum() if "success" in df.columns else total_files
            failed = total_files - successful
            retried = df["retries"].sum() if "retries" in df.columns else 0
            
            score_class = "score-high" if avg_score >= 7 else "score-medium" if avg_score >= 5 else "score-low"
            
            f.write(f"""
            <div class="summary-card">
                <h3>{combo_name}</h3>
                <div class="score-container">
                    <div class="score-box {score_class}">
                        <h4>{get_readable_metric_name("average_score")}</h4>
                        <div style="font-size: 24px;">{avg_score:.1f}</div>
                    </div>
                    <div class="score-box">
                        <h4>Files</h4>
                        <div>Total: {total_files}</div>
                        <div>Success: {successful}</div>
                        <div>Failed: {failed}</div>
                    </div>
                    <div class="score-box">
                        <h4>Retries</h4>
                        <div>{retried}</div>
                    </div>
                </div>
            """)
            
            # Add metric scores
            if not df.empty:
                metric_scores = {}
                for metric in metrics:
                    if metric in df.columns and metric != "average_score":
                        metric_scores[metric] = df[metric].mean()
                
                if metric_scores:
                    f.write("<h4>Metrics</h4><div class='score-container'>")
                    for metric, score in metric_scores.items():
                        metric_class = "score-high" if score >= 7 else "score-medium" if score >= 5 else "score-low"
                        metric_name = get_readable_metric_name(metric)
                        f.write(f"""
                        <div class="score-box {metric_class}">
                            <div>{metric_name}</div>
                            <div style="font-size: 18px;">{score:.1f}</div>
                        </div>
                        """)
                    f.write("</div>")
            
            f.write("</div>")
        
        # Add charts
        f.write("""
                <h2>Visualization Charts</h2>
                
                <h3>Overall Comparison</h3>
                <div class="chart-container">
                    <img src="overall_comparison.png" alt="Overall Comparison">
                </div>
                
                <h3>Detailed Metrics</h3>
                <div class="chart-container">
                    <img src="detailed_metrics.png" alt="Detailed Metrics">
                </div>
                
                <h3>Metrics Heatmap</h3>
                <div class="chart-container">
                    <img src="metrics_heatmap.png" alt="Metrics Heatmap">
                </div>
        """)
        
        # Add audio file comparison if available
        if os.path.exists(os.path.join(output_dir, "audio_file_comparison.png")):
            f.write("""
                <h3>Audio File Comparison</h3>
                <div class="chart-container">
                    <img src="audio_file_comparison.png" alt="Audio File Comparison">
                </div>
            """)
        
        # Add detailed results for each combination
        f.write("<h2>Detailed Results by Combination</h2>")
        
        for combo_name, df in results.items():
            if not df.empty:
                f.write(f"<h3>{combo_name}</h3>")
                # Select relevant columns for display
                display_cols = ["chunk_number", "filename", "audio_file", "average_score", "verbatim_match_score", 
                               "sentence_preservation_score", "content_duplication_score", "content_loss_score",
                               "join_transition_score", "contextual_flow_score"]
                # Only keep columns that exist
                valid_cols = [col for col in display_cols if col in df.columns]
                display_df = df[valid_cols] if valid_cols else df
                
                # Rename columns to more readable format
                renamed_df = display_df.copy()
                for col in renamed_df.columns:
                    if col in metrics:
                        renamed_df.rename(columns={col: get_readable_metric_name(col)}, inplace=True)
                
                f.write(renamed_df.to_html(index=False, classes="dataframe"))
        
        f.write("""
            </div>
        </body>
        </html>
        """)

# Define a function to get human-readable metric names
def get_readable_metric_name(metric: str) -> str:
    """
    Convert metric code names to human-readable display names
    
    Args:
        metric: The internal metric name
        
    Returns:
        Human-readable display name for the metric
    """
    metric_mapping = {
        "verbatim_match_score": "Verbatim Match",
        "sentence_preservation_score": "Sentence Preservation",
        "content_duplication_score": "Content Duplication",
        "content_loss_score": "Content Loss",
        "join_transition_score": "Join Transition",
        "contextual_flow_score": "Contextual Flow",
        "average_score": "Average Score"
    }
    return metric_mapping.get(metric, metric.replace('_', ' ').title())

def main():
    parser = argparse.ArgumentParser(description="Visualize transcription analysis results")
    parser.add_argument("run_dir", help="Directory containing analysis results")
    parser.add_argument("--aggregate", action="store_true", help="Generate aggregate visualizations across all batches")
    args = parser.parse_args()
    
    if args.aggregate:
        print(f"Generating aggregate visualizations from multiple batches in {args.run_dir}")
        results, summary = load_multiple_batch_results(args.run_dir)
        generate_aggregate_visualizations(results, args.run_dir)
        generate_comprehensive_report(results, summary, args.run_dir)
    else:
        print(f"Generating visualizations for {args.run_dir}")
        results, summary = load_all_grading_results(args.run_dir)
        generate_comparison_charts(results, args.run_dir)
        generate_html_report(results, summary, os.path.join(args.run_dir, "visualizations"))

if __name__ == "__main__":
    main() 