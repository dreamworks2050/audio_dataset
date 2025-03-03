import os
import sys
import time
import threading
import json
import glob
import logging
import re
import shutil
import random
from datetime import datetime  # Fix import to use datetime class directly
import asyncio
from typing import List, Dict, Tuple, Optional, Any, Union
import gradio as gr
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local imports
from .optimizer import AudioOptimizer
from .analyzer import TranscriptionAnalyzer
from utils.logger import logger
from transcribe.transcriber import TranscriptionService

# Define version locally instead of importing to avoid circular imports
__version__ = "1.0.0"

# Configure logging
logger = logging.getLogger("ai_optimize.ui")

# Global variables
audio_optimizer = AudioOptimizer()
stop_transcription = False
is_transcribing = False

# Initialize the audio optimizer
audio_optimizer = AudioOptimizer()

# Define constants
AUDIO_AI_OPTIMIZED_DIR = "audio_ai_optimized"
AI_ANALYSIS_DIR = os.path.join(AUDIO_AI_OPTIMIZED_DIR, "analysis")  # New constant for analysis output

# Global variables for transcription and analysis
transcription_service = None
is_transcribing = False
transcription_thread = None
stop_transcription = False  # Flag to signal transcription should stop

# New global variables for AI analysis
analyzer = None
is_analyzing = False
analysis_thread = None
stop_analysis = False  # Flag to signal analysis should stop

# Helper function to get readable metric names
def get_readable_metric_name(metric):
    """Convert metric keys to human-readable names"""
    metric_names = {
        "verbatim_match_score": "Verbatim Match",
        "sentence_preservation_score": "Sentence Preservation",
        "content_duplication_score": "Content Duplication",
        "content_loss_score": "Content Loss",
        "join_transition_score": "Join Transition",
        "contextual_flow_score": "Contextual Flow",
        "average_score": "Average Score"
    }
    return metric_names.get(metric, metric.replace("_", " ").title())

def create_ai_optimize_tab():
    """Create the AI Optimize tab UI.
    
    Returns:
        gr.TabItem: The AI Optimize tab
    """
    # Initialize processing times database by collecting data from existing files
    try:
        processed_count = collect_processing_times_from_existing_files()
        if processed_count:
            logger.info(f"Initialized processing times database with {processed_count} entries")
    except Exception as e:
        logger.error(f"Error initializing processing times database: {str(e)}")
    
    with gr.TabItem("AI Optimize") as ai_optimize_tab:
        with gr.Row():
            # Main content column (2/3 width)
            with gr.Column(scale=2):
                # Audio file selection
                gr.Markdown("### Audio Selection")
                
                # Dropdown for audio files
                audio_files = gr.Dropdown(
                    label="Select Audio File",
                    choices=audio_optimizer.get_audio_files(),
                    type="value",
                    interactive=True
                )
                
                # Refresh button for audio files
                refresh_btn = gr.Button("Refresh Audio Files")
                
                # Add a new row for audio duration selection
                with gr.Row():
                    audio_duration = gr.Radio(
                        label="Audio Duration to Process",
                        choices=["First 1 minute", "First 5 minutes", "First 10 minutes", "Full Audio"],
                        value="First 5 minutes",
                        interactive=True
                    )
                
                # Chunk and Overlap sections side by side
                with gr.Row():
                    # Left half - Chunk lengths section
                    with gr.Column(scale=1):
                        gr.Markdown("### Chunk Lengths")
                        
                        # Checkboxes for chunk lengths in a single row, including custom
                        with gr.Row():
                            chunk_10s = gr.Checkbox(label="10s", value=True)
                            chunk_15s = gr.Checkbox(label="15s", value=False)
                            chunk_20s = gr.Checkbox(label="20s", value=True)
                            chunk_30s = gr.Checkbox(label="30s", value=True)
                            chunk_45s = gr.Checkbox(label="45s", value=False)
                            chunk_100s = gr.Checkbox(label="100s", value=False)
                            chunk_200s = gr.Checkbox(label="200s", value=False)
                            chunk_custom = gr.Checkbox(label="Custom", value=False)
                            chunk_custom_value = gr.Number(
                                label="",
                                value=25,
                                minimum=1,
                                maximum=600,
                                step=1,
                                interactive=True,
                                visible=False
                            )
                    
                    # Right half - Overlap lengths section
                    with gr.Column(scale=1):
                        gr.Markdown("### Overlap Lengths")
                        
                        # Checkboxes for overlap lengths in a single row, including custom
                        with gr.Row():
                            overlap_0s = gr.Checkbox(label="0s", value=True)
                            overlap_1s = gr.Checkbox(label="1s", value=False)
                            overlap_3s = gr.Checkbox(label="3s", value=True)
                            overlap_5s = gr.Checkbox(label="5s", value=True)
                            overlap_10s = gr.Checkbox(label="10s", value=False)
                            overlap_15s = gr.Checkbox(label="15s", value=False)
                            overlap_20s = gr.Checkbox(label="20s", value=False)
                            overlap_custom = gr.Checkbox(label="Custom", value=False)
                            overlap_custom_value = gr.Number(
                                label="",
                                value=2,
                                minimum=0,
                                maximum=60,
                                step=1,
                                interactive=True,
                                visible=False
                            )
                
                # Buttons row
                with gr.Row():
                    calculate_btn = gr.Button("Calculate Combinations")
                    prepare_btn = gr.Button("Prepare Audios")
                    transcribe_btn = gr.Button("Transcribe Chunks")
                    analyze_btn = gr.Button("AI Analysis")  # New AI Analysis button
                
                # Transcription options - Make visible by default
                with gr.Row(visible=True) as transcription_options:
                    with gr.Column(scale=1):
                        model_path = gr.Dropdown(
                            label="Whisper Model",
                            choices=[
                                "/Users/macbook/audio_dataset/whisper.cpp/models/ggml-large-v3-turbo.bin",
                                "models/ggml-base.en.bin",
                                "models/ggml-small.en.bin",
                                "models/ggml-medium.en.bin",
                                "models/ggml-large.bin"
                            ],
                            value="/Users/macbook/audio_dataset/whisper.cpp/models/ggml-large-v3-turbo.bin",
                            type="value"
                        )
                    
                    with gr.Column(scale=1):
                        language = gr.Dropdown(
                            label="Language",
                            choices=["English", "Korean", "Chinese", "Vietnamese", "Spanish"],
                            value="English",
                            type="value",
                            interactive=True
                        )
                
                # Add prompt options
                with gr.Row(visible=True) as prompt_options:
                    with gr.Column(scale=1):
                        use_prompt = gr.Radio(
                            label="Prompt Settings",
                            choices=["Don't use prompt", "Use Prompt"],
                            value="Don't use prompt",
                            type="value",
                            interactive=True
                        )
                    
                    with gr.Column(scale=1):
                        prompt_length = gr.Number(
                            label="Prompt Length (characters)",
                            value=200,
                            minimum=0,
                            maximum=1000,
                            step=50,
                            interactive=True,
                            visible=False
                        )
                
                # Results display with tabs
                with gr.Row():
                    with gr.Tabs() as result_tabs:
                        with gr.TabItem("Audio Split Results") as audio_split_tab:
                            result_text = gr.Textbox(
                                label="Audio Processing Results",
                                placeholder="Processing results will appear here",
                                lines=15,
                                max_lines=30,
                                interactive=False,
                                show_copy_button=True
                            )
                        
                        with gr.TabItem("Transcription Results") as transcription_tab:
                            with gr.Row():
                                refresh_transcription_btn = gr.Button("Refresh Transcription Status", variant="secondary", scale=1)
                                stop_transcription_btn = gr.Button("Stop Transcription", variant="stop", scale=1)
                            
                            transcription_text = gr.Textbox(
                                label="Transcription Results",
                                placeholder="Transcription results will appear here",
                                lines=15,
                                max_lines=30,
                                interactive=False,
                                show_copy_button=True
                            )
                            
                        # New tab for AI Analysis Results
                        with gr.TabItem("AI Analyze") as analysis_tab:
                            with gr.Row():
                                refresh_analysis_btn = gr.Button("Refresh Analysis Status", variant="secondary", scale=1)
                                stop_analysis_btn = gr.Button("Stop Analysis", variant="stop", scale=1)
                            
                            # Add model selector for Ollama
                            with gr.Row():
                                # Try to get available models from Ollama
                                available_models = [
                                    "mistral",
                                    "mistral-small:24b-instruct-2501-q4_K_M",
                                    "llama3",
                                    "llama3:8b",
                                    "gemma2:9b",
                                    "phi3:mini",
                                ]
                                
                                try:
                                    import subprocess
                                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                                    if result.returncode == 0:
                                        models = []
                                        for line in result.stdout.strip().split("\n"):
                                            if line.strip() and not line.startswith("NAME"):
                                                parts = line.split()
                                                if parts and parts[0] not in ["", "ID", "SIZE", "MODIFIED"]:
                                                    models.append(parts[0])
                                        if models:
                                            available_models = models
                                except Exception as e:
                                    logger.warning(f"Failed to get available Ollama models: {str(e)}")
                                
                                model_dropdown = gr.Dropdown(
                                    label="Ollama Model",
                                    choices=available_models,
                                    value=available_models[0] if available_models else "mistral",
                                    type="value",
                                    interactive=True,
                                    info="Select the Ollama model to use for analysis"
                                )
                            
                            analysis_text = gr.Textbox(
                                label="AI Analysis Results",
                                placeholder="Analysis results will appear here",
                                lines=15,
                                max_lines=40,
                                max_length=4096,
                                interactive=False,
                                show_copy_button=True
                            )
                        
                        # New visualization tab for charts
                        with gr.TabItem("AI Analysis Results") as analysis_charts_tab:
                            with gr.Row():
                                # Create the button to generate aggregate visualizations
                                generate_visualizations_btn = gr.Button("Generate Aggregate Visualizations", size="sm")
                                generate_visualizations_output = gr.Markdown("")
                                
                                # Add button to save processing times from the current batch
                                save_processing_times_btn = gr.Button("Save Processing Times", size="sm")
                                save_processing_times_output = gr.Markdown("")
                                
                                # Create the refresh button for charts
                                refresh_charts_btn = gr.Button("Refresh Charts", size="sm")
                            
                            # Function to load analysis data and create visualizations
                            def load_analysis_results():
                                try:
                                    # Import visualization utilities
                                    import matplotlib.pyplot as plt
                                    import seaborn as sns
                                    import pandas as pd
                                    import os
                                    import json
                                    
                                    # Function to load all grading results (simplified version)
                                    def load_results(analysis_dir):
                                        results = {}
                                        # Get all parameter combination directories
                                        combo_dirs = [d for d in os.listdir(analysis_dir) 
                                                    if os.path.isdir(os.path.join(analysis_dir, d)) 
                                                    and d.startswith("chunk")]
                                        
                                        for combo in combo_dirs:
                                            combo_dir = os.path.join(analysis_dir, combo)
                                            # Look for summary.json file
                                            summary_file = os.path.join(combo_dir, "summary.json")
                                            if os.path.exists(summary_file):
                                                with open(summary_file, 'r') as f:
                                                    summary_data = json.load(f)
                                                
                                                # Get chunk analysis files
                                                chunk_files = glob.glob(os.path.join(combo_dir, "chunk_*_analysis.json"))
                                                chunk_data = []
                                                
                                                for chunk_file in chunk_files:
                                                    try:
                                                        with open(chunk_file, 'r') as f:
                                                            data = json.load(f)
                                                        
                                                        # Extract chunk number from filename
                                                        chunk_num = int(re.search(r'chunk_(\d+)_analysis', os.path.basename(chunk_file)).group(1))
                                                        
                                                        # Create a row for this chunk
                                                        row = {
                                                            "chunk_number": chunk_num,
                                                            "verbatim_match_score": data.get("analysis_result", {}).get("verbatim_match_score", 0),
                                                            "sentence_preservation_score": data.get("analysis_result", {}).get("sentence_preservation_score", 0),
                                                            "content_duplication_score": data.get("analysis_result", {}).get("content_duplication_score", 0),
                                                            "content_loss_score": data.get("analysis_result", {}).get("content_loss_score", 0),
                                                            "join_transition_score": data.get("analysis_result", {}).get("join_transition_score", 0),
                                                            "contextual_flow_score": data.get("analysis_result", {}).get("contextual_flow_score", 0),
                                                        }
                                                        
                                                        # Calculate average score
                                                        scores = [row["verbatim_match_score"], row["sentence_preservation_score"],
                                                                row["content_duplication_score"], row["content_loss_score"],
                                                                row["join_transition_score"], row["contextual_flow_score"]]
                                                        row["average_score"] = sum(scores) / len(scores)
                                                        
                                                        chunk_data.append(row)
                                                    except Exception as e:
                                                        logger.error(f"Error processing {chunk_file}: {str(e)}")
                                                
                                                if chunk_data:
                                                    results[combo] = pd.DataFrame(chunk_data)
                                        
                                        return results
                                    
                                    # Load all the results
                                    analysis_dir = os.path.join("audio_ai_optimized", "analysis")
                                    results = load_results(analysis_dir)
                                    
                                    if not results:
                                        return None, None, "No analysis results found."
                                    
                                    # Create a combined DataFrame with a 'combination' column
                                    combined_data = []
                                    for combo_name, df in results.items():
                                        if not df.empty:
                                            df_copy = df.copy()
                                            df_copy["combination"] = combo_name
                                            combined_data.append(df_copy)
                                    
                                    all_data = pd.concat(combined_data)
                                    
                                    # Create DataFrame for the overview scatter plot
                                    metrics = ["verbatim_match_score", "sentence_preservation_score", 
                                              "content_duplication_score", "content_loss_score", 
                                              "join_transition_score", "contextual_flow_score"]
                                    
                                    # Create a summary DataFrame with average scores for each metric by combination
                                    summary = all_data.groupby("combination")[metrics + ["average_score"]].mean().reset_index()
                                    
                                    # Create overview scatter plot data
                                    # Extract chunk and overlap values from combination names
                                    summary['chunk_length'] = summary['combination'].apply(
                                        lambda x: int(x.split('chunk')[1].split('s_overlap')[0]))
                                    summary['overlap_length'] = summary['combination'].apply(
                                        lambda x: int(x.split('overlap')[1].split('s')[0]))
                                    
                                    # Create individual scatter plots for each parameter set
                                    param_sets = summary['combination'].unique()
                                    individual_plots = {}
                                    
                                    for param in param_sets:
                                        param_data = all_data[all_data['combination'] == param]
                                        if not param_data.empty:
                                            param_df = pd.melt(
                                                param_data, 
                                                id_vars=['chunk_number'], 
                                                value_vars=metrics,
                                                var_name='metric', 
                                                value_name='score'
                                            )
                                            param_df['readable_metric'] = param_df['metric'].apply(get_readable_metric_name)
                                            individual_plots[param] = param_df
                                    
                                    return summary, individual_plots, "Charts refreshed successfully."
                                except Exception as e:
                                    logger.error(f"Error loading analysis results: {str(e)}")
                                    return None, None, f"Error loading analysis results: {str(e)}"
                            
                            # Main visualization area
                            with gr.Column():
                                gr.Markdown("### Analysis Results Visualization")
                                
                                # Status message for the charts
                                charts_status = gr.Markdown("Click 'Refresh Charts' to load visualization data")
                                
                                # Main scatter plot showing all parameter sets
                                gr.Markdown("#### Overview: All Parameter Sets")
                                with gr.Row():
                                    overview_plot = gr.ScatterPlot(
                                        label="Parameter Overview", 
                                        x="chunk_length",
                                        y="average_score",
                                        color="overlap_length",
                                        tooltip=["combination", "average_score"],
                                        width=800,
                                        height=400,
                                        title="Average Scores by Chunk Length and Overlap"
                                    )
                                
                                # Container for individual parameter set plots
                                gr.Markdown("#### Individual Parameter Set Details")
                                individual_plots_container = gr.Column()
                                
                                # New section for detailed chunk analysis
                                gr.Markdown("#### Detailed Chunk Analysis")
                                with gr.Row():
                                    # Dropdown for selecting parameter combinations
                                    combination_dropdown = gr.Dropdown(
                                        label="Select Parameter Combination",
                                        choices=[],
                                        interactive=True
                                    )
                                
                                # Container for chunk details plot
                                chunk_detail_plot = gr.Plot(
                                    label="Chunk Scores by Category"
                                )
                                
                                # Add metrics line plot
                                gr.Markdown("#### Metric Performance by Chunk")
                                
                                # Add a dedicated dropdown for metrics plot
                                with gr.Row():
                                    metrics_combination_dropdown = gr.Dropdown(
                                        label="Select Parameter Combination for Metrics Plot",
                                        choices=[],
                                        interactive=True
                                    )
                                
                                metrics_line_plot = gr.Plot(
                                    label="Metrics Comparison"
                                )
                                
                                # Function to update chunk detail plot based on selected combination
                                def update_chunk_detail_plot(combination):
                                    if not combination:
                                        return None
                                    
                                    try:
                                        import matplotlib.pyplot as plt
                                        import numpy as np
                                        
                                        # Find the directory for the selected combination
                                        analysis_dir = os.path.join("audio_ai_optimized", "analysis", combination)
                                        if not os.path.exists(analysis_dir):
                                            return None
                                        
                                        # Load all chunk analysis files
                                        chunk_files = sorted(glob.glob(os.path.join(analysis_dir, "chunk_*_analysis.json")), 
                                                           key=lambda x: int(re.search(r'chunk_(\d+)_analysis', x).group(1)))
                                        
                                        if not chunk_files:
                                            return None
                                        
                                        # Metric names for readable labels
                                        metric_names = {
                                            "verbatim_match_score": "Verbatim Match",
                                            "sentence_preservation_score": "Sentence Preservation",
                                            "content_duplication_score": "Content Duplication",
                                            "content_loss_score": "Content Loss",
                                            "join_transition_score": "Join Transition",
                                            "contextual_flow_score": "Contextual Flow"
                                        }
                                        
                                        # Colors for different metrics
                                        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
                                        
                                        # Create the figure
                                        fig, ax = plt.subplots(figsize=(12, 8))
                                        
                                        # Store all data points to identify overlaps
                                        all_points = {}  # (chunk_num, score) -> count
                                        
                                        # First pass: collect all points to identify overlaps
                                        for metric_idx, (metric, metric_name) in enumerate(metric_names.items()):
                                            for chunk_file in chunk_files:
                                                try:
                                                    with open(chunk_file, 'r', encoding='utf-8') as f:
                                                        data = json.load(f)
                                                    
                                                    # Extract chunk number
                                                    chunk_num = int(re.search(r'chunk_(\d+)_analysis', chunk_file).group(1))
                                                    
                                                    # Extract score for this metric
                                                    score = None
                                                    if 'scores' in data and metric in data['scores']:
                                                        score = data['scores'][metric]
                                                    elif 'analysis_result' in data and metric in data['analysis_result']:
                                                        score = data['analysis_result'][metric]
                                                    
                                                    if score is not None:
                                                        point = (chunk_num, score)
                                                        all_points[point] = all_points.get(point, 0) + 1
                                                except Exception as e:
                                                    logger.error(f"Error reading {chunk_file}: {str(e)}")
                                        
                                        # Calculate jitter for overlapping points
                                        jitter_map = {}
                                        for point, count in all_points.items():
                                            if count > 1:
                                                # Use a much smaller marker_radius to make dots touch with no space
                                                # This needs to be significantly smaller than the default
                                                marker_radius = 0.025
                                                
                                                # Calculate positions so dots touch each other exactly
                                                total_width = (count - 1) * 2 * marker_radius
                                                start = -total_width / 2
                                                
                                                # Create offsets with exact spacing for dots to touch
                                                offsets = [start + i * 2 * marker_radius for i in range(count)]
                                                jitter_map[point] = offsets
                                                
                                        # Counter to track which offset to use for each overlapping point
                                        overlap_counters = {point: 0 for point in jitter_map.keys()}
                                        
                                        # Second pass: plot with jittering for overlapping points
                                        for metric_idx, (metric, metric_name) in enumerate(metric_names.items()):
                                            x_values = []  # Chunk numbers
                                            y_values = []  # Score values
                                            
                                            for chunk_file in chunk_files:
                                                try:
                                                    with open(chunk_file, 'r', encoding='utf-8') as f:
                                                        data = json.load(f)
                                                    
                                                    # Extract chunk number
                                                    chunk_num = int(re.search(r'chunk_(\d+)_analysis', chunk_file).group(1))
                                                    
                                                    # Extract score for this metric
                                                    score = None
                                                    if 'scores' in data and metric in data['scores']:
                                                        score = data['scores'][metric]
                                                    elif 'analysis_result' in data and metric in data['analysis_result']:
                                                        score = data['analysis_result'][metric]
                                                    
                                                    if score is not None:
                                                        # Apply jitter if this is an overlapping point
                                                        x_val = chunk_num
                                                        point = (chunk_num, score)
                                                        if point in jitter_map:
                                                            counter = overlap_counters[point]
                                                            x_val = chunk_num + jitter_map[point][counter]
                                                            overlap_counters[point] += 1
                                                            
                                                        x_values.append(x_val)
                                                        y_values.append(score)
                                                except Exception as e:
                                                    logger.error(f"Error reading {chunk_file}: {str(e)}")
                                            
                                            if x_values and y_values:
                                                ax.scatter(x_values, y_values, label=metric_name, color=colors[metric_idx % len(colors)], s=100, alpha=0.7)
                                        
                                        # Set plot title and labels
                                        ax.set_title(f"Detailed Scores by Category for {combination}", fontsize=16)
                                        ax.set_xlabel("Chunk Number", fontsize=14)
                                        ax.set_ylabel("Score Value (0-10)", fontsize=14)
                                        ax.set_ylim(0, 10.5)
                                        ax.grid(True, linestyle='--', alpha=0.7)
                                        
                                        # Add legend
                                        ax.legend(fontsize=12)
                                        
                                        # Add horizontal line at score 5 for reference
                                        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
                                        
                                        plt.tight_layout()
                                        return fig
                                    except Exception as e:
                                        logger.error(f"Error updating chunk detail plot: {str(e)}")
                                        return None
                                
                                # Function to create metrics line plot for the selected combination
                                def create_metrics_line_plot(combination):
                                    if not combination:
                                        return None
                                        
                                    try:
                                        import matplotlib.pyplot as plt
                                        import numpy as np
                                        import pandas as pd
                                        
                                        # Find the directory for the selected combination
                                        analysis_dir = os.path.join("audio_ai_optimized", "analysis", combination)
                                        if not os.path.exists(analysis_dir):
                                            return None
                                            
                                        # Load all chunk analysis files
                                        chunk_files = sorted(glob.glob(os.path.join(analysis_dir, "chunk_*_analysis.json")), 
                                                           key=lambda x: int(re.search(r'chunk_(\d+)_analysis', x).group(1)))
                                        
                                        if not chunk_files:
                                            return None
                                            
                                        # Metric names and order
                                        metrics = [
                                            "verbatim_match_score",
                                            "sentence_preservation_score",
                                            "content_duplication_score",
                                            "content_loss_score",
                                            "join_transition_score",
                                            "contextual_flow_score"
                                        ]
                                        
                                        metric_names = {
                                            "verbatim_match_score": "Verbatim\nMatch",
                                            "sentence_preservation_score": "Sentence\nPreservation",
                                            "content_duplication_score": "Content\nDuplication",
                                            "content_loss_score": "Content\nLoss",
                                            "join_transition_score": "Join\nTransition",
                                            "contextual_flow_score": "Contextual\nFlow"
                                        }
                                        
                                        # Prepare data structure
                                        data = {}  # chunk_num -> {metric -> score}
                                        
                                        # Extract data
                                        for chunk_file in chunk_files:
                                            try:
                                                with open(chunk_file, 'r', encoding='utf-8') as f:
                                                    chunk_data = json.load(f)
                                                
                                                # Extract chunk number
                                                chunk_num = int(re.search(r'chunk_(\d+)_analysis', chunk_file).group(1))
                                                data[chunk_num] = {}
                                                
                                                # Get scores for each metric
                                                for metric in metrics:
                                                    score = None
                                                    if 'scores' in chunk_data and metric in chunk_data['scores']:
                                                        score = chunk_data['scores'][metric]
                                                    elif 'analysis_result' in chunk_data and metric in chunk_data['analysis_result']:
                                                        score = chunk_data['analysis_result'][metric]
                                                    
                                                    if score is not None:
                                                        data[chunk_num][metric] = score
                                                    else:
                                                        data[chunk_num][metric] = 0  # Default value if missing
                                            except Exception as e:
                                                logger.error(f"Error reading {chunk_file}: {str(e)}")
                                        
                                        if not data:
                                            return None
                                            
                                        # Create figure
                                        fig, ax = plt.subplots(figsize=(12, 6))
                                        
                                        # Define x positions and labels for the metrics
                                        x_pos = list(range(len(metrics)))
                                        x_labels = [metric_names[m] for m in metrics]
                                        
                                        # Plot one line for each chunk
                                        chunk_nums = sorted(data.keys())
                                        colors = plt.cm.tab10(np.linspace(0, 1, len(chunk_nums)))
                                        
                                        for i, chunk_num in enumerate(chunk_nums):
                                            # Get scores for this chunk across all metrics
                                            y_values = [data[chunk_num].get(metric, 0) for metric in metrics]
                                            
                                            # Plot the line
                                            ax.plot(x_pos, y_values, 'o-', linewidth=2, markersize=8, 
                                                   label=f"Chunk {chunk_num}", color=colors[i])
                                        
                                        # Set plot properties
                                        ax.set_title(f"Metric Scores by Chunk for {combination}", fontsize=16)
                                        ax.set_ylim(0, 10.5)
                                        ax.set_xticks(x_pos)
                                        ax.set_xticklabels(x_labels)
                                        ax.set_ylabel("Score Value (0-10)", fontsize=14)
                                        ax.grid(True, linestyle='--', alpha=0.7)
                                        
                                        # Add horizontal line at score 5 for reference
                                        ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
                                        
                                        # Add legend
                                        ax.legend(loc='upper right', fontsize=10)
                                        
                                        plt.tight_layout()
                                        return fig
                                    except Exception as e:
                                        logger.error(f"Error creating metrics line plot: {str(e)}")
                                        return None
                                
                                # Connect the dropdown to the update function
                                combination_dropdown.change(
                                    fn=lambda x: [update_chunk_detail_plot(x), create_metrics_line_plot(x)],
                                    inputs=[combination_dropdown],
                                    outputs=[chunk_detail_plot, metrics_line_plot]
                                )
                                
                                # Connect the metrics dropdown to update just the metrics plot
                                metrics_combination_dropdown.change(
                                    fn=create_metrics_line_plot,
                                    inputs=[metrics_combination_dropdown],
                                    outputs=[metrics_line_plot]
                                )
                                
                                # Function to update the visualizations
                                def update_charts():
                                    try:
                                        summary_df, individual_dfs, status = load_analysis_results()
                                        
                                        # Get all available combinations for dropdown
                                        available_combinations = []
                                        analysis_dir = os.path.join("audio_ai_optimized", "analysis")
                                        if os.path.exists(analysis_dir):
                                            available_combinations = [d for d in os.listdir(analysis_dir) 
                                                                    if os.path.isdir(os.path.join(analysis_dir, d)) 
                                                                    and d.startswith("chunk")]
                                        
                                        # Sort combinations by chunk length first, then by overlap
                                        def extract_values(combo_name):
                                            # Extract chunk length and overlap values
                                            try:
                                                chunk_length = int(combo_name.split('chunk')[1].split('s_overlap')[0])
                                                overlap = int(combo_name.split('overlap')[1].split('s')[0])
                                                return (chunk_length, overlap)
                                            except:
                                                # If parsing fails, return a large value to put it at the end
                                                return (999999, 999999)
                                        
                                        # Sort the combinations
                                        if available_combinations:
                                            available_combinations.sort(key=extract_values)
                                        
                                        # Update dropdown choices
                                        dropdown_update = gr.update(choices=available_combinations)
                                        metrics_dropdown_update = gr.update(choices=available_combinations)
                                        
                                        if available_combinations:
                                            dropdown_update = gr.update(choices=available_combinations, value=available_combinations[0])
                                            metrics_dropdown_update = gr.update(choices=available_combinations, value=available_combinations[0])
                                            # Also update the chunk detail plot with the first combination
                                            chunk_plot = update_chunk_detail_plot(available_combinations[0])
                                            metrics_plot = create_metrics_line_plot(available_combinations[0])
                                        else:
                                            chunk_plot = None
                                            metrics_plot = None
                                        
                                        if summary_df is None:
                                            return None, gr.Column(), dropdown_update, chunk_plot, metrics_dropdown_update, metrics_plot, f"⚠️ {status}"
                                        
                                        # Create the individual plots container
                                        with gr.Column() as plots_container:
                                            for param, df in individual_dfs.items():
                                                with gr.Row():
                                                    gr.Markdown(f"##### {param}")
                                                with gr.Row():
                                                    gr.ScatterPlot(
                                                        value=df,
                                                        x="chunk_number",
                                                        y="score",
                                                        color="readable_metric",
                                                        tooltip=["readable_metric", "score"],
                                                        title=f"Scores by Chunk: {param}",
                                                        width=800,
                                                        height=300
                                                    )
                                        
                                        return summary_df, plots_container, dropdown_update, chunk_plot, metrics_dropdown_update, metrics_plot, f"✅ {status}"
                                    except Exception as e:
                                        logger.error(f"Error updating charts: {str(e)}")
                                        return None, gr.Column(), gr.update(), None, gr.update(), None, f"⚠️ Error updating charts: {str(e)}"
                                
                                # Connect the refresh button
                                refresh_charts_btn.click(
                                    fn=update_charts,
                                    inputs=[],
                                    outputs=[overview_plot, individual_plots_container, combination_dropdown, chunk_detail_plot, metrics_combination_dropdown, metrics_line_plot, charts_status]
                                )
                                
                                # Connect generate visualizations button
                                generate_visualizations_btn.click(
                                    fn=generate_aggregate_visualizations,
                                    inputs=[],
                                    outputs=[generate_visualizations_output]
                                )
                                
                                # Connect save processing times button
                                save_processing_times_btn.click(
                                    fn=save_batch_processing_times,
                                    inputs=[],
                                    outputs=[save_processing_times_output]
                                )
                                
                                # Initialize charts when tab is selected - COMMENTED OUT DUE TO GRADIO ERROR
                                # analysis_charts_tab.select(
                                #     fn=update_charts,
                                #     inputs=[],
                                #     outputs=[overview_plot, individual_plots_container, charts_status]
                                # )
                
                # Status display (small notification area)
                status_display = gr.Markdown("Ready to prepare optimized audio files")
            
            # Summary column (1/3 width)
            with gr.Column(scale=1):
                gr.Markdown("### Processing Summary")
                summary_display = gr.Textbox(
                    label="Combinations Summary",
                    placeholder="Click 'Calculate Combinations' to see what will be processed",
                    lines=20,
                    interactive=False
                )
                
                # Add Clear Folder button
                clear_folder_btn = gr.Button("Clear AI Optimize Folder", variant="secondary")
                
                # Add Generate Aggregate Visualizations button
                generate_vis_btn = gr.Button("Generate Aggregate Visualizations", variant="secondary")
        
        # Function to update custom value visibility
        def update_custom_visibility(is_checked):
            return gr.update(visible=is_checked)
        
        # Connect custom checkbox to visibility
        chunk_custom.change(
            fn=update_custom_visibility,
            inputs=[chunk_custom],
            outputs=[chunk_custom_value]
        )
        
        overlap_custom.change(
            fn=update_custom_visibility,
            inputs=[overlap_custom],
            outputs=[overlap_custom_value]
        )
        
        # Function to clear the audio_ai_optimized directory
        def clear_ai_optimize_folder():
            """Delete all files and subdirectories in the audio_ai_optimized directory.
            
            Returns:
                str: Status message
            """
            try:
                # Check if directory exists
                if not os.path.exists(AUDIO_AI_OPTIMIZED_DIR):
                    os.makedirs(AUDIO_AI_OPTIMIZED_DIR)
                    return "Directory was empty. Nothing to clear."
                
                # Ensure analysis dir exists
                if not os.path.exists(AI_ANALYSIS_DIR):
                    os.makedirs(AI_ANALYSIS_DIR)
                
                # Count items before deletion
                total_items = 0
                for root, dirs, files in os.walk(AUDIO_AI_OPTIMIZED_DIR):
                    total_items += len(files) + len(dirs)
                
                # Delete all files and subdirectories
                for root, dirs, files in os.walk(AUDIO_AI_OPTIMIZED_DIR, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.error(f"Error removing file {file_path}: {str(e)}")
                    
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        try:
                            os.rmdir(dir_path)
                        except Exception as e:
                            logger.error(f"Error removing directory {dir_path}: {str(e)}")
                
                # Keep the main directory but remove all contents
                return f"✅ Successfully cleared {total_items} items from {AUDIO_AI_OPTIMIZED_DIR} directory"
            except Exception as e:
                logger.error(f"Error clearing audio_ai_optimized directory: {str(e)}")
                return f"❌ Error clearing directory: {str(e)}"
        
        # Function to refresh audio files
        def refresh_audio_files():
            return gr.update(choices=audio_optimizer.get_audio_files())
        
        # Connect refresh button
        refresh_btn.click(
            fn=refresh_audio_files,
            inputs=[],
            outputs=[audio_files]
        )
        
        # Update audio files when tab is selected
        ai_optimize_tab.select(
            fn=refresh_audio_files,
            inputs=[],
            outputs=[audio_files]
        )
        
        # Function to validate chunk and overlap calculations
        def validate_chunk_calculations(audio_duration_ms, chunk_length_ms, overlap_ms):
            """Validate the chunk and overlap calculations.
            
            Args:
                audio_duration_ms (int): Duration of the audio in milliseconds
                chunk_length_ms (int): Length of each chunk in milliseconds
                overlap_ms (int): Length of overlap in milliseconds
                
            Returns:
                tuple: (is_valid, total_chunks, validation_message)
            """
            # Check if overlap is less than chunk length
            if overlap_ms >= chunk_length_ms:
                return False, 0, "Overlap must be less than chunk length"
            
            # Calculate effective chunk length (accounting for overlap)
            effective_chunk_length = chunk_length_ms - overlap_ms
            
            # Calculate total chunks
            if effective_chunk_length <= 0:
                return False, 0, "Effective chunk length must be positive"
                
            total_chunks = max(1, int((audio_duration_ms - overlap_ms) / effective_chunk_length) + 1)
            
            # Validate that the chunks will cover the entire audio
            last_chunk_start = (total_chunks - 1) * effective_chunk_length
            last_chunk_end = min(last_chunk_start + chunk_length_ms, audio_duration_ms)
            
            # Check if the last chunk covers the end of the audio
            if last_chunk_end < audio_duration_ms:
                return False, total_chunks, f"Chunks don't cover entire audio (missing {(audio_duration_ms - last_chunk_end)/1000:.2f}s at the end)"
            
            return True, total_chunks, "Valid configuration"
        
        # Function to calculate combinations
        def calculate_combinations(
            audio_file,
            chunk_10s, chunk_15s, chunk_20s, chunk_30s, chunk_45s, chunk_100s, chunk_200s,
            chunk_custom, chunk_custom_value,
            overlap_0s, overlap_1s, overlap_3s, overlap_5s, overlap_10s, overlap_15s, overlap_20s,
            overlap_custom, overlap_custom_value,
            audio_duration="Full Audio",  # Add audio_duration parameter with default value
            language="English"  # Add language parameter with default value
        ):
            # Start timing the calculation
            start_time = time.time()
            
            # Debug: Print language dropdown information
            try:
                logger.warning("---LANGUAGE DROPDOWN DEBUG INFO---")
                # Print whether 'language' exists in global scope
                logger.warning(f"'language' exists in globals: {('language' in globals())}")
                
                # Attempt to access the language dropdown
                if 'language' in globals():
                    lang_obj = globals()['language']
                    logger.warning(f"Type of language object: {type(lang_obj)}")
                    logger.warning(f"Attributes of language object: {dir(lang_obj)}")
                    if hasattr(lang_obj, 'value'):
                        logger.warning(f"Current language value: {lang_obj.value}")
            except Exception as e:
                logger.error(f"Error in language debug code: {str(e)}")
            
            try:
                audio_path = os.path.join("audio", audio_file)
                audio = AudioSegment.from_file(audio_path)
                audio_duration_ms = len(audio)
                audio_duration_s = audio_duration_ms / 1000
                
                # Apply duration limit if selected
                if audio_duration == "First 1 minute":
                    max_duration_ms = 60 * 1000  # 1 minute = 60 seconds
                    if audio_duration_ms > max_duration_ms:
                        audio_duration_ms = max_duration_ms
                        audio_duration_s = 60
                elif audio_duration == "First 5 minutes":
                    max_duration_ms = 300 * 1000  # 5 minutes = 300 seconds
                    if audio_duration_ms > max_duration_ms:
                        audio_duration_ms = max_duration_ms
                        audio_duration_s = 300
                elif audio_duration == "First 10 minutes":
                    max_duration_ms = 600 * 1000  # 10 minutes = 600 seconds
                    if audio_duration_ms > max_duration_ms:
                        audio_duration_ms = max_duration_ms
                        audio_duration_s = 600
            except Exception as e:
                logger.error(f"Error loading audio file for calculation: {str(e)}")
                error_msg = f"Error loading audio file: {str(e)}"
                return error_msg, error_msg
            
            # Generate summary
            valid_combinations = []
            skipped_combinations = []
            
            # Detect language for processing time estimation
            selected_language = language  # Use the language parameter directly
            
            # Log the language for debugging
            logger.warning(f"*** LANGUAGE VERIFICATION FOR CALCULATIONS ***")
            logger.warning(f"* Selected language from dropdown: {selected_language}")
            
            # Create summary header
            summary = f"Audio file: {audio_file}\n"
            summary += f"Duration: {audio_duration_s:.2f} seconds"
            if audio_duration != "Full Audio":
                summary += f" (limited to first {audio_duration.lower()})"
            summary += f"\nSelected language: {selected_language}\n\n"
            
            # Get all selected chunk and overlap sizes
            chunk_lengths = []
            if chunk_10s: chunk_lengths.append(10)
            if chunk_15s: chunk_lengths.append(15)
            if chunk_20s: chunk_lengths.append(20)
            if chunk_30s: chunk_lengths.append(30)
            if chunk_45s: chunk_lengths.append(45)
            if chunk_100s: chunk_lengths.append(100)
            if chunk_200s: chunk_lengths.append(200)
            if chunk_custom and chunk_custom_value > 0:
                chunk_lengths.append(chunk_custom_value)
            
            overlap_lengths = []
            if overlap_0s: overlap_lengths.append(0)
            if overlap_1s: overlap_lengths.append(1)
            if overlap_3s: overlap_lengths.append(3)
            if overlap_5s: overlap_lengths.append(5)
            if overlap_10s: overlap_lengths.append(10)
            if overlap_15s: overlap_lengths.append(15)
            if overlap_20s: overlap_lengths.append(20)
            if overlap_custom and overlap_custom_value > 0:
                overlap_lengths.append(overlap_custom_value)
            
            # Process each combination
            total_processing_time = 0  # Initialize total processing time
            for chunk_length in chunk_lengths:
                for overlap_length in overlap_lengths:
                    # Skip invalid combinations where overlap >= chunk length
                    if overlap_length >= chunk_length:
                        skipped_combinations.append((chunk_length, overlap_length))
                        continue
                    
                    # Validate the chunk calculations
                    chunk_length_ms = chunk_length * 1000
                    overlap_ms = overlap_length * 1000
                    
                    is_valid, total_chunks, validation_msg = validate_chunk_calculations(
                        audio_duration_ms, chunk_length_ms, overlap_ms
                    )
                    
                    if is_valid:
                        # Estimate processing time if data is available
                        # Note: estimate_processing_time already multiplies by num_chunks
                        est_time = estimate_processing_time(chunk_length, overlap_length, selected_language, total_chunks)
                        if est_time is not None:
                            # est_time is already in seconds for all chunks
                            total_processing_time += est_time  # Add to total in seconds
                            time_in_minutes = est_time / 60  # Convert to minutes for display
                            valid_combinations.append((chunk_length, overlap_length, total_chunks, validation_msg, time_in_minutes))
                        else:
                            valid_combinations.append((chunk_length, overlap_length, total_chunks, validation_msg, None))
                    else:
                        skipped_combinations.append((chunk_length, overlap_length, validation_msg))
            
            if valid_combinations:
                summary += f"Valid combinations to process ({len(valid_combinations)}):\n"
                for chunk, overlap, total_chunks, validation_msg, est_time in valid_combinations:
                    time_info = ""
                    if est_time is not None:
                        time_info = f" - Est. processing time: {est_time:.1f} minutes"
                    else:
                        # No timing data available
                        time_info = " - Est. processing time: unknown"
                    
                    summary += f"- Chunk: {chunk}s, Overlap: {overlap}s - {total_chunks} chunks (VALID){time_info}\n"
                summary += "\n"
            else:
                summary += "No valid combinations to process.\n\n"
            
            if skipped_combinations:
                summary += f"Skipped combinations ({len(skipped_combinations)}):\n"
                for combo in skipped_combinations:
                    if len(combo) == 2:
                        chunk, overlap = combo
                        summary += f"- Chunk: {chunk}s, Overlap: {overlap}s (Overlap must be less than chunk length)\n"
                    else:
                        chunk, overlap, reason = combo
                        summary += f"- Chunk: {chunk}s, Overlap: {overlap}s (Invalid: {reason})\n"
            
            # Format and add total processing time
            if total_processing_time > 0:
                hours = int(total_processing_time // 3600)
                minutes = int((total_processing_time % 3600) // 60)
                seconds = int(total_processing_time % 60)
                
                time_str = ""
                if hours > 0:
                    time_str += f"{hours} hours, "
                if minutes > 0 or hours > 0:
                    time_str += f"{minutes} minutes, "
                time_str += f"{seconds} seconds"
                
                summary += f"\nTotal estimated processing time: {time_str}\n"
            
            return summary, summary
        
        # Function to prepare optimized audio files
        def prepare_optimized_audio(
            audio_file,
            chunk_10s, chunk_15s, chunk_20s, chunk_30s, chunk_45s, chunk_100s, chunk_200s,
            chunk_custom, chunk_custom_value,
            overlap_0s, overlap_1s, overlap_3s, overlap_5s, overlap_10s, overlap_15s, overlap_20s,
            overlap_custom, overlap_custom_value,
            audio_duration
        ):
            # Validate input
            if not audio_file:
                return "Please select an audio file", "Error: Please select an audio file"
            
            # Collect selected chunk lengths
            chunk_lengths = []
            if chunk_10s: chunk_lengths.append(10)
            if chunk_15s: chunk_lengths.append(15)
            if chunk_20s: chunk_lengths.append(20)
            if chunk_30s: chunk_lengths.append(30)
            if chunk_45s: chunk_lengths.append(45)
            if chunk_100s: chunk_lengths.append(100)
            if chunk_200s: chunk_lengths.append(200)
            if chunk_custom and chunk_custom_value > 0:
                chunk_lengths.append(int(chunk_custom_value))
            
            # Collect selected overlap lengths
            overlap_lengths = []
            if overlap_0s: overlap_lengths.append(0)
            if overlap_1s: overlap_lengths.append(1)
            if overlap_3s: overlap_lengths.append(3)
            if overlap_5s: overlap_lengths.append(5)
            if overlap_10s: overlap_lengths.append(10)
            if overlap_15s: overlap_lengths.append(15)
            if overlap_20s: overlap_lengths.append(20)
            if overlap_custom and overlap_custom_value >= 0:
                overlap_lengths.append(int(overlap_custom_value))
            
            # Validate selections
            if not chunk_lengths:
                return "Please select at least one chunk length", "Error: Please select at least one chunk length"
            if not overlap_lengths:
                return "Please select at least one overlap length", "Error: Please select at least one overlap length"
            
            # Sort the lengths
            chunk_lengths.sort()
            overlap_lengths.sort()
            
            try:
                # Prepare the audio file path
                audio_path = os.path.join("audio", audio_file)
                
                # Start the optimization process
                success, summary = audio_optimizer.optimize_audio(
                    audio_path,
                    chunk_lengths,
                    overlap_lengths,
                    max_duration=None if audio_duration == "Full Audio" else (60 if audio_duration == "First 1 minute" else 300)  # 1 minute = 60 seconds, 5 minutes = 300 seconds
                )
                
                if success:
                    status = "✅ Audio optimization completed successfully"
                    return summary, status
                else:
                    status = "❌ Error during audio optimization"
                    return summary, status
            except Exception as e:
                logger.error(f"Error preparing optimized audio: {str(e)}")
                error_msg = f"Error: {str(e)}"
                return error_msg, "❌ Error during audio optimization"
        
        # Function to transcribe chunks
        def transcribe_chunks(selected_language=None, use_prompt_value=None, prompt_chars=None, latest_run_dir=None):
            global transcription_service, is_transcribing, transcription_thread, stop_transcription
            
            # Check if we're already transcribing
            if is_transcribing:
                return "Transcription is already in progress", "⚠️ Transcription already in progress"
            
            # Reset stop flag when starting a new transcription
            stop_transcription = False
            
            # Use the passed language parameter or fall back to the dropdown value
            if not selected_language:
                selected_language = language.value
                
            if not selected_language:
                return "Please select a language for transcription", "❌ No language selected"
                
            # Log the selected language with extra visibility
            logger.info(f"SELECTED LANGUAGE FOR TRANSCRIPTION: {selected_language} (from direct input)")
            
            # Get prompt settings if not provided
            if use_prompt_value is None:
                use_prompt_value = use_prompt.value
            
            if prompt_chars is None and use_prompt_value == "Use Prompt":
                prompt_chars = prompt_length.value
            
            # Log prompt settings
            if use_prompt_value == "Use Prompt":
                logger.info(f"Using prompt with {prompt_chars} characters from previous chunk")
            else:
                logger.info("Not using prompts for transcription")
            
            # Find the latest run directory if not provided
            if not latest_run_dir:
                if not os.path.exists(AUDIO_AI_OPTIMIZED_DIR):
                    return "No optimized audio files found. Please run 'Prepare Audios' first.", "❌ No optimized audio files found"
                
                # Get all run directories
                run_dirs = [d for d in os.listdir(AUDIO_AI_OPTIMIZED_DIR) 
                           if os.path.isdir(os.path.join(AUDIO_AI_OPTIMIZED_DIR, d)) 
                           and d.startswith("run_")]
                
                if not run_dirs:
                    return "No optimization runs found. Please run 'Prepare Audios' first.", "❌ No optimization runs found"
                
                # Sort by timestamp (newest first)
                run_dirs.sort(reverse=True)
                latest_run_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, run_dirs[0])
            
            # Initialize transcription service if needed
            if not transcription_service:
                try:
                    model_path_value = model_path.value
                    transcription_service = TranscriptionService(model_path_value)
                except Exception as e:
                    logger.error(f"Failed to initialize transcription service: {str(e)}")
                    return f"Error initializing transcription service: {str(e)}", "❌ Transcription initialization failed"
            
            # Start transcription in a separate thread
            is_transcribing = True
            transcription_thread = threading.Thread(
                target=transcribe_chunks_thread,
                args=(latest_run_dir, selected_language, use_prompt_value, prompt_chars)
            )
            transcription_thread.daemon = True
            transcription_thread.start()
            
            # Include selected language and prompt info in the status message
            prompt_info = f", With {prompt_chars} character prompts" if use_prompt_value == "Use Prompt" else ""
            status_message = f"🔄 Transcription in progress (Language: {selected_language}{prompt_info})"
            return f"Transcription started with language: {selected_language}{prompt_info}. This may take some time...", status_message
        
        def transcribe_chunks_thread(run_dir, language_name, use_prompt_value="Don't use prompt", prompt_chars=0):
            global is_transcribing, stop_transcription
            
            try:
                # Reset stop flag at the beginning of a new transcription
                stop_transcription = False
                
                # Create a dictionary to track skip reasons
                skip_reasons = {}
                
                # Map language name to code
                language_map = {
                    "Korean": "ko",
                    "English": "en",
                    "Chinese": "zh",
                    "Vietnamese": "vi",
                    "Spanish": "es"
                }
                
                # Get language code and log for debugging
                language_code = language_map.get(language_name, "en")
                language_name_actual = language_name  # Keep original language name for comparison
                
                # Debug log to verify language selection
                logger.warning(f"*** LANGUAGE VERIFICATION ***")
                logger.warning(f"* Selected language name: {language_name_actual}")
                logger.warning(f"* Mapped language code: {language_code}")
                logger.warning(f"* Expected code for Spanish: es")
                
                # Force Spanish if that's what the user selected in UI but somehow got changed
                if language_name.lower() == "spanish":
                    language_code = "es"
                    logger.warning("* Forcing Spanish language code (es) based on language name")
                
                # Log selected language with emphasis and highlighting
                logger.info(f"*** TRANSCRIBING WITH LANGUAGE: {language_name} (CODE: {language_code}) ***")
                logger.debug(f"*** LANGUAGE SELECTION: Using language {language_name} with code {language_code} for transcription ***")
                
                # Log prompt settings
                using_prompts = use_prompt_value == "Use Prompt" and prompt_chars > 0
                if using_prompts:
                    logger.info(f"*** PROMPT SETTINGS: Using {prompt_chars} characters from previous chunk as prompt ***")
                else:
                    logger.info("*** PROMPT SETTINGS: Not using prompts for transcription ***")
                
                # Create a directory for transcriptions and initialize progress file
                transcription_dir = os.path.join(run_dir, "transcriptions")
                os.makedirs(transcription_dir, exist_ok=True)
                
                # Load metadata
                metadata_path = os.path.join(run_dir, "optimization_metadata.json")
                if not os.path.exists(metadata_path):
                    logger.error(f"Metadata file not found: {metadata_path}")
                    return
                
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Prepare results
                results = []
                total_combinations = len(metadata['combinations'])
                processed_combinations = 0
                
                # Statistics for tracking
                total_chunks = 0
                successful_chunks = 0
                failed_chunks = 0
                skipped_chunks = 0
                
                # Calculate total number of chunks across all combinations
                for combo in metadata['combinations']:
                    total_chunks += len(combo['files'])
                
                # Create/update the progress file to track transcription progress
                progress_path = os.path.join(transcription_dir, "transcription_progress.txt")
                with open(progress_path, 'w', encoding='utf-8') as f:
                    f.write(f"Starting transcription with language: {language_name} (code: {language_code})\n")
                    if using_prompts:
                        f.write(f"Using prompts: Yes, {prompt_chars} characters from previous chunk\n")
                    else:
                        f.write("Using prompts: No\n")
                    f.write(f"Total combinations to process: {total_combinations}\n")
                    f.write(f"Total audio chunks to process: {total_chunks}\n\n")
                
                # Start time for overall process
                transcription_start_time = time.time()
                
                # Process each combination
                for combo in metadata['combinations']:
                    # Check if stop was requested
                    if stop_transcription:
                        logger.warning("Transcription process stopped by user")
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write("\n⚠️ TRANSCRIPTION STOPPED BY USER\n")
                            f.write(f"Stopped at combination {processed_combinations}/{total_combinations}\n")
                        break
                    
                    processed_combinations += 1
                    chunk_length = combo['chunk_length']
                    overlap_length = combo['overlap_length']
                    combo_dir = os.path.join(run_dir, combo['directory'])
                    
                    # Start time for this combination
                    combo_start_time = time.time()
                    
                    # Update progress
                    with open(progress_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n===== Processing combination {processed_combinations}/{total_combinations}: chunk{chunk_length}s_overlap{overlap_length}s =====\n")
                        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    # Create a subdirectory for this combination's transcriptions
                    combo_transcription_dir = os.path.join(transcription_dir, f"chunk{chunk_length}s_overlap{overlap_length}s")
                    os.makedirs(combo_transcription_dir, exist_ok=True)
                    
                    # Process each file in this combination
                    combo_results = []
                    combo_successful = 0
                    combo_failed = 0
                    combo_skipped = 0
                    
                    total_files = len(combo['files'])
                    processed_files = 0
                    
                    # Create a concatenated transcription for this combination
                    all_transcriptions = []
                    
                    # Sort files by chunk number for proper ordering
                    sorted_files = sorted(combo['files'], key=lambda x: x['chunk_number'])
                    
                    # Track the last successful activity time to detect stalls
                    last_activity_time = time.time()
                    
                    # Track previous chunk's transcription for prompts
                    previous_transcription = ""
                    
                    for file_info in sorted_files:
                        # Check if stop was requested
                        if stop_transcription:
                            logger.warning(f"Transcription process stopped by user during combination {processed_combinations}/{total_combinations}")
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"\n⚠️ TRANSCRIPTION STOPPED BY USER\n")
                                f.write(f"Stopped at chunk {processed_files}/{total_files}\n")
                            break
                        
                        # Check for process stalling (no activity for more than 10 minutes)
                        if time.time() - last_activity_time > 600:  # 10 minutes
                            logger.warning(f"Transcription process may be stalled - no activity for 10 minutes at chunk {processed_files+1}/{total_files}\n")
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"\n⚠️ POSSIBLE STALL DETECTED: No activity for 10 minutes at chunk {processed_files+1}/{total_files}\n")
                        
                        processed_files += 1
                        filename = file_info['filename']
                        file_path = os.path.join(combo_dir, filename)
                        
                        # Start time for this chunk
                        chunk_start_time = time.time()
                        
                        # Update progress for this file
                        chunk_log_message = f"  Processing chunk {processed_files}/{total_files}: {filename}"
                        logger.info(chunk_log_message)
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"{chunk_log_message} (started at {datetime.now().strftime('%H:%M:%S')})\n")
                        
                        # Skip if file doesn't exist
                        if not os.path.exists(file_path):
                            skip_reason = "file_not_found"
                            skip_reason_desc = "Audio file not found"
                            error_msg = f"{skip_reason_desc}: {file_path}"
                            logger.warning(error_msg)
                            skipped_chunks += 1
                            combo_skipped += 1
                            
                            # Track skip reason
                            if skip_reason not in skip_reasons:
                                skip_reasons[skip_reason] = {
                                    "description": skip_reason_desc,
                                    "count": 0,
                                    "chunks": []
                                }
                            skip_reasons[skip_reason]["count"] += 1
                            skip_reasons[skip_reason]["chunks"].append({
                                "chunk_number": file_info["chunk_number"],
                                "filename": filename,
                                "combination": f"chunk{chunk_length}s_overlap{overlap_length}s"
                            })
                            
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"    ❌ SKIPPED: {error_msg}\n")
                                f.write(f"    ℹ️ INFO: This could be due to the file being deleted or never properly created.\n")
                                f.write(f"    🔍 EXPLANATION: Check if the optimization process completed correctly for this combination.\n")
                            continue
                        
                        # Load audio
                        try:
                            audio = AudioSegment.from_file(file_path)
                            
                            # Check if audio is valid (not empty or too short)
                            if len(audio) < 100:  # Less than 100ms is probably invalid
                                skip_reason = "audio_too_short"
                                skip_reason_desc = "Audio file too short"
                                error_msg = f"{skip_reason_desc}: {file_path} ({len(audio)}ms)"
                                logger.warning(error_msg)
                                skipped_chunks += 1
                                combo_skipped += 1
                                
                                # Track skip reason with additional details
                                if skip_reason not in skip_reasons:
                                    skip_reasons[skip_reason] = {
                                        "description": skip_reason_desc,
                                        "count": 0,
                                        "chunks": []
                                    }
                                skip_reasons[skip_reason]["count"] += 1
                                skip_reasons[skip_reason]["chunks"].append({
                                    "chunk_number": file_info["chunk_number"],
                                    "filename": filename,
                                    "duration_ms": len(audio),
                                    "base_start_time": file_info.get("base_start_time", "N/A"),
                                    "base_end_time": file_info.get("base_end_time", "N/A"),
                                    "combination": f"chunk{chunk_length}s_overlap{overlap_length}s"
                                })
                                
                                with open(progress_path, 'a', encoding='utf-8') as f:
                                    f.write(f"    ❌ SKIPPED: {error_msg}\n")
                                    f.write(f"    ℹ️ INFO: This is likely an empty or corrupted audio file.\n")
                                    if file_info["chunk_number"] == len(sorted_files) - 1:
                                        f.write(f"    🔍 EXPLANATION: This is the last chunk in this combination (chunk #{file_info['chunk_number']}), which is often very short or empty when the audio doesn't divide evenly into {chunk_length}s chunks.\n")
                                    else:
                                        f.write(f"    🔍 EXPLANATION: The audio at this position may be silent or corrupted. Check the original audio file at around {file_info.get('base_start_time', 'N/A')}s.\n")
                                continue
                            
                            # Prepare prompt if enabled and not the first chunk
                            initial_prompt = None
                            if using_prompts and previous_transcription and file_info['chunk_number'] > 0:
                                # Get the last N characters, but make sure not to cut words
                                if len(previous_transcription) <= prompt_chars:
                                    initial_prompt = previous_transcription
                                else:
                                    # Start from the last prompt_chars characters
                                    prompt_text = previous_transcription[-prompt_chars:]
                                    # Find the first space to avoid cutting words
                                    first_space = prompt_text.find(' ')
                                    if first_space > 0:
                                        # If there's a space, start from there
                                        initial_prompt = prompt_text[first_space+1:]
                                    else:
                                        # If no space, just use the whole segment
                                        initial_prompt = prompt_text
                                
                                if initial_prompt:
                                    logger.info(f"Using prompt for chunk {file_info['chunk_number']}: '{initial_prompt[:50]}...' ({len(initial_prompt)} chars)")
                                    with open(progress_path, 'a', encoding='utf-8') as f:
                                        f.write(f"    Using prompt: [{len(initial_prompt)} chars] {initial_prompt[:50]}...\n")
                            
                            # Transcribe with error handling
                            try:
                                transcription = transcription_service.transcribe_chunk(audio, lang_code=language_code, initial_prompt=initial_prompt)
                                
                                # Update last activity time since we successfully processed this chunk
                                last_activity_time = time.time()
                                
                                if transcription:
                                    # Save current transcription for next chunk's prompt
                                    previous_transcription = transcription
                                    
                                    # Calculate elapsed time for this chunk
                                    chunk_elapsed_time = time.time() - chunk_start_time
                                    
                                    # Prepare transcription text with prompt information if used
                                    full_transcription = transcription
                                    if initial_prompt:
                                        full_transcription = f"[PROMPT] {initial_prompt}\n\n{transcription}"
                                    
                                    # Save individual transcription to file
                                    txt_filename = os.path.splitext(filename)[0] + ".txt"
                                    txt_path = os.path.join(combo_transcription_dir, txt_filename)
                                    
                                    with open(txt_path, 'w', encoding='utf-8') as f:
                                        # Only save the raw transcription text, without any prompt info
                                        f.write(transcription)
                                    
                                    # Add to concatenated transcriptions
                                    all_transcriptions.append({
                                        'chunk_number': file_info['chunk_number'],
                                        'start_time': file_info['base_start_time'],
                                        'end_time': file_info['base_end_time'],
                                        'text': transcription,
                                        'prompt_used': bool(initial_prompt),
                                        'prompt_text': initial_prompt if initial_prompt else None,
                                        'prompt_length': len(initial_prompt) if initial_prompt else 0
                                    })
                                    
                                    # Add to results
                                    combo_results.append({
                                        'chunk_number': file_info['chunk_number'],
                                        'filename': filename,
                                        'txt_filename': txt_filename,
                                        'base_start_time': file_info['base_start_time'],
                                        'base_end_time': file_info['base_end_time'],
                                        'actual_start_time': file_info['actual_start_time'],
                                        'actual_end_time': file_info['actual_end_time'],
                                        'transcription': transcription,
                                        'prompt_used': bool(initial_prompt),
                                        'prompt_text': initial_prompt if initial_prompt else None,
                                        'elapsed_time': chunk_elapsed_time
                                    })
                                    
                                    # Log success
                                    prompt_info = f" (with prompt)" if initial_prompt else ""
                                    success_msg = f"    ✅ SUCCESS{prompt_info}: Transcribed in {chunk_elapsed_time:.2f}s"
                                    logger.info(f"Successfully transcribed chunk {processed_files}/{total_files} in {chunk_elapsed_time:.2f}s{prompt_info}")
                                    with open(progress_path, 'a', encoding='utf-8') as f:
                                        f.write(f"{success_msg}\n")
                                    
                                    successful_chunks += 1
                                    combo_successful += 1
                                else:
                                    # Transcription failed but didn't raise an exception
                                    error_msg = f"Failed to transcribe (no result returned)"
                                    logger.warning(f"Transcription failed for chunk {processed_files}/{total_files}: {error_msg}")
                                    with open(progress_path, 'a', encoding='utf-8') as f:
                                        f.write(f"    ❌ FAILED: {error_msg} after {time.time() - chunk_start_time:.2f}s\n")
                                    failed_chunks += 1
                                    combo_failed += 1
                            except Exception as e:
                                # Exception during transcription
                                error_msg = f"Error during transcription: {str(e)}"
                                logger.error(error_msg, exc_info=True)
                                with open(progress_path, 'a', encoding='utf-8') as f:
                                    f.write(f"    ❌ ERROR: {error_msg} after {time.time() - chunk_start_time:.2f}s\n")
                                failed_chunks += 1
                                combo_failed += 1
                        except Exception as e:
                            # Exception during audio loading
                            error_msg = f"Failed to process audio file {filename}: {str(e)}"
                            logger.error(error_msg, exc_info=True)
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"    ❌ ERROR: {error_msg}\n")
                            failed_chunks += 1
                            combo_failed += 1
                    
                    # Calculate elapsed time for this combination
                    combo_elapsed_time = time.time() - combo_start_time
                    
                    # Log combination completion
                    combo_summary = f"Combination completed in {combo_elapsed_time:.2f}s: {combo_successful} successful, {combo_failed} failed, {combo_skipped} skipped"
                    logger.info(combo_summary)
                    with open(progress_path, 'a', encoding='utf-8') as f:
                        f.write(f"\nCombination summary: {combo_summary}\n")
                    
                    # Save concatenated transcription for this combination
                    if all_transcriptions:
                        concatenated_file_path = os.path.join(combo_dir, f"combined_transcription.txt")
                        concatenated_json_path = os.path.join(combo_dir, f"combined_transcription.json")
                        
                        # Sort by chunk number to ensure correct order
                        all_transcriptions.sort(key=lambda x: x['chunk_number'])
                        
                        # Save as plain text file
                        with open(concatenated_file_path, 'w', encoding='utf-8') as f:
                            for item in all_transcriptions:
                                # Convert seconds to HH:MM:SS format
                                start_hours, start_remainder = divmod(item['start_time'], 3600)
                                start_minutes, start_seconds = divmod(start_remainder, 60)
                                start_time_str = f"{int(start_hours):02d}:{int(start_minutes):02d}:{int(start_seconds):02d}"
                                
                                end_hours, end_remainder = divmod(item['end_time'], 3600)
                                end_minutes, end_seconds = divmod(end_remainder, 60)
                                end_time_str = f"{int(end_hours):02d}:{int(end_minutes):02d}:{int(end_seconds):02d}"
                                
                                time_str = f"[{start_time_str} - {end_time_str}]"
                                
                                # First write prompt if used (before timecode)
                                if item.get('prompt_used') and item.get('prompt_text'):
                                    # Preserve line breaks in the prompt text
                                    f.write(f"[PROMPT] {item['prompt_text']}\n")
                                
                                # Then write timecode followed by transcription text
                                f.write(f"{time_str} {item['text']}\n\n")
                        
                        # Save as JSON for more structured data
                        with open(concatenated_json_path, 'w', encoding='utf-8') as f:
                            json.dump(all_transcriptions, f, indent=2)
                        
                        # Add combination results
                        results.append({
                            'chunk_length': chunk_length,
                            'overlap_length': overlap_length,
                            'directory': combo['directory'],
                            'transcription_directory': os.path.relpath(combo_transcription_dir, run_dir),
                            'concatenated_file': os.path.relpath(concatenated_file_path, run_dir),
                            'concatenated_json': os.path.relpath(concatenated_json_path, run_dir),
                            'files': combo_results,
                            'successful': combo_successful,
                            'failed': combo_failed,
                            'skipped': combo_skipped,
                            'elapsed_time': combo_elapsed_time,
                            'using_prompts': using_prompts,
                            'prompt_length': prompt_chars if using_prompts else 0
                        })
                        
                        # Update progress with concatenated file info
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"  Created combined transcription file: {os.path.basename(concatenated_file_path)}\n")
                    else:
                        # No successful transcriptions for this combination
                        logger.warning(f"No successful transcriptions for combination chunk{chunk_length}s_overlap{overlap_length}s")
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"  ⚠️ No successful transcriptions for this combination, no combined file created\n")
                
                # Calculate total elapsed time
                total_elapsed_time = time.time() - transcription_start_time
                
                # Save transcription metadata
                transcription_metadata = {
                    'original_metadata': metadata_path,
                    'language': language_name,
                    'language_code': language_code,
                    'using_prompts': using_prompts,
                    'prompt_length': prompt_chars if using_prompts else 0,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'combinations': results,
                    'statistics': {
                        'total_chunks': total_chunks,
                        'successful': successful_chunks,
                        'failed': failed_chunks,
                        'skipped': skipped_chunks,
                        'elapsed_time': total_elapsed_time
                    },
                    'skip_reasons': skip_reasons
                }
                
                transcription_metadata_path = os.path.join(transcription_dir, "transcription_metadata.json")
                with open(transcription_metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription_metadata, f, indent=2)
                
                # Generate summary
                summary = generate_transcription_summary(transcription_metadata, run_dir)
                
                # Write final summary to progress file
                with open(progress_path, 'a', encoding='utf-8') as f:
                    f.write("\n==============================================\n")
                    if stop_transcription:
                        f.write("⛔️ TRANSCRIPTION STOPPED BY USER ⛔️\n")
                    else:
                        f.write("✅✅✅ TRANSCRIPTION COMPLETE ✅✅✅\n")
                    f.write(f"Total elapsed time: {total_elapsed_time:.2f} seconds\n")
                    f.write(f"Total chunks: {total_chunks}\n")
                    f.write(f"  ✅ Successful: {successful_chunks}\n")
                    f.write(f"  ❌ Failed: {failed_chunks}\n")
                    f.write(f"  ⚠️ Skipped: {skipped_chunks}\n")
                    if total_chunks > 0:
                        f.write(f"Success rate: {successful_chunks/total_chunks*100:.1f}%\n")
                    f.write(f"Created {len(results)} combined transcription files, one for each combination\n")
                
                # We can't directly update the UI from a thread, so we'll write to a file that will be polled
                summary_path = os.path.join(transcription_dir, "transcription_summary.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                
                # Create a skipped chunks report if any were skipped
                if skipped_chunks > 0 and 'skip_reasons' in transcription_metadata:
                    skipped_report_path = os.path.join(transcription_dir, "skipped_chunks_report.txt")
                    with open(skipped_report_path, 'w', encoding='utf-8') as f:
                        f.write("===== SKIPPED CHUNKS DETAILED REPORT =====\n\n")
                        f.write(f"Total skipped chunks: {skipped_chunks} out of {total_chunks} ({skipped_chunks/total_chunks*100:.1f}%)\n\n")
                        
                        # Group by reason
                        f.write("=== SKIPPED CHUNKS BY REASON ===\n")
                        for reason_key, reason_data in transcription_metadata['skip_reasons'].items():
                            f.write(f"\n• {reason_data['description']} - {reason_data['count']} chunks\n")
                            
                            # List chunks with this reason
                            for chunk in reason_data['chunks']:
                                combo = chunk['combination']
                                chunk_num = chunk['chunk_number']
                                filename = chunk['filename']
                                
                                chunk_info = f"  - Chunk #{chunk_num} in {combo}: {filename}"
                                
                                # Add duration info if available
                                if 'duration_ms' in chunk:
                                    chunk_info += f" (duration: {chunk['duration_ms']}ms)"
                                
                                # Add time info if available
                                if 'base_start_time' in chunk and 'base_end_time' in chunk:
                                    if chunk['base_start_time'] != "N/A" and chunk['base_end_time'] != "N/A":
                                        chunk_info += f" - audio region {chunk['base_start_time']:.2f}s-{chunk['base_end_time']:.2f}s"
                                
                                f.write(f"{chunk_info}\n")
                            
                        # Group by combination
                        f.write("\n\n=== SKIPPED CHUNKS BY COMBINATION ===\n")
                        combo_skips = {}
                        
                        # Collect chunks by combination
                        for reason_data in transcription_metadata['skip_reasons'].values():
                            for chunk in reason_data['chunks']:
                                combo = chunk['combination']
                                if combo not in combo_skips:
                                    combo_skips[combo] = []
                                    
                                combo_skips[combo].append({
                                    'chunk_number': chunk['chunk_number'],
                                    'reason': reason_data['description'],
                                    'filename': chunk['filename']
                                })
                        
                        # Output chunks by combination
                        for combo, chunks in sorted(combo_skips.items()):
                            f.write(f"\n• {combo} - {len(chunks)} skipped chunks\n")
                            
                            # Sort by chunk number
                            for chunk in sorted(chunks, key=lambda x: x['chunk_number']):
                                f.write(f"  - Chunk #{chunk['chunk_number']}: {chunk['reason']} - {chunk['filename']}\n")
                        
                        # Identify patterns
                        f.write("\n\n=== SKIP PATTERNS DETECTED ===\n")
                        
                        # Check for skips at the end of combinations (common pattern)
                        last_chunk_skips = 0
                        total_combos = len(metadata['combinations'])
                        combos_with_last_chunk_skipped = set()
                        
                        for combo in metadata['combinations']:
                            combo_name = f"chunk{combo['chunk_length']}s_overlap{combo['overlap_length']}s"
                            if combo_name in combo_skips:
                                combo_file_count = len([f for f in combo['files'] if isinstance(f, dict) and 'chunk_number' in f])
                                
                                for skip in combo_skips[combo_name]:
                                    if skip['chunk_number'] == combo_file_count - 1:
                                        last_chunk_skips += 1
                                        combos_with_last_chunk_skipped.add(combo_name)
                        
                        if last_chunk_skips > 0:
                            f.write(f"• Last-chunk skips: {last_chunk_skips} out of {len(combos_with_last_chunk_skipped)} combinations have the last chunk skipped\n")
                            f.write("  This is normal behavior when the audio doesn't divide evenly into the chosen chunk size.\n")
                            f.write("  The last chunk is often very short or empty, which causes it to be skipped.\n")
                            f.write("  ℹ️ Solution: This is expected behavior and doesn't require any action.\n")
                        
                        # Check for chunks with very short duration
                        very_short_chunks = []
                        for reason_data in transcription_metadata['skip_reasons'].values():
                            if reason_data['description'] == "Audio file too short":
                                for chunk in reason_data['chunks']:
                                    if 'duration_ms' in chunk and chunk['duration_ms'] < 50:
                                        very_short_chunks.append(chunk)
                        
                        if very_short_chunks:
                            f.write(f"\n• Very short chunks: {len(very_short_chunks)} chunks are extremely short (less than 50ms)\n")
                            f.write("  This is typically caused by one of the following:\n")
                            f.write("  1. The audio doesn't divide evenly into the chunk size\n")
                            f.write("  2. The original audio has silent portions\n")
                            f.write("  3. The chunk algorithm created some zero-length chunks\n")
                            f.write("  ℹ️ Solution: Try different chunk sizes or check the original audio for silent sections.\n")
                
                logger.info(f"Transcription complete. Processed {total_chunks} chunks with {successful_chunks} successful ({successful_chunks/total_chunks*100:.1f}% success rate)")
                
            except Exception as e:
                logger.error(f"Critical error during transcription process: {str(e)}", exc_info=True)
                # Write error to progress file
                with open(progress_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n❌ CRITICAL ERROR: {str(e)}\n")
                    f.write("Transcription process interrupted. Please check the logs for details.\n")
            finally:
                is_transcribing = False
                stop_transcription = False  # Reset stop flag
        
        def generate_transcription_summary(metadata, run_dir):
            """Generate a summary of the transcription results."""
            summary = "Transcription Results\n\n"
            
            # Add basic info
            summary += f"Language: {metadata['language']} ({metadata['language_code']})\n"
            summary += f"Timestamp: {metadata['timestamp']}\n"
            summary += f"Directory: {os.path.basename(run_dir)}\n"
            
            # Add prompt information
            if metadata.get('using_prompts', False):
                summary += f"Using prompts: Yes, {metadata.get('prompt_length', 0)} characters from previous chunk\n"
            else:
                summary += "Using prompts: No\n"
            
            # Check if transcription was stopped
            was_stopped = stop_transcription
            if was_stopped:
                summary += f"\n⚠️ TRANSCRIPTION WAS STOPPED BY USER\n"
            
            summary += "\n"
            
            # Add statistics
            if 'statistics' in metadata:
                stats = metadata['statistics']
                total_chunks = stats['total_chunks']
                successful = stats['successful']
                failed = stats['failed']
                skipped = stats['skipped']
                elapsed_time = stats['elapsed_time']
                
                summary += f"Transcription Statistics:\n"
                summary += f"- Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n"
                summary += f"- Total chunks: {total_chunks}\n"
                summary += f"  ✅ Successfully transcribed: {successful} ({successful/total_chunks*100:.1f}%)\n"
                summary += f"  ❌ Failed to transcribe: {failed} ({failed/total_chunks*100:.1f}%)\n"
                summary += f"  ⚠️ Skipped chunks: {skipped} ({skipped/total_chunks*100:.1f}%)\n"
                
                # Add more detailed information about skipped chunks in a prominent section
                if skipped > 0:
                    # Always add a header for skipped chunks section
                    summary += f"\n==== SKIPPED CHUNKS INFORMATION ====\n"
                    
                    if 'skip_reasons' in metadata:
                        skip_reasons = metadata['skip_reasons']
                        
                        # Add breakdown by reason
                        summary += f"Skipped chunks by reason:\n"
                        for reason, data in skip_reasons.items():
                            percentage = data['count']/skipped*100
                            summary += f"• {data['description']}: {data['count']} chunks ({percentage:.1f}%)\n"
                        
                        # Check for last chunk skips (common pattern)
                        last_chunk_skips = 0
                        combos_with_last_chunk_skipped = set()
                        
                        for combo in metadata['combinations']:
                            combo_pattern = f"chunk{combo['chunk_length']}s_overlap{combo['overlap_length']}s"
                            combo_file_count = len([f for f in combo['files'] if isinstance(f, dict) and 'chunk_number' in f])
                            
                            # Look through all skip reasons for this combo
                            for reason_data in skip_reasons.values():
                                for chunk in reason_data['chunks']:
                                    if chunk.get('combination') == combo_pattern and chunk['chunk_number'] == combo_file_count - 1:
                                        last_chunk_skips += 1
                                        combos_with_last_chunk_skipped.add(combo_pattern)
                        
                        # If we detected last chunk skips, highlight this common pattern
                        if last_chunk_skips > 0:
                            summary += f"\n⚠️ Common Pattern Detected: {last_chunk_skips} combinations have their last chunk skipped\n"
                            summary += f"  This is normal behavior when the audio doesn't divide evenly into chunks.\n"
                            summary += f"  The last chunk in each combination is often very short or empty.\n"
                            
                        # Add info about skipped chunks report
                        summary += f"\nA detailed report of all skipped chunks has been generated at:\n"
                        summary += f"- {os.path.join(run_dir, 'transcriptions', 'skipped_chunks_report.txt')}\n"
                    else:
                        # Generic message if we don't have detailed skip reasons
                        summary += f"\nSome chunks were skipped, most likely because they were empty or too short.\n"
                        summary += f"This commonly happens with the last chunk in a combination when the audio doesn't divide evenly.\n"
                    
                    # Add a horizontal line to separate this section
                    summary += f"\n" + "-" * 50 + "\n"
                
                if successful > 0:
                    avg_time_per_chunk = elapsed_time / successful
                    summary += f"- Average time per chunk: {avg_time_per_chunk:.2f} seconds\n"
            else:
                # Count total files (for backward compatibility)
                total_files = sum(len(combo['files']) for combo in metadata['combinations'])
                
                summary += f"Transcribed {total_files} audio chunks across {len(metadata['combinations'])} combinations\n"
            
            summary += f"Created {len(metadata['combinations'])} combined transcription files, one for each combination\n\n"
            
            # Add details for each combination
            summary += "Combination Details:\n"
            for combo in metadata['combinations']:
                chunk_length = combo['chunk_length']
                overlap_length = combo['overlap_length']
                
                summary += f"\n=== Chunk {chunk_length}s, Overlap {overlap_length}s ===\n"
                
                # Add statistics if available
                if 'successful' in combo:
                    total_combo_chunks = combo['successful'] + combo['failed'] + combo['skipped']
                    summary += f"Files: {total_combo_chunks} chunks (✅ {combo['successful']} successful, "
                    summary += f"❌ {combo['failed']} failed, ⚠️ {combo['skipped']} skipped)\n"
                    
                    # Add detailed note about skipped files in this combo
                    if combo['skipped'] > 0:
                        summary += f"Note: {combo['skipped']} chunk(s) skipped in this combination.\n"
                        
                        # Try to find which chunks were skipped in this combination
                        if 'skip_reasons' in metadata:
                            combo_pattern = f"chunk{chunk_length}s_overlap{overlap_length}s"
                            skipped_in_combo = []
                            
                            for reason, data in metadata['skip_reasons'].items():
                                for chunk in data['chunks']:
                                    if chunk.get('combination') == combo_pattern:
                                        skipped_in_combo.append({
                                            'chunk_number': chunk['chunk_number'],
                                            'reason': data['description']
                                        })
                            
                            if skipped_in_combo:
                                summary += "  Skipped chunks in this combination:\n"
                                for skip_info in sorted(skipped_in_combo, key=lambda x: x['chunk_number']):
                                    summary += f"    - Chunk #{skip_info['chunk_number']}: {skip_info['reason']}\n"
                                    
                                # If last chunk was skipped, add explanation
                                last_chunks = [s for s in skipped_in_combo if s['chunk_number'] == total_combo_chunks - 1]
                                if last_chunks:
                                    summary += f"  Note: The last chunk was skipped, likely because the audio doesn't divide evenly into {chunk_length}s chunks.\n"
                    
                    if 'elapsed_time' in combo:
                        summary += f"Processing time: {combo['elapsed_time']:.2f} seconds\n"
                else:
                    summary += f"Files: {len(combo['files'])} chunks\n"
                
                # Add info about the combined transcription file
                if 'concatenated_file' in combo:
                    summary += f"Combined transcription: {os.path.join(run_dir, combo['concatenated_file'])}\n"
                
                # Show sample transcriptions (first 2 chunks)
                if combo['files']:
                    summary += "Sample transcriptions:\n"
                    for i, file_info in enumerate(sorted(combo['files'], key=lambda x: x['chunk_number'])[:2]):
                        summary += f"  Chunk {file_info['chunk_number']}: "
                        # Truncate long transcriptions
                        transcription = file_info['transcription']
                        if len(transcription) > 100:
                            transcription = transcription[:97] + "..."
                        summary += f"{transcription}\n"
                    
                    if len(combo['files']) > 2:
                        summary += f"  ... and {len(combo['files']) - 2} more chunks\n"
            
            # Add path to transcription files
            summary += f"\nTranscription Files Location:\n"
            summary += f"- Base directory: {os.path.join(run_dir, 'transcriptions')}\n"
            summary += f"- Each combination has its own subdirectory with individual .txt files for each audio chunk\n"
            summary += f"- Combined transcription files (one per combination) are saved both in each combination's directory\n"
            summary += f"  and in the transcriptions directory\n"
            summary += f"- Full JSON metadata available at: {os.path.join(run_dir, 'transcriptions', 'transcription_metadata.json')}\n"
            
            # Add troubleshooting info
            if failed > 0 or skipped > 0:
                summary += f"\nTroubleshooting Info:\n"
                if skipped > 0:
                    summary += f"- Skipped chunks are usually due to empty or very short audio files (less than 100ms)\n"
                    summary += f"- This is normal for the last chunk in a combination if the audio doesn't divide evenly\n"
                    
                    # Add common solutions for skipped chunks
                    summary += f"- Common solutions for excessive skipped chunks:\n"
                    summary += f"  1. Try different chunk lengths that divide more evenly into your audio length\n"
                    summary += f"  2. Check the original audio file for silent sections or corrupted segments\n"
                    summary += f"  3. For last-chunk skips (most common), this is normal and can be ignored\n"
                    
                summary += f"- Check the transcription_progress.txt file for detailed error messages and info about skipped chunks\n"
                summary += f"- Progress file: {os.path.join(run_dir, 'transcriptions', 'transcription_progress.txt')}\n"
                summary += f"- See application logs for additional details on failures\n"
            
            return summary
        
        # Function to check transcription progress
        def check_transcription_progress(dummy=None):
            """Check the progress of the transcription process.
            
            Args:
                dummy: Dummy parameter to satisfy Gradio's event handler requirements
            """
            # Find the latest run directory
            run_dirs = [d for d in os.listdir(AUDIO_AI_OPTIMIZED_DIR) 
                       if os.path.isdir(os.path.join(AUDIO_AI_OPTIMIZED_DIR, d)) 
                       and d.startswith("run_")]
            
            if not run_dirs:
                return None
            
            # Sort by timestamp (newest first)
            run_dirs.sort(reverse=True)
            latest_run_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, run_dirs[0])
            
            # Check for summary file first (indicates completion)
            summary_path = os.path.join(latest_run_dir, "transcriptions", "transcription_summary.txt")
            if os.path.exists(summary_path):
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = f.read()
                
                # Add a clear completion message at the top if it's not already there
                if "✅ TRANSCRIPTION COMPLETED SUCCESSFULLY ✅" not in summary:
                    # Check if we have statistics to determine if it was a successful completion
                    if "Transcription Statistics:" in summary and not is_transcribing:
                        summary = "✅ TRANSCRIPTION COMPLETED SUCCESSFULLY ✅\n\n" + summary
                
                # Check if there's a skipped chunks report
                skipped_report_path = os.path.join(latest_run_dir, "transcriptions", "skipped_chunks_report.txt")
                if os.path.exists(skipped_report_path) and "Skipped chunks:" in summary:
                    # Extract skipped count from summary
                    try:
                        skipped_count = 0
                        for line in summary.split('\n'):
                            if "⚠️ Skipped chunks:" in line:
                                parts = line.split(':')
                                if len(parts) > 1:
                                    parts = parts[1].strip().split(' ')
                                    if len(parts) > 0:
                                        skipped_count = int(parts[0])
                                break
                        
                        if skipped_count > 0:
                            # Add a notice about the skipped chunks report
                            # Only add this if it doesn't already exist
                            if "DETAILED SKIPPED CHUNKS REPORT AVAILABLE" not in summary:
                                report_notice = f"\n⚠️ DETAILED SKIPPED CHUNKS REPORT AVAILABLE ⚠️\n"
                                report_notice += f"{skipped_count} chunks were skipped during transcription.\n"
                                report_notice += f"See: {os.path.basename(skipped_report_path)} for complete details about each skipped chunk.\n\n"
                                
                                # Insert after the statistics section
                                if "Transcription Statistics:" in summary:
                                    parts = summary.split("Transcription Statistics:")
                                    if len(parts) > 1:
                                        stats_section_end = parts[1].find("\nCreated ")
                                        if stats_section_end > 0:
                                            summary = parts[0] + "Transcription Statistics:" + parts[1][:stats_section_end] + report_notice + parts[1][stats_section_end:]
                                        else:
                                            summary += report_notice
                                    else:
                                        summary += report_notice
                    except Exception as e:
                        logger.error(f"Error adding skipped chunks report notice: {str(e)}")
                
                return summary
            
            # If no summary, check for progress file
            progress_path = os.path.join(latest_run_dir, "transcriptions", "transcription_progress.txt")
            if os.path.exists(progress_path):
                with open(progress_path, 'r', encoding='utf-8') as f:
                    progress = f.read()
                    
                # Check if transcription is complete based on the content
                if "TRANSCRIPTION COMPLETE" in progress and not is_transcribing:
                    progress = "✅ TRANSCRIPTION COMPLETED SUCCESSFULLY ✅\n\n" + progress
                elif "TRANSCRIPTION STOPPED BY USER" in progress and not is_transcribing:
                    progress = "⛔️ TRANSCRIPTION WAS STOPPED BY USER ⛔️\n\n" + progress
                elif is_transcribing:
                    progress = "🔄 TRANSCRIPTION IN PROGRESS 🔄\n\n" + progress
                
                return progress
            
            # If we're still transcribing but no files exist yet
            if is_transcribing:
                return "🔄 TRANSCRIPTION IN PROGRESS - Initializing transcription process... 🔄"
            
            return None
        
        # Function to stop transcription
        def stop_transcription_process():
            """Signal the transcription process to stop.
            
            Returns:
                str: Status message
            """
            global stop_transcription, is_transcribing
            
            if not is_transcribing:
                return "No transcription is currently running."
            
            stop_transcription = True
            logger.warning("User requested to stop transcription process")
            return "⚠️ Stopping transcription... This may take a moment to complete."
        
        # Connect transcribe button
        transcribe_btn.click(
            fn=transcribe_chunks,
            inputs=[language, use_prompt, prompt_length],
            outputs=[transcription_text, status_display]
        )
        
        # Connect refresh transcription button
        refresh_transcription_btn.click(
            fn=check_transcription_progress,
            inputs=None,
            outputs=transcription_text
        )
        
        # Set up polling for transcription progress - COMMENTED OUT DUE TO GRADIO ERROR
        # transcription_tab.select(
        #     fn=check_transcription_progress,
        #     inputs=None,
        #     outputs=transcription_text
        # )
        
        # Connect calculate button
        calculate_btn.click(
            fn=calculate_combinations,
            inputs=[
                audio_files,
                chunk_10s, chunk_15s, chunk_20s, chunk_30s, chunk_45s, chunk_100s, chunk_200s,
                chunk_custom, chunk_custom_value,
                overlap_0s, overlap_1s, overlap_3s, overlap_5s, overlap_10s, overlap_15s, overlap_20s,
                overlap_custom, overlap_custom_value,
                audio_duration,  # Add audio_duration to inputs
                language  # Add language dropdown to inputs
            ],
            outputs=[summary_display, result_text]
        )
        
        # Connect prepare button
        prepare_btn.click(
            fn=prepare_optimized_audio,
            inputs=[
                audio_files,
                chunk_10s, chunk_15s, chunk_20s, chunk_30s, chunk_45s, chunk_100s, chunk_200s,
                chunk_custom, chunk_custom_value,
                overlap_0s, overlap_1s, overlap_3s, overlap_5s, overlap_10s, overlap_15s, overlap_20s,
                overlap_custom, overlap_custom_value,
                audio_duration
            ],
            outputs=[result_text, status_display]
        )
        
        # Connect clear folder button
        clear_folder_btn.click(
            fn=clear_ai_optimize_folder,
            inputs=[],
            outputs=[status_display]
        )
        
        # Connect generate visualizations button
        generate_vis_btn.click(
            fn=generate_aggregate_visualizations,
            inputs=[],
            outputs=[status_display]
        )
        
        # Connect stop transcription button
        stop_transcription_btn.click(
            fn=stop_transcription_process,
            inputs=[],
            outputs=[status_display]
        )
        
        # Connect use_prompt to show/hide prompt_length
        use_prompt.change(
            fn=lambda x: gr.update(visible=(x == "Use Prompt")),
            inputs=[use_prompt],
            outputs=[prompt_length]
        )
        
        # Function to analyze transcriptions
        def analyze_transcriptions(model_name="mistral", combinations_to_analyze=None):
            global analyzer, is_analyzing, analysis_thread, stop_analysis
            
            # Check if we're already analyzing
            if is_analyzing:
                return "Analysis is already in progress", "⚠️ Analysis already in progress"
            
            # Reset stop flag when starting a new analysis
            stop_analysis = False
            
            # Find the latest run directory if not provided
            if not os.path.exists(AUDIO_AI_OPTIMIZED_DIR):
                return "No optimized audio files found. Please run 'Prepare Audios' first.", "❌ No optimized audio files found"
            
            # Get all run directories
            run_dirs = [d for d in os.listdir(AUDIO_AI_OPTIMIZED_DIR) 
                       if os.path.isdir(os.path.join(AUDIO_AI_OPTIMIZED_DIR, d)) 
                       and d.startswith("run_")]
            
            if not run_dirs:
                return "No optimization runs found. Please run 'Prepare Audios' first.", "❌ No optimization runs found"
            
            # Sort by timestamp (newest first)
            run_dirs.sort(reverse=True)
            latest_run_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, run_dirs[0])
            
            # Initialize analyzer with the selected model
            model_name = model_name if model_name else "mistral"
            logger.info(f"Initializing TranscriptionAnalyzer with model: {model_name}")
            
            try:
                # Create the analyzer with the specified model
                global analyzer
                analyzer = TranscriptionAnalyzer(model_name=model_name)
                logger.info(f"Successfully initialized analyzer with model: {model_name}")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error initializing analyzer with model '{model_name}': {error_msg}")
                
                # If the model was not found, suggest alternatives
                if "model not found" in error_msg.lower() or "no such model" in error_msg.lower():
                    # Try to get a list of available models
                    try:
                        import subprocess
                        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                        if result.returncode == 0:
                            models = [line.split()[0] for line in result.stdout.strip().split("\n") if line.strip() and not line.startswith("NAME")]
                            if models:
                                model_suggestions = ", ".join(models[:5])  # Show up to 5 models
                                suggestion = (
                                    f"Error: Model '{model_name}' not found in Ollama.\n\n"
                                    f"Available models: {model_suggestions}\n\n"
                                    f"Please pull the model with: ollama pull {model_name}\n"
                                    f"Or choose one of the available models from the dropdown."
                                )
                                return suggestion, f"❌ Model '{model_name}' not found"
                    except Exception:
                        pass
                
                # For connection errors, check if Ollama is running
                if "connection refused" in error_msg.lower() or "failed to connect" in error_msg.lower():
                    suggestion = (
                        "Error: Could not connect to the Ollama service.\n\n"
                        "To fix this:\n"
                        "1. Open a terminal window\n"
                        "2. Start the Ollama service with: `ollama serve`\n"
                        "3. Try running the analysis again"
                    )
                    return suggestion, "❌ Ollama service not running"
                
                return f"Error initializing analyzer with model '{model_name}': {error_msg}", f"❌ Error initializing analyzer with model '{model_name}'"
            
            # Start analysis in a separate thread
            is_analyzing = True
            analysis_thread = threading.Thread(
                target=analysis_thread_func,
                args=(latest_run_dir,),
                daemon=True
            )
            analysis_thread.start()
            
            return f"Starting AI analysis of transcriptions with model: {model_name}...", "⏱️ Analysis started"
        
        # Thread function for analysis
        def analysis_thread_func(run_dir):
            global is_analyzing, stop_analysis
            
            try:
                # Initialize tracking variables
                errors = []  # List to track all errors for summary reporting
                
                # Determine if this is an example run or a real run
                is_example_run = "example" in run_dir.lower()
                
                # For real runs, use the AI_ANALYSIS_DIR instead of placing results in the combination directories
                analysis_base_dir = TranscriptionAnalyzer.get_analysis_dir(run_dir)
                
                # Create a progress file
                progress_path = os.path.join(analysis_base_dir, "analysis_progress.txt")
                os.makedirs(os.path.dirname(progress_path), exist_ok=True)
                
                with open(progress_path, 'w', encoding='utf-8') as f:
                    f.write(f"AI Analysis started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Using Ollama model: {analyzer.model_name}\n")
                    f.write(f"Analysis run directory: {run_dir}\n")
                    f.write(f"Analysis output directory: {analysis_base_dir}\n\n")
                
                # Look for transcription data
                transcriptions_dir = os.path.join(run_dir, "transcriptions")
                if not os.path.exists(transcriptions_dir):
                    with open(progress_path, 'a', encoding='utf-8') as f:
                        f.write("❌ ERROR: No transcriptions directory found.\n")
                        f.write(f"Expected path: {transcriptions_dir}\n")
                        f.write("Please run the transcription process first.\n")
                    is_analyzing = False
                    return
                
                # Get all combinations in the transcriptions directory
                combinations = [d for d in os.listdir(transcriptions_dir) 
                               if os.path.isdir(os.path.join(transcriptions_dir, d))]
                
                if not combinations:
                    with open(progress_path, 'a', encoding='utf-8') as f:
                        f.write("❌ ERROR: No transcription combinations found.\n")
                        f.write(f"Directory exists but is empty: {transcriptions_dir}\n")
                        f.write("Please run the transcription process first.\n")
                    is_analyzing = False
                    return
                
                with open(progress_path, 'a', encoding='utf-8') as f:
                    f.write(f"Found {len(combinations)} combinations to analyze:\n")
                    for combo in combinations:
                        f.write(f"- {combo}\n")
                    f.write("\n")
                
                # Track overall statistics
                total_combinations = len(combinations)
                successful_combinations = 0
                failed_combinations = 0
                skipped_combinations = 0
                total_chunks_analyzed = 0
                successful_chunks = 0
                failed_chunks = 0
                
                # Process each combination
                for combo_index, combination in enumerate(combinations):
                    if stop_analysis:
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"\n❌ ANALYSIS STOPPED BY USER at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Completed {combo_index}/{total_combinations} combinations before stopping\n")
                        is_analyzing = False
                        return
                    
                    combo_dir = os.path.join(transcriptions_dir, combination)
                    with open(progress_path, 'a', encoding='utf-8') as f:
                        f.write(f"Processing combination {combo_index+1}/{total_combinations}: {combination}\n")
                        f.write("-" * 50 + "\n")
                    
                    # Find transcription data - we need either combined_transcription.json or individual chunk files
                    combined_json_file = os.path.join(combo_dir, "combined_transcription.json")
                    
                    # Check for transcription data in the transcriptions directory first
                    if not os.path.exists(combined_json_file):
                        # If not in transcriptions subdirectory, check in the main run directory
                        main_dir_combo_path = os.path.join(run_dir, combination)
                        main_dir_json_file = os.path.join(main_dir_combo_path, "combined_transcription.json")
                        
                        if os.path.exists(main_dir_json_file):
                            # Found it in the main directory
                            combined_json_file = main_dir_json_file
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"  ℹ️ Found combined_transcription.json in main directory: {main_dir_json_file}\n")
                        else:
                            # Not found in either location
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"  ⚠️ No combined_transcription.json found for {combination}\n")
                                f.write(f"  Checked paths:\n")
                                f.write(f"  - {combined_json_file}\n")
                                f.write(f"  - {main_dir_json_file}\n")
                                f.write(f"  This combination may not have been transcribed yet. Skipping.\n\n")
                            skipped_combinations += 1
                            continue
                    
                    # Load transcription data
                    try:
                        with open(combined_json_file, 'r', encoding='utf-8') as f:
                            transcription_data = json.load(f)
                        
                        if not transcription_data:
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"  ⚠️ Empty transcription data in {combined_json_file}. Skipping.\n\n")
                            skipped_combinations += 1
                            continue
                        
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"  ✅ Successfully loaded transcription data with {len(transcription_data)} chunks\n")
                        
                        # Create output directory for analysis results within the analysis base directory
                        analysis_dir = os.path.join(analysis_base_dir, combination)
                        os.makedirs(analysis_dir, exist_ok=True)
                        
                        # Create a detailed log file for this combination
                        combo_log_path = os.path.join(analysis_dir, "detailed_analysis.log")
                        with open(combo_log_path, 'w', encoding='utf-8') as f:
                            f.write(f"Analysis started for {combination} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Total chunks to analyze: {len(transcription_data)}\n")
                            f.write("=" * 50 + "\n\n")
                        
                        # Sort chunks by chunk number
                        sorted_chunks = sorted(transcription_data, key=lambda x: x.get("chunk_number", 0))
                        total_chunks = len(sorted_chunks)
                        
                        # Prepare for batch analysis
                        results = []
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"  📊 Starting analysis of {len(sorted_chunks)} chunks...\n")
                        
                        # Process each chunk
                        for i, chunk in enumerate(sorted_chunks):
                            if stop_analysis:
                                with open(progress_path, 'a', encoding='utf-8') as f:
                                    f.write(f"  ❌ Analysis stopped by user during chunk {i+1}/{total_chunks}\n\n")
                                with open(combo_log_path, 'a', encoding='utf-8') as f:
                                    f.write(f"\n❌ ANALYSIS STOPPED BY USER at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                break
                            
                            chunk_number = chunk.get("chunk_number", i)
                            chunk_text = chunk.get("text", "")
                            prompt_used = chunk.get("prompt_text", None) if chunk.get("prompt_used", False) else None
                            
                            # Log the start of processing for this chunk
                            with open(combo_log_path, 'a', encoding='utf-8') as f:
                                f.write(f"Processing chunk {chunk_number} ({i+1}/{total_chunks})\n")
                                f.write(f"Transcription length: {len(chunk_text)} characters\n")
                                if prompt_used:
                                    f.write(f"Prompt used: {prompt_used[:100]}{'...' if len(prompt_used) > 100 else ''}\n")
                            
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"  🔍 Analyzing chunk {i+1}/{total_chunks}: #{chunk_number}\n")
                            
                            # Get previous and next chunks if available
                            previous_chunk = None
                            next_chunk = None
                            
                            if i > 0:
                                previous_chunk = sorted_chunks[i-1].get("text", "")
                            
                            if i < len(sorted_chunks) - 1:
                                next_chunk = sorted_chunks[i+1].get("text", "")
                            
                            try:
                                with open(combo_log_path, 'a', encoding='utf-8') as f:
                                    f.write(f"Starting analysis with model: {analyzer.model_name}\n")
                                    f.write(f"Context: {'First chunk' if previous_chunk is None and next_chunk is not None else 'Last chunk' if previous_chunk is not None and next_chunk is None else 'Middle chunk'}\n")
                                
                                total_chunks_analyzed += 1
                                
                                # Analyze the chunk
                                analysis_start_time = time.time()
                                try:
                                    analysis_result = loop.run_until_complete(
                                        analyzer.analyze_chunk(
                                            current_chunk=chunk_text,
                                            previous_chunk=previous_chunk,
                                            next_chunk=next_chunk,
                                            prompt_used=prompt_used
                                        )
                                    )
                                    analysis_duration = time.time() - analysis_start_time
                                    
                                    # Extract chunk length and overlap from the combination name
                                    chunk_length = 0
                                    overlap_length = 0
                                    chunk_match = re.search(r'chunk(\d+)s_overlap(\d+)s', combination)
                                    if chunk_match:
                                        chunk_length = int(chunk_match.group(1))
                                        overlap_length = int(chunk_match.group(2))
                                    
                                    # Get language information if available
                                    language = "unknown"
                                    if "detected_language" in chunk:
                                        language = chunk["detected_language"]
                                    elif "language" in chunk:
                                        language = chunk["language"]
                                    
                                    # If language is still unknown, check the transcription metadata file
                                    if language == "unknown":
                                        metadata_file = os.path.join(run_dir, "transcriptions", "transcription_metadata.json")
                                        if os.path.exists(metadata_file):
                                            try:
                                                with open(metadata_file, 'r', encoding='utf-8') as f:
                                                    metadata = json.load(f)
                                                    if "language" in metadata:
                                                        language = metadata["language"]
                                                    elif "language_code" in metadata:
                                                        language = metadata["language_code"]
                                            except Exception as e:
                                                logger.warning(f"Error reading metadata file {metadata_file}: {str(e)}")
                                    
                                    # Save processing time data for future estimates if we have valid information
                                    # Only save processing times for non-last chunks as they are standardized
                                    if chunk_length > 0 and overlap_length >= 0 and i < len(sorted_chunks) - 1:
                                        save_processing_time(chunk_length, overlap_length, language, analysis_duration, save_to_static=False)
                                        logger.info(f"Saved processing time data: chunk={chunk_length}s, overlap={overlap_length}s, language={language}, duration={analysis_duration:.2f}s")
                                    elif i == len(sorted_chunks) - 1:
                                        logger.info(f"Skipped saving processing time for last chunk (#{chunk_number}) as it may not be full length")
                                    
                                    # Check if analysis_result is a tuple instead of a GradingResult
                                    if isinstance(analysis_result, tuple):
                                        logger.warning(f"Received tuple instead of GradingResult object. Using first element of tuple.")
                                        # If it's a tuple, assume the first element is the GradingResult and second is raw_response
                                        grading_obj = analysis_result[0]
                                        raw_response = analysis_result[1] if len(analysis_result) > 1 else None
                                    else:
                                        grading_obj = analysis_result
                                        raw_response = None
                                    
                                    # Create a result dictionary
                                    result = {
                                        "chunk_number": chunk_number,
                                        "analysis_result": grading_obj.model_dump(),
                                        "average_score": grading_obj.average_score,
                                        "analysis_duration_seconds": round(analysis_duration, 2),
                                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    
                                    # Save raw response if available
                                    if raw_response:
                                        # Create raw_responses directory
                                        raw_responses_dir = os.path.join(analysis_base_dir, "raw_responses")
                                        os.makedirs(raw_responses_dir, exist_ok=True)
                                        
                                        # Create combination subdirectory
                                        combo_responses_dir = os.path.join(raw_responses_dir, combination)
                                        os.makedirs(combo_responses_dir, exist_ok=True)
                                        
                                        # Create a comprehensive response record
                                        raw_response_with_metadata = {
                                            "chunk_number": chunk_number,
                                            "timestamp": datetime.now().isoformat(),
                                            "analysis_duration_seconds": round(analysis_duration, 2),
                                            "raw_response": raw_response,
                                            "original_model_content": raw_response.get("original_content", "Not available")
                                        }
                                        
                                        # Save individual raw response
                                        response_filename = f"chunk_{chunk_number}_raw_response.json"
                                        with open(os.path.join(combo_responses_dir, response_filename), "w") as f:
                                            json.dump(raw_response_with_metadata, f, indent=2, default=str)
                                        
                                        # Log that we saved the raw response
                                        with open(combo_log_path, 'a', encoding='utf-8') as f:
                                            f.write(f"Saved raw response to {response_filename}\n")
                                    
                                    results.append(result)
                                    successful_chunks += 1
                                    
                                    # Log success
                                    with open(combo_log_path, 'a', encoding='utf-8') as f:
                                        f.write(f"✅ Analysis completed in {analysis_duration:.2f} seconds\n")
                                        f.write(f"Average score: {grading_obj.average_score}/10\n")
                                        f.write("Scores: ")
                                        for key, value in grading_obj.model_dump().items():
                                            if isinstance(value, (int, float)) and key != "average_score":
                                                f.write(f"{key.replace('_', ' ').title()}={value}, ")
                                        f.write("\n\n")
                                    
                                    with open(progress_path, 'a', encoding='utf-8') as f:
                                        f.write(f"    ✅ Score: {grading_obj.average_score}/10 in {analysis_duration:.2f}s\n")
                                    
                                    # Save individual result to file
                                    result_file = os.path.join(analysis_dir, f"chunk_{chunk_number}_analysis.json")
                                    with open(result_file, 'w', encoding='utf-8') as f:
                                        json.dump(result, f, indent=2)
                                    
                                except Exception as e:
                                    error_msg = str(e)
                                    error_type = "analysis error"
                                    retry_suggestion = ""
                                    
                                    # Track the error for summary reporting
                                    errors.append(e)
                                    
                                    # Check for common Ollama errors
                                    if "ollama is not running" in error_msg.lower() or "connection refused" in error_msg.lower():
                                        error_type = "Ollama connection error"
                                        retry_suggestion = "Please start Ollama with 'ollama serve' and try again."
                                    elif "no such model" in error_msg.lower() or "model not found" in error_msg.lower():
                                        error_type = "Ollama model error"
                                        model_name = analyzer.model_name
                                        retry_suggestion = f"Please install the model first with 'ollama pull {model_name}'."
                                    
                                    logger.error(f"Error analyzing chunk {i+1}/{total_chunks}: {error_msg}")
                                    
                                    with open(combo_log_path, 'a', encoding='utf-8') as f:
                                        f.write(f"❌ ERROR: {error_type.upper()} - {error_msg}\n")
                                        if retry_suggestion:
                                            f.write(f"SUGGESTION: {retry_suggestion}\n")
                                        f.write("\n")
                                    
                                    with open(progress_path, 'a', encoding='utf-8') as f:
                                        f.write(f"    ❌ ERROR: {error_type.upper()} - {error_msg}\n")
                                        if retry_suggestion:
                                            f.write(f"    SUGGESTION: {retry_suggestion}\n")
                                    
                                    # Update statistics
                                    failed_chunks += 1
                                    
                                    # If it's an Ollama connection error, stop the entire analysis
                                    if "ollama is not running" in error_msg.lower() or "connection refused" in error_msg.lower():
                                        with open(progress_path, 'a', encoding='utf-8') as f:
                                            f.write(f"\n❌ STOPPING ANALYSIS: Ollama service is not available\n")
                                            f.write(f"Please start Ollama with 'ollama serve' and try again.\n")
                                        
                                        is_analyzing = False
                                        stop_analysis = True
                                        return
                            
                            except Exception as e:
                                logger.error(f"Unexpected error processing chunk {chunk_number}: {str(e)}", exc_info=True)
                                with open(combo_log_path, 'a', encoding='utf-8') as f:
                                    f.write(f"❌ CRITICAL ERROR: {str(e)}\n\n")
                                failed_chunks += 1
                        
                        loop.close()
                        
                        # Save all raw responses to a combined file if any exist
                        raw_responses_dir = os.path.join(analysis_base_dir, "raw_responses", combination)
                        if os.path.exists(raw_responses_dir) and os.listdir(raw_responses_dir):
                            all_raw_responses = []
                            
                            # Collect all individual raw responses
                            for filename in os.listdir(raw_responses_dir):
                                if filename.endswith("_raw_response.json"):
                                    try:
                                        with open(os.path.join(raw_responses_dir, filename), 'r') as f:
                                            response_data = json.load(f)
                                            all_raw_responses.append(response_data)
                                    except Exception as e:
                                        logger.error(f"Error reading raw response file {filename}: {str(e)}")
                            
                            # Save combined raw responses
                            if all_raw_responses:
                                combined_file = os.path.join(raw_responses_dir, "all_raw_responses.json")
                                with open(combined_file, 'w') as f:
                                    json.dump(all_raw_responses, f, indent=2, default=str)
                                
                                with open(combo_log_path, 'a', encoding='utf-8') as f:
                                    f.write(f"\nSaved {len(all_raw_responses)} raw responses to all_raw_responses.json\n")
                        
                        # Create summary for this combination
                        if results:
                            successful_combination_chunks = [r for r in results if "error" not in r]
                            failed_combination_chunks = [r for r in results if "error" in r]
                            
                            # Calculate overall average score
                            overall_average = 0
                            if successful_combination_chunks:
                                overall_average = sum(r["average_score"] for r in successful_combination_chunks) / len(successful_combination_chunks)
                                overall_average = round(overall_average, 2)
                            
                            # Create summary
                            summary = {
                                "combination": combination,
                                "total_chunks": len(results),
                                "successful_chunks": len(successful_combination_chunks),
                                "failed_chunks": len(failed_combination_chunks),
                                "overall_average_score": overall_average,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "results": results
                            }
                            
                            # Save summary to file
                            summary_file = os.path.join(analysis_dir, "summary.json")
                            with open(summary_file, 'w', encoding='utf-8') as f:
                                json.dump(summary, f, indent=2)
                            
                            # Create human-readable summary
                            with open(os.path.join(analysis_dir, "summary.txt"), 'w', encoding='utf-8') as f:
                                f.write(f"Analysis Summary for {combination}\n")
                                f.write("=" * 50 + "\n\n")
                                f.write(f"Total chunks: {len(results)}\n")
                                f.write(f"Successful analyses: {len(successful_combination_chunks)}\n")
                                f.write(f"Failed analyses: {len(failed_combination_chunks)}\n")
                                
                                if successful_combination_chunks:
                                    f.write(f"Overall average score: {overall_average}/10\n\n")
                                    
                                    if len(successful_combination_chunks) > 0:
                                        f.write("Top scoring chunks:\n")
                                        top_chunks = sorted(successful_combination_chunks, key=lambda x: x["average_score"], reverse=True)[:3]
                                        for i, chunk in enumerate(top_chunks):
                                            f.write(f"  {i+1}. Chunk #{chunk['chunk_number']}: {chunk['average_score']}/10\n")
                                        
                                        f.write("\nLowest scoring chunks:\n")
                                        bottom_chunks = sorted(successful_combination_chunks, key=lambda x: x["average_score"])[:3]
                                        for i, chunk in enumerate(bottom_chunks):
                                            f.write(f"  {i+1}. Chunk #{chunk['chunk_number']}: {chunk['average_score']}/10\n")
                            
                                if failed_combination_chunks:
                                    f.write("\nFailed chunks:\n")
                                    for chunk in failed_combination_chunks[:5]:  # Show at most 5 failed chunks
                                        f.write(f"  - Chunk #{chunk['chunk_number']}: {chunk['error'][:100]}...\n")
                            
                            # Log completion
                            with open(combo_log_path, 'a', encoding='utf-8') as f:
                                f.write(f"\n✅ Analysis completed for {combination} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"Total chunks: {len(results)}\n")
                                f.write(f"Successful analyses: {len(successful_combination_chunks)}\n")
                                f.write(f"Failed analyses: {len(failed_combination_chunks)}\n")
                                if successful_combination_chunks:
                                    f.write(f"Overall average score: {overall_average}/10\n")
                            
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"  ✅ Completed analysis of {combination}\n")
                                f.write(f"    - Total chunks: {len(results)}\n")
                                f.write(f"    - Successful: {len(successful_combination_chunks)}\n")
                                f.write(f"    - Failed: {len(failed_combination_chunks)}\n")
                                if successful_combination_chunks:
                                    f.write(f"    - Overall score: {overall_average}/10\n")
                                f.write("\n")
                            
                            if failed_combination_chunks:
                                failed_combinations += 1
                            else:
                                successful_combinations += 1
                            
                        else:
                            with open(combo_log_path, 'a', encoding='utf-8') as f:
                                f.write(f"\n⚠️ No results generated for {combination}\n")
                            
                            with open(progress_path, 'a', encoding='utf-8') as f:
                                f.write(f"  ⚠️ No results generated for {combination}\n\n")
                            
                            failed_combinations += 1
                    
                    except Exception as e:
                        logger.error(f"Error processing combination {combination}: {str(e)}", exc_info=True)
                        with open(progress_path, 'a', encoding='utf-8') as f:
                            f.write(f"  ❌ ERROR processing {combination}: {str(e)}\n\n")
                        failed_combinations += 1
                        
                        # Track the error for summary reporting
                        errors.append(e)
                
                # Create overall summary
                with open(progress_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n✅ Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("SUMMARY:\n")
                    f.write(f"Total combinations found: {total_combinations}\n")
                    f.write(f"Successful combinations: {successful_combinations}\n")
                    f.write(f"Failed combinations: {failed_combinations}\n")
                    f.write(f"Skipped combinations: {skipped_combinations}\n")
                    f.write(f"Total chunks analyzed: {total_chunks_analyzed}\n")
                    f.write(f"Successful chunk analyses: {successful_chunks}\n")
                    f.write(f"Failed chunk analyses: {failed_chunks}\n")
                    
                    if successful_chunks > 0:
                        success_rate = (successful_chunks / total_chunks_analyzed) * 100
                        f.write(f"Success rate: {success_rate:.1f}%\n")
                    
                    f.write("\nTo view detailed results, check the 'analysis' folder in each combination directory.\n")
                    
                    if skipped_combinations > 0:
                        f.write("\nNOTE: Some combinations were skipped because they haven't been transcribed yet.\n")
                        f.write("To analyze these combinations, please run the transcription process for them first.\n")
                
                # Create a user-friendly summary file
                summary_path = os.path.join(analysis_base_dir, "analysis_summary.txt")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("# AI ANALYSIS SUMMARY\n\n")
                    
                    # Check for specific errors
                    if stop_analysis and any("ollama is not running" in str(e) or "connection refused" in str(e) for e in errors):
                        f.write("## ❌ ERROR: OLLAMA SERVICE NOT RUNNING\n\n")
                        f.write("The analysis was stopped because the Ollama service is not running.\n\n")
                        f.write("**To fix this:**\n")
                        f.write("1. Open a terminal window\n")
                        f.write("2. Start the Ollama service with: `ollama serve`\n")
                        f.write("3. Try running the analysis again\n\n")
                        f.write("For more information, please refer to the [Ollama documentation](https://github.com/ollama/ollama).\n\n")
                    elif stop_analysis and any("no such model" in str(e) or "model not found" in str(e) for e in errors):
                        f.write(f"## ❌ ERROR: MODEL NOT FOUND\n\n")
                        f.write(f"The analysis was stopped because the selected model '{analyzer.model_name}' was not found in Ollama.\n\n")
                        f.write("**To fix this:**\n")
                        f.write("1. Open a terminal window\n")
                        f.write(f"2. Pull the model with: `ollama pull {analyzer.model_name}`\n")
                        f.write("3. Try running the analysis again\n\n")
                        f.write("For more information, please refer to the [Ollama documentation](https://github.com/ollama/ollama).\n\n")
                    elif skipped_combinations > 0 and successful_combinations == 0 and failed_combinations == 0:
                        f.write("## ⚠️ NO COMBINATIONS WERE ANALYZED\n\n")
                        f.write("All combinations were skipped because no transcription files were found.\n\n")
                        f.write("**To fix this:**\n")
                        f.write("1. Go to the Transcription tab\n")
                        f.write("2. Select the combinations you want to transcribe\n")
                        f.write("3. Click \"Transcribe\"\n")
                        f.write("4. Return to the AI Analysis tab once transcription is complete\n\n")
                    else:
                        f.write("## ANALYSIS RESULTS\n\n")
                        f.write(f"- **Total combinations found:** {total_combinations}\n")
                        f.write(f"- **Successfully analyzed:** {successful_combinations} combinations\n")
                        f.write(f"- **Failed:** {failed_combinations} combinations\n")
                        f.write(f"- **Skipped:** {skipped_combinations} combinations (not transcribed)\n\n")
                        
                        if total_chunks_analyzed > 0:
                            f.write("### CHUNK DETAILS\n\n")
                            f.write(f"- **Total chunks analyzed:** {total_chunks_analyzed}\n")
                            f.write(f"- **Successful:** {successful_chunks} chunks\n")
                            f.write(f"- **Failed:** {failed_chunks} chunks\n")
                            
                            if successful_chunks > 0:
                                success_rate = (successful_chunks / total_chunks_analyzed) * 100
                                f.write(f"- **Success rate:** {success_rate:.1f}%\n\n")
                        
                        if skipped_combinations > 0:
                            f.write("\n### NOTE\n")
                            f.write("Some combinations were skipped because they haven't been transcribed yet.\n")
                            f.write("To analyze these combinations, please run the transcription process for them first.\n\n")
                        
                        if failed_chunks > 0:
                            f.write("\n### ERRORS\n")
                            f.write("Some chunks failed to analyze. Check the logs for more details.\n")
                            f.write("Common issues include:\n")
                            f.write("- Network connectivity problems\n")
                            f.write("- Ollama service interruptions\n")
                            f.write("- Invalid or empty transcription data\n\n")
                logger.info(f"Analysis completed. Processed {total_combinations} combinations with {successful_chunks} successful chunk analyses")
            
            except Exception as e:
                logger.error(f"Error in analysis thread: {str(e)}")
                try:
                    with open(progress_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n❌ ERROR in analysis process: {str(e)}\n")
                        f.write(f"Error type: {type(e).__name__}\n")
                        f.write(f"This is a critical error in the analysis thread itself.\n")
                except:
                    logger.error("Could not write to progress file")
            
            finally:
                is_analyzing = False
        
        # Function to check analysis progress
        def check_analysis_progress(dummy=None):
            """Check the progress of the analysis process.
            
            Args:
                dummy: Dummy parameter to satisfy Gradio's event handler requirements
            """
            # Find the latest run directory
            if not os.path.exists(AUDIO_AI_OPTIMIZED_DIR):
                return "No optimized audio directory found. Please run 'Prepare Audios' first."
            
            run_dirs = [d for d in os.listdir(AUDIO_AI_OPTIMIZED_DIR) 
                       if os.path.isdir(os.path.join(AUDIO_AI_OPTIMIZED_DIR, d)) 
                       and d.startswith("run_")]
            
            if not run_dirs:
                return "No optimization runs found. Please run 'Prepare Audios' first."
            
            # Sort by timestamp (newest first)
            run_dirs.sort(reverse=True)
            latest_run_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, run_dirs[0])
            
            # Determine the appropriate analysis directory
            analysis_base_dir = TranscriptionAnalyzer.get_analysis_dir(latest_run_dir)
            
            # Check for overall summary/progress
            progress_path = os.path.join(analysis_base_dir, "analysis_progress.txt")
            
            if not os.path.exists(progress_path):
                # No analysis started yet
                if is_analyzing:
                    return "Analysis is running, but no progress file has been created yet...\nPlease wait while the analysis initializes."
                else:
                    return "No analysis has been run yet. Click 'Start Analysis' to begin."
            
            try:
                with open(progress_path, 'r', encoding='utf-8') as f:
                    progress = f.read()
                
                # Check if we have a summary file
                summary_path = os.path.join(analysis_base_dir, "analysis_summary.txt")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary_text = f.read()
                    
                    # Enhanced summary with scoring details
                    enhanced_summary = format_enhanced_summary(analysis_base_dir, summary_text)
                    return enhanced_summary
                
                # Add a status indicator
                status_message = ""
                if "Analysis completed at" in progress:
                    status_message = "\n✅ ANALYSIS COMPLETED SUCCESSFULLY ✅\n\n"
                elif "ANALYSIS STOPPED BY USER" in progress:
                    status_message = "\n⚠️ ANALYSIS STOPPED BY USER ⚠️\n\n"
                elif is_analyzing:
                    status_message = "\n⏱️ ANALYSIS IN PROGRESS... ⏱️\n\n"
                else:
                    status_message = "\n⚠️ ANALYSIS PROCESS ENDED UNEXPECTEDLY ⚠️\n\n"
                
                # Get the last few lines for a summary
                lines = progress.splitlines()
                
                # Extract key information
                start_time = None
                for line in lines:
                    if "AI Analysis started at" in line:
                        start_time = line.replace("AI Analysis started at", "").strip()
                        break
                
                # Count combinations and chunks
                total_combinations = 0
                processed_combinations = 0
                skipped_combinations = 0
                
                for line in lines:
                    if "Found" in line and "combinations to analyze" in line:
                        match = re.search(r"Found (\d+) combinations", line)
                        if match:
                            total_combinations = int(match.group(1))
                    elif "Processing combination" in line:
                        match = re.search(r"Processing combination (\d+)/(\d+)", line)
                        if match:
                            processed_combinations = int(match.group(1))
                    elif "This combination may not have been transcribed yet" in line:
                        skipped_combinations += 1
                
                # Calculate progress percentage
                progress_percentage = 0
                if total_combinations > 0:
                    progress_percentage = min(100, int((processed_combinations / total_combinations) * 100))
                
                # Create a summary
                summary = f"{status_message}Analysis Status:\n"
                if start_time:
                    summary += f"- Started: {start_time}\n"
                
                if total_combinations > 0:
                    summary += f"- Progress: {processed_combinations}/{total_combinations} combinations ({progress_percentage}%)\n"
                    if skipped_combinations > 0:
                        summary += f"- Skipped: {skipped_combinations} combinations (not transcribed)\n"
                
                # Check for summary section
                summary_section = ""
                in_summary = False
                for line in lines:
                    if "SUMMARY:" in line:
                        in_summary = True
                        summary_section = "SUMMARY:\n"
                    elif in_summary and line.strip():
                        summary_section += line + "\n"
                
                if summary_section:
                    summary += f"\n{summary_section}\n"
                
                # Add the last 15 lines of the log for context, but skip the summary section if it's already included
                if not summary_section:
                    summary += "\nRecent Log Entries:\n"
                    summary += "\n".join(lines[-15:])
                
                return summary
            
            except Exception as e:
                logger.error(f"Error reading analysis progress: {str(e)}")
                return f"Error reading analysis progress: {str(e)}"
                
        def format_enhanced_summary(analysis_dir, basic_summary):
            """Create an enhanced summary with detailed scoring information.
            
            Args:
                analysis_dir: Directory containing analysis results
                basic_summary: The basic summary text to enhance
            
            Returns:
                Enhanced summary with detailed scoring information
            """
            # Start with the basic summary
            enhanced = basic_summary
            
            # Find all combination directories
            combo_dirs = [d for d in os.listdir(analysis_dir) 
                         if os.path.isdir(os.path.join(analysis_dir, d))
                         and d.startswith("chunk")]
            
            if not combo_dirs:
                return enhanced + "\nNo combination directories found in analysis folder."
            
            # Add a divider and section header
            enhanced += "\n" + "=" * 50 + "\n"
            enhanced += "📊 DETAILED SCORING BREAKDOWN 📊\n" + "=" * 50 + "\n\n"
            
            # Create an aggregated comparison of combinations
            combo_scores = {}
            combo_detailed_scores = {}
            
            for combo in combo_dirs:
                combo_path = os.path.join(analysis_dir, combo)
                summary_json_path = os.path.join(combo_path, "summary.json")
                
                if not os.path.exists(summary_json_path):
                    continue
                
                try:
                    with open(summary_json_path, 'r') as f:
                        data = json.load(f)
                    
                    if "overall_average_score" in data:
                        combo_scores[combo] = data["overall_average_score"]
                    
                    # Extract detailed metrics if available
                    if "detailed_metrics" in data:
                        combo_detailed_scores[combo] = data["detailed_metrics"]
                except Exception as e:
                    logger.error(f"Error reading summary JSON for {combo}: {str(e)}")
            
            # If we have score data, create a comparison section
            if combo_scores:
                # Sort combinations by score (highest first)
                sorted_combos = sorted(combo_scores.items(), key=lambda x: x[1], reverse=True)
                
                enhanced += "🏆 COMBINATION RANKING 🏆\n\n"
                
                for i, (combo, score) in enumerate(sorted_combos):
                    # Get medal emoji based on rank
                    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                    
                    # Format the score with color indicator
                    score_indicator = get_score_indicator(score)
                    
                    enhanced += f"{medal} {combo}: {score_indicator} ({score:.2f}/10)\n"
                
                enhanced += "\n" + "-" * 50 + "\n\n"
                
                # Add detailed metrics comparison
                enhanced += "📈 DETAILED METRICS COMPARISON 📈\n\n"
                
                # Define metrics to display in order
                metrics_display_order = [
                    "verbatim_match_score", 
                    "sentence_preservation_score", 
                    "content_duplication_score", 
                    "content_loss_score", 
                    "join_transition_score", 
                    "contextual_flow_score"
                ]
                
                # Get readable names for metrics
                metrics_display_names = {
                    "verbatim_match_score": "Verbatim Match",
                    "sentence_preservation_score": "Sentence Preservation",
                    "content_duplication_score": "Content Duplication",
                    "content_loss_score": "Content Loss",
                    "join_transition_score": "Join Transition",
                    "contextual_flow_score": "Contextual Flow"
                }
                
                # Create a header row for the comparison table with combination names
                enhanced += "Metric".ljust(25) + " | " + " | ".join([combo.ljust(20) for combo, _ in sorted_combos]) + "\n"
                enhanced += "-" * 25 + "-+-" + "-+-".join(["-" * 20 for _ in sorted_combos]) + "\n"
                
                # Add rows for each metric
                for metric in metrics_display_order:
                    if any(metric in combo_detailed_scores[combo] for combo, _ in sorted_combos if combo in combo_detailed_scores):
                        metric_name = metrics_display_names.get(metric, metric.replace("_", " ").title())
                        row = metric_name.ljust(25) + " | "
                        
                        for combo, _ in sorted_combos:
                            if combo in combo_detailed_scores and metric in combo_detailed_scores[combo]:
                                score = combo_detailed_scores[combo][metric]
                                score_indicator = get_score_indicator(score)
                                row += f"{score_indicator} ({score:.2f}/10)".ljust(20) + " | "
                            else:
                                row += "N/A".ljust(20) + " | "
                        
                        enhanced += row.rstrip(" |") + "\n"
                
                # Add a note about the best option
                if sorted_combos:
                    best_combo = sorted_combos[0][0]
                    enhanced += f"\n✨ RECOMMENDATION: Based on analysis, '{best_combo}' provides the best overall transcription quality. ✨\n"
            
            return enhanced

        def get_score_indicator(score):
            """Create a visual indicator for a score.
            
            Args:
                score: Numeric score value (0-10)
                
            Returns:
                String with visual indicator
            """
            if score >= 9:
                return "🟢 Excellent"
            elif score >= 7.5:
                return "🟢 Very Good"
            elif score >= 6:
                return "🟡 Good"
            elif score >= 5:
                return "🟡 Average"
            elif score >= 3.5:
                return "🟠 Below Average"
            else:
                return "🔴 Poor"
        
        # Function to stop analysis process
        def stop_analysis_process():
            """Stop the analysis process."""
            global stop_analysis, is_analyzing
            
            if not is_analyzing:
                return "No analysis is currently running."
            
            stop_analysis = True
            return "Stopping analysis... This may take a moment to complete."
        
        # Add event handlers for analysis functions
        analyze_btn.click(
            analyze_transcriptions,
            inputs=[model_dropdown],
            outputs=[analysis_text, status_display]
        )
        
        refresh_analysis_btn.click(
            check_analysis_progress,
            inputs=[],
            outputs=[analysis_text]
        )
        
        stop_analysis_btn.click(
            stop_analysis_process,
            inputs=[],
            outputs=[analysis_text]
        )
        
        # Setup a timer to check analysis progress periodically
        def timer_func():
            while True:
                time.sleep(5)
                if is_analyzing:
                    progress = check_analysis_progress()
                    if progress is not None:
                        # Set the value property directly and trigger an update
                        analysis_text.value = progress
                        # Some UI libraries need an update call after setting value
                        try:
                            # Only call update() if it exists without parameters
                            analysis_text.update()
                        except (AttributeError, TypeError):
                            # If update() doesn't exist or doesn't support being called without parameters, we've already set the value
                            pass
        
        threading.Thread(target=timer_func, daemon=True).start()
    
    return ai_optimize_tab 

# Function to generate aggregate visualizations
def generate_aggregate_visualizations():
    """Generate visualizations of the aggregate analysis results."""
    try:
        # Load results from all analysis directories
        analysis_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, "analysis")
        if not os.path.exists(analysis_dir):
            return "No analysis directory found. Please run an analysis first."
        
        combinations = [d for d in os.listdir(analysis_dir) 
                       if os.path.isdir(os.path.join(analysis_dir, d)) and d.startswith("chunk")]
        
        if not combinations:
            return "No analysis combinations found."
        
        # Load results from each combination
        all_results = []
        for combo in combinations:
            combo_dir = os.path.join(analysis_dir, combo)
            results = load_results(combo_dir)
            if results:
                all_results.append({
                    "combination": combo,
                    "results": results
                })
        
        if not all_results:
            return "No valid analysis results found."
        
        # Create output directory
        output_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create scatter plot of average scores
        average_scores = []
        for combo_result in all_results:
            combo = combo_result["combination"]
            
            # Extract chunk length and overlap from combo name
            chunk_match = re.search(r'chunk(\d+)s_overlap(\d+)s', combo)
            if chunk_match:
                chunk_length = int(chunk_match.group(1))
                overlap_length = int(chunk_match.group(2))
                
                # Calculate average score for this combination
                avg_score = 0
                if combo_result["results"]:
                    avg_score = sum(r["average_score"] for r in combo_result["results"]) / len(combo_result["results"])
                
                average_scores.append({
                    "combination": combo,
                    "chunk_length": chunk_length,
                    "overlap_length": overlap_length,
                    "average_score": avg_score
                })
        
        # Sort by chunk length and overlap length
        average_scores.sort(key=lambda x: (x["chunk_length"], x["overlap_length"]))
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        chunk_lengths = [score["chunk_length"] for score in average_scores]
        overlap_lengths = [score["overlap_length"] for score in average_scores]
        avg_scores = [score["average_score"] for score in average_scores]
        
        # Create scatter plot with size proportional to average score
        scatter = plt.scatter(chunk_lengths, overlap_lengths, c=avg_scores, 
                             s=[score * 20 for score in avg_scores], cmap='viridis', 
                             alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Average Score')
        
        # Add labels for each point
        for i, score in enumerate(average_scores):
            plt.annotate(f"{score['average_score']:.1f}",
                        (chunk_lengths[i], overlap_lengths[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
        plt.xlabel('Chunk Length (seconds)')
        plt.ylabel('Overlap Length (seconds)')
        plt.title('Average Scores by Chunk and Overlap Length')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        scatter_plot_path = os.path.join(output_dir, "average_scores_scatter.png")
        plt.savefig(scatter_plot_path)
        plt.close()
        
        # Create individual plots for each metric
        metrics = ["verbatim_match_score", "sentence_preservation_score", 
                  "content_duplication_score", "content_loss_score", 
                  "join_transition_score", "contextual_flow_score"]
        
        for metric in metrics:
            metric_name = get_readable_metric_name(metric)
            
            plt.figure(figsize=(12, 8))
            
            metric_scores = []
            for combo_result in all_results:
                combo = combo_result["combination"]
                
                # Extract chunk length and overlap from combo name
                chunk_match = re.search(r'chunk(\d+)s_overlap(\d+)s', combo)
                if chunk_match:
                    chunk_length = int(chunk_match.group(1))
                    overlap_length = int(chunk_match.group(2))
                    
                    # Calculate average metric score for this combination
                    avg_metric_score = 0
                    valid_scores = [r["analysis_result"][metric] for r in combo_result["results"] 
                                   if metric in r["analysis_result"]]
                    
                    if valid_scores:
                        avg_metric_score = sum(valid_scores) / len(valid_scores)
                    
                    metric_scores.append({
                        "combination": combo,
                        "chunk_length": chunk_length,
                        "overlap_length": overlap_length,
                        "score": avg_metric_score
                    })
            
            # Sort by chunk length and overlap length
            metric_scores.sort(key=lambda x: (x["chunk_length"], x["overlap_length"]))
            
            # Create scatter plot
            chunk_lengths = [score["chunk_length"] for score in metric_scores]
            overlap_lengths = [score["overlap_length"] for score in metric_scores]
            scores = [score["score"] for score in metric_scores]
            
            scatter = plt.scatter(chunk_lengths, overlap_lengths, c=scores, 
                                 s=[score * 20 for score in scores], cmap='viridis', 
                                 alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label(f'{metric_name} Score')
            
            # Add labels for each point
            for i, score in enumerate(metric_scores):
                plt.annotate(f"{score['score']:.1f}",
                            (chunk_lengths[i], overlap_lengths[i]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center')
            
            plt.xlabel('Chunk Length (seconds)')
            plt.ylabel('Overlap Length (seconds)')
            plt.title(f'{metric_name} by Chunk and Overlap Length')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save plot
            metric_plot_path = os.path.join(output_dir, f"{metric.lower()}_scatter.png")
            plt.savefig(metric_plot_path)
            plt.close()
        
        return f"Visualizations generated successfully in {output_dir}"
    
    except Exception as e:
        logger.error(f"Error generating aggregate visualizations: {e}")
        return f"Error generating visualizations: {str(e)}"

def save_batch_processing_times():
    """Save processing times from completed analysis batch to the static processing times file."""
    try:
        # Load current processing times from the temporary file
        processing_times = load_processing_times()
        
        # If no processing times data is available, check if the analysis directory exists
        if not processing_times or not processing_times.get("combinations"):
            analysis_dir = os.path.join(AUDIO_AI_OPTIMIZED_DIR, "analysis")
            if not os.path.exists(analysis_dir):
                return "No analysis directory or processing times found. Please run an analysis first."
            
            # Get all combination directories
            combinations = [d for d in os.listdir(analysis_dir) 
                           if os.path.isdir(os.path.join(analysis_dir, d)) and d.startswith("chunk")]
            
            if not combinations:
                return "No analysis combinations found."
            
            # If we reach here, there's an analysis directory but no processing times data
            # This is an unusual state, but we'll collect data from the analysis files
            return "No processing times data found. Please run another analysis or check the audio_ai_optimized directory."
        
        # Load static processing times to update
        static_times = load_static_processing_times()
        
        # Track stats for the summary
        total_combinations = len(processing_times["combinations"])
        saved_combinations = 0
        summary_data = []
        
        # Process each combination in the processing_times file
        for combo_key, combo_data in processing_times["combinations"].items():
            if "average" in combo_data and combo_data["average"] > 0:
                # Parse the combination key to get chunk length, overlap, and language
                # Format is like: "chunk200s_overlap5s_Spanish"
                match = re.match(r'chunk(\d+)s_overlap(\d+)s_(.+)', combo_key)
                if match:
                    chunk_length = int(match.group(1))
                    overlap_length = int(match.group(2))
                    language = match.group(3)
                    
                    # Get the average processing time
                    avg_time = combo_data["average"]
                    
                    # Update the static processing times with 1 decimal place rounding
                    if combo_key not in static_times:
                        # New entry - just record the duration directly
                        static_times[combo_key] = round(avg_time, 1)
                    else:
                        # Update existing entry with a weighted average
                        # 75% new data, 25% old data for smooth transitions
                        static_times[combo_key] = round((avg_time * 0.75) + (static_times[combo_key] * 0.25), 1)
                    
                    saved_combinations += 1
                    
                    # Add to summary data
                    summary_data.append({
                        "combination": combo_key,
                        "chunk_length": chunk_length,
                        "overlap_length": overlap_length,
                        "language": language,
                        "average_processing_time": avg_time,
                        "saved_value": static_times[combo_key]
                    })
        
        # Save the updated static processing times
        static_times_file = os.path.join('ai_optimize', 'static_processing_times.json')
        try:
            with open(static_times_file, 'w') as f:
                json.dump(static_times, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving static processing times: {str(e)}")
            return f"Error saving static processing times: {str(e)}"
        
        # Generate summary report
        if not summary_data:
            return "No processing time data could be saved to the static file."
            
        # Sort summary data by processing time (ascending)
        summary_data.sort(key=lambda x: x["average_processing_time"])
        
        # Generate detailed report
        summary = f"Successfully saved processing times to static file:\n"
        summary += f"Saved {saved_combinations} out of {total_combinations} combinations\n"
        summary += f"Average processing times by combination:\n"
        
        for data in summary_data:
            combo = data["combination"]
            avg_time = data["average_processing_time"]
            saved_time = data["saved_value"]
            summary += f"{combo}: {avg_time:.1f}s → {saved_time}s (rounded and saved)\n"
            
        return summary
            
    except Exception as e:
        logger.error(f"Error saving static processing times: {str(e)}")
        return f"Error saving static processing times: {str(e)}"

# Function to load and save processing times
def load_processing_times():
    """Load processing time data from JSON file or create new if not exists"""
    processing_times_file = os.path.join(AUDIO_AI_OPTIMIZED_DIR, 'processing_times.json')
    
    if os.path.exists(processing_times_file):
        try:
            with open(processing_times_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading processing times: {str(e)}")
            return {}
    else:
        # Initialize with empty structure
        return {
            "chunk_length": {},
            "overlap_length": {},
            "language": {},
            "combinations": {}
        }

def load_static_processing_times():
    """Load simplified static processing time data from JSON file or create new if not exists"""
    static_times_file = os.path.join('ai_optimize', 'static_processing_times.json')
    
    if os.path.exists(static_times_file):
        try:
            with open(static_times_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading static processing times: {str(e)}")
            return {}
    else:
        # Initialize with empty dictionary
        return {}

def save_processing_time(chunk_length, overlap_length, language, duration, save_to_static=False):
    """
    Save a new processing time entry to the current run's storage only.
    Only saves to static storage if explicitly requested with save_to_static=True.
    """
    # PART 1: Save to detailed processing times with statistics (existing functionality)
    processing_times = load_processing_times()
    
    # Format the keys consistently
    chunk_key = str(chunk_length)
    overlap_key = str(overlap_length)
    
    # Create combination key
    combo_key = f"chunk{chunk_key}s_overlap{overlap_key}s_{language}"
    
    # Update averages for chunk length
    if chunk_key not in processing_times["chunk_length"]:
        processing_times["chunk_length"][chunk_key] = {"total_time": 0, "count": 0, "average": 0}
    
    processing_times["chunk_length"][chunk_key]["total_time"] += duration
    processing_times["chunk_length"][chunk_key]["count"] += 1
    processing_times["chunk_length"][chunk_key]["average"] = (
        processing_times["chunk_length"][chunk_key]["total_time"] / 
        processing_times["chunk_length"][chunk_key]["count"]
    )
    
    # Update averages for overlap length
    if overlap_key not in processing_times["overlap_length"]:
        processing_times["overlap_length"][overlap_key] = {"total_time": 0, "count": 0, "average": 0}
    
    processing_times["overlap_length"][overlap_key]["total_time"] += duration
    processing_times["overlap_length"][overlap_key]["count"] += 1
    processing_times["overlap_length"][overlap_key]["average"] = (
        processing_times["overlap_length"][overlap_key]["total_time"] / 
        processing_times["overlap_length"][overlap_key]["count"]
    )
    
    # Update averages for language
    if language not in processing_times["language"]:
        processing_times["language"][language] = {"total_time": 0, "count": 0, "average": 0}
    
    processing_times["language"][language]["total_time"] += duration
    processing_times["language"][language]["count"] += 1
    processing_times["language"][language]["average"] = (
        processing_times["language"][language]["total_time"] / 
        processing_times["language"][language]["count"]
    )
    
    # Update specific combination
    if combo_key not in processing_times["combinations"]:
        processing_times["combinations"][combo_key] = {"total_time": 0, "count": 0, "average": 0}
    
    processing_times["combinations"][combo_key]["total_time"] += duration
    processing_times["combinations"][combo_key]["count"] += 1
    processing_times["combinations"][combo_key]["average"] = (
        processing_times["combinations"][combo_key]["total_time"] / 
        processing_times["combinations"][combo_key]["count"]
    )
    
    # Create directory if it doesn't exist
    if not os.path.exists(AUDIO_AI_OPTIMIZED_DIR):
        os.makedirs(AUDIO_AI_OPTIMIZED_DIR)
    
    # Save to file
    processing_times_file = os.path.join(AUDIO_AI_OPTIMIZED_DIR, 'processing_times.json')
    try:
        with open(processing_times_file, 'w') as f:
            json.dump(processing_times, f, indent=2)
    except Exception as e:
        print(f"Error saving processing times: {str(e)}")
    
    # PART 2: Save to static record ONLY if explicitly requested
    if save_to_static:
        static_times = load_static_processing_times()
        
        # Use the same combo key format for consistency
        if combo_key not in static_times:
            # New entry - just record the duration directly with 1 decimal place
            static_times[combo_key] = round(duration, 1)
        else:
            # Update existing entry - use a weighted average to smooth out values
            # This gives more weight to newer measurements (75% new, 25% old)
            # Round to 1 decimal place
            static_times[combo_key] = round((duration * 0.75) + (static_times[combo_key] * 0.25), 1)
        
        # Save simplified times
        static_times_file = os.path.join('ai_optimize', 'static_processing_times.json')
        try:
            with open(static_times_file, 'w') as f:
                json.dump(static_times, f, indent=2)
        except Exception as e:
            print(f"Error saving static processing times: {str(e)}")

def estimate_processing_time(chunk_length, overlap_length, language, num_chunks):
    """
    Estimate processing time based on static record only.
    This function only uses data from the static_processing_times.json file
    and does not fall back to any other data source.
    """
    # Log actual parameters being used for lookup
    logger.info(f"Looking up processing time for: chunk_length={chunk_length}, overlap_length={overlap_length}, language='{language}', num_chunks={num_chunks}")
    
    # Only use simplified static record
    static_times = load_static_processing_times()
    
    # Log available keys in static_times for debugging
    logger.info(f"Available keys in static_processing_times.json: {list(static_times.keys())}")
    
    # Format keys
    chunk_key = str(chunk_length)
    overlap_key = str(overlap_length)
    
    # Try exact match first
    combo_key = f"chunk{chunk_key}s_overlap{overlap_key}s_{language}"
    logger.info(f"Trying exact match with key: '{combo_key}'")
    
    # If not found, try case-insensitive matching
    if combo_key not in static_times:
        logger.info(f"Exact match not found for '{combo_key}', trying case-insensitive match")
        # Try to find a case-insensitive match
        for key in static_times:
            key_parts = key.split('_')
            if len(key_parts) >= 3:
                key_lang = key_parts[-1]
                key_chunk = key_parts[0].replace('chunk', '').replace('s', '')
                key_overlap = key_parts[1].replace('overlap', '').replace('s', '')
                
                logger.info(f"Comparing: key_chunk='{key_chunk}' with '{chunk_key}', key_overlap='{key_overlap}' with '{overlap_key}', key_lang='{key_lang}' with '{language}'")
                logger.info(f"Case-insensitive comparison: '{key_lang.lower()}' == '{language.lower()}' is {key_lang.lower() == language.lower()}")
                
                if (key_chunk == chunk_key and 
                    key_overlap == overlap_key and 
                    key_lang.lower() == language.lower()):
                    combo_key = key
                    logger.info(f"Found case-insensitive match: '{combo_key}'")
                    break
    
    # Check if we have this combination in the static record
    if combo_key in static_times:
        # Use the direct time from static record
        result = static_times[combo_key] * num_chunks
        logger.info(f"Found processing time data: {static_times[combo_key]} seconds per chunk, total: {result} seconds for {num_chunks} chunks")
        return result
    
    # No match found - returning None to show "unknown"
    logger.info(f"No processing time data found for chunk={chunk_length}s, overlap={overlap_length}s, language={language} - showing 'unknown' to collect real data")
        
    # If the exact combination is not found, we return None to indicate
    # that we don't have data for this combination
    return None

def collect_processing_times_from_existing_files():
    """Scan existing analysis files and collect processing times"""
    # Get all analysis directories
    base_dir = 'audio_ai_optimized'
    if not os.path.exists(base_dir):
        logger.info("No audio_ai_optimized directory found")
        return
    
    analysis_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'analysis' in root.lower() and files:
            analysis_dirs.append(root)
    
    if not analysis_dirs:
        logger.info("No existing analysis files found for processing times database")
        return
    
    combinations_found = 0
    
    for analysis_dir in analysis_dirs:
        # Find all chunk analysis files
        chunk_files = []
        for root, dirs, files in os.walk(analysis_dir):
            for file in files:
                if file.endswith('_analysis.json'):
                    chunk_files.append(os.path.join(root, file))
        
        if not chunk_files:
            continue
        
        # Get combination parameters from directory structure
        try:
            # Extract combo name from directory path
            combo = os.path.basename(analysis_dir)
            if not combo.startswith('chunk'):
                combo = os.path.basename(os.path.dirname(analysis_dir))
            
            # Extract chunk length, overlap length, and language
            parts = combo.split('_')
            if len(parts) < 4:
                continue
            
            # Example format: chunk_10000ms_overlap_0ms_en
            chunk_ms = ""
            overlap_ms = ""
            language = "unknown"
            
            for i, part in enumerate(parts):
                if part == "chunk" and i+1 < len(parts) and "ms" in parts[i+1]:
                    chunk_ms = parts[i+1].replace("ms", "")
                elif part == "overlap" and i+1 < len(parts) and "ms" in parts[i+1]:
                    overlap_ms = parts[i+1].replace("ms", "")
                elif part in ["en", "de", "fr", "es", "it", "pt", "nl", "ru", "zh", "ja", "auto"]:
                    language = part
            
            if not chunk_ms or not overlap_ms:
                continue
            
            # Convert to integers
            chunk_length = int(chunk_ms)
            overlap_length = int(overlap_ms)
            
            # Try to locate the corresponding run directory for transcription info
            if language == "unknown":
                # Look for a combined transcription file that might exist in the run directory
                run_dirs = [d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]
                
                # Sort run directories by creation time (newest first)
                run_dirs.sort(reverse=True)
                
                # Try each run directory until we find a match
                combo_without_language = "_".join([p for p in parts if p not in ["en", "de", "fr", "es", "it", "pt", "nl", "ru", "zh", "ja", "auto"]])
                
                for run_dir in run_dirs:
                    run_path = os.path.join(base_dir, run_dir)
                    
                    # First check for transcription metadata file
                    metadata_file = os.path.join(run_path, "transcriptions", "transcription_metadata.json")
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                if "language" in metadata:
                                    language = metadata["language"]
                                    logger.info(f"Found language '{language}' in transcription metadata")
                                    break
                                elif "selected_language" in metadata:
                                    language = metadata["selected_language"]
                                    logger.info(f"Found selected_language '{language}' in transcription metadata")
                                    break
                        except Exception as e:
                            logger.warning(f"Error reading metadata file {metadata_file}: {str(e)}")
                    
                    # If still unknown, check combo directories
                    if language == "unknown":
                        # Check for possible combo directories in this run
                        for item in os.listdir(run_path):
                            combo_dir = os.path.join(run_path, item)
                            if os.path.isdir(combo_dir) and item.startswith(combo_without_language):
                                combined_json = os.path.join(combo_dir, "combined_transcription.json")
                                if os.path.exists(combined_json):
                                    try:
                                        with open(combined_json, 'r', encoding='utf-8') as f:
                                            transcription_data = json.load(f)
                                    
                                        # Look for language information in the first chunk
                                        if transcription_data and len(transcription_data) > 0:
                                            if "detected_language" in transcription_data[0]:
                                                language = transcription_data[0]["detected_language"]
                                                break
                                            elif "language" in transcription_data[0]:
                                                language = transcription_data[0]["language"]
                                                break
                                    except:
                                        pass
                    
                    # If we found language info, no need to check other run directories
                    if language != "unknown":
                        break
            
            # Process each chunk file
            processing_times = []
            
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                    
                    # Check if processing_duration or analysis_duration_seconds is available
                    # Extract chunk number to identify if it's the last chunk
                    chunk_number = -1
                    try:
                        chunk_file_name = os.path.basename(chunk_file)
                        # Extract chunk number from filename (e.g., chunk_005_analysis.json -> 5)
                        chunk_match = re.search(r'chunk_(\d+)_', chunk_file_name)
                        if chunk_match:
                            chunk_number = int(chunk_match.group(1))
                    except:
                        pass
                        
                    if "processing_duration" in chunk_data:
                        processing_times.append((chunk_number, chunk_data["processing_duration"]))
                    elif "analysis_duration_seconds" in chunk_data:
                        processing_times.append((chunk_number, chunk_data["analysis_duration_seconds"]))
                except Exception as e:
                    logger.warning(f"Error reading chunk file {chunk_file}: {str(e)}")
            
            # Calculate average processing time, excluding the last chunk
            if processing_times:
                # Sort by chunk number to identify the last chunk reliably
                processing_times.sort(key=lambda x: x[0])
                
                # Filter out the last chunk if we have more than one chunk
                if len(processing_times) > 1:
                    # Remove the last chunk
                    filtered_times = [time for _, time in processing_times[:-1]]
                else:
                    # If only one chunk, use it (better than nothing)
                    filtered_times = [time for _, time in processing_times]
                
                if filtered_times:  # Check again in case all were filtered
                    avg_time = sum(filtered_times) / len(filtered_times)
                    # Save to processing times database
                    save_processing_time(chunk_length, overlap_length, language, avg_time, save_to_static=False)
                    combinations_found += 1
                    logger.info(f"Saved processing time for {combo} with language '{language}': {avg_time:.2f}s (from {len(filtered_times)} chunks)")
        except Exception as e:
            logger.warning(f"Error processing directory {analysis_dir}: {str(e)}")
    
    logger.info(f"Collected processing times from {combinations_found} existing analysis files")
    return combinations_found

# Function to load AI analysis results from a directory
def load_results(analysis_dir):
    """Load all analysis results from a directory of chunk files"""
    results = []
    chunk_files = glob.glob(os.path.join(analysis_dir, "chunk_*_analysis.json"))
    
    # Sort chunk files by chunk number
    chunk_files.sort(key=lambda x: int(re.search(r'chunk_(\d+)_', os.path.basename(x)).group(1)))
    
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            # Check if this is a valid chunk result
            if "average_score" in chunk_data and "chunk_number" in chunk_data:
                results.append(chunk_data)
        except Exception as e:
            logger.error(f"Error reading chunk file {chunk_file}: {str(e)}")
    
    return results