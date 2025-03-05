import os
import json
import glob
import re
import logging
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logger = logging.getLogger("category_scores_comparison")

def create_category_scores_comparison():
    """
    Create a line plot comparing scores across all categories for each parameter set
    Each line represents a parameter set (combination of chunk length and overlap)
    """
    try:
        # Find all analysis directories
        analysis_dir = os.path.join("audio_ai_optimized", "analysis")
        if not os.path.exists(analysis_dir):
            return None
            
        # Find all parameter sets (combinations)
        combinations = [d for d in os.listdir(analysis_dir) 
                        if os.path.isdir(os.path.join(analysis_dir, d)) 
                        and d.startswith("chunk")]
        
        if not combinations:
            return None
            
        # Sort combinations by chunk length first, then by overlap
        def extract_values(combo_name):
            try:
                chunk_length = int(combo_name.split('chunk')[1].split('s_overlap')[0])
                overlap = int(combo_name.split('overlap')[1].split('s')[0])
                return (chunk_length, overlap)
            except:
                return (999999, 999999)
        
        combinations.sort(key=extract_values)
        
        # Define metrics to compare
        metrics = [
            "verbatim_match_score",
            "sentence_preservation_score",
            "content_duplication_score",
            "content_loss_score",
            "join_transition_score",
            "contextual_flow_score"
        ]
        
        # Create readable metric names for x-axis
        metric_names = {
            "verbatim_match_score": "Verbatim\nMatch",
            "sentence_preservation_score": "Sentence\nPreservation",
            "content_duplication_score": "Content\nDuplication",
            "content_loss_score": "Content\nLoss",
            "join_transition_score": "Join\nTransition",
            "contextual_flow_score": "Contextual\nFlow"
        }
        
        # Set dark theme style
        plt.style.use('dark_background')
        
        # Create figure with dark background
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('#121212')  # Dark background for the figure
        ax.set_facecolor('#1E1E1E')  # Slightly lighter dark background for the plot area
        
        # Define x positions and labels for the metrics
        x_pos = list(range(len(metrics)))
        x_labels = [metric_names[m] for m in metrics]
        
        # Markers and colors for different parameter sets - use brighter colors for dark background
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        # Use a colorful, bright palette for better visibility on dark background
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(combinations)))
        
        # Plot a line for each parameter set (combination)
        legend_items = []
        
        for idx, combination in enumerate(combinations):
            # Calculate average scores for this combination across all chunks
            summary_file = os.path.join(analysis_dir, combination, "summary.json")
            if not os.path.exists(summary_file):
                continue
                
            # Load summary data
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading {summary_file}: {str(e)}")
                continue
            
            # Initialize metric averages
            metric_averages = {metric: 0 for metric in metrics}
            count = 0
            
            # Calculate average scores across all chunks
            if 'results' in summary_data:
                # Calculate total for each metric
                for chunk_result in summary_data['results']:
                    if 'analysis_result' in chunk_result:
                        analysis = chunk_result['analysis_result']
                        for metric in metrics:
                            if metric in analysis:
                                metric_averages[metric] += analysis[metric]
                        count += 1
                
                # Calculate average if we have data
                if count > 0:
                    for metric in metrics:
                        metric_averages[metric] /= count
                    
                    # Extract values for plotting
                    y_values = [metric_averages[metric] for metric in metrics]
                    
                    # Plot the line - make it thicker and brighter for dark theme
                    line, = ax.plot(x_pos, y_values, marker=markers[idx % len(markers)], 
                                    color=colors[idx], linewidth=2.5, markersize=10,
                                    markeredgecolor='white', markeredgewidth=0.5)
                    legend_items.append(line)
            
        # Set plot properties with light colors for text and grid
        ax.set_title("Average Scores by Category Across Parameter Sets", 
                    fontsize=18, color='white', fontweight='bold')
        ax.set_ylim(0, 10.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, color='white', fontsize=12)
        ax.set_ylabel("Average Score (0-10)", fontsize=14, color='white')
        
        # Style the axes and ticks
        ax.spines['bottom'].set_color('#888888')
        ax.spines['left'].set_color('#888888')
        ax.spines['top'].set_color('#1E1E1E')
        ax.spines['right'].set_color('#1E1E1E')
        ax.tick_params(axis='both', colors='white')
        
        # Add grid with lighter color
        ax.grid(True, linestyle='--', alpha=0.3, color='#888888')
        
        # Add horizontal line at score 5 for reference - more visible on dark background
        ax.axhline(y=5, color='#AAAAAA', linestyle='--', alpha=0.7)
        
        # Add legend with light text
        legend = ax.legend(legend_items, combinations, title="Parameter Sets", 
                         loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3)
        legend.get_title().set_color('white')
        for text in legend.get_texts():
            text.set_color('white')
        
        plt.tight_layout()
        
        # Save the figure to a file
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "category_scores_comparison.png"), dpi=300)
        
        return fig
    except Exception as e:
        logger.error(f"Error creating category scores comparison plot: {str(e)}")
        return None

if __name__ == "__main__":
    # Test the function
    create_category_scores_comparison()
    print("Category scores comparison plot created and saved to visualizations/category_scores_comparison.png") 