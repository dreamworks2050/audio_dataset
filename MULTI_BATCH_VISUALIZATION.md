# Multi-Batch Visualization Feature

## Overview

The new multi-batch visualization feature allows you to analyze and compare results across multiple batches of transcription analyses. This is particularly useful when you want to:

1. Compare the performance of different parameter sets across multiple batches
2. See the overall effectiveness of various models and settings
3. Generate comprehensive reports that aggregate data from all your test runs

## Key Features

- **Cross-Batch Comparisons**: Compare metrics across different batches
- **Parameter Performance Analysis**: See which parameters perform best across all batches
- **Aggregate Visualizations**: Generate charts showing overall performance trends
- **Comprehensive HTML Report**: Detailed interactive report with all visualizations and metrics

## How to Use

### 1. Basic Usage

```bash
python example_multi_batch_visualization.py example_output
```

This will scan all directories matching the pattern `*batch*` in `example_output` and generate aggregate visualizations.

### 2. Custom Batch Pattern

You can specify a custom pattern to match specific batch directories:

```bash
python example_multi_batch_visualization.py example_output --pattern "*temperature_0.1*"
```

### 3. Programmatic Usage

```python
from ai_optimize.visualize_results import (
    load_multiple_batch_results,
    generate_aggregate_visualizations,
    generate_comprehensive_report
)

# Load results from multiple batches
results, summary = load_multiple_batch_results("example_output", batch_pattern="*batch*")

# Generate aggregate visualizations
generate_aggregate_visualizations(results, "example_output")

# Generate comprehensive report
generate_comprehensive_report(results, summary, "example_output")
```

## Output Files

The feature generates the following files in the `aggregate_visualizations` directory:

- `batch_comparison.png`: Boxplot comparison of average scores across batches
- `parameter_comparison.png`: Boxplot comparison of average scores across parameter sets
- `batch_parameter_comparison.png`: Combined view of batches and parameters
- `metrics_comparison.png`: Comparison of different metrics across all data
- `batch_metrics_heatmap.png`: Heatmap of metrics by batch
- `parameter_metrics_heatmap.png`: Heatmap of metrics by parameter set
- `comprehensive_report.html`: Interactive HTML report with all visualizations and details

## Tips for Best Results

1. **Consistent Naming**: Use consistent naming conventions for your batch directories to make patterns more effective
2. **Multiple Parameter Sets**: Run multiple parameter sets within each batch for better comparisons
3. **Sufficient Data**: Ensure you have enough transcription samples in each batch for meaningful results
4. **Review the Full Report**: The comprehensive HTML report provides the most detailed insights 