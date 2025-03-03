# Enhanced Audio Transcription Analyzer

This system is designed to analyze audio transcriptions for quality and coherence. It evaluates transcription accuracy, smoothness of transitions between chunks, and other key metrics to help identify the most effective transcription parameters.

## Key Features

- **Individual Response Files**: JSON files for each analyzed audio file
- **Parameter-grouped Results**: Results organized by parameter combinations
- **Enhanced Summary Statistics**: Detailed metrics for each parameter set
- **Robust Error Handling**: Graceful handling of API failures
- **Visualization Tools**: Generate comparison charts and reports
  - **NEW: Multi-batch Visualization**: Aggregate and compare results across multiple batches
- **Detailed Logging**: Comprehensive logging for debugging

## Directory Structure

After running the analyzer, results are organized as follows:

```
example_output/
  ├── batch_test/                       # Example batch run
  │   ├── analysis/                     # Analysis results
  │   │   ├── combined/                 # Combined transcription data
  │   │   ├── raw_responses/            # Raw LLM responses
  │   │   │   └── example_parameters/   # Parameter-specific responses
  │   │   │       └── all_grading_results.json   # All grading results
  │   │   └── parameters/               # Parameter summaries
  │   │       └── example_parameters/   # Specific parameter combination
  │   │           ├── results.json      # Detailed results
  │   │           └── summary.json      # Summary statistics
  │   └── visualizations/               # Individual batch visualizations
  │       ├── detailed_metrics.png      # Detailed metrics chart
  │       ├── metrics_heatmap.png       # Heatmap of metrics
  │       ├── overall_comparison.png    # Overall comparison chart
  │       └── report.html               # HTML report for this batch
  │
  └── aggregate_visualizations/         # Cross-batch visualizations
      ├── batch_comparison.png          # Comparison across batches
      ├── parameter_comparison.png      # Parameter comparison
      ├── batch_parameter_comparison.png # Combined batch/parameter view
      └── comprehensive_report.html     # Comprehensive HTML report
```

## Usage

### Basic Analysis (Single Transcription)

```python
from ai_optimize.analyzer import TranscriptionAnalyzer

async def analyze_single():
    analyzer = await TranscriptionAnalyzer.create()
    result = await analyzer.analyze_chunk("This is a sample transcription text.", "example.wav")
    print(f"Analysis score: {result.get('average_score', 0)}")
```

### Batch Analysis (Multiple Transcriptions)

```python
from ai_optimize.analyzer import TranscriptionAnalyzer

async def analyze_batch():
    analyzer = await TranscriptionAnalyzer.create()
    
    # Analysis parameters
    parameters = {
        "model": "mistral-small:24b-instruct-2501-q4_K_M",
        "temperature": 0.1,
        "num_predict": 1000
    }
    
    # Run batch analysis
    results = await analyzer.analyze_batch(
        audio_dir="audio/samples", 
        output_dir="example_output/batch_test",
        parameters=parameters
    )
    
    print(f"Processed {len(results)} files")
```

### Running Analysis on a Directory

```bash
python example_usage.py
```

### Generating Visualizations for a Single Batch

```bash
python example_visualization.py example_output/batch_test
```

### NEW: Generating Aggregate Visualizations Across Multiple Batches

```bash
python example_multi_batch_visualization.py example_output
```

This will:
1. Load results from all batch directories in `example_output`
2. Generate aggregate visualizations comparing performance across batches
3. Create a comprehensive HTML report at `example_output/aggregate_visualizations/comprehensive_report.html`

You can specify a custom pattern to match specific batch directories:
```bash
python example_multi_batch_visualization.py example_output --pattern "*small*"
```

## Example Scripts

- `test_analyzer.py`: Basic initialization test
- `example_usage.py`: Comprehensive example of analyzing transcriptions
- `example_visualization.py`: Generate visualizations for a single batch
- `example_multi_batch_visualization.py`: Generate aggregate visualizations across multiple batches

## Configuration

The analyzer supports various configuration options:

- `model`: LLM model to use for analysis (e.g., "mistral-small:24b-instruct-2501-q4_K_M")
- `temperature`: Controls randomness in responses (lower = more deterministic)
- `chunk_size`: Maximum size of text chunks for analysis
- `retry_count`: Number of retries on API failure

## Requirements

- Python 3.8+
- Ollama API
- pandas
- matplotlib
- seaborn

## Troubleshooting

- **API Connection Issues**: Ensure Ollama is running locally
- **Missing Results**: Check the logs for specific error messages
- **Visualization Errors**: Ensure matplotlib and seaborn are installed 