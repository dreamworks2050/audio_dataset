#!/usr/bin/env python3
"""
Example script showing how to use the enhanced TranscriptionAnalyzer
to analyze audio transcriptions and save individual and grouped results.
"""

import os
import asyncio
import json
import logging
from ai_optimize.analyzer import TranscriptionAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("example_script")

async def analyze_single_transcription():
    """Example of analyzing a single transcription"""
    analyzer = TranscriptionAnalyzer()
    
    # Create sample transcription data
    current_chunk = "This is a sample transcription to test the analysis functionality."
    next_chunk = "The next chunk continues the thought process with more information."
    
    # Example overlap seconds value
    overlap_seconds = 5
    
    print("\n=== Analyzing Single Chunk ===")
    result, raw_response = await analyzer.analyze_chunk(
        current_chunk=current_chunk,
        next_chunk=next_chunk,
        overlap_seconds=overlap_seconds
    )
    
    # Print the result
    print(f"Analysis result average score: {result.average_score}")
    print("Individual scores:")
    print(f"- Verbatim Match Score: {result.verbatim_match_score}")
    print(f"- Sentence Preservation Score: {result.sentence_preservation_score}")
    print(f"- Content Duplication Score: {result.content_duplication_score}")
    print(f"- Content Loss Score: {result.content_loss_score}")
    print(f"- Join Transition Score: {result.join_transition_score}")
    print(f"- Contextual Flow Score: {result.contextual_flow_score}")
    
    # Save the raw response to a file
    os.makedirs("example_output", exist_ok=True)
    with open("example_output/single_chunk_response.json", "w") as f:
        json.dump(raw_response, f, indent=2, default=str)
    
    print(f"Raw response saved to example_output/single_chunk_response.json")
    return result

async def analyze_batch_of_transcriptions():
    """Example of analyzing a batch of transcriptions with parameter grouping"""
    analyzer = TranscriptionAnalyzer()
    
    # Create a sample batch of transcriptions
    transcriptions = []
    
    # Example overlap seconds value
    overlap_seconds = 5
    
    # First chunk
    transcriptions.append({
        "chunk_number": 0,
        "filename": "chunk_0.wav",
        "audio_file": "sample_audio_1.wav",
        "transcription": "This is the first chunk of our sample transcription for testing.",
        "prompt": "Transcribe the following audio",
        "overlap_seconds": overlap_seconds
    })
    
    # Middle chunk
    transcriptions.append({
        "chunk_number": 1,
        "filename": "chunk_1.wav",
        "audio_file": "sample_audio_1.wav",
        "transcription": "This is the middle chunk that continues from the first one.",
        "prompt": "Transcribe the following audio",
        "overlap_seconds": overlap_seconds
    })
    
    # Last chunk
    transcriptions.append({
        "chunk_number": 2,
        "filename": "chunk_2.wav",
        "audio_file": "sample_audio_1.wav",
        "transcription": "This is the final chunk that concludes our transcription.",
        "prompt": "Transcribe the following audio",
        "overlap_seconds": overlap_seconds
    })
    
    # Create output directory
    output_dir = "example_output/batch_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the batch analysis
    print("\n=== Analyzing Batch of Transcriptions ===")
    print(f"Analyzing {len(transcriptions)} chunks...")
    
    result = await analyzer.analyze_batch(
        transcriptions=transcriptions,
        output_dir=output_dir,
        params_name="example_parameters"
    )
    
    print("\nBatch analysis completed!")
    print(f"Results stored in: {output_dir}/analysis")
    print(f"Raw responses stored in: {output_dir}/analysis/raw_responses/example_parameters")
    
    # Print summary
    print("\nSummary:")
    print(f"- Total files: {result['total_files']}")
    print(f"- Successful analyses: {result['successful_analyses']}")
    print(f"- Failed analyses: {result['failed_analyses']}")
    print(f"- Overall average score: {result['overall_average_score']}")
    
    # Print detailed metrics
    print("\nDetailed metrics:")
    for metric, value in result['detailed_metrics'].items():
        print(f"- {metric.replace('_', ' ').title()}: {value}")
    
    return result

async def run_analysis_on_directory():
    """Example of running analysis on a directory with multiple parameter combinations"""
    # This is a placeholder function - in a real scenario, you would have
    # folders with actual transcription files to analyze
    print("\n=== Running Analysis on Directory ===")
    print("This example requires an actual run directory with transcription files.")
    print("To use this functionality:")
    print("  analyzer = TranscriptionAnalyzer()")
    print("  result = await analyzer.run_analysis('path/to/run_directory')")
    
    # The directory structure would typically be:
    # run_directory/
    #   ├── chunk100s_overlap10s/
    #   │   └── combined_transcription.json
    #   ├── chunk150s_overlap20s/
    #   │   └── combined_transcription.json
    #   └── ...
    
    print("\nNOTE: The run_analysis method will:")
    print("1. Automatically read the overlap_seconds value from the split state file")
    print("2. Pass this overlap_seconds value to analyze_batch for each transcription set")
    print("3. Use the overlap_seconds value in the prompt templates for better analysis")

async def main():
    """Run all example functions"""
    print("=== TranscriptionAnalyzer Enhanced Usage Examples ===")
    
    # Example 1: Analyze a single transcription
    await analyze_single_transcription()
    
    # Example 2: Analyze a batch of transcriptions with parameter grouping
    await analyze_batch_of_transcriptions()
    
    # Example 3: Run analysis on a directory with multiple parameter combinations
    await run_analysis_on_directory()
    
    print("\n=== Examples Completed ===")
    print("Check the 'example_output' directory for results")

if __name__ == "__main__":
    asyncio.run(main()) 