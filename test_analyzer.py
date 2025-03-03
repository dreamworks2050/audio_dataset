import asyncio
from ai_optimize.analyzer import TranscriptionAnalyzer

async def test_analyzer():
    print("Initializing TranscriptionAnalyzer...")
    analyzer = TranscriptionAnalyzer()
    print(f"Analyzer initialized with model: {analyzer.model_name}")
    print("Initialization successful!")
    return analyzer

if __name__ == "__main__":
    asyncio.run(test_analyzer()) 