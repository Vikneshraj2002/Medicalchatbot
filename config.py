## config.py

class Config:
    # API keys
    GROQ_API_KEY = "YOUR API KEY"

    # # Optional: Keep OpenAI API key as fallback or for comparison
    # OPENAI_API_KEY = "your-openai-api-key-here"

    # API endpoints
    GROQ_API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    # Model settings - Make sure to use the same model name throughout the codebase
    # GROQ_MODEL = "llama3-70b-8192-vision"  # Updated to match groq_utils.py
    GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # or another available model

    # Response settings
    MAX_TOKENS = 600
    TEMPERATURE = 0.7  # Optional: Controls randomness in responses

    # Application settings
    DEBUG = True  # Enable/disable debug mode
    TIMEOUT = 60  # Increased API request timeout in seconds

    # Paths
    DATA_PATH = 'D:/project/data/rocov2/'  # Make sure this path exists
    TEMP_DIR = 'D:/project/temp/'  # Make sure this path exists
