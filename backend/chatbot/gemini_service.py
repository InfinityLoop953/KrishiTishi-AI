import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize the new standard SDK Client
# It automatically picks up the GEMINI_API_KEY environment variable from Render
client = genai.Client()

def ask_gemini(user_message: str) -> str:
    # Set system instruction directly using the standard GenerateContentConfig
    config = types.GenerateContentConfig(
        system_instruction="সবসময় বাংলায় উত্তর দিবে।"
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=user_message,
        config=config,
    )
    
    return response.text