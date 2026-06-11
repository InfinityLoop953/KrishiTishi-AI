from google import genai
import os

# Initialize the modern Gemini Client
# It automatically looks for the GEMINI_API_KEY environment variable in os.environ
client = genai.Client()

def ask_gemini(prompt: str) -> str:
    """
    Sends an agricultural query to the Gemini model and returns the response.
    """
    try:
        # Using the standard recommended model for text tasks
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        # Return the clean text response
        if response.text:
            return response.text
        return "দুঃখিত, কোনো উত্তর পাওয়া যায়নি।"
        
    except Exception as e:
        print(f"❌ Gemini API Error: {str(e)}")
        return "দুঃখিত, এই মুহূর্তে এআই সহকারী সক্রিয় নেই। অনুগ্রহ করে কিছুক্ষণ পর আবার চেষ্টা করুন।"