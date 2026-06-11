import os
import google.generativeai as genai

# We skip load_dotenv() because Render injects environment variables natively!

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "models/gemini-2.5-flash"
)

def ask_gemini(user_message: str):
    response = model.generate_content(
        f"""
        সবসময় বাংলায় উত্তর দিবে।

        প্রশ্ন:
        {user_message}
        """
    )
    return response.text