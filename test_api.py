import os
from dotenv import load_dotenv
from google import genai

# Load environment variables (to ensure GOOGLE_API_KEY is available)
load_dotenv() 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY not found.")
    exit()

try:
    print("--- Starting API Connectivity Test ---")
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    # Use the same model as your router
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Respond with only the single word: 'TESTED'"
    )
    
    print("\n--- Test Result ---")
    print(f"Status: SUCCESS (API connected)")
    print(f"Response Text: {response.text.strip()}")
    
except Exception as e:
    print("\n--- Test Result ---")
    print(f"Status: FAILURE (API or Network Error)")
    print(f"Error Details: {e}")
    # This will often show a clear error like "400 Bad Request" (invalid key) or a timeout error (network block)