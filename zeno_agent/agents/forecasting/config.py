import os
from google import genai

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set.")

client = genai.Client(api_key=GOOGLE_API_KEY)

EXCHANGE_RATES = {
    "KES": 0.0075,
    "ETB": 0.0087,
    "RWF": 0.00076,
}

SUPPORTED_COMMODITIES = ["dry maize", "green maize", "Coffee", "Tea", "Maize Flour"]
SUPPORTED_METRICS = ["export_volume", "price", "revenue"]
SUPPORTED_COUNTRIES = ["kenya", "ethiopia", "rwanda"]

ALIAS_MAP = {
    "maize": "dry maize",
    "dry maize": "dry maize",
    "green maize": "green maize",
    "coffee_arabica": "Coffee",
    "coffee_robusta": "Coffee",
    "arabica": "Coffee",
    "robusta": "Coffee",
    "coffee": "Coffee",
    "tea": "Tea",
    "maize flour": "Maize Flour",
}
