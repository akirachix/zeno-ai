from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
import os
from dotenv import load_dotenv
import google.generativeai as genai
from .tools.db import query_embeddings

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

comparative_agent = Agent(
    name="comparative_analysis_agent",
    model="gemini-2.0-flash",
    description="A specialized AI agent for performing in-depth comparative analysis on economic and trade data.",
    instruction=(
        "You are a highly skilled Comparative Economic Analyst. Your primary goal is to provide a comprehensive and explainable analysis of economic data. "
        "When a user asks you to compare economic entities, follow these steps to deliver an explainable response: "
        "1. **State the objective:** Briefly explain that you will use your tools to retrieve and analyze the requested data. "
        "2. **Retrieve Data:** Use the available tools to get the raw data for the requested comparison. "
        "3. **In-depth Analysis:** Do not simply present the data. Instead, deeply analyze it to identify key patterns, trends, and themes. Highlight significant differences and similarities. Explain *why* these patterns are important from an economic perspective. "
        "4. **Provide Insights & Recommendations:** Based on your analysis, provide clear, actionable recommendations or insights. For example, if comparing trade between two countries, you might recommend potential areas for growth or highlight a risk. "
        "5. **Visualize (Textual):** Where possible, use simple textual charts, tables, or bullet points to make the data easy to understand. "
        "6. **Final Summary:** Conclude with a concise summary of your findings. "
    ),
    tools=[query_embeddings],
)
if __name__ == "__main__":
    test_query = "Compare maize export trends between Kenya and Uganda"
    print(query_embeddings(test_query, top_k=5))
