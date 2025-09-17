from google.adk.agents import Agent
import os
from google.adk.tools.agent_tool import AgentTool
from .comparative import comparative_agent
# from .forecasting import forecast_trade_agent
# from .scenario import scenario_analysis_agent

comparative_tool = AgentTool(comparative_agent)
# forecast_tool = AgentTool(forecast_trade_agent)

root_agent = Agent(
   name="zeno_root_agent",
   model="gemini-2.5-flash",
   description=(
       "The Zeno Root Agent is the central orchestrator of the Zeno AI economic analysis platform. "
       "It interprets economist queries and delegates them to specialized sub-agents for forecasting, scenario analysis, and comparative analytics."
   ),
   instruction=(
       "Hello! I am Zeno, your lead economic analysis AI, here to help you with complex economic queries. My primary role is to act as a router, delegating user queries to the most appropriate sub-agent based on the user's intent. "
       "I am rational, friendly, and have a deep understanding of economic principles. "
       "Carefully analyze each query for keywords that indicate a specific type of analysis is needed. "
       "Here are the rules for delegation: "
       "1. Use the 'comparative_analysis_agent' tool when the user's query asks for a direct comparison, contrast, or analysis between two or more economic entities (e.g., 'compare exports of...', 'how do X and Y differ...'). "
       "2. Use the 'forecast_trade' agent (as a tool) for any query that asks for future predictions or trends over time (e.g., 'what is the future trade outlook...', 'predict the next quarter's GDP...'). "
       "3. Use the 'scenario_analysis' agent (as a tool) when the user poses a hypothetical 'what-if' question or asks to simulate an economic event (e.g., 'what if the oil price doubles...', 'simulate the impact of...'). "
       "If no sub-agent is a perfect match, you should still attempt to use the most relevant one and provide a well-structured response based on that agent's capabilities."
   ),
   tools=[
       comparative_tool,
       # forecast_tool,
       # scenario_tool
   ],
)

def main():
   if not os.getenv("GOOGLE_API_KEY"):
       print("CRITICAL ERROR: GOOGLE_API_KEY environment variable is not set. The agent will not be able to connect to the model.")
       return

   print("Zeno Root Agent ready. Type 'quit' to exit.")
   while True:
       user_input = input("You: ")
       if user_input.strip().lower() in {"quit", "exit"}:
           print("Exiting. Goodbye!")
           break
       try:
           response = root_agent.run(user_input, tool_call_config={"allowed_tools": "any"})
           final_response = getattr(response, "text", response)
           print("Zeno:", final_response)
       except Exception as e:
           print(f"ERROR: An exception occurred during the agent's run.")
           print(f"ERROR: Exception details: {e}")

if __name__ == "__main__":
   main()
