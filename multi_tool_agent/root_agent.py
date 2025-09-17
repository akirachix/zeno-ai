from google.adk.agents import Agent
from .forecasting import ForecastingAgent
from .rag_tools import ask_knowledgebase
from typing import Optional, Dict, Any
import re

forecasting_agent = ForecastingAgent()

def forecast_trade(
    commodity: str,
    metric: str,
    timeframe: str,
    country: str,
    model_type: Optional[str] = None,
    conversation_id: Optional[int] = None,
    run_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Forecast trade metrics (e.g., export_volume, price) for a commodity and country.
    Normalizes commodity inputs (e.g., 'coffee' to 'coffee_arabica' for Kenya).
    Validates inputs, calls ForecastingAgent, and formats results with explanations.
    """

    if not all([commodity, metric, timeframe, country]):
        return {"error": "Missing required parameters: commodity, metric, timeframe, country"}
    
    commodity = commodity.lower().strip()
    metric = metric.lower().replace(" ", "_") 
    country = country.lower().strip()
    
    commodity_mapping = {
        "coffee": "coffee_arabica" if country == "kenya" else "coffee_robusta",  
        "tea": "tea"
    }
    normalized_commodity = commodity_mapping.get(commodity, commodity)
    
    if not re.match(r"next \d+ (years|months)", timeframe.lower()):
        return {"error": "Invalid timeframe format. Use 'next X years' or 'next X months'"}

    params = {
        "commodity": normalized_commodity,
        "metric": metric,
        "timeframe": timeframe,
        "country": country,
        "model_type": model_type,
        "conversation_id": conversation_id,
        "run_id": run_id,
        "original_commodity": commodity  
    }
    
    try:
        result = forecasting_agent.run(params)
        if "error" in result:
            return {"error": result["error"]}
        
        forecast_value = result.get("forecast_value", "Unknown")
        confidence = result.get("confidence", "Medium")
        reasoning = result.get("reasoning", "")
        sources = result.get("sources", [])
        
        explanation = (
            f"Based on analysis of recent reports:\n"
            f" **Forecast**: {forecast_value}\n"
            f" **Confidence**: {confidence}\n"
            f" **Insight**: {reasoning}\n"
            f" **Sources**: {', '.join(sources[:3])}{'...' if len(sources) > 3 else ''}"
        )
        
        return {
            "result": result,
            "explanation": explanation
        }
    except Exception as e:
        return {"error": f"Forecasting failed: {str(e)}"}

root_agent = Agent(
    name="zeno_root_agent",
    model="gemini-1.5-flash",  
    description=(
        "The Zeno Root Agent is an AI Economist Assistant specialized in Kenyan Agricultural trade dynamics. "
        "It combines deep economic theory knowledge with practical analytical tools for forecasting, scenario simulation, and cross-country comparison. "
        "It serves economists by answering conceptual questions, explaining policy impacts, and executing data-driven simulations — all with transparency and source grounding."
    ),
    instruction=(
        "You are Zeno, an AI Economist Assistant for Kenya Agricultural Trade. "
        "Your role is to support Agro-trade economists by answering questions with economic rigor, executing analytical tasks, and providing actionable, explainable insights. "
        "You have two modes of operation:\n\n"
        "MODE 1: DIRECT KNOWLEDGE RESPONSE (No Tool Needed)\n"
        " If the user asks about economic concepts, theories, policies, or general explanations (e.g., 'What is comparative advantage?', 'Explain the impact of tariffs on smallholder farmers', 'How does CGE modeling work?'), "
        " RESPOND DIRECTLY using your fine-tuned economic knowledge. "
        " Structure your answer: 1) Define the concept, 2) Give a Kenya/EAC-relevant example, 3) Mention policy implications, 4) Cite sources if known (e.g., 'According to World Bank 2023...', 'As modeled in KNBS reports...'). "
        " If uncertain, say: 'I don't have enough context to answer confidently, but here's what I know...' — DO NOT HALLUCINATE.\n\n"
        "MODE 2: TOOL-BASED ANALYSIS (Forecasting, Scenario, Comparative)\n"
        " If the user asks for data-driven tasks (forecasting, simulating shocks, comparing countries), delegate to the correct tool:\n"
        "   • FORECASTING: Use forecast_trade for predicting prices, volumes, revenues. Automatically select the best model (ARIMA for small datasets, Prophet for seasonality, XGBoost for large datasets, or Ensemble for robustness) unless specified. "
        "   • SCENARIO: Use scenario_analysis for 'what-if' shocks (drought, tariffs, fuel prices). Always retrieve baseline first.\n"
        "   • COMPARATIVE: Use comparative_analysis for Kenya vs. Uganda/Ethiopia/Tanzania comparisons on exports, policies, productivity.\n"
        " Extract parameters: commodity (maize/coffee/tea, defaulting to context-appropriate specifics like coffee_arabica for Kenya), country, metric (export_volume, price, revenue), timeframe, shock type/magnitude.\n"
        " If commodity is ambiguous (e.g., 'coffee'), assume a default (e.g., Arabica for Kenya, Robusta for Ethiopia) and note the assumption in the response.\n"
        " Validate entities against supported lists before calling tools.\n"
        " After tool returns results, SYNTHESIZE them into a narrative: 'Here's what the model shows...', 'Key insight: ...', 'Policy implication: ...', 'Assumptions: ...', 'Sources: ...'. Explain model choice (e.g., 'Prophet was used due to seasonality') and commodity assumptions (e.g., 'Assumed Arabica coffee for Kenya').\n\n"
        " ALWAYS:\n"
        " Ground responses in real data or economic theory — never guess.\n"
        " Explain assumptions and limitations (e.g., 'This forecast assumes no new policy changes').\n"
        " Use Kenya/EAC context: reference KNBS, EAC treaties, AfCFTA, local commodity boards.\n"
        " Offer to visualize, export, or dive deeper: 'Would you like to see this as a chart?', 'Should I simulate how this changes under drought?'\n"
        " For multi-turn chats: remember context, refer to past answers, and build on them.\n"
        " If user uploads a file (PDF/CSV), acknowledge it and integrate insights if relevant.\n"
        " End with a question or suggestion to keep conversation going: 'Would you like to compare this with Ethiopia's approach?', 'Should we simulate a 10% tariff shock next?'\n\n"
        " NEVER:\n"
        " Ask the user for technical details like model_type unless absolutely necessary.\n"
        " Make up data or sources.\n"
        " Give financial or policy advice without disclaimers.\n"
        " Ignore user's request for explanations or sources.\n"
        " Assume user knows economic jargon — explain terms like 'elasticity', 'CGE', 'VAR' when first used."
    ),
    tools=[
        forecast_trade,
        ask_knowledgebase,
    ],
)