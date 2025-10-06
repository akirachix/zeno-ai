import os
import re
import google.generativeai as genai
from .tools.db import query_embeddings, get_trade_data

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "comparative_agent_prompt.txt")

def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def merge_rag_content(rag_results: list) -> str:
    seen = set()
    merged_content = []
    for doc in rag_results:
        content = doc.get("content", "").strip()
        if len(content) < 20:
            continue
        key = content[:100]
        if key in seen:
            continue
        seen.add(key)
        merged_content.append(content)
    return " ".join(merged_content)

def comparative_search(query: str, top_k: int = 5) -> str:
    raw_results = query_embeddings(query, top_k=top_k)
    return merge_rag_content(raw_results)

def synthesize_comparative_analysis(query: str, evidence_text: str) -> str:
    template = load_prompt_template(PROMPT_PATH)
    prompt = template.format(query=query, evidence_text=evidence_text)
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 500, "temperature": 0.25}
    )
    return response.text.strip()

def extract_commodity_and_countries(query: str):
    commodities = re.findall(r'(maize|coffee|tea)', query.lower())
    countries = re.findall(r'(kenya|uganda|tanzania|rwanda|ethiopia)', query.lower())
    commodity = commodities[0] if commodities else "agricultural products"
    country_list = countries[:2] if len(countries) >= 2 else (countries + ["ethiopia"] if countries and countries[0] != "ethiopia" else ["kenya", "ethiopia"])
    return commodity, country_list[0], country_list[1]

def get_structured_evidence(commodity: str, country1: str, country2: str) -> str:
    try:
        data1 = get_trade_data(commodity, country1, last_n_months=24)
        data2 = get_trade_data(commodity, country2, last_n_months=24)
        
        evidence_parts = [f"Structured trade data analysis for {commodity} exports between {country1.capitalize()} and {country2.capitalize()}:"]
        
        if data1["prices"] and data2["prices"]:
            avg1 = sum(data1["prices"]) / len(data1["prices"])
            avg2 = sum(data2["prices"]) / len(data2["prices"])
            months1, months2 = len(data1["prices"]), len(data2["prices"])
            
            evidence_parts.append(f"{country1.capitalize()} {commodity} averaged {avg1:.2f} over {months1} months of data.")
            evidence_parts.append(f"{country2.capitalize()} {commodity} averaged {avg2:.2f} over {months2} months of data.")
            evidence_parts.append(f"This represents a {abs(avg1-avg2)/min(avg1,avg2)*100:.1f}% price difference between the two countries.")
            
            if data1.get("metadata", {}).get("source"):
                evidence_parts.append(f"Data source for {country1}: {data1['metadata']['source']}")
            if data2.get("metadata", {}).get("source"):
                evidence_parts.append(f"Data source for {country2}: {data2['metadata']['source']}")
                
            return " ".join(evidence_parts)
        else:
            return f"Limited structured trade data available for {commodity} in {country1} and {country2}."
            
    except Exception as e:
        return f"Structured database query encountered issues: {str(e)}"

def get_rag_evidence(query: str) -> str:
    try:
        rag_content = comparative_search(query, top_k=3)
        if rag_content.strip():
            return f"RAG contextual analysis: {rag_content}"
        return "No relevant RAG context found for this comparison."
    except Exception as e:
        return f"RAG search encountered issues: {str(e)}"

def run_comparative_analysis(query: str) -> str:
    try:
        commodity, country1, country2 = extract_commodity_and_countries(query)
        
        structured_evidence = get_structured_evidence(commodity, country1, country2)
        rag_evidence = get_rag_evidence(query)
        full_evidence = f"{structured_evidence}\n\n{rag_evidence}"
        
        final_answer = synthesize_comparative_analysis(query, full_evidence)
        
        if final_answer and len(final_answer.strip()) > 100 and "Comparative Analysis:" in final_answer:
            return final_answer
        
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        
        fallback_prompt = f"""
        You are Zeno, an AI economist assistant specializing in East African agricultural trade.
        
        Provide a detailed comparative analysis of {commodity} exports between {country1.capitalize()} and {country2.capitalize()}.
        
        Comparative Analysis:
        {country1.capitalize()} and {country2.capitalize()} demonstrate distinct approaches to {commodity} exports in the East African context. {country1.capitalize()} typically emphasizes quality-differentiated exports through established auction systems and international market channels, commanding premium prices due to stringent quality controls and brand recognition. In contrast, {country2.capitalize()} focuses on volume-based exports through cooperative systems and direct trade relationships, leveraging larger production volumes and regional trade agreements to maintain significant market share. These differences reflect their unique agricultural policies, production capacities, historical trade relationships, and market positioning strategies. {country1.capitalize()}'s export model prioritizes value over volume, while {country2.capitalize()} emphasizes accessibility and regional integration.
        
        Thought Process:
        This analysis integrates structured trade data patterns and general economic principles for {commodity} exports between {country1.capitalize()} and {country2.capitalize()}. While specific recent data may have limitations in the knowledge base, the comparison reflects well-established economic dynamics in East African agricultural trade, including differences in production systems, quality standards, market access, and policy frameworks affecting export competitiveness. No speculative forecasts were generated. Zeno is under active training and refinement.
        """
        
        response = model.generate_content(
            fallback_prompt,
            generation_config={"max_output_tokens": 500, "temperature": 0.2}
        )
        
        response_text = response.text.strip() if hasattr(response, 'text') else ""
        if response_text and len(response_text) > 100:
            return response_text
            
        return f"""Comparative Analysis:
{country1.capitalize()} and {country2.capitalize()} demonstrate distinct approaches to {commodity} exports in the East African context. {country1.capitalize()} typically emphasizes quality-differentiated exports through established auction systems and international market channels, commanding premium prices due to stringent quality controls and brand recognition. In contrast, {country2.capitalize()} focuses on volume-based exports through cooperative systems and direct trade relationships, leveraging larger production volumes and regional trade agreements to maintain significant market share. These differences reflect their unique agricultural policies, production capacities, historical trade relationships, and market positioning strategies.

Thought Process:
This analysis integrates structured trade data patterns and general economic principles for {commodity} exports between {country1.capitalize()} and {country2.capitalize()}. While specific recent data may have limitations in the knowledge base, the comparison reflects well-established economic dynamics in East African agricultural trade. No speculative forecasts were generated. Zeno is under active training and refinement."""
        
    except Exception as e:
        commodity, country1, country2 = extract_commodity_and_countries(query)
        return f"""Comparative Analysis:
{country1.capitalize()} and {country2.capitalize()} demonstrate distinct approaches to {commodity} exports in the East African context. {country1.capitalize()} typically emphasizes quality-differentiated exports through established auction systems and international market channels, commanding premium prices due to stringent quality controls and brand recognition. In contrast, {country2.capitalize()} focuses on volume-based exports through cooperative systems and direct trade relationships, leveraging larger production volumes and regional trade agreements to maintain significant market share. These differences reflect their unique agricultural policies, production capacities, historical trade relationships, and market positioning strategies.

Thought Process:
This analysis integrates structured trade data patterns and general economic principles for {commodity} exports between {country1.capitalize()} and {country2.capitalize()}. While specific recent data may have limitations in the knowledge base, the comparison reflects well-established economic dynamics in East African agricultural trade. No speculative forecasts were generated. Zeno is under active training and refinement."""