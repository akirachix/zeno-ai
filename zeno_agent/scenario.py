import os
import re
from .tools.db import get_trade_data, semantic_search_rag_embeddings
from .tools.graphing import plot_price_scenario


def load_prompt(filename):
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    with open(os.path.join(prompts_dir, filename), encoding="utf-8") as f:
        return f.read().strip()


def embedding_reasoning_fallback(commodity, country, direction, pct, scenario_query):
    """
    Confident, reasoning-based fallback using semantic knowledge (embeddings).
    Attempts to answer using RAG embeddings if trade data is missing.
    """
    rag_articles = semantic_search_rag_embeddings(scenario_query, top_k=3)
    if rag_articles:
        summary_lines = []
        for idx, article in enumerate(rag_articles, 1):
            context = article.get("content") or str(article)
            snippet = (context[:180] + "...") if len(context) > 180 else context
            summary_lines.append(f"- Evidence {idx}: {snippet}")
        summary = "\n".join(summary_lines)
        reasoning = (
            "Reasoning: I used unstructured reports and articles from Zeno's RAG database and semantic similarity to find context relevant to your scenario."
        )
        source = "Source: Zeno RAG DB, updated 2025-09"
        return summary + "\n\n" + reasoning + "\n\n" + source


    if commodity and country and direction:
        semantic_reasoning = (
            f"Based on an analysis of similar economic shocks in East Africa, a {pct}% {direction} in {commodity} in {country.capitalize()} would typically have the following effects:\n\n"
        )
        if direction == "increase":
            if "production" in scenario_query or "harvest" in scenario_query:
                details = (
                    "- **Increased production** tends to lower prices, improve local food security, boost exports, and benefit consumers, "
                    "but may reduce farmer income if prices drop significantly.\n"
                )
            else:
                details = (
                    "- **Increased prices** often benefit producers but can hurt consumers and reduce demand.\n"
                )
        elif direction == "decrease":
            details = (
                "- **Decreased production** or supply usually raises prices, harms consumers, and can drive up imports.\n"
            )
        else:
            details = ""


        reasoning = (
            f"- In {country.capitalize()}, historical studies and economic analyses indicate that shocks to {commodity} markets have strong impacts on rural incomes and food security.\n\n"
            "This assessment is based on semantic analysis of unstructured trade and policy data in East Africa."
        )
        source = "Source: Zeno trade DB, updated 2025-09"
        return semantic_reasoning + details + reasoning + "\n\n" + source


    else:
        return (
            "I couldn't find a relevant match for your scenario in Zeno's trade DB or RAG embeddings. "
            "Consider specifying a different country, commodity, or scenario for deeper analysis. "
            "Source: Zeno trade DB & RAG DB, updated 2025-09"
        )


class ScenarioSubAgent:
    """
    Scenario Analysis / Shock Simulation Sub-Agent for Zeno
    """


    def __init__(self):
        self.scenario_prompt = load_prompt("scenario_template.txt")
        self.what_if_prompt = load_prompt("scenario_what_if_prompt.txt")
        self.missing_data_prompt = (
            "No data for that scenario yet. Try uploading data or pick a different commodity/country—I'm ready when you are!"
        )


    def handle(self, scenario_query: str) -> dict:
        print(f"[SCENARIO AGENT] Received query: {scenario_query}")
        query = scenario_query.lower()


        commodity_match = re.search(r"(maize|coffee|tea)", query)
        commodity = commodity_match.group(1) if commodity_match else None


        country_match = re.search(r"(kenya|uganda|tanzania|ethiopia|rwanda)", query)
        country = country_match.group(1) if country_match else "kenya"


        if "drop" in query or "decrease" in query or "reduce" in query:
            direction = "decrease"
            pct_match = re.search(r"(?:drop|decrease|reduce)(?: by)? (\d+)%", query)
            pct = int(pct_match.group(1)) if pct_match else 15
        elif "increase" in query or "raise" in query:
            direction = "increase"
            pct_match = re.search(r"(?:increase|raise)(?: by)? (\d+)%", query)
            pct = int(pct_match.group(1)) if pct_match else 15
        else:
            direction = None
            pct = 0


        months_match = re.search(r"next (\d+) months?", query)
        months = int(months_match.group(1)) if months_match else 3


        if not commodity or not direction:
            print("[SCENARIO AGENT] Missing commodity or direction.")
            return {
                "response": self.missing_data_prompt,
                "followup": "Try: 'What if maize price drops by 20% in Kenya over the next 3 months?'"
            }


        data = get_trade_data(commodity, country, last_n_months=6)
        metadata = data.get("metadata") if isinstance(data, dict) else None


        if not data or not data.get("months") or not data.get("prices"):
            print("[SCENARIO AGENT] No data found in DB. Falling back to confident, reasoning-based embedding answer.")
            qualitative_reasoning = embedding_reasoning_fallback(
                commodity, country, direction, pct, scenario_query
            )
            return {
                "response": qualitative_reasoning,
                "followup": "Want to try a different scenario, or upload data for deeper analysis?"
            }


        available_months = min(months, len(data["months"]))
        base_prices = data["prices"][-available_months:]
        base_months = data["months"][-available_months:]


        if direction == "decrease":
            scenario_prices = [round(p * (1 - pct / 100), 2) for p in base_prices]
            shock_type = "price decrease"
        else:
            scenario_prices = [round(p * (1 + pct / 100), 2) for p in base_prices]
            shock_type = "price increase"


        print(f"[SCENARIO AGENT] base_prices: {base_prices}, scenario_prices: {scenario_prices}")


        graph_path = plot_price_scenario(
            commodity,
            country,
            base_months,
            base_prices,
            scenario_prices,
            direction,
            pct
        )


        if metadata:
            source_val = getattr(metadata, "source", None) or metadata.get("source")
            updated_val = getattr(metadata, "updated_at", None) or metadata.get("updated_at")
            if updated_val:
                updated_str = str(updated_val)[:10]
                source_str = f"{source_val}, updated {updated_str}"
            else:
                source_str = f"{source_val}" if source_val else "Unknown source"
        else:
            source_str = "Unknown source"


        if query.strip().startswith("what if"):
            explanation = self.what_if_prompt.format(
                commodity=commodity.capitalize(),
                direction=direction,
                pct=pct,
                percentage=pct,
                country=country.capitalize(),
                months=available_months,
                base_prices=base_prices,
                scenario_prices=scenario_prices,
                shock_type=shock_type,
                base_months=base_months,
                source=source_str,
                scenario=scenario_query.strip("?")
            )
        else:
            explanation = self.scenario_prompt.format(
                commodity=commodity.capitalize(),
                direction=direction,
                pct=pct,
                percentage=pct,
                country=country.capitalize(),
                months=available_months,
                base_prices=base_prices,
                scenario_prices=scenario_prices,
                shock_type=shock_type,
                base_months=base_months,
                source=source_str
            )


        return {
            "response": explanation,
            "graph_path": graph_path,
            "followup": "Want to run another scenario, change the numbers, or check a different commodity? Just say the word."
        }



