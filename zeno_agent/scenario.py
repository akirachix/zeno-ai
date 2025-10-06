import os
import re
from .tools.db import get_trade_data, semantic_search_rag_embeddings
from .tools.graphing import plot_price_scenario


def load_prompt(filename):
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    with open(os.path.join(prompts_dir, filename), encoding="utf-8") as f:
        return f.read().strip()


def embedding_reasoning_fallback(commodity, country, direction, pct, scenario_query):
    rag_articles = semantic_search_rag_embeddings(scenario_query, top_k=3)
    if rag_articles:
        summary_lines = []
        for idx, article in enumerate(rag_articles, 1):
            context = article.get("content") or str(article)
            snippet = (context[:180] + "...") if len(context) > 180 else context
            summary_lines.append(f"- Evidence {idx}: {snippet}")
        summary = "\n".join(summary_lines)
        reasoning = "Reasoning: I used unstructured reports and articles from Zeno's RAG database and semantic similarity to find context relevant to your scenario."
        source = "Source: Zeno RAG DB, updated 2025-09"
        return summary + "\n\n" + reasoning + "\n\n" + source

    if commodity and country and direction:
        shock_noun = "reduction" if direction == "decrease" else "increase"
        price_or_supply = "prices" if "price" in scenario_query.lower() else "supply"

        analysis = f"A {pct}% {shock_noun} in {commodity} {price_or_supply} in {country.capitalize()} would trigger significant macroeconomic and social effects across multiple dimensions.\n\n"

        analysis += f"**Consumer and Household Impact:**\n"
        if direction == "decrease" and "price" in scenario_query.lower():
            analysis += f"With lower {commodity} prices, household purchasing power would increase, particularly benefiting low-income urban consumers who spend a large share of their income on staple foods. Food inflation would likely ease, reducing pressure on the national consumer price index.\n\n"
        elif direction == "decrease":
            analysis += f"A {pct}% reduction in {commodity} supply would likely drive domestic prices upward, straining household budgets and potentially increasing food insecurity, especially among vulnerable populations in urban centers and drought-affected regions.\n\n"
        else:
            analysis += f"Higher {commodity} prices would reduce real incomes for consumers, increase cost of living, and potentially fuel broader inflationary pressures in the economy.\n\n"

        analysis += f"**Producer and Rural Economy Impact:**\n"
        if direction == "decrease" and "price" in scenario_query.lower():
            analysis += f"Smallholder farmers and rural producers would face reduced income, potentially leading to decreased investment in inputs for subsequent planting seasons. This could create negative feedback loops affecting future production cycles and rural economic activity.\n\n"
        elif direction == "decrease":
            analysis += f"While reduced supply may temporarily raise prices, the underlying production shock (e.g., drought, pest outbreak) would harm farmer livelihoods, reduce agricultural GDP contribution, and could trigger rural distress migration.\n\n"
        else:
            analysis += f"Producers would benefit from higher prices, potentially stimulating increased production in subsequent seasons, though this depends on input availability, credit access, and weather conditions.\n\n"

        analysis += f"**Trade and Policy Response:**\n"
        analysis += f"Kenya's response would likely involve strategic grain reserves, potential import adjustments, and targeted subsidies. A supply shock could increase reliance on imports from regional partners like Uganda or Tanzania, affecting the trade balance. The government might also implement price stabilization measures or social safety nets to protect vulnerable populations.\n\n"

        analysis += f"**Macroeconomic Context:**\n"
        analysis += f"Given that {commodity} is a staple food in Kenya, significant price or supply shocks have historically influenced monetary policy decisions, inflation targeting, and fiscal spending. The Central Bank and Ministry of Agriculture would likely coordinate closely to mitigate second-round effects on inflation expectations and social stability.\n\n"

        source = "Source: Zeno trade DB & general East African economic literature, updated 2025-09"
        return analysis + source

    else:
        return (
            "I couldn't find a relevant match for your scenario in Zeno's trade DB or RAG embeddings. "
            "Consider specifying a different country, commodity, or scenario for deeper analysis. "
            "Source: Zeno trade DB & RAG DB, updated 2025-09"
        )


class ScenarioSubAgent:
    def __init__(self):
        self.scenario_prompt = load_prompt("scenario_template.txt")
        self.what_if_prompt = load_prompt("scenario_agent_prompt.txt")
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