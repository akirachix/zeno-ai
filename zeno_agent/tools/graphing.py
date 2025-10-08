
import matplotlib.pyplot as plt
import os
import re

def safe_filename(s):
    return re.sub(r'\W+', '_', s.strip().lower())

def plot_price_scenario(commodity, country, months, base_prices, scenario_prices, direction, pct):
    """
    Plots and saves a price scenario graph as a PNG file, returns the web-relative path.
    Args:
        commodity (str): Commodity name.
        country (str): Country name.
        months (list of str or int): Month labels.
        base_prices (list of float): Historical prices.
        scenario_prices (list of float): Scenario prices.
        direction (str): 'increase' or 'decrease'.
        pct (int or float): Scenario percentage.
    Returns:
        str: Web-relative path to the saved PNG plot.
    """
    if not (len(months) == len(base_prices) == len(scenario_prices)):
        raise ValueError("Input lists months, base_prices, scenario_prices must have the same length.")

    months_labels = [str(m) for m in months]

    plt.figure(figsize=(8, 5))
    plt.plot(months_labels, base_prices, marker='o', label="Historical Price")
    plt.plot(months_labels, scenario_prices, marker='x', linestyle='--', label=f"Scenario: {direction} by {pct}%")
    plt.title(f"{commodity.capitalize()} Price Scenario in {country.capitalize()}")
    plt.xlabel("Month")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    filename = f"{safe_filename(commodity)}_{safe_filename(country)}_scenario_{direction}{pct}.png"
    out_dir = "static"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return f"/static/{filename}"
