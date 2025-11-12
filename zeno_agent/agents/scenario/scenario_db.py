"""
Database utilities for scenario analysis context building.

Retrieves and formats structured economic data including trade metrics
and macroeconomic indicators for scenario analysis.
"""

from typing import List, Optional
from zeno_agent.db_utils import (
    get_country_id_by_name,
    get_product_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    get_macro_stats_from_db,
)


def build_structured_context(commodity: str, country: str) -> str:
    """
    Build comprehensive economic context for scenario analysis.
    
    Retrieves trade data (prices, volumes) and macroeconomic indicators
    (GDP, CPI, inflation, trade balance) for the specified commodity and country.
    
    Args:
        commodity: Commodity name (e.g., "maize", "coffee")
        country: Country name (e.g., "kenya", "uganda")
        
    Returns:
        Formatted string containing all available economic data,
        or fallback message if no data is available
    """
    context_parts = []

    # Resolve country and product IDs
    country_id = _get_country_id_safe(country)
    product_id = _get_product_id_safe(commodity)

    # Retrieve trade data if both IDs are valid
    if country_id and product_id:
        context_parts.extend(_get_trade_metrics(country_id, product_id))
    
    # Retrieve macroeconomic indicators
    if country_id:
        context_parts.extend(_get_macro_indicators(country_id))

    if not context_parts:
        return "No structured economic data available for this commodity-country pair."
    
    return " | ".join(context_parts)


def _get_country_id_safe(country: str) -> Optional[int]:
    """
    Safely retrieve country ID, returning None on error.
    
    Args:
        country: Country name
        
    Returns:
        Country ID or None if not found
    """
    try:
        return get_country_id_by_name(country.title())
    except Exception:
        return None


def _get_product_id_safe(commodity: str) -> Optional[int]:
    """
    Safely retrieve product ID, returning None on error.
    
    Args:
        commodity: Commodity name
        
    Returns:
        Product ID or None if not found
    """
    try:
        return get_product_id_by_name(commodity)
    except Exception:
        return None


def _get_trade_metrics(country_id: int, product_id: int) -> List[str]:
    """
    Retrieve trade-related metrics (prices and volumes).
    
    Args:
        country_id: Database country identifier
        product_id: Database product identifier
        
    Returns:
        List of formatted metric strings
    """
    metrics = []
    
    # Price data
    try:
        indicator_id = get_indicator_id_by_metric("price")
        df_price = get_trade_data_from_db(country_id, product_id, indicator_id)
        
        if not df_price.empty and 'price' in df_price.columns:
            avg_price = df_price['price'].mean()
            recent_price = df_price['price'].iloc[-1]
            currency = df_price.get('currency', ['KES']).iloc[0] if 'currency' in df_price.columns else 'KES'
            
            metrics.append(
                f"Price: Avg {currency} {avg_price:,.2f}/unit, "
                f"Recent {currency} {recent_price:,.2f}/unit"
            )
    except Exception:
        pass

    # Volume/quantity data
    try:
        indicator_id = get_indicator_id_by_metric("quantity")
        df_qty = get_trade_data_from_db(country_id, product_id, indicator_id)
        
        if not df_qty.empty and 'quantity' in df_qty.columns:
            total_qty = df_qty['quantity'].sum()
            recent_qty = df_qty['quantity'].iloc[-1]
            
            metrics.append(
                f"Volume: Total {total_qty:,.0f} units, "
                f"Recent period {recent_qty:,.0f} units"
            )
    except Exception:
        pass

    return metrics


def _get_macro_indicators(country_id: int, start_year: int = 2015) -> List[str]:
    """
    Retrieve macroeconomic indicators for scenario context.
    
    Args:
        country_id: Database country identifier
        start_year: Starting year for data retrieval (default: 2015)
        
    Returns:
        List of formatted indicator strings
    """
    indicators = []
    macro_metrics = [
        ("GDP", "GDP Growth"),
        ("CPI", "Consumer Price Index"),
        ("Inflation", "Inflation Rate"),
        ("Trade Balance", "Trade Balance"),
        ("Exchange Rate", "Exchange Rate")
    ]
    
    for metric_name, display_name in macro_metrics:
        try:
            indicator_id = get_indicator_id_by_metric(metric_name)
            df_macro = get_macro_stats_from_db(country_id, indicator_id, start_year=start_year)
            
            if not df_macro.empty and 'year' in df_macro.columns and 'value' in df_macro.columns:
                recent_year = df_macro['year'].max()
                recent_value = df_macro[df_macro['year'] == recent_year]['value'].iloc[0]
                
                # Format based on metric type
                if "Rate" in display_name or "Inflation" in display_name:
                    indicators.append(f"{display_name} ({recent_year}): {recent_value:.2f}%")
                else:
                    indicators.append(f"{display_name} ({recent_year}): {recent_value:,.2f}")
        except Exception:
            continue
    
    return indicators