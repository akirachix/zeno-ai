from zeno_agent.db_utils import (
    get_country_id_by_name,
    get_product_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    get_macro_stats_from_db,
)

def build_structured_context(commodity, country):
    context_parts = []

    try:
        country_id = get_country_id_by_name(country.title())
    except Exception as e:
        print(f"[Scenario] Country lookup error: {e}")
        country_id = None

    try:
        product_id = get_product_id_by_name(commodity)
    except Exception as e:
        print(f"[Scenario] Product lookup error: {e}")
        product_id = None

    if country_id and product_id:
        try:
            indicator_id = get_indicator_id_by_metric("price")
            df_price = get_trade_data_from_db(country_id, product_id, indicator_id)
            if not df_price.empty:
                avg_price = df_price['price'].mean()
                currency = df_price.get('currency', ['KES'])[0]
                context_parts.append(f"Average historical price: {currency} {avg_price:,.2f} per unit.")
        except:
            pass

        try:
            indicator_id = get_indicator_id_by_metric("quantity")
            df_qty = get_trade_data_from_db(country_id, product_id, indicator_id)
            if not df_qty.empty:
                total_qty = df_qty['quantity'].sum()
                context_parts.append(f"Historical export volume: {total_qty:,.0f} units.")
        except:
            pass

    macro_indicators = ["GDP", "CPI", "Inflation", "Trade Balance"]
    for macro in macro_indicators:
        try:
            indicator_id = get_indicator_id_by_metric(macro)
            df_macro = get_macro_stats_from_db(country_id, indicator_id, start_year=2015)
            if not df_macro.empty:
                recent_year = df_macro['year'].max()
                recent_value = df_macro[df_macro['year'] == recent_year]['value'].iloc[0]
                context_parts.append(f"{macro} ({recent_year}): {recent_value:,.2f}")
        except:
            continue

    return " | ".join(context_parts) if context_parts else "No structured economic data available."
