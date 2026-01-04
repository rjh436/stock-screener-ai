from typing import List, Dict

from .generic import GenericStrategy
from .apex_wealth import ApexWealthStrategy
from .apex_sniper import StrategyApexSniper


def load_strategies(configs: List[Dict]) -> List[object]:
    """
    Build strategy instances from config dicts so UI and engine share logic.

    Routing Priority:
    1. Sniper Variant (Explicit check for 'sniper' in name)
    2. Wealth Variant (Legacy check for type='wealth')
    3. Generic (Fallback)
    """
    strategies = []
    for config in configs or []:
        name = config.get("name", "")

        # 1. Route to Sniper Class if name implies it
        if "sniper" in name.lower():
            strategy = StrategyApexSniper(params=config)
            # Manually set name since StrategyApexSniper might not extract it from params automatically
            try:
                setattr(strategy, "name", name)
            except (AttributeError, TypeError):
                pass

        # 2. Route to Legacy Wealth Class
        elif config.get("type") == "wealth":
            strategy = ApexWealthStrategy(config)

        # 3. Default Handler
        else:
            strategy = GenericStrategy(config)

        strategies.append(strategy)

        # Safe printing of loaded type
        safe_name = getattr(strategy, "name", "Unknown Strategy")
        print(f"✅ Loaded {safe_name} as {type(strategy).__name__}")

    return strategies
