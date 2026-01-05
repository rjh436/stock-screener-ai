from typing import List, Dict

from .generic import GenericStrategy
from .apex_wealth import ApexWealthStrategy

def load_strategies(configs: List[Dict]) -> List[object]:
    """
    Build strategy instances from config dicts so UI and engine share logic.
    """
    strategies = []
    for config in configs or []:
        # Revert: Route "Wealth" types to ApexWealthStrategy (Generic-compatible)
        # This aligns with the Optimization run that produced 45% CAGR.
        if config.get("type") == "wealth":
            strategy = ApexWealthStrategy(config)
        else:
            strategy = GenericStrategy(config)
            
        strategies.append(strategy)
        print(f"✅ Loaded {strategy.name} as {type(strategy).__name__}")
        
    return strategies
