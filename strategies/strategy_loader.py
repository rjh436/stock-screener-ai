from typing import List, Dict

from .generic import GenericStrategy
from .apex_wealth import ApexWealthStrategy


def load_strategies(configs: List[Dict]) -> List[GenericStrategy]:
    """
    Build strategy instances from config dicts so UI and engine share logic.
    """
    strategies = []
    for config in configs or []:
        strategy = ApexWealthStrategy(config) if config.get("type") == "wealth" else GenericStrategy(config)
        strategies.append(strategy)
        print(f"✅ Loaded {strategy.name} as {type(strategy).__name__}")
    return strategies
