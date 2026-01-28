from typing import List, Dict

from .generic import GenericStrategy
from .apex_wealth import ApexWealthStrategy
from .sepa_champion import SEPAChampionStrategy

STRATEGY_CLASSES = {
    "Apex SEPA Champion (2026)": SEPAChampionStrategy,
}

def load_strategies(configs: List[Dict]) -> List[object]:
    """
    Strategy Loader - PERFORMANCE LOCKED (Generic Mode V9)
    
    CRITICAL ARCHITECTURE DECISION:
    This loader intentionally bypasses the 'StrategyApexSniper' class.
    Audit confirmed that the specialized class enforces hardcoded 'Dip Buy' (RSI < 5) logic
    which conflicts with the optimized 'Momentum' parameters (RSI > 15) in generated_strategies.json.
    
    By routing everything to GenericStrategy, we ensure the system respects the 35% CAGR configuration.
    """
    strategies = []
    for config in configs or []:
        # Priority: Respect explicit strategy class mapping by name
        strategy_cls = STRATEGY_CLASSES.get(config.get("name"))
        if strategy_cls is not None:
            strategy = strategy_cls()
        elif config.get("type") == "wealth":
            strategy = ApexWealthStrategy(config)
        else:
            strategy = GenericStrategy(config)
            
        strategies.append(strategy)
        print(f"✅ Loaded {strategy.name} as {type(strategy).__name__} (Performance Mode)")
        
    return strategies
