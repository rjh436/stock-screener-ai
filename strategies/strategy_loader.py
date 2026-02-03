from typing import List, Dict

from .minervini_sepa import MinerviniSEPAStrategy
from .superperformance import SuperperformanceStrategy

STRATEGY_CLASSES = {
    "Superperformance": SuperperformanceStrategy,
    "Superperformance Strategy": SuperperformanceStrategy,
    "Minervini SEPA": MinerviniSEPAStrategy,
    "Minervini SEPA (Daily)": MinerviniSEPAStrategy,
}

def load_strategies(configs: List[Dict]) -> List[object]:
    """
    Strategy Loader - Superperformance + Minervini only.
    All other strategies are archived to prevent confusion.
    """
    strategies = []
    for config in configs or []:
        strategy_cls = STRATEGY_CLASSES.get(config.get("name"))
        if strategy_cls is None:
            # Skip unknown strategy configs in this locked-down mode.
            continue

        strategy = strategy_cls(config)
        strategies.append(strategy)
        print(f"✅ Loaded {strategy.name} as {type(strategy).__name__} (Performance Mode)")
        
    return strategies
