from typing import List, Dict, Optional

from .minervini_sepa import MinerviniSEPAStrategy
from .superperformance import SuperperformanceStrategy
from .apex_fusion import ApexFusionStrategy

STRATEGY_CLASSES = {
    "Superperformance": SuperperformanceStrategy,
    "Superperformance Strategy": SuperperformanceStrategy,
    "Superperformance (Practical EOD No Leverage)": SuperperformanceStrategy,
    "Minervini SEPA": MinerviniSEPAStrategy,
    "Minervini SEPA (Daily)": MinerviniSEPAStrategy,
    "Apex Fusion": ApexFusionStrategy,
    "Apex Fusion (Practical)": ApexFusionStrategy,
}

def _resolve_strategy_class(config: Dict) -> Optional[type]:
    name = str(config.get("name", "") or "").strip()
    strategy_type = str(config.get("type", "") or "").strip().lower()
    if strategy_type == "superperformance":
        return SuperperformanceStrategy
    if strategy_type in {"minervini_sepa", "minervini"}:
        return MinerviniSEPAStrategy
    if strategy_type in {"apex_fusion", "apexfusion", "apex"}:
        return ApexFusionStrategy
    return STRATEGY_CLASSES.get(name)

def load_strategies(configs: List[Dict]) -> List[object]:
    """
    Strategy Loader - Superperformance + Minervini only.
    All other strategies are archived to prevent confusion.
    """
    strategies = []
    for config in configs or []:
        strategy_cls = _resolve_strategy_class(config)
        if strategy_cls is None:
            # Skip unknown strategy configs in this locked-down mode.
            continue

        strategy = strategy_cls(config)
        strategies.append(strategy)
        print(f"✅ Loaded {strategy.name} as {type(strategy).__name__} (Performance Mode)")
        
    return strategies
