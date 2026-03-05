from typing import List, Dict, Optional

from .minervini_sepa import MinerviniSEPAStrategy
from .superperformance import SuperperformanceStrategy
from .apex_fusion import ApexFusionStrategy
from .cross_sectional_momentum import CrossSectionalMomentumStrategy
from .low_volatility import LowVolatilityStrategy
from .separate_value_momentum import SeparateValueMomentumStrategy

STRATEGY_CLASSES = {
    "Superperformance": SuperperformanceStrategy,
    "Superperformance Strategy": SuperperformanceStrategy,
    "Superperformance (Practical EOD No Leverage)": SuperperformanceStrategy,
    "Cross-Sectional Momentum": CrossSectionalMomentumStrategy,
    "Low Volatility": LowVolatilityStrategy,
    "Separate Value Momentum": SeparateValueMomentumStrategy,
    "Minervini SEPA": MinerviniSEPAStrategy,
    "Minervini SEPA (Daily)": MinerviniSEPAStrategy,
    "Apex Fusion": ApexFusionStrategy,
    "Apex Fusion (Practical)": ApexFusionStrategy,
}

def _resolve_strategy_class(config: Dict) -> Optional[type]:
    name = str(config.get("name", "") or "").strip()
    strategy_type = str(config.get("type", config.get("strategy_type", "")) or "").strip().lower()
    if strategy_type == "superperformance":
        return SuperperformanceStrategy
    if strategy_type in {"cross_sectional_momentum", "cross-sectional-momentum", "momentum_rank"}:
        return CrossSectionalMomentumStrategy
    if strategy_type in {"low_volatility", "low-volatility", "low_vol"}:
        return LowVolatilityStrategy
    if strategy_type in {"separate_value_momentum", "value_momentum", "value-momentum"}:
        return SeparateValueMomentumStrategy
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
