from strategies.generic import GenericStrategy


class SEPAChampionStrategy(GenericStrategy):
    def __init__(self, config=None):
        if config:
            super().__init__(config)
            return

        # The "DNA" evolved by the Genetic Algorithm (Gen 50/50)
        # 2008 Drawdown: 0.0% | 2020 Return: 25.7%
        genome = {
            "name": "Apex SEPA Champion (2026)",
            "use_fundamentals": False,
            # --- EXECUTION FILTERS ---
            "entry_rules": [
                # The "Elitist" Filter: Only Top 6% of Market
                {"col": "rs_rating", "op": ">", "val": 85},
                # The "Coiled Spring" Filter: 10% Max Volatility Contraction
                {"col": "bb_width", "op": "<", "val": 0.25},
                # Trend Reinforcement
                {"col": "close", "op": ">", "ref": "sma50"},
                # Breakout Trigger: Close > Yesterday's 20-day high
                {"col": "close", "op": ">", "ref": "high_20_prev", "val": 0.99},
            ],
            # --- RISK MANAGEMENT ---
            "risk_parameters": {
                "risk_per_trade": 0.025,  # 2.5% Risk (Aggressive)
                "max_pos_size_pct": 0.25,  # 25% Max Position Size (Concentration)
                "stop_loss_type": "atr",
                "stop_loss_atr": 3.0,  # 3.0x ATR (Wide for Volatility)
                "max_positions": 4,  # Focus on top 4 ideas
            },
            # --- EXIT LOGIC ---
            "execution_parameters": {
                "time_stop": 120,  # 120 days (Ride trends)
                "partial_profit_day": 999,  # DISABLED (Let it ride)
                "partial_profit_r": 100.0, # EFFECTIVELY DISABLED
            },
            "exit_rules": [
                # Trend Following: Ride the 50-day line
                {"col": "close", "op": "<", "ref": "sma50"},
            ],
        }
        super().__init__(genome)
