from strategies.generic import GenericStrategy


class SEPAChampionStrategy(GenericStrategy):
    def __init__(self):
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
            ],
            # --- RISK MANAGEMENT ---
            "risk_parameters": {
                "risk_per_trade": 0.014,  # 1.4% Risk
                "max_pos_size_pct": 0.20,  # 20% Max Position Size (Concentrated)
                "stop_loss_type": "atr",
                "stop_loss_atr": 2.75,  # 2.75x ATR (Room to breathe)
                "max_positions": 5,  # Focus on best ideas
            },
            # --- EXIT LOGIC ---
            "execution_parameters": {
                "time_stop": 30,  # Get out if not working in 30 days
                "partial_profit_day": 5,  # Take some off the table early
            },
            "exit_rules": [
                # Qullamaggie Trail: Exit if Close < 10-day SMA
                {"col": "close", "op": "<", "ref": "sma10"},
            ],
        }
        super().__init__(genome)
