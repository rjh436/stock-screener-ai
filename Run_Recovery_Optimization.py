import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimization.optimizer import Optimizer


def run_recovery():
    print("Starting V9 Performance Recovery Optimization...")
    print("Goal: Recover 40%+ CAGR with Risk Parity Logic")

    # Initialize Optimizer specifically for Apex Sniper Wealth
    opt = Optimizer(
        strategy_name="Apex Sniper Wealth v1.0",
        population_size=30,  # Smaller pop for speed
        generations=10,      # Enough to find convergence
        n_jobs=10            # Use your M3 Max cores
    )

    # Force specific gene ranges for recovery
    opt.set_gene_range("time_stop", 20, 80, int)
    opt.set_gene_range("breakeven_pct", 0.02, 0.10, float)
    opt.set_gene_range("stop_loss_atr", 2.0, 5.0, float)
    opt.set_gene_range("profit_target", 1.10, 1.50, float)

    # Run
    best_dna = opt.run()

    print("\nOptimization Complete. Best DNA found:")
    print(best_dna)


if __name__ == "__main__":
    run_recovery()
