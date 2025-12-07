import optimization.optimizer
if "sharpe_score = min(sharpe, 3.0) * 40.0" in open(optimization.optimizer.__file__).read():
    print("✅ Optimizer Updated")
else:
    print("❌ Optimizer Failed")
