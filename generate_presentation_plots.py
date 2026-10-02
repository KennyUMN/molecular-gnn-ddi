import os
import json
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("paper", exist_ok=True)

# 1. Load history
with open("runs_kaggle/random/history.json") as f:
    rand_hist = json.load(f)
with open("runs_kaggle/cold_start/history.json") as f:
    cold_hist = json.load(f)
with open("runs_kaggle/scaffold/history.json") as f:
    scaf_hist = json.load(f)

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Colors
c_rand = "#1E40AF"  # Royal blue
c_cold = "#059669"  # Emerald
c_scaf = "#DC2626"  # Crimson

# Epochs
r_eps = [h["epoch"] for h in rand_hist]
r_loss_tr = [h["train"]["loss"] for h in rand_hist]
r_loss_va = [h["val"]["loss"] for h in rand_hist]
r_auc_va = [h["val"]["auroc"] for h in rand_hist]

c_eps = [h["epoch"] for h in cold_hist]
c_loss_va = [h["val"]["loss"] for h in cold_hist]
c_auc_va = [h["val"]["auroc"] for h in cold_hist]

s_eps = [h["epoch"] for h in scaf_hist]
s_loss_va = [h["val"]["loss"] for h in scaf_hist]
s_auc_va = [h["val"]["auroc"] for h in scaf_hist]

# Plot 1: Validation Loss Trajectory
ax1.plot(r_eps, r_loss_va, color=c_rand, linewidth=2.2, label="Random Split (Val Loss)")
ax1.plot(c_eps, c_loss_va, color=c_cold, linewidth=2.2, label="Cold-Start Split (Val Loss)")
ax1.plot(s_eps, s_loss_va, color=c_scaf, linewidth=2.2, linestyle="--", label="Scaffold Split (Val Loss)")
ax1.set_title("A. Validation Loss Trajectories", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Epoch", fontsize=10)
ax1.set_ylabel("Binary Cross Entropy Loss", fontsize=10)
ax1.legend(loc="upper right", frameon=True, fontsize=9)
ax1.set_ylim(0.2, 0.8)

# Plot 2: Validation AUROC Progression
ax2.plot(r_eps, r_auc_va, color=c_rand, linewidth=2.2, label="Random Split (Best: 0.9493)")
ax2.plot(c_eps, c_auc_va, color=c_cold, linewidth=2.2, label="Cold-Start Split (Best: 0.7623)")
ax2.plot(s_eps, s_auc_va, color=c_scaf, linewidth=2.2, linestyle="--", label="Scaffold Split (Best: 0.6605)")
ax2.axhline(0.648, color="#6B7280", linestyle=":", label="TDC Baseline Avg (~0.6480)")
ax2.set_title("B. Validation AUROC Learning Curves", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Epoch", fontsize=10)
ax2.set_ylabel("Validation AUROC", fontsize=10)
ax2.legend(loc="lower right", frameon=True, fontsize=9)
ax2.set_ylim(0.5, 1.0)

plt.tight_layout()
fig.savefig("paper/empirical_training_curves.png", dpi=300)
plt.close(fig)
print("Saved paper/empirical_training_curves.png")

# 2. Benchmark Comparison Plot (Scaffold Leakage Cliff)
fig2, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

models = ["SSI-DDI\n(Bioinformatics '21)", "GMPNN-CS\n(Bioinformatics '22)", "SA-DDI\n(Chem Sci '22)", "TDC Baseline Avg\n(NeurIPS '21)", "PharmaGNN\n(Ours)"]
random_scores = [0.9701, 0.9845, 0.9880, 0.9100, 0.9493]
cold_scores = [0.6833, 0.7748, 0.7914, 0.6480, 0.7623]

x = np.arange(len(models))
width = 0.35

rects1 = ax.bar(x - width/2, random_scores, width, label="Transductive Random Split", color="#3B82F6", edgecolor="#1D4ED8")
rects2 = ax.bar(x + width/2, cold_scores, width, label="Inductive Cold-Start / Scaffold Disjoint", color="#10B981", edgecolor="#047857")

# Highlight our model
ax.axvspan(3.5, 4.5, color="#FEF3C7", alpha=0.35, zorder=0, label="PharmaGNN (Tested Rigorously)")

ax.set_ylabel("Test AUROC Score", fontsize=11, fontweight="bold")
ax.set_title("The Generalization Cliff: Performance Drop Across Random vs Cold-Start Splits", fontsize=12, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9.5)
ax.set_ylim(0.5, 1.08)
ax.legend(loc="upper right", frameon=True, fontsize=9.5)

# Add values above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight="semibold")

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
fig2.savefig("paper/split_generalization_comparison.png", dpi=300)
plt.close(fig2)
print("Saved paper/split_generalization_comparison.png")
