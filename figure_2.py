import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("Concrete_Data.csv")

# Reserve rows 501–630 (Python indices 500–629) for testing
train_df = df.drop(index=range(500, 630))

# Select variables
age = train_df["Age (day)"]
strength = train_df["Concrete compressive strength(MPa, megapascals) "]

# Create scatterplot
plt.figure(figsize=(9, 5.5))

sns.scatterplot(
    x=age,
    y=strength,
    alpha=0.55,
    s=35
)

# Logarithmic axis to better display early curing ages
plt.xscale("log")

plt.title("Curing Age vs. Concrete Compressive Strength",
          fontsize=14, fontweight="bold")
plt.xlabel("Curing Age (Days, Log Scale)", fontsize=11)
plt.ylabel("Compressive Strength (MPa)", fontsize=11)
plt.grid(alpha=0.2)
plt.tight_layout()

# Save high-resolution figure
plt.savefig("age_relationship.png", dpi=300, bbox_inches="tight")
plt.show()
