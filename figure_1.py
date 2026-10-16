
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load concrete dataset
df = pd.read_csv("Concrete_Data.csv")

# Reserve rows 501–630 (Python indices 500–629) for testing
train_df = df.drop(index=range(500, 630))

# Select variables
cement = train_df["Cement (component 1)(kg in a m^3 mixture)"]
strength = train_df["Concrete compressive strength(MPa, megapascals) "]

# Create scatterplot
plt.figure(figsize=(9, 5.5))


sns.regplot(
    x=cement,
    y=strength,
    scatter_kws={"alpha": 0.4, "s": 30},
    line_kws={"color": "darkred", "linewidth": 2}
)

plt.title("Cement Content vs. Concrete Compressive Strength",
          fontsize=14, fontweight="bold")
plt.xlabel("Cement Content (kg/m³)", fontsize=11)
plt.ylabel("Compressive Strength (MPa)", fontsize=11)
plt.grid(alpha=0.2)
plt.tight_layout()

# Save high-resolution figure
plt.savefig("cement_relationship.png", dpi=300, bbox_inches="tight")
plt.show()
