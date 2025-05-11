import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# === Step 1: Load Excel data === ### Here the data from OLS_Regression data provided in the GitHub.
input_file = r"YOUR_DATA_HERE"
df = pd.read_excel(input_file)

# === Step 2: Function to run OLS and Breusch-Pagan test ===
def run_regression(score_group, group_label):
    data = df[df["Score"] == score_group].dropna(subset=["Excess_Return", "Mkt"])

    X = sm.add_constant(data["Mkt"])  # add intercept
    y = data["Excess_Return"]

    model = sm.OLS(y, X).fit()

    print(f"\n=== OLS Results for Score {score_group} ({group_label}) ===")
    print(model.summary())

    # === Breusch-Pagan Test ===
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

    print("\n--- Breusch-Pagan Test ---")
    for label, value in zip(bp_labels, bp_test):
        print(f"{label}: {value:.4f}")

    return model

# === Step 3: Run for each score group ===
reg1 = run_regression(1, "Lower 10%")
reg2 = run_regression(2, "Middle 80%")
reg3 = run_regression(3, "Upper 10%")