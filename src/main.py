import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1: Load data and preprocess ---

data_dir = "./data/"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
data_list = []

for file in csv_files:
    try:
        df = pd.read_csv(file, encoding="utf-8")
    except Exception as e:
        print(f"Loading Error {file}: {e}")
        continue

    # Zero point correction
    # baseline1 = df["Stress(Pascal)"].iloc[0]
    # df["Stress(Pascal)"] = df["Stress(Pascal)"] - baseline1
    # baseline2 = df["Stress(kPa)"].iloc[0]
    # df["Stress(kPa)"] = df["Stress(kPa)"] - baseline2
    # Eliminate negative value
    # df = df[df["Force_N"] > 0]

    # Get sample infomation from file name
    basename = os.path.basename(file)
    if "1-1-0" in basename:
        cd_ratio = 0.0
        pei_ratio = 1.0
    elif "1-1-0.25" in basename:
        cd_ratio = 0.25
        pei_ratio = 1.0
    elif "1-1-0.5" in basename:
        cd_ratio = 0.5
        pei_ratio = 1.0
    elif "1-1-1" in basename:
        cd_ratio = 1.0
        pei_ratio = 1.0
    elif "1-2-0" in basename:
        cd_ratio = 0.0
        pei_ratio = 2.0
    elif "1-3-1" in basename:
        cd_ratio = 0.0
        pei_ratio = 3.0
    else:
        cd_ratio = np.nan
        pei_ratio = np.nan

    if "beta" in basename.lower():
        cd_type = "β-CD"
    elif "gamma" in basename.lower():
        cd_type = "γ-CD"
    else:
        cd_type = "unknown"

    # Add sample information to new columns
    df["CD_ratio"] = cd_ratio
    df["PEI_ratio"] = pei_ratio
    df["CD_type"] = cd_type
    df["Sample_ID"] = basename

    data_list.append(df)

if len(data_list) == 0:
    raise ValueError("Empty Data Error")

# Combine all csv data
all_data = pd.concat(data_list, ignore_index=True)


# --- Step 2: Feature Extraction ---

def extract_features(df_sample):
    # Maximum Stress
    max_stress = df_sample["Stress(kPa)"].max()
    max_strain = df_sample["Strain"].max()

    # Linear fitting of elastic modulus from data in the initial linear region (Strain: 0 to 0.15)
    df_linear = df_sample[(df_sample["Strain"] >= 0.0) &
                          (df_sample["Strain"] <= 0.15)]
    if len(df_linear) >= 2:
        slope, intercept = np.polyfit(
            df_linear["Strain"], df_linear["Stress(kPa)"], 1)
    else:
        slope, intercept = np.nan, np.nan

    # Energy absorption (area under stress-strain curve) is calculated by trapezoidal integration
    energy_absorption = np.trapz(df_sample["Stress(kPa)"], df_sample["Strain"])

    # Sample Information
    cd_ratio = df_sample["CD_ratio"].iloc[0]
    pei_ratio = df_sample["PEI_ratio"].iloc[0]
    cd_type = df_sample["CD_type"].iloc[0]
    sample_id = df_sample["Sample_ID"].iloc[0]

    return pd.Series({
        "Sample_ID": sample_id,
        "CD_ratio": cd_ratio,
        "PEI_ratio": pei_ratio,
        "CD_type": cd_type,
        "max_stress": max_stress,
        "max_strain": max_strain,
        "E_modulus": slope,
        "energy_absorption": energy_absorption
    })


# Group by sample (Sample_ID) and extract features
features_df = all_data.groupby("Sample_ID").apply(
    extract_features).reset_index(drop=True)
# Drop rows with missing values in critical features
features_df = features_df.dropna(
    subset=["CD_ratio", "PEI_ratio", "CD_type", "E_modulus"])


# --- Step 3: Build Models ---

features_df["CD_type_num"] = features_df["CD_type"].map({"β-CD": 0, "γ-CD": 1})
features_df["robustness"] = features_df["max_stress"]

x = features_df[["PEI_ratio", "CD_ratio", "CD_type_num", "E_modulus"]]
y = features_df["robustness"]

# Split into training and testing (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Gradient Boosting Regressor with hyperparameter tuning using GridSearchCV
gb_model = GradientBoostingRegressor(random_state=42)
param_grid = {
    "gradientboostingregressor__n_estimators": [50, 100, 200],
    "gradientboostingregressor__learning_rate": [0.01, 0.05, 0.1],
    "gradientboostingregressor__max_depth": [3, 4, 5]
}
pipeline_gb = make_pipeline(StandardScaler(), gb_model)
grid_gb = GridSearchCV(pipeline_gb, param_grid, cv=5,
                       scoring="neg_mean_squared_error")
grid_gb.fit(x_train, y_train)
y_pred_gb = grid_gb.predict(x_test)
print("Best Params for Gradient Boosting:", grid_gb.best_params_)

# Display model evaluation
print("\nGradient Boosting Regressor:")
print("  MSE:", mean_squared_error(y_test, y_pred_gb))
print("  R^2:", r2_score(y_test, y_pred_gb))


# --- Step 4: Optimization and trade-off analysis ---

mean_E_modulus = features_df["E_modulus"].mean()


def predict_robustness(model, pei_ratio, cd_ratio, cd_type_num, E_modulus):
    x_query = pd.DataFrame([[pei_ratio, cd_ratio, cd_type_num, E_modulus]], columns=[
                           "PEI_ratio", "CD_ratio", "CD_type_num", "E_modulus"])
    return model.predict(x_query)[0]


# Define candidate CD ratios uniformly in the range [0, 1]
ratio_candidates = np.linspace(0, 1, 101)
print("Candidate PVA/CD Ratios:", ratio_candidates)
# Define candidate PEI ratios uniformly between the minimum and maximum observed values.
pei_candidates = np.linspace(
    features_df["PEI_ratio"].min(), features_df["PEI_ratio"].max(), 101)
print("Candidate PEI Ratios:", pei_candidates)

optimal_results = {}
for cd_type_num, cd_label in zip([0, 1], ["β-CD", "γ-CD"]):
    # Compute mean E_modulus separately for each CD type.
    mean_E_modulus_cd = features_df.loc[features_df["CD_type_num"]
                                        == cd_type_num, "E_modulus"].mean()
    best_obj = -np.inf
    best_cd_ratio = None
    best_pei_ratio = None
    # Optimize over both CD ratio and PEI ratio (with fixed PVA = 1).
    for cd_val in ratio_candidates:
        for pei_val in pei_candidates:
            obj = predict_robustness(
                grid_gb.best_estimator_, pei_val, cd_val, cd_type_num, mean_E_modulus_cd)
            if obj > best_obj:
                best_obj = obj
                best_cd_ratio = cd_val
                best_pei_ratio = pei_val
    optimal_results[cd_label] = {
        "optimal_pei_ratio": best_pei_ratio,
        "optimal_cd_ratio": best_cd_ratio,
        "predicted_robustness": best_obj}
    plt.plot(ratio_candidates, [predict_robustness(grid_gb.best_estimator_, best_pei_ratio, cd, cd_type_num, mean_E_modulus_cd) for cd in ratio_candidates],
             label=f"{cd_label} (PEI={best_pei_ratio:.2f})")

plt.xlabel("PEI/CD Ratio")
plt.ylabel("Predicted Robustness")
plt.title("PEI/CD Ratio and Predicted Maximum Stress by CD Type")
plt.legend()
plt.show(block=False)

print("Optimization Result:")
for cd_label, result in optimal_results.items():
    print(f"{cd_label}: Optimal Composition -> PVA: 1, PEI: {result['optimal_pei_ratio']:.2f}, CD: {
          result['optimal_cd_ratio']:.2f}, Predicted Robustness = {result['predicted_robustness']:.2f}")

# Determine the overall best CD type and composition based on predicted robustness.
best_cd_label, best_info = max(optimal_results.items(
), key=lambda item: item[1]['predicted_robustness'])
print("\nFinal Optimal Composition:")
print(f"CD Type: {best_cd_label}, Composition: PVA=1, PEI={
      best_info['optimal_pei_ratio']:.2f}, CD={best_info['optimal_cd_ratio']:.2f}")

input("Enter to teminate...")
