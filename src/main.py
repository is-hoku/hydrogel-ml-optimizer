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
    baseline1 = df["Stress(Pascal)"].iloc[0]
    df["Stress(Pascal)"] = df["Stress(Pascal)"] - baseline1
    baseline2 = df["Stress(kPa)"].iloc[0]
    df["Stress(kPa)"] = df["Stress(kPa)"] - baseline2
    # Eliminate negative value
    df = df[df["Force_N"] > 0]

    # Get sample infomation from file name
    basename = os.path.basename(file)
    if "1-0" in basename:
        ratio_num = 0.0
    elif "1-0.25" in basename:
        ratio_num = 0.25
    elif "1-0.5" in basename:
        ratio_num = 0.5
    elif "1-1" in basename:
        ratio_num = 1.0
    else:
        ratio_num = np.nan

    if "beta" in basename.lower():
        cd_type = "β-CD"
    elif "gamma" in basename.lower():
        cd_type = "γ-CD"
    else:
        cd_type = "unknown"

    # Add sample information to new columns
    df["PVA_CD_ratio"] = ratio_num
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
    ratio = df_sample["PVA_CD_ratio"].iloc[0]
    cd_type = df_sample["CD_type"].iloc[0]
    sample_id = df_sample["Sample_ID"].iloc[0]

    return pd.Series({
        "Sample_ID": sample_id,
        "PVA_CD_ratio": ratio,
        "CD_type": cd_type,
        "max_stress": max_stress,
        "max_strain": max_strain,
        "E_modulus": slope,
        "energy_absorption": energy_absorption
    })


# Group by sample (Sample_ID) and extract features
features_df = all_data.groupby("Sample_ID").apply(
    extract_features).reset_index(drop=True)


# --- Step 3: Build Models ---

features_df["CD_type_num"] = features_df["CD_type"].map({"β-CD": 0, "γ-CD": 1})

# Set explanatory variables and objective variables
# Explanatory variables: PVA_CD_ratio, CD_type_num, E_modulus
# Objective variable: Robustness defined as max_stress * max_strain
x = features_df[["PVA_CD_ratio", "CD_type_num", "E_modulus"]]
y = features_df["max_stress"] * features_df["max_strain"]

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

# To evaluate the predicted maximum stresses at varying PVA/CD ratios for each CD type here,
# average E_modulus and energy_absorption (averaged over all samples) as representative values.
mean_E_modulus = features_df["E_modulus"].mean()


def predict_robustness(model, ratio, cd_type_num, mean_E_modulus):
    x_query = pd.DataFrame([[ratio, cd_type_num, mean_E_modulus]], columns=[
                           "PVA_CD_ratio", "CD_type_num", "E_modulus"])
    return model.predict(x_query)[0]


min_ratio = features_df.loc[features_df["PVA_CD_ratio"]
                            > 0, "PVA_CD_ratio"].min()
# Use candidate ratios by interpolating between the unique experimental values observed (excluding 0)
unique_ratios = np.sort(features_df["PVA_CD_ratio"].unique())
interpolated_candidates = []
for i in range(len(unique_ratios) - 1):
    # Generate 20 evenly-spaced values between each pair of adjacent unique ratios (excluding the upper bound)
    segment = np.linspace(
        unique_ratios[i], unique_ratios[i+1], 20, endpoint=False)
    interpolated_candidates.extend(segment)
# Append the last unique ratio
interpolated_candidates.append(unique_ratios[-1])
# Convert to a NumPy array and sort
ratio_candidates = np.array(sorted(interpolated_candidates))
print("Candidate PVA/CD Ratios:", ratio_candidates)

# To avoid extreme values (0 or 1), introduce a penalty term to favor intermediate ratios.
# Define target ratio as the midpoint between the minimum experimental ratio and maximum candidate.
max_ratio_val = ratio_candidates[-1]
target_ratio = (min_ratio + max_ratio_val) / 2
lambda_penalty = 0.2  # Adjust this penalty parameter as needed

optimal_results = {}
for cd_type_num, cd_label in zip([0, 1], ["β-CD", "γ-CD"]):
    # Compute mean E_modulus separately for each CD type to better capture their differences.
    mean_E_modulus_cd = features_df.loc[features_df["CD_type_num"]
                                        == cd_type_num, "E_modulus"].mean()
    composite_objs = [predict_robustness(grid_gb.best_estimator_, r, cd_type_num, mean_E_modulus_cd)
                      - lambda_penalty * (r - target_ratio) ** 2
                      for r in ratio_candidates]
    best_idx = np.argmax(composite_objs)
    best_ratio = ratio_candidates[best_idx]
    best_pred = predict_robustness(
        grid_gb.best_estimator_, best_ratio, cd_type_num, mean_E_modulus_cd)

    optimal_results[cd_label] = {
        "optimal_ratio": best_ratio, "predicted_robustness": best_pred}
    plt.plot(ratio_candidates, composite_objs, label=cd_label)

plt.xlabel("PVA/CD Ratio")
plt.ylabel("Predicted Robustness")
plt.title("PVA/CD Ratio and Predicted Maximum Stress by CD Type")
plt.legend()
plt.show(block=False)

print("Optimization Result:")
for cd_label, result in optimal_results.items():
    print(f"{cd_label}: Optimal PVA/CD Ratio = {
          result['optimal_ratio']:.2f}, Predicted Robustness = {result['predicted_robustness']:.2f}")

# Determine the overall best CD type and ratio based on predicted robustness.
best_cd_label, best_info = max(optimal_results.items(
), key=lambda item: item[1]['predicted_robustness'])
print("\nFinal Optimal Composition:")
print(f"CD Type: {
      best_cd_label}, Optimal PVA/CD Ratio = {best_info['optimal_ratio']:.2f}")

input("Enter to teminate...")
