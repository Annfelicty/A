# %% [markdown]
# # IMPORT ALL RELEVANT LIBRARIES

# %%
# ------- [Import all relevant libraries] -------

# Utilities
import warnings
warnings.filterwarnings('ignore')

# Usual Suspects
import numpy as np           # Mathematical operations
import pandas as pd          # Data manipulation

# Visualization
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
import seaborn as sns

# String manipulation
import re

# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ML
# Models
import xgboost as xgb
import lightgbm as lgb

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet, ElasticNetCV, Ridge

# ML Model Evaluation
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Time Series
#!pip install optuna
import optuna

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#!pip install catboost
from catboost import CatBoostRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Display settings
pd.set_option('display.max_colwidth', None)
from IPython.display import display

# %% [markdown]
# # LOAD DATA AND DO QUICK IDE

# %%
data = pd.read_csv("../Clean Data/data_no_feature_engineering.csv")

# Create a copy
df = data.copy(deep=True)

# --- IDE ---

# Check dataset shape
print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

# Check columns
print('\n'+'--'*50)
print("Columns:")
display(df.columns)

# Check metadata
print('\n'+'--'*50)
print("Metadata Check:")
display(df.info())

# Check and remove duplicates
print('\n'+'--'*50)
print("Duplicates:", df.duplicated().sum())

# Drop duplicates
df.drop_duplicates(inplace=True)
print("Duplicates after dropping:", df.duplicated().sum())

# Check data completeness
print('\n'+'--'*50)
print("Missingness check:")
display(df.isna().sum())

# %% [markdown]
# # DATA WRANGLING

# %%
# Columns to drop
cols_to_drop = [
    'country_name',
    'transaction_date',
    'current_dollar_amount',
    'implementing_partner_name',
    'managing_subagency_or_bureau_name',
    'implementing_partner_category_name',
    'foreign_assistance_objective_name'
]

# Drop
df = df.drop(columns=cols_to_drop)
print("Remaining columns:")
display(df.columns)


# %%
df['is_refund'] = (df['constant_dollar_amount'] < 0).astype(int)

# Convert all dollar amounts to positive magnitudes
df['constant_dollar_amount'] = df['constant_dollar_amount'].abs()

print("Negative values remaining:", (df['constant_dollar_amount'] < 0).sum())
print("Refund transactions:", df['is_refund'].sum())

# %% [markdown]
# # BASIC EDA

# %%
# --- Inspect managing agency counts ---
agency_counts = df['managing_agency_name'].value_counts()
display(agency_counts)

# --- Plot distribution ---
plt.figure(figsize=(12,7.5))
sns.barplot(x=agency_counts.index, y=agency_counts.values, palette="magma")
plt.xticks(rotation=90)
plt.title("Distribution of Projects by Managing Agency")
plt.ylabel("Number of Projects")
plt.xlabel("Managing Agency")
plt.tight_layout()
plt.show()

# %%
# Inspect funding agency name
agency_counts = df['funding_agency_name'].value_counts()
display(agency_counts)

# --- Plot distribution ---
plt.figure(figsize=(10,7.5))
sns.barplot(x=agency_counts.index, y=agency_counts.values, palette="viridis")
plt.xticks(rotation=90)
plt.title("Distribution of Funding Agencies")
plt.ylabel("Number of Projects")
plt.xlabel("Funding Agency")
plt.tight_layout()
plt.show()

# %%
# inspect us category name
df['us_category_name'].value_counts()

# %%
# Inspect us_sector_name
df['us_sector_name'].value_counts()

# %%
sector_mapping = {
    # --- Health ---
    'HIV/AIDS': 'Health',
    'Malaria': 'Health',
    'Tuberculosis': 'Health',
    'Maternal and Child Health': 'Health',
    'Family Planning and Reproductive Health': 'Health',
    'Health - General': 'Health',
    'Other Public Health Threats': 'Health',
    'Pandemic Influenza and Other Emerging Threats (PIOET)': 'Health',
    'Nutrition': 'Health',

    # --- Education ---
    'Basic Education': 'Education',
    'Higher Education': 'Education',
    'Education and Social Services - General': 'Education',

    # --- Governance & Human Rights ---
    'Rule of Law and Human Rights': 'Governance & Human Rights',
    'Good Governance': 'Governance & Human Rights',
    'Democracy, Human Rights, and Governance - General': 'Governance & Human Rights',
    'Civil Society': 'Governance & Human Rights',
    'Political Competition and Consensus-Building': 'Governance & Human Rights',
    'Monitoring and Evaluation': 'Governance & Human Rights',

    # --- Agriculture & Food Security ---
    'Agriculture': 'Agriculture & Food Security',

    # --- Economic Growth & Development ---
    'Economic Opportunity': 'Economic Growth & Development',
    'Economic Development - General': 'Economic Growth & Development',
    'Financial Sector': 'Economic Growth & Development',
    'Private Sector Competitiveness': 'Economic Growth & Development',
    'Trade and Investment': 'Economic Growth & Development',
    'Macroeconomic Foundation for Growth': 'Economic Growth & Development',
    'Policies, Regulations, and Systems': 'Economic Growth & Development',

    # --- Infrastructure & Environment ---
    'Infrastructure': 'Infrastructure & Environment',
    'Water Supply and Sanitation': 'Infrastructure & Environment',
    'Clean Productive Environment': 'Infrastructure & Environment',
    'Environment - General': 'Infrastructure & Environment',
    'Natural Resources and Biodiversity': 'Infrastructure & Environment',
    'Manufacturing': 'Infrastructure & Environment',
    'Mining and Natural Resources': 'Infrastructure & Environment',
    'Environment': 'Infrastructure & Environment',

    # --- Peace & Security ---
    'Stabilization Operations and Security Sector Reform': 'Peace & Security',
    'Counter-Terrorism': 'Peace & Security',
    'Counter-Narcotics': 'Peace & Security',
    'Conflict Mitigation and Reconciliation': 'Peace & Security',
    'Peace and Security - General': 'Peace & Security',
    'Transnational Crime': 'Peace & Security',
    'Combating Weapons of Mass Destruction (WMD)': 'Peace & Security',

    # --- Humanitarian & Social Protection ---
    'Protection, Assistance and Solutions': 'Humanitarian & Social Protection',
    'Disaster Readiness': 'Humanitarian & Social Protection',
    'Migration Management': 'Humanitarian & Social Protection',
    'Humanitarian Assistance - General': 'Humanitarian & Social Protection',
    'Social Services': 'Humanitarian & Social Protection',
    'Social Assistance': 'Humanitarian & Social Protection',

    # --- Governance & Administration ---
    'Direct Administrative Costs': 'Governance & Administration',
    'Multi-sector - Unspecified': 'Governance & Administration',
    'International Contributions': 'Governance & Administration',
    'Labor Policies and Markets': 'Governance & Administration',
}

# Apply mapping
df['sector'] = df['us_sector_name'].map(sector_mapping)
df.drop(columns='us_sector_name', inplace=True)

# --- Check distribution ---
sector_counts = df['sector'].value_counts(dropna=False)
display(sector_counts)

# --- Plot distribution ---
plt.figure(figsize=(10,6))
sns.barplot(x=sector_counts.index, y=sector_counts.values, palette="coolwarm")
plt.xticks(rotation=90, ha='center')
plt.title("Distribution of Projects by Sector")
plt.ylabel("Number of Projects")
plt.xlabel("Sector")
plt.tight_layout()
plt.show()

# %%
# --- Count projects per fiscal year ---
fy_counts = df['fiscal_year'].value_counts().sort_index()  # chronological order
print(fy_counts)

# --- Line plot ---
plt.figure(figsize=(10,4))
sns.lineplot(x=fy_counts.index, y=fy_counts.values, marker='o', color='teal')
plt.title("Trend of Projects by Fiscal Year")
plt.xlabel("Fiscal Year")
plt.ylabel("Number of Projects")
plt.grid(True)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# %%
# Drop us category name
df.drop(columns='us_category_name', inplace=True)

# %%
df = df.sort_values(['managing_agency_name', 'sector', 'fiscal_year']).reset_index(drop=True)
df.head()

# %% [markdown]
# # FEATURE ENGINEERING

# %%
# --- FEATURE ENGINEERING ---
group_cols = ['managing_agency_name', 'sector']
target_col = 'constant_dollar_amount'

# Define lag periods and rolling windows
lags = [1, 2]
rolling_windows = [3]

# --- Lag features ---
for lag in lags:
    df[f'lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)

# --- Rolling mean and std ---
for window in rolling_windows:
    df[f'rolling_mean_{window}yr'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    df[f'rolling_std_{window}yr'] = df.groupby(group_cols)[target_col].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).std()
    )

# --- Growth rate ---
df['funding_growth_rate'] = df.groupby(group_cols)[target_col].pct_change()

# --- Fill NaNs with 0 ---
feat_cols = [f'lag_{lag}' for lag in lags] + \
            [f'rolling_mean_{w}yr' for w in rolling_windows] + \
            [f'rolling_std_{w}yr' for w in rolling_windows] + \
            ['funding_growth_rate']

df[feat_cols] = df[feat_cols].fillna(0)

# Preview
df.head(10)

# %% [markdown]
# ## VANILLA ENSEMBLE REGRESSORS
# 
# We train unparameterised versions of Random Forest, XGBoost, LightGBM and CatBoost regressors to predict funding allocations and evaluate them on R-Squared (explained variance), RMSE and MAE.
# 
# We also train stacked ensembles of our best tree regressors and evaluate their performance against our chosen baseline model.

# %%
# --- ML PIPELINES FOR RANDOM FOREST, XGBOOST, LIGHTGBM AND CATBOOST ---

# --- 1. Separate features and target ---
X = df.drop('constant_dollar_amount', axis=1)
y = np.log1p(df['constant_dollar_amount'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical and numeric features
cat_cols = x_train.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in x_train.columns if c not in cat_cols]

# --- 2. Column transformer for preprocessing ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

# --- 3. Define pipelines for each model ---
pipelines = {
    'RandomForest': Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))
    ]),
    'LightGBM': Pipeline([
        ('preprocessor', preprocessor),
        ('model', lgb.LGBMRegressor(random_state=42))
    ]),
    'Catboost': Pipeline([
        ('preprocessor', preprocessor),
        ('model', CatBoostRegressor(random_state=42, verbose=0))
    ])
}

# --- 4. Train and evaluate ---
for name, pipe in pipelines.items():
    pipe.fit(x_train, y_train)
    y_pred_log = pipe.predict(x_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"--- {name} ---")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²: {r2:.4f}\n")

# %% [markdown]
# #### *Observation:*
# 
# The models explain a descent amount of the variance with CatBoost explaining the most at 89% followed by XGBoost at 78%. Rndom forest is the weakest one which will make a great baseline. 
# 
# The RMSE and MAE values are too high, even though we know that the data is in hundreds of millions and billions. Tuning and hyperparameter optimization will fix that.

# %% [markdown]
# ## VANILLA STACKED ENSEMBLES
# 
# We will now combine XGBoost and LightGBM to create our stacked ensemble. Ideally, we would want a stacked ensemble with XGBoost and Catboost but Catboost cannot handle 1D arrays that our encoded categoricals. 
# 
# Additionally, it is computationally intensive so we stuck with XGBoost and LightGBM.

# %%
# === VANILLA STACKING REGRESSION (XGB + LGB + RIDGE) ===

# --- 1. Split features and target ---
X = df.drop('constant_dollar_amount', axis=1)
y = np.log1p(df['constant_dollar_amount'])  # Log-transform for stability

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --- 2. Identify categorical and numeric columns ---
cat_cols = x_train.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in x_train.columns if c not in cat_cols]

# --- 3. Preprocessor ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

# --- 4. Base Models ---
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
lgb_model = lgb.LGBMRegressor(random_state=42)

# --- 5. Meta Model ---
meta_model = Ridge()

# --- 6. Stacking Regressor ---
stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    final_estimator=meta_model,
    n_jobs=-1
)

# --- 7. Pipeline ---
stack_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', stack_model)
])

# --- 8. Fit and Evaluate ---
print("Training vanilla stacking model...\n")
stack_pipe.fit(x_train, y_train)

# Predictions
y_pred_log = stack_pipe.predict(x_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

# Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

# --- 9. Report ---
print("=== VANILLA STACKING ENSEMBLE ===")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²: {r2:.4f}")

# %% [markdown]
# #### *Observation:*
# 
# The stacked ensemble does exemplary! It is now our second best model.

# %% [markdown]
# ## OPTIMIZING STACKED ENSEMBLES USING RANDOMIZEDSEARCHCV

# %%
  # === TUNED STACKING REGRESSION (XGB + LGB + RIDGE with RandomizedSearchCV) ===

# --- 1. Split features and target ---
X = df.drop('constant_dollar_amount', axis=1)
y = np.log1p(df['constant_dollar_amount'])

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --- 2. Identify categorical and numeric columns ---
cat_cols = x_train.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in x_train.columns if c not in cat_cols]

# --- 3. Preprocessor ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

# --- 4. Base Models ---
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
lgb_model = lgb.LGBMRegressor(random_state=42)
meta_model = Ridge()

stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    final_estimator=meta_model,
    n_jobs=-1
)

stack_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', stack_model)
])

# --- 5. Parameter Grid for RandomizedSearchCV ---
param_grid = {
    # XGBoost
    'model__xgb__n_estimators': [100, 200, 300],
    'model__xgb__max_depth': [3, 4, 5, 6, 8],
    'model__xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__xgb__subsample': [0.6, 0.8, 1.0],
    'model__xgb__colsample_bytree': [0.6, 0.8, 1.0],

    # LightGBM
    'model__lgb__n_estimators': [100, 200, 300],
    'model__lgb__num_leaves': [20, 31, 50, 70],
    'model__lgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__lgb__subsample': [0.6, 0.8, 1.0],
    'model__lgb__colsample_bytree': [0.6, 0.8, 1.0],

    # Ridge meta-model
    'model__final_estimator__alpha': np.logspace(-3, 2, 20)
}

# --- 6. Randomized Search ---
random_search = RandomizedSearchCV(
    stack_pipe,
    param_distributions=param_grid,
    n_iter=25,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("Running Randomized Search for Stacking Ensemble...\n")
random_search.fit(x_train, y_train)

# --- 7. Evaluation ---
best_model = random_search.best_estimator_

y_pred_log = best_model.predict(x_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n=== TUNED XGBOOST AND LGM STACKING ENSEMBLE RESULTS ===")
print(f"Best Params: {random_search.best_params_}")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²: {r2:.4f}")


# %% [markdown]
# #### *Observation:*
# 
# Interesting. While we would expect our stacked ensemble to improve in performance, it actually degraded. Let's see how CatBoost and XGBoost do.

# %%
# Retrieve trained base models
xgb_best = best_model.named_steps['model'].named_estimators_['xgb']
lgb_best = best_model.named_steps['model'].named_estimators_['lgb']

# --- 1. Get feature names from preprocessor ---
cat_features = best_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(cat_cols)
num_features = num_cols
all_features = np.concatenate([cat_features, num_features])

# --- 2. Individual model importances ---
xgb_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': xgb_best.feature_importances_,
    'Model': 'XGBoost'
})

lgb_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': lgb_best.feature_importances_,
    'Model': 'LightGBM'
})

# --- 3. Combine and average ---
combined_importance = pd.concat([xgb_importance, lgb_importance])
avg_importance = (
    combined_importance.groupby('Feature')['Importance']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

# --- 4. Plot top 15 features ---
plt.figure(figsize=(12.5, 5.5))
sns.barplot(
    data=avg_importance.head(15),
    x='Importance',
    y='Feature',
    palette='Blues_d'
)
plt.title('Top 15 Feature Importances (Average of XGBoost + LightGBM)', fontsize=14)
plt.xlabel('Average Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### *Interpretation:*
# 
# - **funding_growth_rate** - Most influential predictor; year-over-year changes in funding drive future aid levels.  
#   => Rapid increases or decreases are strong signals for prediction.
# 
# - **lag_1 and lag_2** - Past funding levels heavily influence current allocations.  
#   => Confirms strong temporal dependency and autoregressive behavior in funding trends.
# 
# - **rolling_std_3yr & rolling_mean_3yr** - Three-year variability and average trends capture medium-term funding stability.  
#   => Projects with steady funding patterns are more predictable.
# 
# - **fiscal_year** - Reflects underlying time trends or global policy shifts over the years.
# 
# - **transaction_type_name_Obligations** - Indicates financial commitments; obligations signal ongoing or new project investments.
# 
# - **is_refund** - Refund transactions affect net disbursements, influencing annual funding totals.#
# 
# - **sector_Governance & Administration** and **sector_Health** - Certain sectors consistently impact prediction outcomes, reflecting priority funding areas.
# 
# - **Agency identifiers**  
#   - **funding_agency_name_U.S. Agency for International Development**  
#   - **funding_agency_name_Department of State**  
#   - **managing_agency_name_Department of Energy**, etc.  
#   => Institutional sources matter, but less than historical and trend-based features.
# 
# #### **Key Insight**
# - Temporal and trend-based features (**growth rate, lags, rolling stats**) dominate the model.  
# - Institutional and categorical features play secondary roles.  
# - The pattern reveals that **foreign aid follows historical continuity**, not random fluctuations - 
#   funding decisions are driven more by **momentum and policy trends** than by agency identity.

# %% [markdown]
# ## OPTIMIZE CATBOOST AND XGBOOST USING RANDOMIZEDSEARCHCV

# %%
# === TUNED XGBOOST AND CATBOOST USING RANDOMIZEDSEARCHCV ===

# --- 1. TimeSeriesSplit setup ---
tscv = TimeSeriesSplit(n_splits=8)

# --- 2. Model parameter grids ---
xgb_params = {
    'model__n_estimators': [100, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [4, 6, 8],
    'model__subsample': [0.7, 0.9, 1.0],
    'model__colsample_bytree': [0.7, 0.9, 1.0],
    'model__reg_lambda': [1, 2, 5]
}

cat_params = {
    'model__iterations': [100, 300],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__depth': [4, 6, 8],
    'model__l2_leaf_reg': [1, 3, 5],
    'model__bagging_temperature': [0.5, 1, 2]
}

# --- 3. Model pipelines ---
pipelines = {
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(random_state=42, objective='reg:squarederror'))
    ]),
    'CatBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('model', CatBoostRegressor(random_state=42, verbose=0))
    ])
}

# --- 4. Randomized Search ---
param_grids = {'XGBoost': xgb_params, 'CatBoost': cat_params}

best_models = {}
for name, pipe in pipelines.items():
    print(f"\nTuning {name} with TimeSeriesSplit...\n")

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grids[name],
        n_iter=25,               # test 25 random combinations
        cv=tscv,
        scoring='r2',
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X, y)
    best_models[name] = search.best_estimator_

    print(f"Best R² (CV): {search.best_score_:.4f}")
    print(f"Best Params: {search.best_params_}\n")

# --- 5. Evaluate on final holdout (last few years, e.g. 2020–2025) ---
holdout = df[df['fiscal_year'] >= 2020]
train = df[df['fiscal_year'] < 2020]

X_train = train.drop('constant_dollar_amount', axis=1)
y_train = np.log1p(train['constant_dollar_amount'])
X_test = holdout.drop('constant_dollar_amount', axis=1)
y_test = np.log1p(holdout['constant_dollar_amount'])

for name, model in best_models.items():
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"--- {name} (Final Holdout Evaluation) ---")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²: {r2:.4f}\n")

# %% [markdown]
# #### *Observation:*
# 
# Outstanding! XGBoost with RandomSearchCV optimization now becomes our best performing model and explains nearly all the variance (99.19%) with the least RMSE and MAE. 
# 
# CatBoost is a close second. Now we might end it here but we want to see if we can reduce that RMSE into tens of thousands. So we try out Optuna.
# 
# RandomizedSearchCV takes random samples of the data and trains different models on them. For sequential data such as ours (time series), we know that that this might not be the best approach. We understand that historical data does affect current and future data. That's what Optuna is for.

# %% [markdown]
# Feature Importance

# %%
# === CATBOOST FEATURE IMPORTANCE ===
cat_model = best_models['CatBoost']  # retrieve best CatBoost pipeline
catboost_model = cat_model.named_steps['model']  # extract the CatBoostRegressor

# Get feature names from the preprocessor
feature_names = cat_model.named_steps['preprocessor'].get_feature_names_out()

# Retrieve importances
importances = catboost_model.get_feature_importance()
feat_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Display top features
display(feat_importance.head(20))

# Optional: visualize
plt.figure(figsize=(13,5))
plt.barh(feat_importance.head(15)['Feature'][::-1],
         feat_importance.head(15)['Importance'][::-1])
plt.title('CatBoost Feature Importance (Top 15)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()


# %% [markdown]
# ### *Interpretation:*
# 
# The CatBoost model’s feature importance results indicate which factors most strongly influence the model’s ability to **forecast overall funding**. The distribution of importance scores highlights a strong emphasis on historical and trend-based predictors over categorical descriptors.  
# 
# #### Key Insights
# 
# - **Funding Growth Rate (49.33%)**
#   - This is the single most influential variable, showing that the *rate at which funding changes over time* is the dominant signal for future predictions.  
#   - It implies that the model heavily relies on recent momentum—rapid increases or decreases in funding strongly influence expected future amounts.
# 
# - **Lag 1 (46.71%)**
#   - The immediate past value of funding (previous year) is nearly as powerful as the growth rate itself.  
#   - This indicates that *funding exhibits strong temporal continuity*—past funding levels are a solid indicator of what follows next.
# 
# - **Rolling Mean (3-Year) (2.13%)**
#   - The 3-year average smooths fluctuations, capturing medium-term trends.  
#   - Although less dominant, it reinforces the idea that multi-year funding stability still carries predictive weight.
# 
# - **Lag 2 & Rolling Std (0.87% and 0.35%)**
#   - These features capture deeper lag effects and volatility in funding patterns, suggesting minor yet meaningful contributions to understanding cyclical or irregular movements.
# 
# - **Fiscal Year (0.26%)**
#   - A very small but non-zero influence, hinting that the timing or period itself may affect funding levels slightly—possibly due to policy cycles or budget schedules.
# 
# - **Categorical Variables (each <0.1%)**
#   - Agency and sector identifiers (like “U.S. Agency for International Development” or “Health”) have minimal influence.  
#   - This shows the model prioritizes *temporal and numerical trends* over institutional or categorical distinctions—consistent with your goal of forecasting **aggregate funding**, not sector-specific amounts.
# 
# #### Summary
# 
# - The CatBoost model behaves much like a **time series learner**, deriving almost all its predictive power from **funding dynamics over time** (growth rate, lagged values, and rolling averages).  
# - Sectoral and agency identifiers add marginal nuance but do not drive the core prediction.  
# - This pattern confirms that **historical funding behavior** is the best predictor of future funding levels in this dataset.

# %% [markdown]
# ## OPTIMIZE CATBOOST AND XGBOOST USING OPTUNA OPTIMIZATION
# 
# Unlike GridSearchCV which uses a brute force approach to find the best parameters and RandomizedSearchCV which takes different samples, Optuna trains different models in the defined hyperparameter space sequentially to find the best parameters. 
# 
# Not only does that serve us well given the nature of our data, it is also faster and less computationally intensive.

# %%
 # === OPTUNA OPTIMIZATION FOR XGBOOST ===
# --- 1. Setup ---
tscv = TimeSeriesSplit(n_splits=8)

# === XGBOOST OPTIMIZATION ===
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    }

    model = xgb.XGBRegressor(**params)
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=tscv, scoring='r2', n_jobs=-1)
    return np.mean(scores)

print("=== OPTIMIZING XGBOOST ===")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=40, show_progress_bar=True)

print(f"\nBest R² (XGB CV): {study_xgb.best_value:.4f}")
print("Best Params (XGB):", study_xgb.best_params)

# --- Fit Best XGB ---
best_xgb = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb.XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', random_state=42, n_jobs=-1))
])
best_xgb.fit(X, y)

# === CATBOOST OPTIMIZATION ===
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 100, 500),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_state': 42,
        'verbose': 0
    }

    model = CatBoostRegressor(**params)
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=tscv, scoring='r2', n_jobs=-1)
    return np.mean(scores)

print("\n=== OPTIMIZING CATBOOST ===")
study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=40, show_progress_bar=True)

print(f"\nBest R² (CatBoost CV): {study_cat.best_value:.4f}")
print("Best Params (CatBoost):", study_cat.best_params)

# --- Fit Best CatBoost ---
best_cat = Pipeline([
    ('preprocessor', preprocessor),
    ('model', CatBoostRegressor(**study_cat.best_params, random_state=42, verbose=0))
])
best_cat.fit(X, y)

# === FINAL HOLDOUT EVALUATION (2020–2025) ===
holdout = df[df['fiscal_year'] >= 2020]
train = df[df['fiscal_year'] < 2020]

X_train = train.drop('constant_dollar_amount', axis=1)
y_train = np.log1p(train['constant_dollar_amount'])
X_test = holdout.drop('constant_dollar_amount', axis=1)
y_test = np.log1p(holdout['constant_dollar_amount'])

# --- XGBOOST EVAL ---
y_pred_log_xgb = best_xgb.predict(X_test)
y_pred_xgb = np.expm1(y_pred_log_xgb)
y_true = np.expm1(y_test)

mae_xgb = mean_absolute_error(y_true, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_true, y_pred_xgb))
r2_xgb = r2_score(y_true, y_pred_xgb)

print(f"\n--- XGBOOST HOLDOUT RESULTS ---")
print(f"MAE: {mae_xgb:,.2f}")
print(f"RMSE: {rmse_xgb:,.2f}")
print(f"R²: {r2_xgb:.4f}")

# --- CATBOOST EVAL ---
y_pred_log_cat = best_cat.predict(X_test)
y_pred_cat = np.expm1(y_pred_log_cat)

mae_cat = mean_absolute_error(y_true, y_pred_cat)
rmse_cat = np.sqrt(mean_squared_error(y_true, y_pred_cat))
r2_cat = r2_score(y_true, y_pred_cat)

print(f"\n--- CATBOOST HOLDOUT RESULTS ---")
print(f"MAE: {mae_cat:,.2f}")
print(f"RMSE: {rmse_cat:,.2f}")
print(f"R²: {r2_cat:.4f}")

# %% [markdown]
# Throughout all our experimantation and optimization, XGBoost with hyperparameter oprimization using RandomizedSearchCV has emerged the top with the highed explained variance- R-Squared (99.1%) and lowest RMSE and MAE. Catboost is a close second. It has been great throughout but couldnt beat XGBoost in the end. 
# 
# As such, we will save the XGBoost model with RandomizedSearchCV and deploy it!

# %% [markdown]
# ## FEATURE IMPORTANCES FOR OUR BEST MODEL

# %%
# Extract the tuned XGBoost pipeline
best_xgb = best_models['XGBoost']

# Extract trained model and feature names
xgb_model = best_xgb.named_steps['model']
feature_names = best_xgb.named_steps['preprocessor'].get_feature_names_out()

# Compute importances
importances = xgb_model.feature_importances_
feat_imp_xgb = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 15
plt.figure(figsize=(14,6.5))
plt.barh(feat_imp_xgb.head(20)['Feature'], feat_imp_xgb.head(20)['Importance'])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

feat_imp_xgb.head(20)

# %% [markdown]
# ### *Interpretation:*
# 
# The XGBoost model highlights that **historical funding behavior** is the key driver of future allocations, while agency or sector attributes play a much smaller role.
# 
# #### **Key Takeaways**
# 
# - **rolling_mean_3yr (0.416)**  
#   - The 3-year moving average of aid dominates the model.  
#   - Funding tends to follow recent trends - high or low spending patterns persist across years.
# 
# - **funding_growth_rate (0.295)**  
#   - The rate at which funding changes is highly influential.  
#   - Strong upward or downward movements in recent years heavily shape future expectations.
# 
# - **lag_1 (0.221)**  
#   - The previous year’s funding significantly impacts the next.  
#   - This reflects continuity and institutional momentum in funding decisions.
# 
# - **Agency indicators (0.004–0.009)**  
#   - Different agencies (e.g., USAID, DFC, MCC) have small but measurable effects.  
#   - However, which agency handles the funds matters far less than overall trends.
# 
# - **is_refund (0.007)**  
#   - Refund transactions play a minor role, suggesting occasional adjustments or corrections in the funding process.
# 
# - **fiscal_year (0.003)**  
#   - The fiscal year itself has low predictive value - time effects are already captured through rolling averages and lagged terms.
# 
# - **rolling_std_3yr (0.001)**  
#   - Variability in recent funding is not a strong predictor.  
#   - Stable, consistent patterns matter more than short-term volatility.
# 
# - **Sector indicators (~0.001)**  
#   - Sector-based effects are minimal, implying that funding moves together across sectors rather than diverging sharply.
# 
# #### **Implication:**
# 
# - The model emphasizes **momentum and historical consistency** in aid allocation.  
# - **Temporal features** (trends, averages, and lags) overwhelmingly explain future funding.  
# - **Categorical features** like sector or agency add nuance but do not drive the main pattern.  
# - Policymakers seem to adjust budgets **gradually** rather than making abrupt shifts year to year.

# %% [markdown]
# # SAVE BEST MODEL
# 
# We save the entire modeling pipeline together with the preprocessing steps

# %%
import joblib

# Save the full pipeline
joblib.dump(best_xgb, 'best_xgb_pipeline.pkl')

print("XGBoost pipeline saved successfully as 'best_xgb_pipeline.pkl'")


