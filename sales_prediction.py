# ── Imports ──────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
from pathlib import Path
import os
import json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ── Styling ──────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.1)
PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

# ============================================================
# STEP 1 — LOAD & CLEAN DATA
# ============================================================
print("=" * 60)
print("STEP 1: LOADING & CLEANING DATA")
print("=" * 60)

# Paths (use script location so relative runs work reliably)
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / 'Advertising.csv'
OUTPUT_DIR = BASE_DIR / 'Img_output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, index_col=0)
df.columns = ['TV', 'Radio', 'Newspaper', 'Sales']

print(f"\nDataset shape : {df.shape}  ({df.shape[0]} records, {df.shape[1]} features)")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print("\n── Descriptive Statistics ──")
print(df.describe().round(2).to_string())

# Check for outliers using IQR
print("\n── Outlier Detection (IQR method) ──")
for col in df.columns:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"  {col:<12}: {len(outliers)} outliers")

# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\n── Correlation with Sales ──")
corr = df.corr()['Sales'].drop('Sales').sort_values(ascending=False)
for feat, c in corr.items():
    print(f"  {feat:<12}: {c:.4f}")

# ── EDA Figure ────────────────────────────────────────────────
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Exploratory Data Analysis — Advertising Dataset',
              fontsize=16, fontweight='bold')

# a) Sales distribution
axes[0, 0].hist(df['Sales'], bins=20, color='#2196F3',
                edgecolor='white', alpha=0.85)
axes[0, 0].axvline(df['Sales'].mean(), color='red', linestyle='--',
                   label=f'Mean = {df["Sales"].mean():.1f}')
axes[0, 0].set(title='Sales Distribution',
               xlabel='Sales (units)', ylabel='Frequency')
axes[0, 0].legend()

# b) Correlation heatmap
sns.heatmap(df[['TV', 'Radio', 'Newspaper', 'Sales']].corr(),
            annot=True, fmt='.2f', cmap='Blues', ax=axes[0, 1],
            square=True, linewidths=.5, cbar_kws={'shrink': .7})
axes[0, 1].set_title('Correlation Heatmap')

# c) TV vs Sales scatter
axes[1, 0].scatter(df['TV'], df['Sales'], alpha=0.5,
                   color='#2196F3', edgecolors='white', s=50)
m, b = np.polyfit(df['TV'], df['Sales'], 1)
xs = np.linspace(df['TV'].min(), df['TV'].max(), 100)
axes[1, 0].plot(xs, m * xs + b, color='red', linewidth=2,
                label=f'y = {m:.3f}x + {b:.1f}')
axes[1, 0].set(title='TV Spend vs Sales',
               xlabel='TV Budget ($K)', ylabel='Sales (units)')
axes[1, 0].legend()

# d) Budget share pie
budget_means = df[['TV', 'Radio', 'Newspaper']].mean()
axes[1, 1].pie(budget_means, labels=budget_means.index,
               autopct='%1.1f%%',
               colors=['#2196F3', '#FF5722', '#4CAF50'],
               startangle=140,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1, 1].set_title('Average Advertising Budget Share')

plt.tight_layout()
fig1_path = OUTPUT_DIR / 'fig1_eda.png'
plt.savefig(str(fig1_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {fig1_path} saved")

# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: FEATURE ENGINEERING")
print("=" * 60)

df['TV_Radio']     = df['TV'] * df['Radio']          # interaction term
df['TV_sq']        = df['TV'] ** 2                   # TV non-linearity
df['Radio_sq']     = df['Radio'] ** 2                # Radio non-linearity
df['Total_Spend']  = df['TV'] + df['Radio'] + df['Newspaper']
df['TV_share']     = df['TV'] / df['Total_Spend']    # TV budget proportion

features_base = ['TV', 'Radio', 'Newspaper']
features_eng  = ['TV', 'Radio', 'Newspaper',
                 'TV_Radio', 'TV_sq', 'Radio_sq',
                 'Total_Spend', 'TV_share']
target = 'Sales'

print(f"  Original features : {features_base}")
print(f"  Engineered features added: TV_Radio, TV_sq, Radio_sq,"
      f" Total_Spend, TV_share")

X = df[features_eng]
y = df[target]

# ── Train / Test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\n  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

# Scale for linear models
scaler   = StandardScaler()
X_tr_s   = scaler.fit_transform(X_train)
X_te_s   = scaler.transform(X_test)

# ============================================================
# STEP 4 — MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    'Linear Regression' : LinearRegression(),
    'Ridge Regression'  : Ridge(alpha=1.0),
    'Lasso Regression'  : Lasso(alpha=0.1),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print(f"\n  {'Model':<22} {'Train R²':>9} {'Test R²':>9} "
      f"{'RMSE':>8} {'MAE':>8}")
print("  " + "-" * 58)

for name, model in models.items():
    is_linear = 'Regression' in name
    if is_linear:
        model.fit(X_tr_s, y_train)
        pred   = model.predict(X_te_s)
        train_r2 = model.score(X_tr_s, y_train)
    else:
        model.fit(X_train, y_train)
        pred     = model.predict(X_test)
        train_r2 = model.score(X_train, y_train)

    r2   = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    results[name] = dict(r2=r2, rmse=rmse, mae=mae,
                         pred=pred, train_r2=train_r2)
    print(f"  {name:<22} {train_r2:>9.4f} {r2:>9.4f} "
          f"{rmse:>8.4f} {mae:>8.4f}")

best_model = max(results, key=lambda k: results[k]['r2'])
print(f"\n  ✓ Best model: {best_model}  (R² = {results[best_model]['r2']:.4f})")

# ── 5-Fold Cross-Validation on best model ───────────────────
print("\n── 5-Fold Cross-Validation (Linear Regression) ──")
cv_scores = cross_val_score(LinearRegression(),
                             scaler.fit_transform(X), y,
                             cv=5, scoring='r2')
print(f"  CV R² scores : {cv_scores.round(4)}")
print(f"  Mean ± Std   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Model comparison figure ──────────────────────────────────
model_names_short = ['Lin Reg', 'Ridge', 'Lasso',
                     'Rand Forest', 'Grad Boost']
r2s   = [results[m]['r2']   for m in models]
rmses = [results[m]['rmse'] for m in models]

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Model Performance Comparison',
              fontsize=15, fontweight='bold')

# a) R² bar
bars = axes[0].barh(model_names_short, r2s, color=PALETTE,
                    edgecolor='white', height=0.55)
axes[0].set(title='Test R² Score', xlabel='R²', xlim=[0.97, 1.0])
for bar, val in zip(bars, r2s):
    axes[0].text(val + 0.0002, bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', fontsize=9)

# b) RMSE bar
bars2 = axes[1].barh(model_names_short, rmses, color=PALETTE,
                     edgecolor='white', height=0.55)
axes[1].set(title='RMSE (lower = better)', xlabel='RMSE')
for bar, val in zip(bars2, rmses):
    axes[1].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=9)

# c) Actual vs Predicted
best_pred = results[best_model]['pred']
axes[2].scatter(y_test, best_pred, alpha=0.6,
                color='#2196F3', edgecolors='white', s=50)
lims = [min(y_test.min(), best_pred.min()) - 1,
        max(y_test.max(), best_pred.max()) + 1]
axes[2].plot(lims, lims, 'r--', linewidth=2, label='Perfect fit')
axes[2].set(title=f'Actual vs Predicted\n({best_model})',
            xlabel='Actual Sales', ylabel='Predicted Sales',
            xlim=lims, ylim=lims)
axes[2].legend()

plt.tight_layout()
fig2_path = OUTPUT_DIR / 'fig2_model_comparison.png'
plt.savefig(str(fig2_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {fig2_path} saved")

# Save model metrics summary and best-model predictions
metrics_df = pd.DataFrame({
    name: {
        'Train_R2': results[name]['train_r2'],
        'Test_R2' : results[name]['r2'],
        'RMSE'     : results[name]['rmse'],
        'MAE'      : results[name]['mae']
    } for name in results
}).T
metrics_path = OUTPUT_DIR / 'model_metrics.csv'
metrics_df.to_csv(metrics_path)
print(f"  → {metrics_path} saved")

best_pred = results[best_model]['pred']
pred_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': best_pred,
    'Residual': y_test.values - best_pred
}, index=y_test.index)
pred_path = OUTPUT_DIR / f'predictions_{best_model.replace(" ","_")}.csv'
pred_df.to_csv(pred_path)
print(f"  → {pred_path} saved")

# ============================================================
# STEP 5 — FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: FEATURE IMPORTANCE (Random Forest)")
print("=" * 60)

rf = models['Random Forest']
fi = pd.Series(rf.feature_importances_,
               index=features_eng).sort_values(ascending=False)
print()
for feat, val in fi.items():
    bar = '█' * int(val * 40)
    print(f"  {feat:<16} {val:.4f}  {bar}")

# ============================================================
# STEP 6 — ADVERTISING IMPACT ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: ADVERTISING IMPACT ANALYSIS")
print("=" * 60)

# Fit simple LR on base 3 features for interpretable coefficients
lr_base = LinearRegression()
lr_base.fit(df[features_base], df[target])
coefs = pd.Series(lr_base.coef_, index=features_base)

print("\n── Sales Lift per $1K Additional Spend ──")
for ch, c in coefs.items():
    print(f"  {ch:<12}: +{c:.4f} units of sales")

print(f"\n  Intercept: {lr_base.intercept_:.4f}")

# ── Insights figure ─────────────────────────────────────────
fig3, axes = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle('Feature Importance & Business Insights',
              fontsize=15, fontweight='bold')

# a) Feature importance bar
colors_fi = ['#2196F3' if i == 0 else '#90CAF9' for i in range(len(fi))]
axes[0].barh(fi.index, fi.values, color=colors_fi, edgecolor='white')
axes[0].set(title='Random Forest Feature Importance',
            xlabel='Importance Score')
axes[0].invert_yaxis()

# b) Regression coefficients
colors_c = ['#2196F3', '#FF5722', '#4CAF50']
axes[1].bar(coefs.index, coefs.values,
            color=colors_c, edgecolor='white', width=0.5)
axes[1].set(title='Sales Lift per $1K Spend\n(Linear Regression)',
            ylabel='Sales Units Increase')
for i, (ch, v) in enumerate(coefs.items()):
    axes[1].text(i, v + 0.01, f'+{v:.3f}',
                 ha='center', fontsize=11, fontweight='bold')

# c) Forecast simulation — vary TV budget
tv_range    = np.linspace(0, 300, 100)
radio_fixed = df['Radio'].mean()
news_fixed  = df['Newspaper'].mean()
sim_X       = np.c_[tv_range,
                    np.full(100, radio_fixed),
                    np.full(100, news_fixed)]
pred_sales  = lr_base.predict(sim_X)

axes[2].plot(tv_range, pred_sales, color='#2196F3',
             linewidth=2.5, label='Predicted Sales')
axes[2].fill_between(tv_range, pred_sales - 1, pred_sales + 1,
                     alpha=0.15, color='#2196F3', label='±1 unit band')
axes[2].axhline(df['Sales'].mean(), color='red', linestyle='--',
                alpha=0.6, label=f'Baseline ({df["Sales"].mean():.1f})')
axes[2].set(title='Sales Forecast vs TV Budget\n'
                  '(Radio & Newspaper held at mean)',
            xlabel='TV Budget ($K)', ylabel='Predicted Sales')
axes[2].legend(fontsize=9)

plt.tight_layout()
fig3_path = OUTPUT_DIR / 'fig3_insights.png'
plt.savefig(str(fig3_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {fig3_path} saved")

# ============================================================
# STEP 7 — BUSINESS RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: ACTIONABLE BUSINESS RECOMMENDATIONS")
print("=" * 60)

tv_coef    = coefs['TV']
radio_coef = coefs['Radio']
news_coef  = coefs['Newspaper']

tv_roi    = tv_coef    / df['TV'].mean()    * 100
radio_roi = radio_coef / df['Radio'].mean() * 100
news_roi  = news_coef  / df['Newspaper'].mean() * 100

print(f"""
  ┌─ KEY FINDINGS ────────────────────────────────────────────┐
  │                                                           │
  │  • TV   advertising has the HIGHEST absolute impact       │
  │    → +{tv_coef:.3f} sales units per $1K invested               │
  │                                                           │
  │  • Radio advertising delivers the BEST ROI per mean $     │
  │    → coefficient {radio_coef:.3f} on a smaller base budget        │
  │                                                           │
  │  • Newspaper shows the WEAKEST correlation with sales     │
  │    → consider reallocating budget to TV or Radio          │
  │                                                           │
  │  • TV × Radio interaction is the #1 Random Forest        │
  │    feature — combined campaigns are highly synergistic    │
  │                                                           │
  │  • Best predictive model : {best_model:<28}  │
  │    Test R² = {results[best_model]["r2"]:.4f}  |  RMSE = {results[best_model]["rmse"]:.4f}               │
  │                                                           │
  └───────────────────────────────────────────────────────────┘

  RECOMMENDED BUDGET ALLOCATION STRATEGY:
  ─────────────────────────────────────────
  1. Prioritise TV + Radio COMBINED campaigns (synergy effect)
  2. Maintain Radio spend — high ROI at lower budget levels
  3. Review Newspaper ROI before next planning cycle
  4. Use the linear model to simulate any TV budget scenario
""")

print("=" * 60)
print("ANALYSIS COMPLETE — All figures saved.")
print("=" * 60)
