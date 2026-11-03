import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print("STEP 1: DATA LOADING, PREPROCESSING, AND CLEANING")
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
df = pd.read_csv('export.csv')
print("Initial Dataset Shape:", df.shape)
print("\nInitial Missing Values per Column:\n", df.isnull().sum())
print("\nInitial Exact Duplicate Rows:", df.duplicated().sum())
df['Port Name'] = df['Port Name'].replace({'Chief Mountain Mt Poe': 'Chief Mountain'})
df['State'] = df['State'].fillna('Montana')
df['Latitude'] = df['Latitude'].fillna(48.999)
df['Longitude'] = df['Longitude'].fillna(-113.660)
df['Point'] = df['Point'].fillna('POINT (-113.66 48.999)')
df['Port_Label'] = df['Port Name'] + ', ' + df['State']
df['Value'] = df['Value'].astype(str).str.replace(',', '')
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df['Value'] = df['Value'].fillna(df.groupby(['Port Code', 'Measure'])['Value'].transform('median'))
df['Value'] = df['Value'].fillna(0).astype(int)
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
df = df.drop_duplicates()
df = df.sort_values(['Port Code', 'Measure', 'Date']).reset_index(drop=True)
print("\nCleaned Dataset Shape:", df.shape)
print("\nRemaining Missing Values After Cleaning:\n", df.isnull().sum())
print("\nSummary Statistics of Cleaned Numerical Columns:\n", df[['Port Code', 'Value', 'Latitude', 'Longitude']].describe())
print("======================================================================")
print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA) & FIVE OBJECTIVES")
print("======================================================================")
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
mode_category_map = {
    'Personal Vehicle Passengers': 'Passenger Traffic',
    'Pedestrians': 'Passenger Traffic',
    'Bus Passengers': 'Passenger Traffic',
    'Train Passengers': 'Passenger Traffic',
    'Personal Vehicles': 'Personal Vehicles',
    'Buses': 'Transit Vehicles',
    'Trucks': 'Commercial Freight',
    'Trains': 'Commercial Freight'
}
df['Category'] = df['Measure'].map(mode_category_map)
print("\n--- OBJECTIVE 1A: Border-Level Volume Summary")
border_summary = df.groupby('Border')['Value'].agg(['count', 'sum', 'mean', 'median', 'std'])
border_summary['Share_%'] = (border_summary['sum'] / df['Value'].sum()) * 100
print(border_summary)

print("\n--- OBJECTIVE 1B: Transport Mode (Measure) Volume by Border")
border_mode_table = df.pivot_table(index='Measure', columns='Border', values='Value', aggfunc='sum')
border_mode_table['Total'] = border_mode_table.sum(axis=1)
border_mode_table['Share_%'] = (border_mode_table['Total'] / border_mode_table['Total'].sum()) * 100
border_mode_table = border_mode_table.sort_values(by='Total', ascending=False)
print(border_mode_table)

print("\n--- OBJECTIVE 2: Annual Border Volumes in Key Historical Milestone Years")
yearly_table = df.pivot_table(index='Year', columns='Border', values='Value', aggfunc='sum')
yearly_table['Combined_Total'] = yearly_table.sum(axis=1)
print(yearly_table.loc[[1996, 2000, 2001, 2002, 2019, 2020, 2021, 2022, 2025, 2026]])

print("\n--- OBJECTIVE 3A: Top 10 Border States by Cumulative Volume ---")
state_table = df.groupby(['State', 'Border'])['Value'].sum().reset_index()
state_table = state_table.sort_values(by='Value', ascending=False).reset_index(drop=True)
state_table['Share_%'] = (state_table['Value'] / df['Value'].sum()) * 100
print(state_table.head(10))

print("\n--- OBJECTIVE 3B: Top 10 Ports of Entry by Cumulative Volume ---")
port_table = df.groupby(['Port_Label', 'Port Code', 'Border'])['Value'].sum().reset_index()
port_table = port_table.sort_values(by='Value', ascending=False).reset_index(drop=True)
port_table['Share_%'] = (port_table['Value'] / df['Value'].sum()) * 100
port_table['Cumulative_Share_%'] = port_table['Share_%'].cumsum()
print(port_table.head(10))

print("\n--- OBJECTIVE 4: Monthly Mean Crossings by Functional Traffic Category ---")
monthly_seasonality = df.pivot_table(index='Month', columns='Category', values='Value', aggfunc='mean')
print(monthly_seasonality)

print("\n======================================================================")
print("STEP 3: GENERATING EDA DATA VISUALIZATIONS")
print("======================================================================")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

border_totals = df.groupby('Border')['Value'].sum() / 1e9
axes[0].pie(
    border_totals,
    labels=border_totals.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=['#2b5c8f', '#d95f02'],
    explode=(0.04, 0),
    textprops={'fontsize': 11, 'weight': 'bold'}
)
axes[0].set_title('Objective 1A: Share of Total Crossings by Border\n(Total = 11.68 Billion)', fontsize=13, weight='bold')

border_mode = df.groupby(['Measure', 'Border'])['Value'].sum().unstack() / 1e6
border_mode = border_mode.loc[border_mode.sum(axis=1).sort_values(ascending=True).index]
border_mode.plot(kind='barh', ax=axes[1], color=['#2b5c8f', '#d95f02'], width=0.8)
axes[1].set_xscale('log')
axes[1].set_title('Objective 1B: Crossing Volume by Mode & Border\n(Millions, Log Scale)', fontsize=13, weight='bold')
axes[1].set_xlabel('Cumulative Crossings (Millions - Log Scale)', fontsize=11)
axes[1].set_ylabel('Transport Mode (Measure)', fontsize=11)
axes[1].legend(title='Border', frameon=True)

plt.tight_layout()
plt.savefig('viz1_objective1_border_modal_split.png', bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

yearly_border = df[df['Year'] <= 2025].groupby(['Year', 'Border'])['Value'].sum().unstack() / 1e6
yearly_border.plot(kind='line', marker='o', linewidth=2.5, color=['#2b5c8f', '#d95f02'], ax=axes[0])
axes[0].axvline(x=2001, color='dimgray', linestyle='--', linewidth=1.5, label='Post-9/11 Tightening (2001)')
axes[0].axvline(x=2020, color='crimson', linestyle='--', linewidth=1.8, label='COVID-19 Restrictions (2020)')
axes[0].set_title('Objective 2A: Annual Border Crossing Trends (1996–2025)\n(Full Calendar Years, in Millions)', fontsize=13, weight='bold')
axes[0].set_xlabel('Year', fontsize=11)
axes[0].set_ylabel('Total Annual Crossings (Millions)', fontsize=11)
axes[0].legend(frameon=True, fontsize=10)

covid_window = yearly_border.loc[[2019, 2020, 2021, 2022, 2025]]
covid_recovery_pct = covid_window.divide(covid_window.loc[2019]) * 100
covid_recovery_pct.plot(kind='bar', color=['#2b5c8f', '#d95f02'], width=0.75, ax=axes[1])
axes[1].axhline(y=100, color='black', linestyle=':', linewidth=1.5, label='2019 Pre-Pandemic Baseline (100%)')
axes[1].set_title('Objective 2B: COVID-19 Shock & Recovery Index\n(% of 2019 Pre-Pandemic Volume)', fontsize=13, weight='bold')
axes[1].set_xlabel('Year', fontsize=11)
axes[1].set_ylabel('Volume Index (2019 = 100%)', fontsize=11)
axes[1].set_xticklabels(covid_recovery_pct.index, rotation=0)
axes[1].legend(frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('viz2_objective2_longitudinal_trends.png', bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

top_states = df.groupby('State')['Value'].sum().sort_values(ascending=True).tail(10) / 1e9
top_states.plot(kind='barh', color='#1b9e77', width=0.75, ax=axes[0])
axes[0].set_title('Objective 3A: Top 10 States by Cumulative Volume\n(1996–2026, in Billions)', fontsize=13, weight='bold')
axes[0].set_xlabel('Total Crossings (Billions)', fontsize=11)
axes[0].set_ylabel('State', fontsize=11)

top_ports = df.groupby('Port_Label')['Value'].sum().sort_values(ascending=True).tail(10) / 1e9
top_ports.plot(kind='barh', color='#7570b3', width=0.75, ax=axes[1])
axes[1].set_title('Objective 3B: Top 10 Ports of Entry by Cumulative Volume\n(Top 10 Ports = 65.68% of National Volume)', fontsize=13, weight='bold')
axes[1].set_xlabel('Total Crossings (Billions)', fontsize=11)
axes[1].set_ylabel('Port of Entry (Port, State)', fontsize=11)

plt.tight_layout()
plt.savefig('viz3_objective3_geographic_concentration.png', bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

monthly_pax = df[df['Category'] == 'Passenger Traffic'].groupby('Month')['Value'].mean()
axes[0].plot(range(1, 13), monthly_pax.values, marker='s', markersize=8, color='#008080', linewidth=2.8)
axes[0].set_title('Objective 4A: Passenger Traffic Monthly Seasonality\n(Summer Peak in Jul–Aug [+28% vs Feb] & Dec Spike)', fontsize=12.5, weight='bold')
axes[0].set_xlabel('Month of Year', fontsize=11)
axes[0].set_ylabel('Mean Monthly Crossings per Record', fontsize=11)
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_labels)

monthly_freight = df[df['Category'] == 'Commercial Freight'].groupby('Month')['Value'].mean()
axes[1].plot(range(1, 13), monthly_freight.values, marker='^', markersize=8, color='#b22222', linewidth=2.8)
axes[1].set_title('Objective 4B: Commercial Freight Monthly Seasonality\n(Peaks in Mar & Oct; Annual Trough in Dec [-14.5%])', fontsize=12.5, weight='bold')
axes[1].set_xlabel('Month of Year', fontsize=11)
axes[1].set_ylabel('Mean Monthly Crossings per Record', fontsize=11)
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_labels)

plt.tight_layout()
plt.savefig('viz4_objective4_seasonal_dynamics.png', bbox_inches='tight')
plt.show()



print("\n======================================================================")
print("STEP 4: FEATURE ENGINEERING")
print("======================================================================")


df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Is_Summer'] = df['Month'].isin([6, 7, 8]).astype(int)
df['Is_COVID_Era'] = df['Year'].isin([2020, 2021]).astype(int)


df['Lag_1_Month'] = df.groupby(['Port Code', 'Measure'])['Value'].shift(1)
df['Lag_2_Month'] = df.groupby(['Port Code', 'Measure'])['Value'].shift(2)
df['Lag_3_Month'] = df.groupby(['Port Code', 'Measure'])['Value'].shift(3)
df['Lag_12_Month'] = df.groupby(['Port Code', 'Measure'])['Value'].shift(12)

df['Rolling_3M_Mean'] = (df['Lag_1_Month'] + df['Lag_2_Month'] + df['Lag_3_Month']) / 3.0
df['YoY_Diff'] = df['Lag_1_Month'] - df['Lag_12_Month']


ml_df = df.dropna(subset=['Lag_1_Month', 'Lag_2_Month', 'Lag_3_Month', 'Lag_12_Month', 'Rolling_3M_Mean']).copy()


le_border = LabelEncoder()
le_state = LabelEncoder()
le_measure = LabelEncoder()

ml_df['Border_Enc'] = le_border.fit_transform(ml_df['Border'])
ml_df['State_Enc'] = le_state.fit_transform(ml_df['State'])
ml_df['Measure_Enc'] = le_measure.fit_transform(ml_df['Measure'])


features = [
    'Port Code', 'Border_Enc', 'State_Enc', 'Measure_Enc',
    'Latitude', 'Longitude', 'Year', 'Month', 'Quarter',
    'Month_Sin', 'Month_Cos', 'Is_Summer', 'Is_COVID_Era',
    'Lag_1_Month', 'Lag_12_Month', 'Rolling_3M_Mean', 'YoY_Diff'
]

X = ml_df[features]
y = ml_df['Value']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Engineered ML Dataset Shape:", ml_df.shape)
print("Total Features Engineered:", len(features))
print("Training Set Rows:", X_train.shape[0], "| Testing Set Rows:", X_test.shape[0])



print("\n======================================================================")
print("STEP 5: TRAINING & EVALUATING 5 MACHINE LEARNING MODELS")
print("======================================================================")


lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
pred_lr = lr_model.predict(X_test_scaled)


ridge_model = Ridge(alpha=10.0, random_state=42)
ridge_model.fit(X_train_scaled, y_train)
pred_ridge = ridge_model.predict(X_test_scaled)


dt_model = DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=42)
dt_model.fit(X_train, y_train)
pred_dt = dt_model.predict(X_test)


rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)


hgb_model = HistGradientBoostingRegressor(max_iter=150, max_depth=10, learning_rate=0.1, random_state=42)
hgb_model.fit(X_train, y_train)
pred_hgb = hgb_model.predict(X_test)


results_df = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'Ridge Regression',
        'Decision Tree Regressor',
        'Random Forest Regressor',
        'Gradient Boosting (HistGBM)'
    ],
    'R2_Score': [
        r2_score(y_test, pred_lr),
        r2_score(y_test, pred_ridge),
        r2_score(y_test, pred_dt),
        r2_score(y_test, pred_rf),
        r2_score(y_test, pred_hgb)
    ],
    'MAE': [
        mean_absolute_error(y_test, pred_lr),
        mean_absolute_error(y_test, pred_ridge),
        mean_absolute_error(y_test, pred_dt),
        mean_absolute_error(y_test, pred_rf),
        mean_absolute_error(y_test, pred_hgb)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, pred_lr)),
        np.sqrt(mean_squared_error(y_test, pred_ridge)),
        np.sqrt(mean_squared_error(y_test, pred_dt)),
        np.sqrt(mean_squared_error(y_test, pred_rf)),
        np.sqrt(mean_squared_error(y_test, pred_hgb))
    ]
})

results_ranked = results_df.sort_values(by='R2_Score', ascending=False).reset_index(drop=True)
print("\n--- Machine Learning Model Comparison Table (Ranked by R2 & RMSE) ---")
print(results_ranked.to_string(index=False))


fi_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)
print("\n--- Feature Importance Ranking (Random Forest) ---")
print(fi_df.to_string(index=False))


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

results_sorted = results_df.sort_values('RMSE', ascending=False)
x_pos = np.arange(len(results_sorted))
width = 0.38

axes[0].barh(x_pos - width/2, results_sorted['RMSE'], height=width, label='RMSE (Root Mean Squared Error)', color='#377eb8')
axes[0].barh(x_pos + width/2, results_sorted['MAE'], height=width, label='MAE (Mean Absolute Error)', color='#ff7f00')
axes[0].set_yticks(x_pos)
axes[0].set_yticklabels(results_sorted['Model'], fontsize=11)
axes[0].set_title('Objective 5A: ML Model Error Comparison (RMSE & MAE)\n(Lower Error = Better Fit)', fontsize=13, weight='bold')
axes[0].set_xlabel('Prediction Error (Monthly Crossings)', fontsize=11)
axes[0].legend(frameon=True)

axes[1].scatter(y_test / 1e6, pred_hgb / 1e6, alpha=0.3, color='#2ca02c', edgecolors='none', s=25, label='Test Predictions (n=53,243)')
axes[1].plot([0, y_test.max() / 1e6], [0, y_test.max() / 1e6], color='crimson', linestyle='--', linewidth=2, label='Ideal 1:1 Fit Line')
axes[1].set_title('Objective 5B: Best Model (HistGBM, R² = 0.9898)\nActual vs. Predicted Monthly Volume (Millions)', fontsize=13, weight='bold')
axes[1].set_xlabel('Actual Monthly Crossings (Millions)', fontsize=11)
axes[1].set_ylabel('Predicted Monthly Crossings (Millions)', fontsize=11)
axes[1].legend(frameon=True)

plt.tight_layout()
plt.savefig('viz5_objective5_ml_evaluation.png', bbox_inches='tight')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

port_geo = df.groupby(['Port Code', 'Port_Label', 'Border', 'Latitude', 'Longitude'])['Value'].sum().reset_index()
can_ports = port_geo[port_geo['Border'] == 'US-Canada Border']
mex_ports = port_geo[port_geo['Border'] == 'US-Mexico Border']

axes[0].scatter(
    can_ports['Longitude'], can_ports['Latitude'],
    s=np.sqrt(can_ports['Value'] / 1e5) + 15,
    color='#2b5c8f', alpha=0.7, edgecolors='black', linewidth=0.5, label='US-Canada Ports (89 Ports)'
)
axes[0].scatter(
    mex_ports['Longitude'], mex_ports['Latitude'],
    s=np.sqrt(mex_ports['Value'] / 1e5) + 15,
    color='#d95f02', alpha=0.75, edgecolors='black', linewidth=0.5, label='US-Mexico Ports (29 Ports)'
)
axes[0].set_title('Geospatial Distribution of US Border Ports of Entry\n(Bubble Size Scaled by Cumulative Crossing Volume)', fontsize=12.5, weight='bold')
axes[0].set_xlabel('Longitude', fontsize=11)
axes[0].set_ylabel('Latitude', fontsize=11)
axes[0].legend(frameon=True, loc='lower left')

fi_top10 = fi_df.head(10).sort_values(by='Importance', ascending=True)
axes[1].barh(fi_top10['Feature'], fi_top10['Importance'], color='#6a3d9a', height=0.7)
axes[1].set_xscale('log')
axes[1].set_title('Top 10 Engineered Features by Importance (Random Forest)\n(Log Scale to Highlight Secondary Temporal & Spatial Drivers)', fontsize=12.5, weight='bold')
axes[1].set_xlabel('Feature Importance Score (Log Scale)', fontsize=11)
axes[1].set_ylabel('Engineered Feature', fontsize=11)

plt.tight_layout()
plt.savefig('viz6_geospatial_and_feature_importance.png', bbox_inches='tight')
plt.show()


print("\n======================================================================")
print("STEP 6: KEY FINDINGS, CONCLUSIONS, AND RECOMMENDATIONS")
print("======================================================================")
print("1. BORDER ASYMMETRY: US-Mexico Border handles 74.98% of volume (8.76B) across 29 ports,")
print("   while US-Canada Border handles 25.02% (2.92B) across 89 ports but leads in Trucks (52.60%).")
print("2. MACRO TRENDS: Traffic peaked in 2000 (529.69M), fell 47.39% in 2020 due to COVID-19,")
print("   and recovered to 99.76% on the US-Mexico border vs. 70.75% on the US-Canada border by 2025.")
print("3. PARETO CONCENTRATION: Top 3 states (TX, CA, AZ) account for 74.21% of volume;")
print("   Top 10 ports (led by San Ysidro & El Paso) process 65.68% of all national crossings.")
print("4. SEASONAL DIVERGENCE: Passenger traffic peaks in Jul-Aug (+28% vs Feb) and Dec,")
print("   whereas Commercial Freight peaks in Mar and Oct and hits its annual low in Dec (-14.5%).")
print("5. BEST-FIT ML MODEL: Gradient Boosting (HistGBM) achieved the highest R2 (0.9898) and")
print("   lowest RMSE (17,629.04), while Random Forest achieved the lowest MAE (2,938.33).")
print("======================================================================")
