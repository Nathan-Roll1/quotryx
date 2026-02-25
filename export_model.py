import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

CSV_PATH = "quotryx_combined_20260225_040559.csv"

NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "size_interior_sqft",
    "lot_size_sqft", "year_built", "parking_spaces",
]
BINARY_FEATURES = [
    "is_house", "is_condo", "has_garage",
    "has_basement", "basement_finished", "has_cooling",
]
HOME_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES
CITY_DUMMIES = ["city_Saskatoon", "city_Regina"]


def parse_size_interior(val):
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    return float(val.replace(" sqft", "").replace(",", "").strip())


def parse_lot_size(val):
    if pd.isna(val) or not isinstance(val, str):
        return np.nan
    val = val.strip()
    if val in ("0.00", "0", "Condo", ""):
        return np.nan
    if "ac" in val.lower():
        num = float(val.lower().replace("ac", "").replace(",", "").strip())
        return num * 43560
    if "sqft" in val.lower():
        return float(val.lower().replace("sqft", "").replace(",", "").strip())
    if "x" in val.lower():
        parts = val.lower().split("x")
        try:
            return float(parts[0]) * float(parts[1])
        except (ValueError, IndexError):
            return np.nan
    return np.nan


df = pd.read_csv(CSV_PATH)
df["size_interior_sqft"] = df["size_interior"].apply(parse_size_interior)
df["lot_size_sqft"] = df["lot_size"].apply(parse_lot_size)
df["is_house"] = (df["building_type"] == "House").astype(int)
df["is_condo"] = df["ownership_type"].fillna("").str.contains("Condominium", case=False).astype(int)
df["has_garage"] = df["parking_type"].fillna("").str.contains("Garage", case=False).astype(int)
df["has_basement"] = (df["basement_type"].fillna("").str.strip() != "").astype(int)
df["basement_finished"] = df["basement_type"].fillna("").str.contains("Finished", case=False).astype(int)
df["has_cooling"] = (df["cooling_type"].fillna("").str.strip() != "").astype(int)

city_dummies = pd.get_dummies(df["city"], prefix="city").astype(int)
for col in CITY_DUMMIES:
    if col not in city_dummies.columns:
        city_dummies[col] = 0
df = pd.concat([df, city_dummies[CITY_DUMMIES]], axis=1)

city_averages = df.groupby("city")["price_numeric"].mean().to_dict()

for col in NUMERIC_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
for col in BINARY_FEATURES:
    df[col] = df[col].fillna(0).astype(int)

feature_medians = df[NUMERIC_FEATURES].median().to_dict()
df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())

y = df["price_numeric"].dropna()
df = df.loc[y.index]

X_home = df[HOME_FEATURES].values
X_city = df[CITY_DUMMIES].values

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_home)
X_full = np.hstack([X_poly, X_city])

mask = np.isfinite(X_full).all(axis=1) & np.isfinite(y.values)
X_full = X_full[mask]
y = y.values[mask]

model = LinearRegression()
model.fit(X_full, y)

X_sm = sm.add_constant(X_full)
ols_result = sm.OLS(y, X_sm).fit()

export = {
    "intercept": float(model.intercept_),
    "coefficients": model.coef_.tolist(),
    "powers": poly.powers_.tolist(),
    "feature_means": feature_medians,
    "city_averages": city_averages,
    "r_squared": round(float(ols_result.rsquared), 4),
    "adj_r_squared": round(float(ols_result.rsquared_adj), 4),
    "n_listings": int(len(y)),
}

with open("docs/model.json", "w") as f:
    json.dump(export, f)

print(f"Exported: {len(export['coefficients'])} coefficients, {len(export['powers'])} poly terms")
print(f"R²={export['r_squared']}, n={export['n_listings']}")
