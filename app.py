import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.stats import norm

app = Flask(__name__)

CSV_PATH = os.path.join(os.path.dirname(__file__), "quotryx_combined_20260225_040559.csv")

NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "size_interior_sqft",
    "lot_size_percentile", "year_built", "parking_spaces",
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


def preprocess(df):
    df = df.copy()
    df["size_interior_sqft"] = df["size_interior"].apply(parse_size_interior)
    df["lot_size_sqft"] = df["lot_size"].apply(parse_lot_size)
    df["lot_size_percentile"] = df["lot_size_sqft"].rank(pct=True) * 100

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

    return df


def train_model():
    df = pd.read_csv(CSV_PATH)
    df = preprocess(df)

    city_averages = df.groupby("city")["price_numeric"].mean().to_dict()

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in BINARY_FEATURES:
        df[col] = df[col].fillna(0).astype(int)

    feature_medians = df[NUMERIC_FEATURES].median().to_dict()
    feature_medians["lot_size_sqft"] = float(df["lot_size_sqft"].median())
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())

    lot_sqfts_sorted = df["lot_size_sqft"].dropna().sort_values().tolist()

    y = df["price_numeric"].dropna()
    df = df.loc[y.index]

    X_home = df[HOME_FEATURES].values
    X_city = df[CITY_DUMMIES].values

    poly = PolynomialFeatures(degree=1, include_bias=False)
    X_poly = poly.fit_transform(X_home)
    X_full = np.hstack([X_poly, X_city])

    mask = np.isfinite(X_full).all(axis=1) & np.isfinite(y.values)
    X_full = X_full[mask]
    y = y.values[mask]

    model = LinearRegression()
    model.fit(X_full, y)

    X_sm = sm.add_constant(X_full)
    ols_result = sm.OLS(y, X_sm).fit()
    r_squared = ols_result.rsquared
    adj_r_squared = ols_result.rsquared_adj

    poly_names = poly.get_feature_names_out(HOME_FEATURES).tolist()
    all_feature_names = poly_names + CITY_DUMMIES

    return {
        "model": model,
        "poly": poly,
        "feature_means": feature_medians,
        "lot_sqfts_sorted": lot_sqfts_sorted,
        "all_feature_names": all_feature_names,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "city_averages": city_averages,
        "n_listings": len(y),
    }


trained = train_model()


@app.route("/")
def index():
    return render_template(
        "index.html",
        feature_means=trained["feature_means"],
        city_averages=trained["city_averages"],
        r_squared=trained["r_squared"],
        adj_r_squared=trained["adj_r_squared"],
        n_listings=trained["n_listings"],
        prediction=None,
        form_data={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {}

    city = request.form.get("city", "Prince Albert")
    form_data["city"] = city

    numeric_values = {}
    for feat in NUMERIC_FEATURES:
        if feat == "lot_size_percentile":
            raw = request.form.get("lot_size_sqft", "")
            form_data["lot_size_sqft"] = raw
            if raw.strip() == "":
                sqft_val = trained["feature_means"].get("lot_size_sqft", 0)
            else:
                sqft_val = float(raw)
            arr = trained["lot_sqfts_sorted"]
            idx = np.searchsorted(arr, sqft_val)
            pct = (idx / len(arr)) * 100 if len(arr) > 0 else 50.0
            numeric_values[feat] = pct
        else:
            raw = request.form.get(feat, "")
            form_data[feat] = raw
            if raw.strip() == "":
                numeric_values[feat] = trained["feature_means"].get(feat, 0)
            else:
                numeric_values[feat] = float(raw)

    binary_values = {}
    for feat in BINARY_FEATURES:
        raw = request.form.get(feat, "")
        form_data[feat] = raw
        binary_values[feat] = 1 if raw == "1" else 0

    home_vector = [numeric_values[f] for f in NUMERIC_FEATURES] + \
                  [binary_values[f] for f in BINARY_FEATURES]
    home_array = np.array(home_vector).reshape(1, -1)

    X_poly = trained["poly"].transform(home_array)
    city_vector = [1 if city == "Saskatoon" else 0, 1 if city == "Regina" else 0]
    X_full = np.hstack([X_poly, np.array(city_vector).reshape(1, -1)])

    pred = trained["model"].predict(X_full)[0]
    pred = max(pred, 0)

    return render_template(
        "index.html",
        feature_means=trained["feature_means"],
        city_averages=trained["city_averages"],
        r_squared=trained["r_squared"],
        adj_r_squared=trained["adj_r_squared"],
        n_listings=trained["n_listings"],
        prediction=pred,
        form_data=form_data,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
