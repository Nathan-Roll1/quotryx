# Fair Value Prediction Model — Design

## Overview

A web app that predicts the "fair value" of a home in Saskatchewan using a degree 2 OLS regression trained on ~1,153 MLS listings across Saskatoon, Regina, and Prince Albert.

## Data

- Source: `quotryx_combined_20260225_040559.csv` (1,153 rows, 42 columns)
- Cities: Saskatoon (503), Regina (503), Prince Albert (146)
- Target: `price_numeric`

## Features

### Numeric (6)

| Feature | Source Column | Parsing |
|---|---|---|
| bedrooms | bedrooms | integer |
| bathrooms | bathrooms | integer |
| size_interior_sqft | size_interior | strip "sqft", float |
| lot_size_sqft | lot_size | convert acres to sqft (1 ac = 43,560 sqft), strip "sqft" |
| year_built | year_built | integer |
| parking_spaces | parking_spaces | integer |

### Binary Flags (6)

| Flag | Derived From | Logic |
|---|---|---|
| is_house | building_type | == "House" |
| is_condo | ownership_type | contains "Condominium" |
| has_garage | parking_type | contains "Garage" |
| has_basement | basement_type | non-empty |
| basement_finished | basement_type | contains "Finished" |
| has_cooling | cooling_type | non-empty |

### Municipality Dummies (2)

- city_Saskatoon, city_Regina (Prince Albert is reference category)

## Missing Value Strategy

- Numeric features: fill with column mean from training data
- Binary flags: default to 0 when source column is blank

## Model

- Degree 2 polynomial expansion on 12 home features (squares + pairwise interactions)
- Municipality dummies enter as linear terms only (intercept shifts, not polynomial-expanded)
- OLS regression fit on the full feature matrix
- ~81 total coefficients, comfortable ratio to 1,153 rows

## Architecture

- Flask app with Jinja2 templates + Tailwind CSS (CDN)
- Model trains on startup from the bundled CSV (<1s)
- Deployed on Railway via gunicorn

### File Structure

```
quotryx_3/
├── app.py                              # Flask app, model training, routes
├── templates/
│   └── index.html                      # Form + results, Tailwind styled
├── requirements.txt                    # flask, pandas, scikit-learn, statsmodels, gunicorn
├── Procfile                            # web: gunicorn app:app
└── quotryx_combined_20260225_040559.csv
```

### Routes

- `GET /` — form with inputs for all features, municipality dropdown
- `POST /predict` — imputes blanks with stored means, runs polynomial pipeline, returns predicted fair value

### UI

1. Select municipality (Saskatoon / Regina / Prince Albert)
2. Fill in known features — leave unknown ones blank
3. Toggle binary flags
4. Submit → see predicted price, model R², municipality average for context
