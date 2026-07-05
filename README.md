# 🌦️ SkyCast — Kathmandu Rain Predictor

Given a set of weather conditions — humidity, temperature, pressure, cloud cover, wind,
solar radiation — this project classifies whether those conditions look like **Rain** or
**No Rain**, based on patterns learned from 10 years of Kathmandu weather data
(2015–2025). It also produces a 2026 monthly rain outlook using two different
forecasting approaches, and ships with an interactive Streamlit app for trying your
own "what-if" conditions.

The project is split across two notebooks and one app:

| File | Role |
|---|---|
| `model1.ipynb` | Cleans the raw data, runs EDA, trains/compares 4 classifiers, and produces a 2026 monthly outlook using **bootstrap resampling**. |
| `model2.ipynb` | Reloads the cleaned data, retrains the same 4 classifiers, produces a 2026 monthly outlook using a **SARIMA time-series model**, and saves the final Random Forest pipeline (`rain_prediction.pkl`). |
| `demo.py` | A Streamlit app ("SkyCast") that loads `rain_prediction.pkl` for live what-if classification, and re-runs the SARIMA model for the seasonal outlook tab. |

> **How to read this:** the classifier isn't given a future date and asked to predict
> its weather — it's given a set of weather *conditions* (real or assumed) and
> classifies whether that combination matches the profile of a rainy day or a dry day
> historically. The two notebooks' 2026 outlooks are a separate thing: they forecast
> from the *calendar* alone, using two different statistical techniques. See
> [How prediction actually works](#how-prediction-actually-works) below.

## Dataset

Source: 10 years of daily weather data for Kathmandu (2015–2025), 3,468 rows, including:

- Temperature (max, min, average — original data is in Fahrenheit, converted to Celsius)
- Humidity, dew point
- Precipitation and precipitation probability
- Wind speed, wind gust, wind direction
- Sea level pressure, cloud cover, visibility
- Solar radiation, solar energy, UV index
- Sunrise / sunset times

Target definition:

```
rain = 1  if precipitation > 0
rain = 0  otherwise
```

Class balance in the cleaned data: 2,107 No Rain / 1,361 Rain (≈61% / 39%).

## `model1.ipynb` — Cleaning, EDA, model comparison, bootstrap outlook

**Data cleaning**
- Drops sparse/unneeded columns: `severerisk`, `preciptype`, `solarenergy`,
  `precipcover`, `winddir`, `sunrise`, `sunset`, `dew`.
- Converts `tempmax`, `tempmin`, `temp` from Fahrenheit to Celsius (`(x − 32) × 5/9`)
  and renames them to `max_temp_C`, `min_temp_C`, `avg_temp_c`.
- Extracts `year`, `month`, `day` from `datetime` and drops the original column.
- Builds the `rain` target and saves the result as `cleaned_dataset.csv`
  (3,468 rows × 17 columns).

**EDA** — average/highest/lowest temperature by year and month, monthly temperature
composition, a correlation heatmap (temperature, humidity, solar radiation, cloud
cover, wind speed), and scatter plots of temperature against humidity and solar
radiation.

**Model comparison** — four classifiers trained on a feature set that **excludes
`precipprob`, `precipitation`, and `rain`** (14 features: `year`, `month`, `day`,
`max_temp_C`, `min_temp_C`, `avg_temp_c`, `humidity`, `windgust`, `windspeed`,
`sealevelpressure`, `cloudcover`, `visibility`, `solarradiation`, `uvindex`), evaluated
with 6-fold stratified cross-validation and then on a held-out 20% test set
(694 rows, plain random split):

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Random Forest** | **0.8415** | **0.7852** | **0.9086** |
| KNN | 0.8415 | 0.7718 | 0.8933 |
| Logistic Regression | 0.8127 | 0.7547 | 0.8846 |
| Decision Tree | 0.7594 | 0.6819 | 0.7488 |

Random Forest is selected as the best model — tied with KNN on accuracy but clearly
ahead on F1 and ROC-AUC.

Feature importance (top 6): humidity (0.199), avg_temp_c (0.101), min_temp_C (0.101),
sealevelpressure (0.100), cloudcover (0.096), solarradiation (0.069).

**Bootstrap method (2026 monthly outlook)** — for each calendar month, 1,000 historical
rows from that same month (across all years) are resampled *with replacement*, `year`
is set to 2026, and the trained Random Forest predicts a rain probability for each
resampled row. The mean of those probabilities becomes the month's forecast, with a
95% interval taken from the 2.5th/97.5th percentiles. This produces a full working
12-month table, e.g.:

| Month | Rain Probability | 95% Range |
|---|---|---|
| January | 12.9% | 0–72% |
| May | 52.7% | 4–96% |
| July | 86.1% | 34–99% |
| September | 69.0% | 18–96% |

## `model2.ipynb` — Same features, SARIMA outlook, saved model

Reloads `cleaned_dataset.csv` and retrains the same four classifiers on the same
14-feature set (`precipprob` still excluded), this time with a **stratified** 80/20
split:

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Random Forest** | **0.8372** | **0.7839** | **0.8956** |
| Logistic Regression | 0.8184 | 0.7667 | 0.8677 |
| KNN | 0.8127 | 0.7379 | 0.8736 |
| Decision Tree | 0.7666 | 0.7011 | 0.7545 |

Feature importance (top 6): humidity (0.194), cloudcover (0.107), avg_temp_c (0.105),
sealevelpressure (0.098), min_temp_C (0.092), solarradiation (0.074).

Includes a worked example: a hypothetical day's conditions (humidity 83%, avg temp
22.8°C, etc.) classified as **Rain**, P(Rain) = 0.690.

**SARIMA method (2026 monthly outlook)** — builds a monthly time series of the
percentage of rainy days per month across 2015–2025, logit-transforms it (so the
forecast can't leave the 0–100% range), fits a seasonal ARIMA `(1,1,1)(1,1,1,12)` with
`statsmodels`, and forecasts forward through December 2026 with an 80% confidence
interval:

| Month | Predicted Rainy Days | 80% CI |
|---|---|---|
| January | 7.6% | 1.5–30.3% |
| May | 57.9% | 19.0–88.9% |
| July | 95.9% | 77.7–99.4% |
| September | 72.1% | 26.2–95.0% |

**Model persistence** — the trained Random Forest pipeline is saved with
`joblib.dump(best_model, 'rain_prediction.pkl')`. This is the exact file the Streamlit
app loads.

## `demo.py` — Streamlit app

Two tabs:

- **Rain Check (your conditions)** — set humidity, temperature, pressure, cloud cover,
  wind, and solar values yourself; `rain_prediction.pkl` (Model 2's Random Forest)
  classifies that specific combination as Rain or No Rain, with a probability gauge
  and a live feature-importance chart. This is a what-if classifier — see below.
- **2026 Seasonal Outlook** — re-runs the SARIMA approach from `model2.ipynb` live
  against `cleaned_dataset.csv`, plotting the monthly forecast with its confidence
  band. (The app does not use model1's bootstrap method — only the SARIMA one.)

### Running the app

```bash
pip install -r requirements.txt
streamlit run demo.py
```

`rain_prediction.pkl` and `cleaned_dataset.csv` must be in the same folder as
`demo.py` — both are loaded directly, and `cleaned_dataset.csv` is only produced once
`model1.ipynb` has been run.

## How prediction actually works

**1. Classifying a set of conditions (Rain Check tab / the Random Forest models).**
The model takes weather readings — humidity, cloud cover, pressure, wind, solar
radiation, etc. — and classifies whether that specific combination looks like a rainy
day or a dry day, based on patterns in 10 years of historical data. It does not need
those readings to belong to any real day; you can enter today's actual numbers, a
forecast's numbers, or hypothetical ones.

**2. Predicting an actual future date's weather.** The model can't do this on its
own — it has no way to know tomorrow's humidity or cloud cover, since those values
don't exist yet. To check a real upcoming day, you'd need to feed it numbers from an
actual weather forecast.

**3. The calendar-only monthly outlooks (bootstrap in model1, SARIMA in model2) are
the exception.** Both take only the month (or run the trained model on resampled
historical readings from that month) and produce a rain-likelihood range for 2026.
They don't need any weather readings as input, but in exchange they can only speak to
*typical* monthly patterns, not any individual day. The two methods disagree somewhat
in places (e.g. January: 12.9% bootstrap vs. 7.6% SARIMA) — a useful reminder that
"forecasting the future from 10 years of the past" has real uncertainty baked in
regardless of method.

## Known Limitations

- **No hyperparameter tuning.** All models use default or lightly-set parameters.
  A `GridSearchCV`/`RandomizedSearchCV` pass on Random Forest would likely help.
- **model1 and model2 use different train/test splits** (plain vs. stratified), so
  their reported metrics aren't perfectly apples-to-apples, though both land Random
  Forest at roughly 0.84 accuracy / 0.90 ROC-AUC.
- **The Fahrenheit→Celsius conversion cell in `model1.ipynb` is mislabeled** as
  "convert temperature from k to C" — the formula used is correct (F→C), but the
  comment says Kelvin.
- **The bootstrap and SARIMA monthly outlooks disagree in places** (see above) —
  neither should be read as a precise forecast, only as a directional seasonal signal.
- **The Streamlit app only exposes the SARIMA outlook**, not model1's bootstrap
  method — someone comparing the two has to run `model1.ipynb` directly.

## Technologies Used

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Decision Tree, Random Forest, KNN)
- Statsmodels (SARIMAX)
- Streamlit, Plotly, joblib

## Conclusion

Both notebooks land on Random Forest as the best classifier (≈0.84 accuracy, ≈0.90
ROC-AUC) using the same 14 weather features, with humidity consistently the strongest
predictor. Where they diverge is in how they extend that model into a 2026 outlook:
model1 resamples historical same-month data through the classifier (bootstrap), while
model2 fits a dedicated seasonal time-series model (SARIMA) directly on the monthly
rain-day percentage. The deployed app uses model2's Random Forest for live
classification and its SARIMA model for the seasonal view.
