# 🌦️ SkyCast — Kathmandu Rain Predictor

Given a set of weather conditions — humidity, temperature, pressure, cloud cover, wind,
solar radiation — this project classifies whether those conditions look like **Rain** or
**No Rain**, based on patterns learned from 10 years of Kathmandu weather data
(2015–2025). It ships with an interactive Streamlit app for trying your own "what-if"
conditions, plus a separate 2026 seasonal outlook based on historical monthly patterns.

Built as two model versions to test — and fix — a data leakage issue found along the way.

> **How to read this:** the model isn't given a future date and asked to predict its
> weather — it's given a set of weather *conditions* (real or assumed) and classifies
> whether that combination matches the profile of a rainy day or a dry day historically.
> See [How prediction actually works](#how-prediction-actually-works) below.

---

## Project Objective

Build a full ML workflow — cleaning, exploratory analysis, feature engineering, model
comparison, and evaluation — to classify days as **Rain** or **No Rain** based on weather
conditions, and to understand which weather variables actually drive rainfall in the
Kathmandu Valley.

## Demo App

The repo includes `demo.py`, a Streamlit app ("SkyCast") with two views:

- **Rain Check** — set a combination of weather conditions (temperature, humidity,
  pressure, cloud cover, wind, solar radiation, UV index) and the trained Random
  Forest model classifies that combination as Rain or No Rain, with a probability
  gauge and feature importance chart. This is a **what-if classifier**, not a
  forecast — see below.
- **2026 Seasonal Outlook** — a SARIMA time-series model trained on 10 years of
  monthly rain frequency, projecting month-by-month rain likelihood through 2026
  with an 80% confidence band. This one *is* purely calendar-based.

## How prediction actually works

It's worth being precise about what "predicts rain" means here, because there are two
different questions being answered:

**1. Classifying known conditions (what the Rain Check tab does).**
The model takes a set of weather readings — humidity, cloud cover, pressure, wind,
solar radiation, etc. — and classifies whether that specific combination looks like a
rainy day or a dry day, based on patterns in 10 years of historical data. It does
**not** need a date to do this; the date fields just add seasonal context. You can
enter today's actual readings, a future forecast's readings, or entirely hypothetical
numbers, and the model will classify whichever combination you give it.

**2. Predicting an actual future date's weather (what it does *not* do).**
The model has no way to know what tomorrow's humidity or cloud cover will actually
be — those values don't exist yet. So it can't independently forecast a specific
future day. To use it for a real upcoming date, you'd need to plug in numbers from an
actual weather forecast; the model then tells you whether *that forecast* looks rain-
prone historically.

**3. The Seasonal Outlook tab is the one exception.**
It takes only the calendar month as input and uses a SARIMA time-series model fit on
10 years of monthly rain frequency to project a rain-likelihood range for each month
of 2026. It doesn't need weather readings at all, but in exchange it can only speak to
*typical* monthly patterns, not any individual day.

### Running the app

```bash
pip install -r requirements.txt
streamlit run demo.py
```

Make sure `rain_prediction.pkl` and `cleaned_dataset.csv` are in the same folder as
`demo.py` — the app loads both directly.

> **Note:** the Rain Check tab classifies a set of conditions you provide — it can't
> forecast future weather on its own. For a genuine forward-looking view, use the
> Seasonal Outlook tab, which is calendar-based rather than condition-based.

## Dataset

Source: 10 years of daily weather data for Kathmandu (2015–2025), including:

- Temperature (max, min, average)
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

## Workflow

```
Data Cleaning
↓
Unit Conversion (Fahrenheit → Celsius)
↓
Date Feature Extraction (year, month)
↓
Exploratory Data Analysis
↓
Target Encoding
↓
Train-Test Split
↓
Feature Scaling
↓
Model Comparison using Cross-Validation
↓
Test Set Evaluation
↓
Feature Importance Analysis
↓
Model Persistence (joblib) → Streamlit App
```

## Data Cleaning

- Dropped columns that were mostly empty or not useful for prediction:
  `severerisk`, `preciptype`, `solarenergy`, `precipcover`, `winddir`, `sunrise`,
  `sunset`, `dew`.
- Converted `tempmax`, `tempmin`, and `temp` from Fahrenheit to Celsius and renamed
  them to `max_temp_C`, `min_temp_C`, `avg_temp_c`.
- Extracted `year` and `month` from the `datetime` column and moved them to the front
  of the dataframe; dropped the original `datetime` column afterward.
- Converted `month` to an ordered categorical type for correct chronological plotting,
  then back to numeric (1–12) before modeling.

## Exploratory Data Analysis

EDA covered both univariate and relationship-level questions:

- Average, highest, and lowest temperature by year and by month
- Monthly temperature composition (stacked bar chart)
- Correlation heatmap between temperature, humidity, solar radiation, cloud cover,
  and wind speed
- Scatter plots of average temperature vs. humidity, solar radiation, and cloud cover
- Multi-panel bar charts comparing average temperature, humidity, and solar radiation
  across months

## Model Training

Four classifiers were compared using 6-fold cross-validation:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

Feature scaling (`StandardScaler`) was fit on the training set only, then applied to
the test set — no leakage from scaling.

## Two Versions — Why

### Model 1 (initial version)

Included `precipprob` (precipitation probability) as an input feature.

```
Random Forest — Test Accuracy: 0.8689 | F1: 0.8312 | ROC-AUC: 0.9288
```

On inspection, `precipprob` correlates strongly with the actual rain outcome
(occurrence-level correlation ≈ 0.63, with a large gap between rain-day and
non-rain-day averages). Since this value is itself a forecast-style estimate of rain
likelihood, using it as an input feature lets the model lean on a value that already
encodes the answer, rather than learning from independent weather conditions.

### Model 2 (final version)

`precipprob` removed. Predicts rain using only independent weather features:
humidity, temperature, pressure, cloud cover, wind speed, visibility, solar
radiation, and UV index.

```
Random Forest — Test Accuracy: 0.8300 | F1: 0.7704 | ROC-AUC: 0.9028
```

Performance drops moderately once the leakage-prone feature is removed, which is
expected and is treated as the more honest, generalizable result.
**Model 2 is the final model, and is what powers the Streamlit app.**

## Model Comparison (Model 2, final feature set)

| Model               | Accuracy   | F1-Score   | ROC-AUC    |
| ------------------- | ---------- | ---------- | ---------- |
| **Random Forest**   | **0.8300** | **0.7704** | **0.9028** |
| KNN                 | 0.8386     | 0.7627     | 0.8855     |
| Logistic Regression | 0.8112     | 0.7533     | 0.8853     |
| Decision Tree       | 0.7680     | 0.6979     | 0.7616     |

Random Forest was selected as the final model based on the best balance of F1-score
and ROC-AUC, even though KNN shows a marginally higher raw accuracy — ROC-AUC and F1
better reflect how well a model separates rain from non-rain days.

## Feature Importance (Model 2)

With `precipprob` removed, **humidity** became the strongest predictor, followed by
minimum/average temperature, sea level pressure, and cloud cover:

| Feature          | Importance |
| ---------------- | ---------- |
| humidity         | 0.206      |
| min_temp_C       | 0.108      |
| avg_temp_c       | 0.103      |
| sealevelpressure | 0.103      |
| cloudcover       | 0.098      |
| solarradiation   | 0.075      |

This is a meaningful result: it shows the model is learning from real, physically
sensible weather relationships (moisture, temperature, pressure, cloud formation)
rather than depending on one dominant, forecast-derived feature.

## Known Limitations

- **No hyperparameter tuning was performed.** All models use default or lightly-set
  parameters. A `GridSearchCV`/`RandomizedSearchCV` pass on Random Forest would
  likely improve on the current numbers.
- **The seasonal forecast in the notebooks is incomplete** — the month-by-month rain
  probability loop in `model2.ipynb` does not currently produce output. The working
  version of this (using SARIMA) lives in `demo.py` instead; folding it back into the
  notebook is a natural next step.
- Class balance is moderate but not extreme (roughly 64% No Rain / 36% Rain in the
  test set) — worth noting when interpreting accuracy alongside F1/ROC-AUC.
- The seasonal outlook reflects historical climate patterns, not year-specific
  anomalies (e.g. an El Niño year), and cannot predict any individual day's weather.

## Files

```
kathmandu_weather_2015 to 2025.csv   — raw dataset
cleaned_dataset.csv                  — cleaned dataset, used by the app
model1.ipynb                         — initial model, includes precipprob
model2.ipynb                         — final model, precipprob removed
rain_prediction.pkl                  — trained Model 2 pipeline (joblib), used by the app
demo.py                              — Streamlit app: live prediction + seasonal outlook
requirements.txt                     — Python dependencies for demo.py
```

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Statsmodels (SARIMA)
- Streamlit, Plotly

## Conclusion

This project's most useful lesson wasn't the final accuracy number — it was catching
that Model 1's strong performance was partly inflated by a feature that too closely
mirrored the target, and rebuilding the model without it. Model 2's more modest but
more honest result (0.83 accuracy, 0.90 ROC-AUC using only independent weather
variables) is the one worth standing behind, and the feature importance results back
it up with physically sensible predictors rather than a single dominant shortcut.
