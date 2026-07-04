# Kathmandu Rain Prediction

Predicts whether it will rain on a given day in Kathmandu using 10 years of historical
weather data (2015–2025). Built as two versions to test — and fix — a data leakage issue
found along the way.

## Project Objective

Build a full ML workflow — cleaning, exploratory analysis, feature engineering, model
comparison, and evaluation — to classify days as **Rain** or **No Rain** based on weather
conditions, and to understand which weather variables actually drive rainfall in the
Kathmandu Valley.

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
```text
rain = 1  if precipitation > 0
rain = 0  otherwise
```

## Workflow

```text
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
```

## Data Cleaning

- Dropped columns that were mostly empty or not useful for prediction:
  `severerisk`, `preciptype`, `solarenergy`, `precipcover`, `winddir`, `sunrise`,
  `sunset`, `dew`.
- Converted `tempmax`, `tempmin`, and `temp` from Fahrenheit to Celsius and renamed them
  to `max_temp_C`, `min_temp_C`, `avg_temp_c`.
- Extracted `year` and `month` from the `datetime` column and moved them to the front of
  the dataframe; dropped the original `datetime` column afterward.
- Converted `month` to an ordered categorical type for correct chronological plotting,
  then back to numeric (1–12) before modeling.

## Exploratory Data Analysis

EDA covered both univariate and relationship-level questions:

- Average, highest, and lowest temperature by year and by month
- Monthly temperature composition (stacked bar chart)
- Correlation heatmap between temperature, humidity, solar radiation, cloud cover, and
  wind speed
- Scatter plots of average temperature vs. humidity, solar radiation, and cloud cover
- Multi-panel bar charts comparing average temperature, humidity, and solar radiation
  across months

## Model Training

Four classifiers were compared using 6-fold cross-validation:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

Feature scaling (`StandardScaler`) was fit on the training set only, then applied to the
test set — no leakage from scaling.

## Two Versions — Why

### Model 1 (initial version)

Included `precipprob` (precipitation probability) as an input feature.

```text
Random Forest — Test Accuracy: 0.8689 | F1: 0.8312 | ROC-AUC: 0.9288
```

On inspection, `precipprob` correlates strongly with the actual rain outcome
(occurrence-level correlation ≈ 0.63, with a large gap between rain-day and non-rain-day
averages). Since this value is itself a forecast-style estimate of rain likelihood, using
it as an input feature lets the model lean on a value that already encodes the answer,
rather than learning from independent weather conditions.

### Model 2 (final version)

`precipprob` removed. Predicts rain using only independent weather features: humidity,
temperature, pressure, cloud cover, wind speed, visibility, solar radiation, and UV index.

```text
Random Forest — Test Accuracy: 0.8300 | F1: 0.7704 | ROC-AUC: 0.9028
```

Performance drops moderately once the leakage-prone feature is removed, which is expected
and is treated as the more honest, generalizable result. **Model 2 is the final model.**

## Model Comparison (Model 2, final feature set)

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Random Forest** | **0.8300** | **0.7704** | **0.9028** |
| KNN | 0.8386 | 0.7627 | 0.8855 |
| Logistic Regression | 0.8112 | 0.7533 | 0.8853 |
| Decision Tree | 0.7680 | 0.6979 | 0.7616 |

Random Forest was selected as the final model based on the best balance of F1-score and
ROC-AUC, even though KNN shows a marginally higher raw accuracy — ROC-AUC and F1 better
reflect how well a model separates rain from non-rain days.

## Feature Importance (Model 2)

With `precipprob` removed, **humidity** became the strongest predictor, followed by
minimum/average temperature, sea level pressure, and cloud cover:

| Feature | Importance |
|---|---|
| humidity | 0.206 |
| min_temp_C | 0.108 |
| avg_temp_c | 0.103 |
| sealevelpressure | 0.103 |
| cloudcover | 0.098 |
| solarradiation | 0.075 |

This is a meaningful result: it shows the model is learning from real, physically
sensible weather relationships (moisture, temperature, pressure, cloud formation) rather
than depending on one dominant, forecast-derived feature.

## Known Limitations

- **No hyperparameter tuning was performed.** All models use default or lightly-set
  parameters. A `GridSearchCV`/`RandomizedSearchCV` pass on Random Forest would likely
  improve on the current numbers.
- **The 2026 monthly forecast feature is incomplete.** An early attempt to generate
  month-by-month rain probability estimates for 2026 by averaging historical monthly
  values is present in both notebooks but was not finished — the loop does not currently
  produce output. This is a natural next step, not a finished feature.
- **No saved model file.** The trained pipeline is not currently persisted (e.g. via
  `joblib`), so predictions require re-running the notebook.
- Class balance is moderate but not extreme (roughly 64% No Rain / 36% Rain in the test
  set) — worth noting when interpreting accuracy alongside F1/ROC-AUC.

## Files

```text
kathmandu_weather_2015 to 2025.csv   — raw dataset
kathmandu_weather_cleaned            — cleaned dataset (CSV format)
model1.ipynb                         — initial model, includes precipprob
model2.ipynb                         — final model, precipprob removed
linechart.png                        — saved plot: lowest average temperature by month
```

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## Conclusion

This project's most useful lesson wasn't the final accuracy number — it was catching that
Model 1's strong performance was partly inflated by a feature that too closely mirrored
the target, and rebuilding the model without it. Model 2's more modest but more honest
result (0.83 accuracy, 0.90 ROC-AUC using only independent weather variables) is the one
worth standing behind, and the feature importance results back it up with physically
sensible predictors rather than a single dominant shortcut.
