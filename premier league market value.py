import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
import os

# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading data...")

players = pd.read_csv("data/players.csv")
valuations = pd.read_csv("data/player_valuations.csv")
appearances = pd.read_csv("data/appearances.csv")
clubs = pd.read_csv("data/clubs.csv")

print("Data loaded!")


# ==========================================
# 2. CONVERT DATES
# ==========================================

players["date_of_birth"] = pd.to_datetime(
    players["date_of_birth"],
    errors="coerce"
)

valuations["date"] = pd.to_datetime(
    valuations["date"],
    errors="coerce"
)

appearances["date"] = pd.to_datetime(
    appearances["date"],
    errors="coerce"
)


# ==========================================
# 3. PREMIER LEAGUE CLUBS
# ==========================================

premier_league_clubs = clubs[
    clubs["domestic_competition_id"] == "GB1"
].copy()

pl_club_names = set(
    premier_league_clubs["name"].dropna()
)

print(
    "\nNumber of Premier League clubs:",
    len(pl_club_names)
)


# ==========================================
# 4. PREMIER LEAGUE APPEARANCES
# ==========================================

pl_appearances = appearances[
    appearances["competition_id"] == "GB1"
].copy()

print(
    "Premier League appearances:",
    len(pl_appearances)
)


# ==========================================
# 5. PREMIER LEAGUE VALUATIONS
# ==========================================

historical_valuations = valuations[
    (valuations["current_club_name"].isin(pl_club_names)) &
    (valuations["date"] >= "2015-01-01")
].copy()

print(
    "Historical Premier League valuations:",
    len(historical_valuations)
)

historical_valuations = historical_valuations.rename(
    columns={
        "current_club_name": "valuation_club_name"
    }
)


# ==========================================
# 6. PLAYER INFORMATION
# ==========================================

player_info = players[
    [
        "player_id",
        "position",
        "sub_position",
        "foot",
        "height_in_cm",
        "date_of_birth",
        "international_caps"
    ]
].copy()

historical_valuations = historical_valuations.merge(
    player_info,
    on="player_id",
    how="left"
)


# ==========================================
# 7. AGE
# ==========================================

historical_valuations["age"] = (
    historical_valuations["date"]
    - historical_valuations["date_of_birth"]
).dt.days / 365.25


# ==========================================
# 8. HISTORICAL PERFORMANCE
# ==========================================

print("\nCalculating historical player statistics...")

stats = []

total_rows = len(historical_valuations)

for count, (_, row) in enumerate(
    historical_valuations.iterrows()
):

    if count % 5000 == 0:
        print(
            f"Processed {count} / {total_rows}"
        )

    player_id = row["player_id"]
    valuation_date = row["date"]

    start_date = (
        valuation_date -
        pd.Timedelta(days=365)
    )

    player_apps = pl_appearances[
        (pl_appearances["player_id"] == player_id) &
        (pl_appearances["date"] < valuation_date) &
        (pl_appearances["date"] >= start_date)
    ]

    stats.append({
        "player_id": player_id,
        "valuation_date": valuation_date,
        "appearances": len(player_apps),
        "goals": player_apps["goals"].sum(),
        "assists": player_apps["assists"].sum(),
        "minutes": player_apps["minutes_played"].sum(),
        "yellow_cards": player_apps["yellow_cards"].sum(),
        "red_cards": player_apps["red_cards"].sum()
    })


stats_df = pd.DataFrame(stats)


# ==========================================
# 9. MERGE PERFORMANCE
# ==========================================

historical_valuations = historical_valuations.merge(
    stats_df,
    left_on=["player_id", "date"],
    right_on=["player_id", "valuation_date"],
    how="left"
)

historical_valuations = historical_valuations.drop(
    columns=["valuation_date"],
    errors="ignore"
)


# ==========================================
# 10. FILL MISSING VALUES
# ==========================================

performance_columns = [
    "appearances",
    "goals",
    "assists",
    "minutes",
    "yellow_cards",
    "red_cards"
]

historical_valuations[performance_columns] = (
    historical_valuations[performance_columns]
    .fillna(0)
)


# ==========================================
# 11. SORT DATA
# ==========================================

historical_valuations = historical_valuations.sort_values(
    ["player_id", "date"]
).reset_index(drop=True)


# ==========================================
# 12. PREVIOUS MARKET VALUE
# ==========================================

# ==========================================
# 12. PREVIOUS MARKET VALUE
# ==========================================

print("\nCalculating previous market values from full career histories...")

# Create a copy containing ALL valuation records.
# This is important because a player's previous valuation
# may have been while playing outside the Premier League.
all_valuations = valuations[
    [
        "player_id",
        "date",
        "market_value_in_eur"
    ]
].copy()

# Remove rows where important information is missing
all_valuations = all_valuations.dropna(
    subset=[
        "player_id",
        "date",
        "market_value_in_eur"
    ]
)

# Sort each player's COMPLETE valuation history
all_valuations = all_valuations.sort_values(
    [
        "player_id",
        "date"
    ]
)

# Previous valuation from anywhere in the player's career
all_valuations["previous_market_value"] = (
    all_valuations
    .groupby("player_id")["market_value_in_eur"]
    .shift(1)
)

# Keep only the information needed for the merge
previous_values = all_valuations[
    [
        "player_id",
        "date",
        "previous_market_value"
    ]
].copy()


# Remove the old previous_market_value column
# if one already exists
historical_valuations = historical_valuations.drop(
    columns=["previous_market_value"],
    errors="ignore"
)


# Merge the correct previous value into our
# Premier League valuation dataset
historical_valuations = historical_valuations.merge(
    previous_values,
    on=[
        "player_id",
        "date"
    ],
    how="left"
)

print(
    "Previous market values calculated!"
)

# ==========================================
# 13. ADVANCED FEATURES
# ==========================================

historical_valuations["goals_per_90"] = np.where(
    historical_valuations["minutes"] > 0,
    historical_valuations["goals"]
    / historical_valuations["minutes"] * 90,
    0
)

historical_valuations["assists_per_90"] = np.where(
    historical_valuations["minutes"] > 0,
    historical_valuations["assists"]
    / historical_valuations["minutes"] * 90,
    0
)

historical_valuations["minutes_per_appearance"] = np.where(
    historical_valuations["appearances"] > 0,
    historical_valuations["minutes"]
    / historical_valuations["appearances"],
    0
)


# ==========================================
# 14. REMOVE FIRST VALUATION FOR EACH PLAYER
# ==========================================

model_data = historical_valuations.dropna(
    subset=["previous_market_value"]
).copy()


# ==========================================
# 15. TRAIN / TEST SPLIT
# ==========================================

train_data = model_data[
    model_data["date"] < "2025-01-01"
].copy()

test_data = model_data[
    model_data["date"] >= "2025-01-01"
].copy()

print("\n--------------------------------")
print("TRAIN / TEST SPLIT")
print("--------------------------------")

print(
    "Training records:",
    len(train_data)
)

print(
    "Testing records:",
    len(test_data)
)


# ==========================================
# 16. MODEL FUNCTION
# ==========================================

def train_and_evaluate(
    model_name,
    features
):

    print("\n")
    print("--------------------------------")
    print(model_name)
    print("--------------------------------")

    categorical_features = [
        feature
        for feature in features
        if feature in [
            "position",
            "sub_position",
            "foot"
        ]
    ]

    numeric_features = [
        feature
        for feature in features
        if feature not in categorical_features
    ]

    X_train = train_data[features]
    X_test = test_data[features]

    y_train = train_data[
        "market_value_in_eur"
    ]

    y_test = test_data[
        "market_value_in_eur"
    ]

    # Log transform target
    y_train_log = np.log1p(y_train)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numeric",
                "passthrough",
                numeric_features
            )
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("Training...")

    pipeline.fit(
        X_train,
        y_train_log
    )

    predictions_log = pipeline.predict(
        X_test
    )

    predictions = np.expm1(
        predictions_log
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(
        f"MAE:  €{mae:,.0f}"
    )

    print(
        f"RMSE: €{rmse:,.0f}"
    )

    print(
        f"R²:   {r2:.3f}"
    )

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ==========================================
# 17. MODEL 1 — PERFORMANCE ONLY
# ==========================================

basic_features = [
    "age",
    "position",
    "sub_position",
    "foot",
    "height_in_cm",
    "international_caps",
    "appearances",
    "goals",
    "assists",
    "minutes",
    "yellow_cards",
    "red_cards"
]

result_1 = train_and_evaluate(
    "MODEL 1 — Performance Only",
    basic_features
)


# ==========================================
# 18. MODEL 2 — + PREVIOUS VALUE
# ==========================================

previous_value_features = basic_features + [
    "previous_market_value"
]

result_2 = train_and_evaluate(
    "MODEL 2 — + Previous Market Value",
    previous_value_features
)


# ==========================================
# 19. MODEL 3 — ADVANCED MODEL
# ==========================================

advanced_features = previous_value_features + [
    "goals_per_90",
    "assists_per_90",
    "minutes_per_appearance"
]

result_3 = train_and_evaluate(
    "MODEL 3 — Advanced",
    advanced_features
)


# ==========================================
# 20. COMPARISON
# ==========================================

comparison = pd.DataFrame([
    result_1,
    result_2,
    result_3
])

print("\n")
print("--------------------------------")
print("MODEL COMPARISON")
print("--------------------------------")

print(
    comparison.to_string(
        index=False
    )
)

# ==========================================
# 21. NAIVE BASELINE
# ==========================================

print("\n")
print("--------------------------------")
print("NAIVE BASELINE — PREVIOUS VALUE")
print("--------------------------------")

actual_values = test_data["market_value_in_eur"]

# Simply assume the player's next value
# will be the same as their previous value
baseline_predictions = test_data[
    "previous_market_value"
]

baseline_mae = mean_absolute_error(
    actual_values,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    actual_values,
    baseline_predictions
)

print(
    f"MAE:  €{baseline_mae:,.0f}"
)

print(
    f"RMSE: €{baseline_rmse:,.0f}"
)

print(
    f"R²:   {baseline_r2:.3f}"
)


# ==========================================
# 22. TRAIN FINAL ADVANCED MODEL
# ==========================================

print("\n")
print("--------------------------------")
print("TRAINING FINAL ADVANCED MODEL")
print("--------------------------------")

features = advanced_features

categorical_features = [
    "position",
    "sub_position",
    "foot"
]

numeric_features = [
    feature
    for feature in features
    if feature not in categorical_features
]

X_train = train_data[features]
X_test = test_data[features]

y_train = train_data[
    "market_value_in_eur"
]

y_test = test_data[
    "market_value_in_eur"
]

y_train_log = np.log1p(y_train)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", final_model)
    ]
)


final_pipeline.fit(
    X_train,
    y_train_log
)


# ==========================================
# 23. FEATURE IMPORTANCE
# ==========================================

print("\n")
print("--------------------------------")
print("FEATURE IMPORTANCE")
print("--------------------------------")

# Get the trained Random Forest
rf_model = final_pipeline.named_steps["model"]

# Get feature names after one-hot encoding
feature_names = (
    final_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)

print(
    feature_importance.head(20).to_string(
        index=False
    )
)


# ==========================================
# 24. PLOT FEATURE IMPORTANCE
# ==========================================

import matplotlib.pyplot as plt

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 15 Features — Premier League Market Value Predictor"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nFeature importance graph saved as feature_importance.png"
)


# ==========================================
# 25. PERFORMANCE BY MARKET VALUE
# ==========================================

print("\n")
print("--------------------------------")
print("PERFORMANCE BY MARKET VALUE")
print("--------------------------------")


# Make predictions using the final model
test_predictions_log = final_pipeline.predict(
    X_test
)

test_predictions = np.expm1(
    test_predictions_log
)


# Create a results dataframe
evaluation = test_data[
    [
        "player_id",
        "date",
        "market_value_in_eur",
        "previous_market_value"
    ]
].copy()

evaluation["prediction"] = test_predictions


# ==========================================
# 26. CREATE VALUE BANDS
# ==========================================

def value_band(value):

    if value < 1_000_000:
        return "Under €1m"

    elif value < 10_000_000:
        return "€1m–€10m"

    elif value < 30_000_000:
        return "€10m–€30m"

    elif value < 50_000_000:
        return "€30m–€50m"

    else:
        return "Over €50m"


evaluation["value_band"] = (
    evaluation["market_value_in_eur"]
    .apply(value_band)
)


# ==========================================
# 27. CALCULATE ERROR BY BAND
# ==========================================

bands = [
    "Under €1m",
    "€1m–€10m",
    "€10m–€30m",
    "€30m–€50m",
    "Over €50m"
]

results = []

for band in bands:

    band_data = evaluation[
        evaluation["value_band"] == band
    ]

    if len(band_data) == 0:
        continue

    actual = band_data[
        "market_value_in_eur"
    ]

    predicted = band_data[
        "prediction"
    ]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # Mean percentage error
    percentage_error = (
        np.abs(actual - predicted)
        / actual
        * 100
    ).mean()

    results.append({
        "Value Band": band,
        "Players": len(band_data),
        "MAE": mae,
        "RMSE": rmse,
        "Mean % Error": percentage_error
    })


band_results = pd.DataFrame(results)


# ==========================================
# 28. DISPLAY RESULTS
# ==========================================

print(
    band_results.to_string(
        index=False,
        formatters={
            "MAE": "€{:,.0f}".format,
            "RMSE": "€{:,.0f}".format,
            "Mean % Error": "{:.1f}%".format
        }
    )
)


# ==========================================
# 29. SHOW SOME PREDICTIONS
# ==========================================

print("\n")
print("--------------------------------")
print("EXAMPLE PREDICTIONS")
print("--------------------------------")

examples = evaluation[
    [
        "market_value_in_eur",
        "prediction",
        "previous_market_value",
        "value_band"
    ]
].copy()

examples = examples.head(20)

examples["market_value_in_eur"] = (
    examples["market_value_in_eur"]
    .apply(lambda x: f"€{x:,.0f}")
)

examples["prediction"] = (
    examples["prediction"]
    .apply(lambda x: f"€{x:,.0f}")
)

examples["previous_market_value"] = (
    examples["previous_market_value"]
    .apply(lambda x: f"€{x:,.0f}")
)

print(
    examples.to_string(index=False)
)

# ==========================================
# 30. ADD PLAYER NAMES TO EVALUATION
# ==========================================

print("\n")
print("--------------------------------")
print("PLAYER-LEVEL ANALYSIS")
print("--------------------------------")

player_names = players[
    ["player_id", "name"]
].drop_duplicates(
    subset=["player_id"]
)

evaluation = evaluation.merge(
    player_names,
    on="player_id",
    how="left"
)


# ==========================================
# 31. CALCULATE PREDICTION ERRORS
# ==========================================

evaluation["error"] = (
    evaluation["prediction"]
    - evaluation["market_value_in_eur"]
)

evaluation["absolute_error"] = (
    evaluation["error"].abs()
)

evaluation["percentage_error"] = (
    evaluation["absolute_error"]
    / evaluation["market_value_in_eur"]
    * 100
)

evaluation["value_change"] = (
    evaluation["market_value_in_eur"]
    - evaluation["previous_market_value"]
)


# ==========================================
# 32. HELPER FUNCTION
# ==========================================

def display_players(data, number=10):

    display = data[
        [
            "name",
            "date",
            "previous_market_value",
            "market_value_in_eur",
            "prediction",
            "absolute_error",
            "value_change"
        ]
    ].head(number).copy()

    money_columns = [
        "previous_market_value",
        "market_value_in_eur",
        "prediction",
        "absolute_error",
        "value_change"
    ]

    for column in money_columns:

        display[column] = display[column].apply(
            lambda x: f"€{x:,.0f}"
        )

    print(
        display.to_string(
            index=False
        )
    )


# ==========================================
# 33. BIGGEST MODEL ERRORS
# ==========================================

print("\n")
print("--------------------------------")
print("10 BIGGEST PREDICTION ERRORS")
print("--------------------------------")

biggest_errors = evaluation.sort_values(
    "absolute_error",
    ascending=False
)

display_players(
    biggest_errors
)


# ==========================================
# 34. MOST ACCURATE PREDICTIONS
# ==========================================

print("\n")
print("--------------------------------")
print("10 MOST ACCURATE PREDICTIONS")
print("--------------------------------")

# Ignore extremely cheap players here so tiny
# absolute errors don't dominate the results
accurate_predictions = evaluation[
    evaluation["market_value_in_eur"]
    >= 5_000_000
].sort_values(
    "percentage_error"
)

display_players(
    accurate_predictions
)


# ==========================================
# 35. BIGGEST MARKET VALUE INCREASES
# ==========================================

print("\n")
print("--------------------------------")
print("10 BIGGEST MARKET VALUE INCREASES")
print("--------------------------------")

biggest_increases = evaluation.sort_values(
    "value_change",
    ascending=False
)

display_players(
    biggest_increases
)


# ==========================================
# 36. BIGGEST MARKET VALUE FALLS
# ==========================================

print("\n")
print("--------------------------------")
print("10 BIGGEST MARKET VALUE FALLS")
print("--------------------------------")

biggest_falls = evaluation.sort_values(
    "value_change",
    ascending=True
)

display_players(
    biggest_falls
)


# ==========================================
# 37. BIGGEST MODEL UNDERESTIMATES
# ==========================================

print("\n")
print("--------------------------------")
print("10 BIGGEST UNDERESTIMATES")
print("--------------------------------")

# Negative error means:
# prediction < actual value

underestimates = evaluation.sort_values(
    "error",
    ascending=True
)

display_players(
    underestimates
)


# ==========================================
# 38. BIGGEST MODEL OVERESTIMATES
# ==========================================

print("\n")
print("--------------------------------")
print("10 BIGGEST OVERESTIMATES")
print("--------------------------------")

# Positive error means:
# prediction > actual value

overestimates = evaluation.sort_values(
    "error",
    ascending=False
)

display_players(
    overestimates
)

# ==========================================
# 39. SAVE FINAL MODEL
# ==========================================

print("\n")
print("--------------------------------")
print("SAVING FINAL MODEL")
print("--------------------------------")

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the entire pipeline
joblib.dump(
    final_pipeline,
    "models/market_value_model.pkl"
)

print(
    "Model saved successfully:"
    " models/market_value_model.pkl"
)