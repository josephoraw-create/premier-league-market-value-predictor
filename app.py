import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Premier League Market Value Predictor",
    page_icon="⚽",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(
        "models/market_value_model_compressed.pkl"
    )


model = load_model()


# --------------------------------------------------
# LOAD PLAYER DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    players = pd.read_csv("app_players.csv")

    players["date_of_birth"] = pd.to_datetime(
        players["date_of_birth"],
        errors="coerce"
    )

    return players


players = load_data()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚽ Premier League Player Market Value Predictor")

st.write(
    """
    Estimate a player's next Transfermarkt market value using their
    previous market value, player profile and recent Premier League
    performance.
    """
)

st.divider()


# --------------------------------------------------
# INPUT MODE
# --------------------------------------------------

input_mode = st.radio(
    "Choose input method",
    [
        "Select Real Player",
        "Enter Player Manually"
    ],
    horizontal=True
)


# --------------------------------------------------
# POSITION OPTIONS
# --------------------------------------------------

position_map = {
    "Goalkeeper": [
        "Goalkeeper"
    ],

    "Defender": [
        "Centre-Back",
        "Left-Back",
        "Right-Back"
    ],

    "Midfield": [
        "Defensive Midfield",
        "Central Midfield",
        "Attacking Midfield",
        "Left Midfield",
        "Right Midfield"
    ],

    "Attack": [
        "Left Winger",
        "Right Winger",
        "Centre-Forward",
        "Second Striker"
    ]
}


# --------------------------------------------------
# DEFAULT VALUES
# --------------------------------------------------

default_age = 25
default_position = "Attack"
default_sub_position = "Centre-Forward"
default_foot = "right"
default_height = 180
default_caps = 0
default_previous_value = 20_000_000

default_appearances = 20
default_goals = 5
default_assists = 3
default_minutes = 1500
default_yellow_cards = 2
default_red_cards = 0

selected_player_name = None
selected_club = None


# --------------------------------------------------
# REAL PLAYER MODE
# --------------------------------------------------

if input_mode == "Select Real Player":

    player_options = players.copy()

    player_options["display_name"] = (
        player_options["name"].fillna("Unknown")
        + " — "
        + player_options["current_club_name"].fillna(
            "Unknown Club"
        )
    )

    player_options = player_options.sort_values(
        "display_name"
    )

    selected_display = st.selectbox(
        "Select player",
        player_options["display_name"].tolist()
    )

    selected_player = player_options[
        player_options["display_name"]
        == selected_display
    ].iloc[0]

    selected_player_name = selected_player["name"]
    selected_club = selected_player["current_club_name"]

    # ----------------------------------------------
    # AGE
    # ----------------------------------------------

    if pd.notna(selected_player["date_of_birth"]):

        dob = selected_player["date_of_birth"]

        today = pd.Timestamp.today()

        calculated_age = int(
            (today - dob).days / 365.25
        )

        default_age = calculated_age

    # ----------------------------------------------
    # POSITION
    # ----------------------------------------------

    if pd.notna(selected_player["position"]):

        player_position = str(
            selected_player["position"]
        )

        if player_position in position_map:
            default_position = player_position

    if pd.notna(selected_player["sub_position"]):
        default_sub_position = str(
            selected_player["sub_position"]
        )

    # ----------------------------------------------
    # FOOT
    # ----------------------------------------------

    if pd.notna(selected_player["foot"]):
        default_foot = str(
            selected_player["foot"]
        ).lower()

    # ----------------------------------------------
    # HEIGHT
    # ----------------------------------------------

    if pd.notna(selected_player["height_in_cm"]):
        default_height = int(
            selected_player["height_in_cm"]
        )

    # ----------------------------------------------
    # INTERNATIONAL CAPS
    # ----------------------------------------------

    if pd.notna(selected_player["international_caps"]):
        default_caps = int(
            selected_player["international_caps"]
        )

    # ----------------------------------------------
    # MARKET VALUE
    # ----------------------------------------------

    if pd.notna(selected_player["market_value_in_eur"]):
        default_previous_value = int(
            selected_player["market_value_in_eur"]
        )

    # ----------------------------------------------
    # PRE-CALCULATED PERFORMANCE DATA
    # ----------------------------------------------

    default_appearances = int(
        selected_player["appearances"]
    )

    default_goals = int(
        selected_player["goals"]
    )

    default_assists = int(
        selected_player["assists"]
    )

    default_minutes = int(
        selected_player["minutes"]
    )

    default_yellow_cards = int(
        selected_player["yellow_cards"]
    )

    default_red_cards = int(
        selected_player["red_cards"]
    )


# --------------------------------------------------
# INPUT COLUMNS
# --------------------------------------------------

left_column, right_column = st.columns(2)


# --------------------------------------------------
# PLAYER PROFILE
# --------------------------------------------------

with left_column:

    st.subheader("👤 Player Profile")

    age = st.number_input(
        "Age",
        min_value=15,
        max_value=45,
        value=int(default_age)
    )

    position_options = list(
        position_map.keys()
    )

    try:
        position_index = position_options.index(
            default_position
        )
    except ValueError:
        position_index = 0

    position = st.selectbox(
        "Position",
        position_options,
        index=position_index
    )

    available_sub_positions = position_map[
        position
    ]

    if default_sub_position in available_sub_positions:

        sub_position_index = (
            available_sub_positions.index(
                default_sub_position
            )
        )

    else:
        sub_position_index = 0

    sub_position = st.selectbox(
        "Specific Position",
        available_sub_positions,
        index=sub_position_index
    )

    foot_options = [
        "right",
        "left",
        "both"
    ]

    if default_foot in foot_options:
        foot_index = foot_options.index(
            default_foot
        )
    else:
        foot_index = 0

    foot = st.selectbox(
        "Preferred Foot",
        foot_options,
        index=foot_index
    )

    height = st.number_input(
        "Height (cm)",
        min_value=150,
        max_value=220,
        value=int(default_height)
    )

    international_caps = st.number_input(
        "International Caps",
        min_value=0,
        max_value=250,
        value=int(default_caps)
    )

    previous_market_value = st.number_input(
        "Previous Market Value (€)",
        min_value=0,
        max_value=300_000_000,
        value=int(default_previous_value),
        step=500_000
    )


# --------------------------------------------------
# PERFORMANCE
# --------------------------------------------------

with right_column:

    st.subheader("📊 Recent Premier League Performance")

    appearances = st.number_input(
        "Appearances",
        min_value=0,
        max_value=60,
        value=int(default_appearances)
    )

    goals = st.number_input(
        "Goals",
        min_value=0,
        max_value=60,
        value=int(default_goals)
    )

    assists = st.number_input(
        "Assists",
        min_value=0,
        max_value=60,
        value=int(default_assists)
    )

    minutes = st.number_input(
        "Minutes Played",
        min_value=0,
        max_value=5000,
        value=int(default_minutes)
    )

    yellow_cards = st.number_input(
        "Yellow Cards",
        min_value=0,
        max_value=30,
        value=int(default_yellow_cards)
    )

    red_cards = st.number_input(
        "Red Cards",
        min_value=0,
        max_value=10,
        value=int(default_red_cards)
    )


st.divider()


# --------------------------------------------------
# DERIVED PERFORMANCE FEATURES
# --------------------------------------------------

if minutes > 0:

    goals_per_90 = (
        goals / minutes
    ) * 90

    assists_per_90 = (
        assists / minutes
    ) * 90

else:

    goals_per_90 = 0
    assists_per_90 = 0


if appearances > 0:

    minutes_per_appearance = (
        minutes / appearances
    )

else:

    minutes_per_appearance = 0


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "Predict Market Value",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame(
        {
            "age": [
                age
            ],

            "position": [
                position
            ],

            "sub_position": [
                sub_position
            ],

            "foot": [
                foot
            ],

            "height_in_cm": [
                height
            ],

            "international_caps": [
                international_caps
            ],

            "appearances": [
                appearances
            ],

            "goals": [
                goals
            ],

            "assists": [
                assists
            ],

            "minutes": [
                minutes
            ],

            "yellow_cards": [
                yellow_cards
            ],

            "red_cards": [
                red_cards
            ],

            "previous_market_value": [
                previous_market_value
            ],

            "goals_per_90": [
                goals_per_90
            ],

            "assists_per_90": [
                assists_per_90
            ],

            "minutes_per_appearance": [
                minutes_per_appearance
            ]
        }
    )

    # Model was trained using log market values
    log_prediction = model.predict(
        input_data
    )[0]

    predicted_value = np.expm1(
        log_prediction
    )

    predicted_value = max(
        predicted_value,
        0
    )

    value_change = (
        predicted_value
        - previous_market_value
    )

    if previous_market_value > 0:

        percentage_change = (
            value_change
            / previous_market_value
        ) * 100

    else:

        percentage_change = 0


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.subheader("💰 Prediction")

    if selected_player_name:

        st.write(
            f"### {selected_player_name}"
        )

        if pd.notna(selected_club):
            st.caption(
                selected_club
            )


    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        st.metric(
            "Previous Market Value",
            f"€{previous_market_value:,.0f}"
        )


    with result_col2:

        st.metric(
            "Predicted Market Value",
            f"€{predicted_value:,.0f}"
        )


    with result_col3:

        st.metric(
            "Estimated Change",
            f"€{value_change:,.0f}",
            f"{percentage_change:+.1f}%"
        )


    # --------------------------------------------------
    # COMPARISON CHART
    # --------------------------------------------------

    st.subheader("Market Value Comparison")

    chart_data = pd.DataFrame(
        {
            "Market Value (€)": [
                previous_market_value,
                predicted_value
            ]
        },
        index=[
            "Previous Value",
            "Predicted Value"
        ]
    )

    st.bar_chart(
        chart_data
    )


    # --------------------------------------------------
    # PERFORMANCE SUMMARY
    # --------------------------------------------------

    st.subheader(
        "Performance Summary"
    )

    stat1, stat2, stat3, stat4 = (
        st.columns(4)
    )


    stat1.metric(
        "Goals / 90",
        f"{goals_per_90:.2f}"
    )

    stat2.metric(
        "Assists / 90",
        f"{assists_per_90:.2f}"
    )

    stat3.metric(
        "Minutes / Appearance",
        f"{minutes_per_appearance:.0f}"
    )

    stat4.metric(
        "International Caps",
        international_caps
    )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

with st.expander(
    "🤖 About the Model"
):

    st.write(
        """
        The prediction model is a Random Forest regression
        model trained using historical Transfermarkt
        Premier League player valuations.

        The model predicts a player's next Transfermarkt
        market valuation using:

        - Previous market value
        - Age
        - Position
        - Preferred foot
        - Height
        - International experience
        - Premier League appearances
        - Goals
        - Assists
        - Minutes played
        - Disciplinary record
        - Goals per 90
        - Assists per 90
        - Minutes per appearance
        """
    )

    st.write("**Test-set performance:**")

    st.write(
        """
        - MAE: €2.43 million
        - RMSE: €4.18 million
        - R²: 0.962
        """
    )


with st.expander(
    "⚠️ Model Limitations"
):

    st.write(
        """
        This model predicts Transfermarkt market valuations,
        not actual transfer fees.

        Previous market value is the strongest predictor,
        meaning the model performs best when player values
        change gradually.

        Sudden rises or falls caused by breakout performances,
        injuries, transfers, contract situations or other
        external factors can be harder for the model to
        predict.

        Performance statistics used for real players are
        based on the Premier League data available when
        app_players.csv was generated.
        """
    )