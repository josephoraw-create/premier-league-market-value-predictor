import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("models/market_value_model_compressed.pkl")


model = load_model()


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    players = pd.read_csv("data/players.csv")

    appearances = pd.read_csv(
        "data/appearances.csv"
    )

    players["date_of_birth"] = pd.to_datetime(
        players["date_of_birth"],
        errors="coerce"
    )

    appearances["date"] = pd.to_datetime(
        appearances["date"],
        errors="coerce"
    )

    return players, appearances


players, appearances = load_data()


# ==========================================
# PAGE SETUP
# ==========================================

st.set_page_config(
    page_title="Premier League Market Value Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title(
    "⚽ Premier League Market Value Predictor"
)

st.write(
    "Estimate a player's next Transfermarkt market value "
    "using their previous valuation and recent "
    "Premier League performance."
)

st.divider()


# ==========================================
# PLAYER SELECTION
# ==========================================

st.subheader("🔎 Player Selection")

selection_mode = st.radio(
    "Input Method",
    [
        "Select Real Player",
        "Enter Player Manually"
    ],
    horizontal=True
)


selected_player = None

if selection_mode == "Select Real Player":

    player_names = (
        players["name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    selected_name = st.selectbox(
        "Search / Select Player",
        player_names
    )

    selected_rows = players[
        players["name"] == selected_name
    ]

    if not selected_rows.empty:
        selected_player = selected_rows.iloc[0]


# ==========================================
# DEFAULT VALUES
# ==========================================

default_age = 24
default_position = "Attack"
default_sub_position = "Left Winger"
default_foot = "right"
default_height = 180
default_caps = 0
default_market_value = 20_000_000

default_appearances = 25
default_goals = 5
default_assists = 5
default_minutes = 2000
default_yellow_cards = 3
default_red_cards = 0


# ==========================================
# AUTO-FILL PLAYER PROFILE
# ==========================================

if selected_player is not None:

    player_id = selected_player["player_id"]

    # ------------------------------
    # AGE
    # ------------------------------

    if pd.notna(
        selected_player["date_of_birth"]
    ):

        today = pd.Timestamp.today()

        default_age = int(
            (
                today
                - selected_player["date_of_birth"]
            ).days / 365.25
        )

    # ------------------------------
    # POSITION
    # ------------------------------

    if pd.notna(
        selected_player["position"]
    ):
        default_position = (
            selected_player["position"]
        )

    # ------------------------------
    # SUB POSITION
    # ------------------------------

    if pd.notna(
        selected_player["sub_position"]
    ):
        default_sub_position = (
            selected_player["sub_position"]
        )

    # ------------------------------
    # FOOT
    # ------------------------------

    if pd.notna(
        selected_player["foot"]
    ):
        default_foot = (
            selected_player["foot"]
        )

    # ------------------------------
    # HEIGHT
    # ------------------------------

    if pd.notna(
        selected_player["height_in_cm"]
    ):

        default_height = int(
            selected_player["height_in_cm"]
        )

    # ------------------------------
    # INTERNATIONAL CAPS
    # ------------------------------

    if pd.notna(
        selected_player["international_caps"]
    ):

        default_caps = int(
            selected_player["international_caps"]
        )

    # ------------------------------
    # CURRENT MARKET VALUE
    # ------------------------------

    if pd.notna(
        selected_player["market_value_in_eur"]
    ):

        default_market_value = int(
            selected_player[
                "market_value_in_eur"
            ]
        )


    # ======================================
    # AUTO-CALCULATE LAST 365 DAYS
    # ======================================

    today = pd.Timestamp.today()

    one_year_ago = (
        today
        - pd.Timedelta(days=365)
    )

    player_appearances = appearances[
        (
            appearances["player_id"]
            == player_id
        )
        &
        (
            appearances["competition_id"]
            == "GB1"
        )
        &
        (
            appearances["date"]
            >= one_year_ago
        )
        &
        (
            appearances["date"]
            <= today
        )
    ].copy()


    # ------------------------------
    # APPEARANCES
    # ------------------------------

    default_appearances = len(
        player_appearances
    )


    # ------------------------------
    # GOALS
    # ------------------------------

    if "goals" in player_appearances.columns:

        default_goals = int(
            player_appearances[
                "goals"
            ].fillna(0).sum()
        )

    else:
        default_goals = 0


    # ------------------------------
    # ASSISTS
    # ------------------------------

    if "assists" in player_appearances.columns:

        default_assists = int(
            player_appearances[
                "assists"
            ].fillna(0).sum()
        )

    else:
        default_assists = 0


    # ------------------------------
    # MINUTES
    # ------------------------------

    if (
        "minutes_played"
        in player_appearances.columns
    ):

        default_minutes = int(
            player_appearances[
                "minutes_played"
            ].fillna(0).sum()
        )

    else:
        default_minutes = 0


    # ------------------------------
    # YELLOW CARDS
    # ------------------------------

    if (
        "yellow_cards"
        in player_appearances.columns
    ):

        default_yellow_cards = int(
            player_appearances[
                "yellow_cards"
            ].fillna(0).sum()
        )

    else:
        default_yellow_cards = 0


    # ------------------------------
    # RED CARDS
    # ------------------------------

    if (
        "red_cards"
        in player_appearances.columns
    ):

        default_red_cards = int(
            player_appearances[
                "red_cards"
            ].fillna(0).sum()
        )

    else:
        default_red_cards = 0


# ==========================================
# POSITION OPTIONS
# ==========================================

position_options = [
    "Goalkeeper",
    "Defender",
    "Midfield",
    "Attack"
]


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


# ==========================================
# INPUT COLUMNS
# ==========================================

left, right = st.columns(2)


# ==========================================
# PLAYER PROFILE
# ==========================================

with left:

    st.subheader(
        "👤 Player Profile"
    )

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=45,
        value=max(
            16,
            min(
                default_age,
                45
            )
        )
    )

    position_index = (
        position_options.index(
            default_position
        )
        if default_position
        in position_options
        else 3
    )

    position = st.selectbox(
        "Position",
        position_options,
        index=position_index
    )

    available_sub_positions = (
        position_map[position]
    )

    if (
        default_sub_position
        in available_sub_positions
    ):

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

    foot_index = (
        foot_options.index(
            default_foot
        )
        if default_foot
        in foot_options
        else 0
    )

    foot = st.selectbox(
        "Preferred Foot",
        foot_options,
        index=foot_index
    )

    height = st.number_input(
        "Height (cm)",
        min_value=150,
        max_value=215,
        value=max(
            150,
            min(
                default_height,
                215
            )
        )
    )

    international_caps = (
        st.number_input(
            "International Caps",
            min_value=0,
            max_value=250,
            value=max(
                default_caps,
                0
            )
        )
    )

    previous_market_value = (
        st.number_input(
            "Previous Market Value (€)",
            min_value=100_000,
            max_value=250_000_000,
            value=max(
                100_000,
                min(
                    default_market_value,
                    250_000_000
                )
            ),
            step=1_000_000
        )
    )


# ==========================================
# PERFORMANCE
# ==========================================

with right:

    st.subheader(
        "📊 Last 12 Months"
    )

    appearances_input = (
        st.number_input(
            "Premier League Appearances",
            min_value=0,
            max_value=60,
            value=max(
                default_appearances,
                0
            )
        )
    )

    goals = st.number_input(
        "Goals",
        min_value=0,
        max_value=60,
        value=max(
            default_goals,
            0
        )
    )

    assists = st.number_input(
        "Assists",
        min_value=0,
        max_value=40,
        value=max(
            default_assists,
            0
        )
    )

    minutes = st.number_input(
        "Minutes Played",
        min_value=0,
        max_value=5000,
        value=max(
            default_minutes,
            0
        )
    )

    yellow_cards = (
        st.number_input(
            "Yellow Cards",
            min_value=0,
            max_value=30,
            value=max(
                default_yellow_cards,
                0
            )
        )
    )

    red_cards = st.number_input(
        "Red Cards",
        min_value=0,
        max_value=10,
        value=max(
            default_red_cards,
            0
        )
    )


# ==========================================
# SHOW AUTO-FILL MESSAGE
# ==========================================

if selection_mode == "Select Real Player":

    # ======================================
    # CURRENT PREMIER LEAGUE PLAYERS
    # ======================================

    premier_league_players = players[
        players[
            "current_club_domestic_competition_id"
        ] == "GB1"
    ].copy()

    premier_league_players = (
        premier_league_players
        .dropna(subset=["name"])
        .drop_duplicates(subset=["player_id"])
    )

    # Create a nicer display name:
    # Bukayo Saka — Arsenal FC
    premier_league_players["display_name"] = (
        premier_league_players["name"]
        + " — "
        + premier_league_players[
            "current_club_name"
        ].fillna("Unknown Club")
    )

    premier_league_players = (
        premier_league_players
        .sort_values("name")
    )

    selected_display_name = st.selectbox(
        "Search / Select Premier League Player",
        premier_league_players[
            "display_name"
        ].tolist()
    )

    selected_rows = (
        premier_league_players[
            premier_league_players[
                "display_name"
            ] == selected_display_name
        ]
    )

    if not selected_rows.empty:

        selected_player = (
            selected_rows.iloc[0]
        )

        st.write(
            f"**{selected_player['name']}**"
        )

        st.caption(
            selected_player[
                "current_club_name"
            ]
        )
# ==========================================
# ADVANCED FEATURES
# ==========================================

if minutes > 0:

    goals_per_90 = (
        goals
        / minutes
        * 90
    )

    assists_per_90 = (
        assists
        / minutes
        * 90
    )

else:

    goals_per_90 = 0
    assists_per_90 = 0


if appearances_input > 0:

    minutes_per_appearance = (
        minutes
        / appearances_input
    )

else:

    minutes_per_appearance = 0


# ==========================================
# VALIDATION
# ==========================================

valid_input = True


if (
    appearances_input == 0
    and minutes > 0
):

    st.warning(
        "⚠️ Minutes played should be 0 "
        "if appearances are 0."
    )

    valid_input = False


if appearances_input > 0:

    if (
        minutes
        > appearances_input * 120
    ):

        st.warning(
            "⚠️ The minutes entered look "
            "unusually high for the number "
            "of appearances."
        )


# ==========================================
# CREATE INPUT DATA
# ==========================================

input_data = pd.DataFrame({

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
        appearances_input
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
})


# ==========================================
# PREDICT
# ==========================================

st.divider()


if st.button(
    "⚽ Predict Market Value",
    type="primary",
    use_container_width=True
):

    if valid_input:

        log_prediction = (
            model.predict(
                input_data
            )[0]
        )

        predicted_value = (
            np.expm1(
                log_prediction
            )
        )

        predicted_value = max(
            predicted_value,
            0
        )

        change = (
            predicted_value
            - previous_market_value
        )

        percentage_change = (
            change
            / previous_market_value
            * 100
        )


        # ==================================
        # RESULTS
        # ==================================

        st.header(
            "💰 Prediction"
        )

        col1, col2, col3 = (
            st.columns(3)
        )

        with col1:

            st.metric(
                "Previous Value",
                f"€{previous_market_value:,.0f}"
            )

        with col2:

            st.metric(
                "Predicted Value",
                f"€{predicted_value:,.0f}",
                f"{percentage_change:+.1f}%"
            )

        with col3:

            st.metric(
                "Estimated Change",
                f"€{change:+,.0f}"
            )


        # ==================================
        # CHART
        # ==================================

        st.subheader(
            "Market Value Comparison"
        )

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


        # ==================================
        # PERFORMANCE SUMMARY
        # ==================================

        st.subheader(
            "Player Performance"
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


        if selected_player is not None:

            st.success(
                f"Prediction generated for "
                f"{selected_player['name']}."
            )


        st.caption(
            "This model estimates a future "
            "Transfermarkt-style market valuation. "
            "It does not predict an actual transfer fee."
        )


# ==========================================
# MODEL INFORMATION
# ==========================================

st.divider()


with st.expander(
    "🧠 About the Machine Learning Model"
):

    st.write(
        """
        The model uses Random Forest regression
        trained on historical Premier League
        valuations and player performance.

        The recent performance features are based
        on the player's Premier League appearances
        during the previous 365 days.

        Important inputs include:

        - Previous market value
        - Age
        - Position
        - Preferred foot
        - Height
        - International caps
        - Premier League appearances
        - Goals
        - Assists
        - Minutes played
        - Goals per 90
        - Assists per 90
        - Discipline
        """
    )

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    metric1.metric(
        "Test MAE",
        "€2.43m"
    )

    metric2.metric(
        "Test RMSE",
        "€4.18m"
    )

    metric3.metric(
        "Test R²",
        "0.962"
    )

    st.write(
        "**Naive baseline MAE:** €2.80m"
    )


with st.expander(
    "⚠️ Model Limitations"
):

    st.write(
        """
        Previous market value is currently the
        most influential feature in the model.

        The model performs particularly well when
        valuations change gradually, but sudden
        rises and falls are harder to predict.

        Performance data currently includes only
        Premier League appearances. A new signing
        from another league may therefore have
        little or no recent performance data in
        this application.

        Transfermarkt market values are estimates
        and should not be treated as actual transfer
        fees.
        """
    )
    