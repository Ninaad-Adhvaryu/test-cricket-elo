import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math

# Set the page configuration
st.set_page_config(
    page_title="ELO Dashboard",
    page_icon=":cricket_game:",
    layout="wide",
)

# Load data
@st.cache_data
def load_data():
    """Load data and ensure the 'date' column is in datetime format."""
    elo_data = pd.read_csv("./elo_data_1946_2024.csv")
    elo_data["date"] = pd.to_datetime(elo_data["date"], errors="coerce")
    return elo_data

@st.cache_data
def load_matches():
    """Load match results data."""
    return pd.read_csv("./test_matches_1946_2024.csv")

# Generate Match Info
@st.cache_data
def generate_match_info(tests_elo):
    tests = load_matches()
    info = []
    for _, row in tests_elo.iterrows():
        match = tests[tests['Scorecard'] == row['Test']]
        if not match.empty:
            match = match.iloc[0]
            home_team = match['Team 1']
            away_team = match['Team 2']
            ground = match['Ground']
            date = match['Match Date']
            winner = match['Winner']
            margin = match['Margin']

            # Handle different outcomes
            if winner == "drawn":
                info.append(f"{home_team} played {away_team} at {ground} on {date}. Match drawn.")
            elif winner == "tied":
                info.append(f"{home_team} played {away_team} at {ground} on {date}. Match tied.")
            elif winner in [home_team, away_team]:
                info.append(f"{home_team} played {away_team} at {ground} on {date}. {winner} won by {margin}.")
            else:
                info.append(f"{home_team} played {away_team} at {ground} on {date}. No result.")
        else:
            info.append("No match information available.")
    return info

# HoverInfo
def hoverInfo(tests_elo):
    """
    Generate hover information for each team in the dataset.

    Parameters:
        tests_elo (pd.DataFrame): The ELO data containing test match information.

    Returns:
        dict: A dictionary containing hover information for each team.
    """
    # Load match data
    tests = pd.read_csv("test_matches_1946_2024.csv")

    # Initialize hover info dictionary
    info = {country: [] for country in tests_elo.columns.tolist()[:13]}

    # Iterate through ELO data to construct hover info
    for i in range(1, len(tests_elo)):
        # Get the scorecard number
        scorecard_number = tests_elo.iloc[i, 14][7:]  # Extract the scorecard number (Test # XXXX)

        # Retrieve the corresponding test match row
        match = tests.loc[tests['Scorecard'] == f'Test # {scorecard_number}']

        if not match.empty:
            match = match.iloc[0]  # Extract the first match (as a Series)
            home_team = match['Team 1']
            away_team = match['Team 2']
            ground = match['Ground']
            date = match['Match Date']
            winner = match['Winner']
            margin = match['Margin']

            # Construct hover text
            result_text = f"{home_team} played {away_team} at {ground} on {date}. {winner} won by {margin}."

            # Update hover info for each team
            for country in tests_elo.columns.tolist()[:13]:
                if country == home_team or country == away_team:
                    info[country].append(result_text)
                else:
                    info[country].append(None)
        else:
            # If no match is found, append None for all teams
            for country in tests_elo.columns.tolist()[:13]:
                info[country].append(None)

    return info


# Match lookup function
def match_lookup(match_number, k=400, hga=64):
    match_data = elo_data[elo_data["Test"] == f"Test # {match_number}"]
    if not match_data.empty:
        match_info = match_data.iloc[0]["Match Info"]

        # Extract home and away teams
        home_team = match_info.split(" played ")[0]
        away_team = match_info.split(" played ")[1].split(" at ")[0]

        # Extract pre-match ELO (from the row before)
        current_index = match_data.index[0]
        if current_index > 0:  # Ensure we aren't at the first match
            pre_match = elo_data.iloc[current_index - 1]
            pre_home_elo = pre_match[home_team]
            pre_away_elo = pre_match[away_team]
        else:
            pre_home_elo = pre_away_elo = "N/A"

        # Extract post-match ELO (from the current row)
        post_home_elo = match_data.iloc[0][home_team]
        post_away_elo = match_data.iloc[0][away_team]

        # Calculate win probability for the home team
        if pre_home_elo != "N/A" and pre_away_elo != "N/A":
            elo_diff = pre_home_elo - pre_away_elo + hga
            win_prob_home = 1 / (1 + 10 ** (-elo_diff / k)) * 100
        else:
            win_prob_home = "N/A"

        return {
            "match_info": match_info,
            "pre_home_elo": pre_home_elo,
            "pre_away_elo": pre_away_elo,
            "post_home_elo": post_home_elo,
            "post_away_elo": post_away_elo,
            "win_prob_home": win_prob_home,
        }
    else:
        return {"error": "Match not found."}

# Add Match Info to ELO Data and save rounded file
elo_data = load_data()
if "Match Info" not in elo_data.columns:
    elo_data["Match Info"] = generate_match_info(elo_data)
    elo_data.iloc[:, 2:15] = elo_data.iloc[:, 2:15].round(0)
    elo_data.to_csv("rounded_elo_data_1946_2024.csv", index=False)  # Updated filename

# Tabs for graph, table, and match lookup
tab1, tab2, tab3, tab4 = st.tabs(
    ["Ratings Graph", "Data Table", "Match Lookup", "Current Rankings"]
)
with tab1:
    st.header("ELO Ratings Graph")

    # Filters for the graph
    col1, col2 = st.columns(2)

    with col1:
        year_start, year_end = st.slider(
            "Select Year Range:",
            min_value=int(elo_data["date"].dt.year.min()),
            max_value=int(elo_data["date"].dt.year.max()),
            value=(int(elo_data["date"].dt.year.min()), int(elo_data["date"].dt.year.max())),
        )

    with col2:
        selected_teams = st.multiselect(
            "Select Teams to Display:",
            options=elo_data.columns[2:15],
            default=[
                "India",
                "Pakistan",
                "Sri Lanka",
                "West Indies",
                "England",
                "New Zealand",
                "South Africa",
                "Australia",
            ],
        )

    # Filter data based on user input
    filtered_data = elo_data[
        (elo_data["date"].dt.year >= year_start) & (elo_data["date"].dt.year <= year_end)
    ]

    # Plot the ELO ratings
    fig = go.Figure()

    TEAM_COLORS = {
        "Afghanistan": "cyan",
        "ICC World XI": "firebrick",
        "India": "blue",
        "Pakistan": "lawngreen",
        "Bangladesh": "olive",
        "Sri Lanka": "darkorange",
        "West Indies": "maroon",
        "Zimbabwe": "red",
        "England": "silver",
        "New Zealand": "black",
        "South Africa": "hotpink",
        "Australia": "goldenrod",
        "Ireland": "springgreen",
    }

    for team in selected_teams:
        if team in filtered_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_data["date"],
                    y=filtered_data[team],
                    mode="lines",
                    text=hoverInfo(filtered_data)[team],
                    name=team,
                    line=dict(color=TEAM_COLORS.get(team, "gray")),
                )
            )

    fig.update_layout(
        title="ELO Ratings Over Time",
        xaxis_title="Date",
        yaxis_title="ELO Rating",
        template="plotly_dark",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("ELO Data Table")

    # Filters for the table
    start_date, end_date = st.date_input(
        "Select date range:",
        [elo_data["date"].min().date(), elo_data["date"].max().date()],
        min_value=elo_data["date"].min().date(),
        max_value=elo_data["date"].max().date(),
    )

    table_teams = st.multiselect(
        "Select teams to display:",
        options=elo_data.columns[2:15],
        default=[
            "India",
            "Pakistan",
            "Sri Lanka",
            "West Indies",
            "England",
            "New Zealand",
            "South Africa",
            "Australia",
        ],
    )

    # Filter data for table
    table_data = elo_data[
        (elo_data["date"] >= pd.to_datetime(start_date))
        & (elo_data["date"] <= pd.to_datetime(end_date))
    ]

    # Sort the data by 'date' in descending order
    table_data = table_data.sort_values(by="date", ascending=False)

    formatted_table = table_data[["date", "Test"] + table_teams + ["Match Info"]].copy()
    formatted_table["date"] = formatted_table["date"].dt.strftime("%Y-%m-%d")

    st.dataframe(formatted_table, use_container_width=True, height=600)

with tab3:
    st.title("Match Lookup")
    match_number = st.text_input("Enter Match Number:", value="2444")

    if match_number:
        result = match_lookup(int(match_number))

        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Match Information")
            st.write(result["match_info"])

            st.subheader("Rating Change")
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label=f"{result['match_info'].split(' played ')[0]}",
                    value=f"{int(result['pre_home_elo'])} -> {int(result['post_home_elo'])}"
                )
                st.metric(
                    label=f"{result['match_info'].split(' played ')[1].split(' at ')[0]}",
                    value=f"{int(result['pre_away_elo'])} -> {int(result['post_away_elo'])}"
                )

            with col2:
                st.metric(
                    label="Home Team Win Probability",
                    value=f"{result['win_prob_home']:.2f}%" if result["win_prob_home"] != "N/A" else "N/A"
                )

with tab4:
    st.header("Current Rankings")

    # Fetch the most recent data row
    most_recent_data = elo_data.sort_values(by="date", ascending=False).head(1)

    # Extract and sort team ratings (excluding unwanted columns)
    team_ratings = (
        most_recent_data.iloc[0, 2:15]  # Exclude non-team columns
        .drop(labels=["ICC World XI"], errors="ignore")  # Exclude ICC World XI
        .apply(pd.to_numeric, errors="coerce")  # Ensure numeric values
        .dropna()  # Remove NaN values
        .sort_values(ascending=False)  # Sort by rating, descending
    )

    # Render the rankings as a numbered list
    st.subheader("Team Rankings")
    for i, (team, rating) in enumerate(team_ratings.items(), start=1):
        st.write(f"**{i}. {team}** - {int(rating)}")

    # Display the latest match information
    st.subheader("Most Recent Match")
    most_recent_date = most_recent_data["date"].dt.strftime("%Y-%m-%d").iloc[0]
    st.write(f"**Date:** {most_recent_date}")
    st.write(f"**Match:** {most_recent_data['Test'].iloc[0]}")
    st.write(f"**Info:** {most_recent_data['Match Info'].iloc[0]}")
