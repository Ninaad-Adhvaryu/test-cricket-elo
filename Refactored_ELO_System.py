
# Consolidated and refactored ELO rating system script

import os
import pandas as pd
import requests
import datetime
import numpy as np
import plotly.graph_objects as go


# Utility functions
def convert_date(date):
    """Convert different date formats to a standardized format."""
    if len(date) <= 15:
        day, month, year = date[4:6], date[:3], date[-4:]
    elif 15 <= len(date) < 26:
        day, month, year = date[-7:-4], date[7:10], date[-4:]
    else:
        day, month, year = date[-7:-4], date[-11:-8], date[-4:]
    
    day = int(day.strip().strip('-,'))
    return datetime.datetime.strptime(f"{day}/{month}/{year}", '%d/%b/%Y').date()


def get_winner(result):
    """Determine winner from test match result."""
    if result.iloc[2] == result.iloc[0]:  # Home win
        return 1
    elif result.iloc[2] == result.iloc[1]:  # Away win
        return 0
    return 0.5  # Draw


def update_elo(rating_home, rating_away, w, k, n, hga):
    """Calculate updated ELO ratings."""
    prob_home = 1 / (1 + 10 ** -((rating_home - rating_away + hga) / n))
    prob_away = 1 - prob_home
    if w == 1:
        rating_home += k * (1 - prob_home)
        rating_away += k * (0 - prob_away)
    elif w == 0.5:
        rating_home += k * (0.5 - prob_home)
        rating_away += k * (0.5 - prob_away)
    else:
        rating_home += k * (0 - prob_home)
        rating_away += k * (1 - prob_away)
    return round(rating_home, 4), round(rating_away, 4)


def all_tests(start_year, end_year):
    """
    Fetches Test match results for all years between start_year and end_year (inclusive).

    Parameters:
        start_year (int): The starting year.
        end_year (int): The ending year.

    Returns:
        pd.DataFrame: A DataFrame containing match results for the specified years.
    """
    all_matches = []
    
    for year in range(start_year, end_year + 1):
        url = f"https://www.espncricinfo.com/records/year/team-match-results/{year}-{year}/test-matches-1"
        
        try:
            # Get HTML content and read the match results table
            html = requests.get(url).content
            df_matches = pd.read_html(html)[0]

            # Extract numeric part of the Scorecard (Test #) and sort by it
            df_matches['Scorecard Number'] = df_matches['Scorecard'].str.extract(r'(\d+)').astype(int)
            df_matches = df_matches.sort_values(by='Scorecard Number').reset_index(drop=True)
            
            
            # Append to the list
            all_matches.append(df_matches)
            
            print(f"Successfully fetched matches for {year}.")
        except Exception as e:
            print(f"Failed to fetch matches for {year}: {e}")
    
    # Combine all yearly dataframes into one
    if all_matches:
        return pd.concat(all_matches, ignore_index=True)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data was fetched

def run_elo(tests, initial_ratings, track=False, k=22.2, n=400, hga=64):
    """Run ELO calculations for test matches."""
    elo_list = initial_ratings.copy()
    elo_track = pd.DataFrame([elo_list], index=[elo_list["Test"].split("#")[-1].strip()])
    for _, match in tests.iterrows():
        w = get_winner(match)
        home_team, away_team = match.iloc[0], match.iloc[1]
        elo_list[home_team], elo_list[away_team] = update_elo(
            elo_list[home_team], elo_list[away_team], w, k, n, hga
        )
        elo_list["date"] = convert_date(match.iloc[5])
        elo_list["Test"] = match.iloc[6]
        if track:
            index_name = match.iloc[6].split("#")[-1].strip()
            elo_track = pd.concat([elo_track, pd.DataFrame([elo_list], index=[index_name])])
    return elo_track if track else elo_list


# Analysis and visualization

def top_rank(data):
    """
    Determine the top-ranked teams over time based on the highest ELO.

    Parameters:
        data (pd.DataFrame): ELO data with teams as columns and 'date' and 'Test' columns.

    Returns:
        pd.DataFrame: A DataFrame indicating periods of top-ranked teams.
    """
    top_rank_df = pd.DataFrame(columns=["Team", "Start Date", "Start Test#", "End Date", "End Test#"])
    current_top = None

    for i, row in data.iterrows():
        # Find the team with the highest ELO
        new_top = row.iloc[:13].idxmax()  # First 13 columns are team ELOs

        if new_top != current_top:
            if current_top:
                # Update the end date and test number for the previous top team
                top_rank_df.loc[top_rank_df.index[-1], ["End Date", "End Test#"]] = row["date"], row["Test"]

            # Add a new row for the new top-ranked team
            top_rank_df = pd.concat(
                [top_rank_df, pd.DataFrame([{
                    "Team": new_top,
                    "Start Date": row["date"],
                    "Start Test#": row["Test"],
                    "End Date": None,
                    "End Test#": None
                }])],
                ignore_index=True
            )
            current_top = new_top

    # Update the end date and test number for the last top team
    if not top_rank_df.empty:
        top_rank_df.loc[top_rank_df.index[-1], ["End Date", "End Test#"]] = data.iloc[-1]["date"], data.iloc[-1]["Test"]

    return top_rank_df

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

def generate_match_info(tests_elo):
    tests = pd.read_csv("test_matches_1946_2024.csv")
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

def elo_plot(data, skip_teams=None, filename="ELO_Rating_Tests.html", smooth=False, smoothing_window=50, highlight_top_teams=False):
    """
    Plot ELO ratings for all teams with optional smoothing, highlights, and hover info.

    Parameters:
        data (pd.DataFrame): ELO ratings data.
        skip_teams (list): List of teams to exclude from the plot.
        filename (str): Name of the output HTML file.
        smooth (bool): Whether to smooth the data using a rolling average.
        smoothing_window (int): Window size for rolling average smoothing (used if smooth=True).
        highlight_top_teams (bool): Whether to highlight periods of top-ranked teams.
    """
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
        "Ireland": "springgreen"
    }

    if skip_teams is None:
        skip_teams = []

    # Create a figure
    fig = go.Figure()

    # Add traces for each team
    for team in data.columns[:13]:
        if team not in skip_teams:
            # Apply smoothing if enabled
            y_values = data[team].rolling(window=smoothing_window, center=True).mean() if smooth else data[team]

            # Add the trace
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=y_values,
                    mode="lines",
                    text=hoverInfo(data)[team],
                    line=dict(color=TEAM_COLORS.get(team, "gray")),
                    name=team
                )
            )

    # Highlight top-ranked teams over time
    if highlight_top_teams:
        ranks = top_rank(data)
        for _, row in ranks.iterrows():
            team = row["Team"]
            if team not in skip_teams:
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=row["Start Date"],
                    x1=row["End Date"],
                    y0=0,
                    y1=1,
                    fillcolor=TEAM_COLORS.get(team, "lightblue"),
                    opacity=0.3,
                    layer="below",
                    line_width=0
                )

    # Update layout
    fig.update_layout(
        title="ELO Ratings Over Time" + (" (Smoothed)" if smooth else ""),
        xaxis_title="Date",
        yaxis_title="ELO Rating",
        template="seaborn"
    )

    # Save to an HTML file
    fig.write_html(filename)




# Initialize and run
initial_elo = {
    "Afghanistan": 1200, "ICC World XI": 1200, "India": 1200, "Pakistan": 1200,
    "Bangladesh": 1200, "Sri Lanka": 1200, "West Indies": 1200, "Zimbabwe": 1200,
    "England": 1200, "New Zealand": 1200, "South Africa": 1200, "Australia": 1200,
    "Ireland": 1200, "date": "0000-00-00", "Test": "Test #0000"
}

year_start, year_end = 1946, 2024
roundElo = True

# File paths for saving match data and ELO data
match_results_file = f"test_matches_{year_start}_{year_end}.csv"
elo_results_file = f"elo_data_{year_start}_{year_end}.csv"

# Check if match data already exists
if os.path.exists(match_results_file):
    print(f"Loading saved match results from {match_results_file}...")
    tests = pd.read_csv(match_results_file)
else:
    print("Downloading match results...")
    tests = all_tests(year_start, year_end)
    print(f"Saving match results to {match_results_file}...")
    tests.to_csv(match_results_file, index=False)

# Check if ELO data already exists
if os.path.exists(elo_results_file):
    print(f"Loading saved ELO data from {elo_results_file}...")
    elo_data = pd.read_csv(elo_results_file)
    elo_data["date"] = pd.to_datetime(elo_data["date"], errors="coerce")  # Convert to datetime
    elo_data = elo_data.dropna(subset=["date"])  # Remove invalid rows
    elo_data["date"] = elo_data["date"].dt.date  # Convert to datetime.date
else:
    print("Calculating ELO ratings...")
    elo_data = run_elo(tests, initial_elo, track=True)

    # Adding match info to the elo CSV
    elo_data["Match Info"] = generate_match_info(elo_data)

    # Rounding the elo values
    elo_data.iloc[:, 2:15] = elo_data.iloc[:, 2:15].round(0)

    print(f"Saving ELO data to {elo_results_file}...")
    elo_data.to_csv(elo_results_file, index=False) #elo will also be rounded now and contain match info


# Plot the ELO ratings
# elo_plot(
#     elo_data,
#     skip_teams=["Afghanistan", "ICC World XI", "Ireland"],
#     filename="ELO_Rating_Plot.html",
#     smooth=False,
#     smoothing_window=10
# )