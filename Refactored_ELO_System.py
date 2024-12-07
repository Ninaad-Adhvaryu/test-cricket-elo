
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

# def top_rank(data):
#     """Determine the top-ranked teams over time."""
#     top_rank_df = pd.DataFrame(columns=["Team", "Start Date", "Start Test#", "End Date", "End Test#"])
#     current_top = None
#     for _, row in data.iterrows():
#         new_top = row.iloc[:13].idxmax()
#         if new_top != current_top:
#             if current_top:
#                 top_rank_df.loc[top_rank_df.index[-1], ["End Date", "End Test#"]] = row["date"], row["Test"]
#             top_rank_df = pd.concat(
#                 [top_rank_df, pd.DataFrame([{
#                     "Team": new_top, "Start Date": row["date"], "Start Test#": row["Test"]
#                 }])]
#             )
#             current_top = new_top
#     return top_rank_df

# def elo_plot(data, skip_teams=None, filename="ELO_Rating_Tests.html"):

#     """Plot ELO ratings for all teams."""
#     if skip_teams is None:
#         skip_teams = []
#     fig = go.Figure()
#     for team in data.columns[:13]:
#         if team not in skip_teams:
#             fig.add_trace(go.Scatter(x=data['date'], y=data[team], mode='lines', name=team))
#     fig.update_layout(title="ELO Ratings Over Time", xaxis_title="Date", yaxis_title="ELO Rating")
#     fig.write_html(filename)


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


def elo_plot(data, skip_teams=None, filename="ELO_Rating_Tests.html", smooth=False, smoothing_window=50, highlight_top_teams=False):
    """
    Plot ELO ratings for all teams with optional smoothing and highlights for top-ranked teams.

    Parameters:
        data (pd.DataFrame): ELO ratings data.
        skip_teams (list): List of teams to exclude from the plot.
        filename (str): Name of the output HTML file.
        smooth (bool): Whether to smooth the data using a rolling average.
        smoothing_window (int): Window size for rolling average smoothing (used if smooth=True).
    """
    TEAM_COLORS = {
    "Afghanistan": "cyan",
    "ICC World XI": "firebrick",
    "India": "blue",
    "Pakistan": "lawngreen",
    "Bangladesh": "olive",
    "Sri Lanka": "darkorange", #navyblue
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

    for team in data.columns[:13]:
        if team not in skip_teams:
            # Check if smoothing is enabled
            if smooth:
                # Apply smoothing using a rolling average
                y_values = data[team].rolling(window=smoothing_window, center=True).mean()
                trace_name = f"{team} (Smoothed)"
            else:
                y_values = data[team]
                trace_name = team

            # Add the data to the plot
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=y_values,
                    mode='lines',
                    name=trace_name,
                    line=dict(color=TEAM_COLORS.get(team, "gray"))  # Use team color or default to gray,
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
                    fillcolor=TEAM_COLORS.get(team, "lightblue"),  # Default to lightblue if team color not found opacity=0.3,
                    layer="below",
                    line_width=0
                )

    # # Add annotations for key matches
    # test_matches = pd.read_csv("test_matches_1946_2024.csv")
    # # Standardize `data['Test']` to match `test_matches['Scorecard Number']`
    # data['Test'] = data['Test'].str.extract(r'(\d+)', expand=False).astype(int)

    # for _, match in test_matches.iterrows():
    #     # Determine the annotation text
    #     if match['Winner'] == match['Team 1']:
    #         annotation_text = f"{match['Winner']} beat {match['Team 2']} by {match['Margin']} at {match['Ground']}, {match['Match Date']} ({match['Scorecard']})"
    #     elif match['Winner'] == match['Team 2']:
    #         annotation_text = f"{match['Winner']} beat {match['Team 1']} by {match['Margin']} at {match['Ground']}, {match['Match Date']} ({match['Scorecard']})"
    #     else:  # Handle draws or other cases
    #         annotation_text = f"{match['Team 1']} vs {match['Team 2']} was drawn at {match['Ground']}, {match['Match Date']} ({match['Scorecard']})"

    #     # Retrieve match details for annotation
    #     if match['Scorecard Number'] in data['Test'].values:
    #         match_date = data.loc[data['Test'] == match['Scorecard Number'], 'date'].values[0]
    #         if match['Winner'] in data.columns:
    #             winner_elo = data.loc[data['Test'] == match['Scorecard Number'], match['Winner']].values[0]

    #             # Add the annotation marker
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=[match_date],
    #                     y=[winner_elo],
    #                     mode='markers',
    #                     hovertext=annotation_text,  # Use hovertext for hover-only display
    #                     hoverinfo="text",  # Show only the text on hover
    #                     marker=dict(color=TEAM_COLORS.get(match['Winner'], "gray"), size=10),
    #                     showlegend=False
    #                 )
    #             )

 

    # Update layout for better visualization
    fig.update_layout(
        title="ELO Ratings Over Time" + (" (Smoothed)" if smooth else ""),
        xaxis_title="Date",
        yaxis_title="ELO Rating",
        template="seaborn"
    )
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
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
else:
    print("Calculating ELO ratings...")
    elo_data = run_elo(tests, initial_elo, track=True)
    print(f"Saving ELO data to {elo_results_file}...")
    elo_data.to_csv(elo_results_file, index=False)

# Plot the ELO ratings
elo_plot(
    elo_data,
    skip_teams=["Afghanistan", "ICC World XI", "Ireland"],
    filename="ELO_Rating_Plot.html",
    smooth=False,
    smoothing_window=10
)

