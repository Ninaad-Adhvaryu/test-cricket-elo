import pandas as pd

# Load the ELO data and match results
elo_data_file = "elo_data_1946_2024.csv"  # Replace with your ELO data file
match_results_file = "test_matches_1946_2024.csv"  # Replace with your match results file
elo_data = pd.read_csv(elo_data_file)
match_results = pd.read_csv(match_results_file)

# Ensure 'date' in ELO data is in datetime format
elo_data['date'] = pd.to_datetime(elo_data['date'], errors='coerce')

# Merge ELO data and match results on the 'Test' column
merged_data = pd.merge(elo_data, match_results, left_on='Test', right_on='Scorecard')

# Calculate greatest upsets
upsets = []
for _, match in merged_data.iterrows():
    winner = match['Winner']
    loser = match['Team 1'] if match['Team 1'] != winner else match['Team 2']
    
    if winner in elo_data.columns and loser in elo_data.columns:
        # Get ELO values for winner and loser before the match
        winner_elo = match[winner]
        loser_elo = match[loser]
        
        # Check if it's an upset (winner's ELO is lower than loser's ELO)
        if winner_elo < loser_elo:
            elo_diff = loser_elo - winner_elo
            upsets.append({
                'Date': match['date'],  # Use elo_data['date']
                'Match': f"{winner} beat {loser} by {match['Margin']} at {match['Ground']} ({match['Scorecard']})",
                'ELO Difference': elo_diff
            })

# Convert upsets to a DataFrame and find the top 5
upsets_df = pd.DataFrame(upsets).sort_values(by='ELO Difference', ascending=False).head(20)

# Print the top 20 greatest upsets
print("\nTop 20 Greatest Upsets (With Match Details):")
print(upsets_df)

# Save to CSV for further analysis
# upsets_df.to_csv("greatest_upsets.csv", index=False)
