import pandas as pd

# Load the ELO data
elo_data_file = "elo_data_1946_2024.csv"  # Replace with your ELO data file
elo_data = pd.read_csv(elo_data_file)

# Ensure 'date' is in datetime format
elo_data['date'] = pd.to_datetime(elo_data['date'], errors='coerce')

# Ensure the data is sorted by date
elo_data = elo_data.sort_values(by='date')

# Teams columns
teams = [col for col in elo_data.columns if col not in ['date', 'Test']]

# Identify the team with the highest ELO on each date
elo_data['No. #1 Team'] = elo_data[teams].idxmax(axis=1)

# Calculate days spent at No. #1
# Group by consecutive periods where the same team is No. #1
elo_data['No. #1 Team Shift'] = (elo_data['No. #1 Team'] != elo_data['No. #1 Team'].shift()).cumsum()

# Group by these consecutive periods and calculate the duration of each period
grouped_periods = elo_data.groupby(['No. #1 Team Shift', 'No. #1 Team']).agg(
    Start_Date=('date', 'min'),
    End_Date=('date', 'max')
).reset_index()

# Calculate the number of days for each period
grouped_periods['Days'] = (grouped_periods['End_Date'] - grouped_periods['Start_Date']).dt.days + 1

# Sum up the days for each team
no_1_days_df = grouped_periods.groupby('No. #1 Team').agg(Total_Days=('Days', 'sum')).reset_index()

# Sort by total days spent at No. #1
no_1_days_df = no_1_days_df.sort_values(by='Total_Days', ascending=False)

# Display the results
print("\nDays Spent at No. #1 by Each Team:")
print(no_1_days_df)

# Save to CSV for further analysis
# no_1_days_df.to_csv("days_at_no_1_corrected.csv", index=False)
