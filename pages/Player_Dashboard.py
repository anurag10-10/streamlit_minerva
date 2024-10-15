import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as path_effects
from mplsoccer import VerticalPitch, Pitch
from matplotlib.lines import Line2D
import main_minerva
import pickle
import json
import streamlit_shadcn_ui as ui

# Load the data for both teams
df1 = pd.read_csv("Minerva_vs_Sudeva_Minerva_data.csv")
df2 = pd.read_csv('Minerva_vs_Sudeva_Sudeva_data.csv')

team_name1 = df1['Team'].iloc[0]
team_name2 = df2['Team'].iloc[0]

shotmap_df1 = df1[df1['Event'] == 'Shot']
shotmap_df2 = df2[df2['Event'] == 'Shot']

# Define a function to process shot data
def process_shots_df(df):
    """
    Processes the shot data from a given DataFrame, adds binning based on custom pitch divisions,
    and resets the index.

    Parameters:
    df (pd.DataFrame): DataFrame containing event data with 'X', 'Y', and 'Event' columns.

    Returns:
    pd.DataFrame: Processed DataFrame with inverted coordinates, shot data, and bin columns.
    """
    
    # Filter the dataframe to get only shots
    shots_df = df[df['Event'] == 'Shot']
    
    # Invert the X and Y coordinates
    shots_df.rename(columns={"X": "Y", "Y": "X"}, inplace=True)

    # Define bins for the pitch divisions
    y_bins = [120, 114, 108, 102, 84, 66]  # Custom bins for Y (formerly X)
    x_bins = [18, 30, 50, 62, 80]          # Custom bins for X (formerly Y)

    # Sort bins to ensure they are in ascending order
    x_bins.sort()
    y_bins.sort()

    # Apply binning for both X and Y coordinates
    shots_df["bins_x"] = pd.cut(shots_df["X"], bins=x_bins)
    shots_df["bins_y"] = pd.cut(shots_df["Y"], bins=y_bins)

    # Reset the index and rename the index column to 'shotID'
    shots_df = shots_df.reset_index().rename(columns={'index': 'shotID'})
    
    return shots_df

# Define zones for shot categorization
zones = {
    'First 6 yd.': {'X_min': 114, 'X_max': 120, 'Y_min': 29, 'Y_max': 51.0},
    'Second 6 yd.': {'X_min': 108, 'X_max': 114, 'Y_min': 30, 'Y_max': 50},
    'Third 6 yd.': {'X_min': 102, 'X_max': 108, 'Y_min': 30, 'Y_max': 50},
    'Zone 14': {'X_min': 84, 'X_max': 102, 'Y_min': 30, 'Y_max': 50},
    'Left Wide Area \n of the box': {'X_min': 102, 'X_max': 120, 'Y_min': 18, 'Y_max': 30},
    'Right Wide Area \n of the box': {'X_min': 102, 'X_max': 120, 'Y_min': 50, 'Y_max': 62},
    'Left Upper Wing Area': {'X_min': 102, 'X_max': 120, 'Y_min': 0, 'Y_max': 18},
    'Right Upper Wing Area': {'X_min': 102, 'X_max': 120, 'Y_min': 62, 'Y_max': 80},
    'Left Att. \n Half Space': {'X_min': 84, 'X_max': 102, 'Y_min': 18, 'Y_max': 30},
    'Right Att. \n Half Space': {'X_min': 84, 'X_max': 102, 'Y_min': 50, 'Y_max': 62},
    'Right mid Wide Area': {'X_min': 84, 'X_max': 102, 'Y_min': 62, 'Y_max': 80}
}

# Function to categorize each shot into a zone
def categorize_shot(row, zones):
    for zone, bounds in zones.items():
        if (bounds['X_min'] <= row['Y'] <= bounds['X_max'] and
            bounds['Y_min'] <= row['X'] <= bounds['Y_max']):
            return zone
    return np.nan  # If the shot doesn't fall into any defined zone

# Function to draw the pitch with divisions
def soc_pitch_divisions(ax):
    pitch = VerticalPitch(
        pitch_type="statsbomb",
        half=True,
        goal_type='box',
        linewidth=1.25,
        line_color='grey',
        pitch_color='white'
    )
    pitch.draw(ax=ax)
    
    ax.plot([18, 62], [84, 84], color='black', linewidth=1)
    ax.plot([0, 80], [102, 102], color='black', linewidth=1)
    ax.plot([30,50], [114, 114], color='black', linewidth=1)
    ax.plot([30,50], [108, 108], color='black', linewidth=1)

    ax.plot([18, 18], [84, 120], color='black', linewidth=1)
    ax.plot([30, 30], [84, 120], color='black', linewidth=1)
    ax.plot([50, 50], [84, 120], color='black', linewidth=1)
    ax.plot([62, 62], [84, 120], color='black', linewidth=1)

    #ax.text(28, 75, f'Total shots: {total_shots}', fontweight='bold')

# Function to plot shots
def plot_shots_map(shots_df, color, ax):
    shots_df['zone'] = shots_df.apply(categorize_shot, zones=zones, axis=1)
    df_shots = shots_df.groupby('zone').size().reset_index(name='shot_count')
    total_shots = df_shots['shot_count'].sum()
    df_shots['shot_share'] = df_shots['shot_count'] / total_shots
    df_shots['shot_share_percent'] = (df_shots['shot_share'] * 100).round(2)

    df_shots['shot_scaled'] = df_shots['shot_share'] / df_shots['shot_share'].max()

    soc_pitch_divisions(ax)

    for counter, zone in enumerate(df_shots['zone']):
        bounds = zones[zone]
        ax.fill_between(
            x=[bounds['Y_min'], bounds['Y_max']],
            y1=bounds['X_min'],
            y2=bounds['X_max'],
            color=color,
            alpha=df_shots['shot_scaled'].iloc[counter],
            zorder=-1,
            lw=0
        )

        if df_shots['shot_share'].iloc[counter] > .02:
            ax.annotate(
                xy=((bounds['Y_max'] + bounds['Y_min']) / 2, (bounds['X_max'] + bounds['X_min']) / 2),
                text=f"{df_shots['shot_share'].iloc[counter]:.2%}",
                ha="center",
                va="center",
                color="black",
                size=8.5,
                weight="bold",
                zorder=3
            )

def shots_map(shots_df_, col, ax=None):
    pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#ffffff', line_color='grey', half=True, goal_type='box')
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
        pitch.draw(ax=ax)
    else:
        fig = ax.get_figure()
        pitch.draw(ax=ax)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Goal', markerfacecolor=col, markersize=15, markeredgecolor='black'),
        Line2D([0], [0], marker='x', color='w', label='Blocked', markerfacecolor=col, markersize=15, alpha=0.2, markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', label='Off Target', markerfacecolor=col, markersize=15, alpha=0.2, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Saved', markerfacecolor=col, markersize=15, alpha=0.2, markeredgecolor='black')
    ]

    for i, row in shots_df_.iterrows():
        if row["Outcome"] == 'Goal':
            pitch.scatter(row.X, row.Y, alpha=1, s=200, color=col, ax=ax, edgecolors="black")
            pitch.annotate(row["Player"], (row.X + 1, row.Y - 2), ax=ax, fontsize=15)
        elif row["Outcome"] == 'Blocked':
            pitch.scatter(row.X, row.Y, alpha=0.2, s=100, color=col, ax=ax, edgecolors="black", marker='x')
        elif row["Outcome"] == 'Off Target':
            pitch.scatter(row.X, row.Y, alpha=0.2, s=100, color=col, ax=ax, edgecolors="black", marker='^')
        elif row["Outcome"] == 'Saved':
            pitch.scatter(row.X, row.Y, alpha=0.2, s=100, color=col, ax=ax, edgecolors="black")

    ax.legend(handles=legend_elements, loc='lower left', fontsize=15)
    return fig, ax

# Streamlit app starts here

st.title("Player Dashboard")

# Define columns for the layout
col1, col2 = st.columns(2)

# Selectbox to choose the team
team = st.selectbox('Select a team', (team_name1, team_name2))

# Selectbox to choose the team
#team = st.selectbox('Select a team', (team_name1, team_name2))

# Load the appropriate dataset based on selection
if team == team_name1:
   selected_player = st.selectbox('Select a player', df1[df1['Team']== team]['Player'].sort_values().unique())
else:
    selected_player = st.selectbox('Select a player', df2[df2['Team']== team]['Player'].sort_values().unique())

# Load the appropriate dataset based on selection
if team == team_name1:
    shots_df = process_shots_df(df1)
    shotmap_df = shotmap_df1  # Use shotmap_df1 for shots_map function
    color = '#05b7e0'  # Customize team color for team_name1
else:
    shots_df = process_shots_df(df2)
    shotmap_df = shotmap_df2  # Use shotmap_df2 for shots_map function
    color = '#ff5733'  # Customize team color for team_name2

# Extract list of players from the dataset
#players = shots_df['Player'].unique()

# Selectbox to choose a player
#selected_player = st.selectbox('Select a player', players)

# Filter shots_df to include only shots from the selected player
player_shots_df = shots_df[shots_df['Player'] == selected_player]

# Plot the zone-based shot map in the first column
with col1:
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    # Call your zone-based plot function here (if you have one)
    plot_shots_map(player_shots_df, color, ax1)
    total_shots = len(player_shots_df)  # Calculate the total shots for the player/team
    ax1.set_title('Shot zones % wise', fontsize=20)  # Show total shots in the plot title
    st.pyplot(fig1)

# Plot the shot map in the second column
with col2:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    # Use the selected shotmap_df (either shotmap_df1 or shotmap_df2)
    shots_map(shotmap_df[shotmap_df['Player'] == selected_player], color, ax2)
    ax2.set_title(f'Total Shots: {total_shots}', fontsize=20)  # Show total shots in the plot title
    st.pyplot(fig2)

##########################
# STATS AND RATINGS #
##########################
st.title(f"Stats for {selected_player}")
# Load the model
model = pickle.load(open('player_ratings_prediction.pickle', 'rb'))

# Load the columns from the JSON file
with open('model_columns.json', 'r') as file:
    model_columns = json.load(file)

# Load event data for both teams (replace with actual file paths)
df_team1 = pd.read_csv('Minerva_vs_Sudeva_Minerva_data.csv')
df_team2 = pd.read_csv('Minerva_vs_Sudeva_Sudeva_data.csv')

df1 = main_minerva.process_data(df_team1)
df2 = main_minerva.process_data(df_team2)



mins_df1 = main_minerva.mins_secs(df1)
mins_df2 = main_minerva.mins_secs(df2)

team1 = main_minerva.add_passer_recipient_columns(mins_df1)
team2 = main_minerva.add_passer_recipient_columns(mins_df2)

# Calculate goals for each team
team1_goals = main_minerva.calculate_goals(df_team1)
team2_goals = main_minerva.calculate_goals(df_team2)

# Assign win/loss based on goals
if team1_goals > team2_goals:
    df_team1['win'] = 1
    df_team1['lost'] = 0
    df_team2['win'] = 0
    df_team2['lost'] = 1
elif team2_goals > team1_goals:
    df_team1['win'] = 0
    df_team1['lost'] = 1
    df_team2['win'] = 1
    df_team2['lost'] = 0
else:
    # In case of a draw
    df_team1['win'] = 0
    df_team1['lost'] = 0
    df_team2['win'] = 0
    df_team2['lost'] = 0


player_stats1 = main_minerva.calculate_player_stats(df1)
player_stats2 = main_minerva.calculate_player_stats(df2)

player_stats1['is_home_team'] = 1 # home team
player_stats2['is_home_team'] = 0 # away team

# Calculate goals for each team
team1_goals = main_minerva.calculate_goals(df_team1)
team2_goals = main_minerva.calculate_goals(df_team2)

# Assign win/loss based on goals
if team1_goals > team2_goals:
    player_stats1['win'] = 1
    player_stats1['lost'] = 0
    player_stats2['win'] = 0
    player_stats2['lost'] = 1
elif team2_goals > team1_goals:
    player_stats1['win'] = 0
    player_stats1['lost'] = 1
    player_stats2['win'] = 1
    player_stats2['lost'] = 0
else:
    # In case of a draw
    player_stats1['win'] = 0
    player_stats1['lost'] = 0
    player_stats2['win'] = 0
    player_stats2['lost'] = 0

network_metrics_df1 = main_minerva.calculate_network_metrics(team1)
network_metrics_df2 = main_minerva.calculate_network_metrics(team2)

player_net_metrics1 = main_minerva.unique_jerseys(team1, network_metrics_df1)
player_net_metrics2 = main_minerva.unique_jerseys(team2, network_metrics_df2)

#st.write(print(type(player_net_metrics1)))
#st.write(print(type(player_stats1)))

def final_df(network_stats_df, player_stats_df):
    final_df = network_stats_df.merge(player_stats_df, on='Player', how='left')
    final_df.drop(columns =['Jersey'])
    final_df = final_df[['Player','goals', 'assists', 'shots_ontarget',
       'shots_offtarget', 'shotsblocked', 'drib_success', 'drib_unsuccess',
       'keypasses', 'touches', 'passes_acc', 'passes_inacc', 'lballs_acc',
       'lballs_inacc', 'grduels_w', 'grduels_l', 'aerials_w', 'aerials_l',
       'fouls', 'clearances', 'interceptions', 'tackles', 'dribbled_past',
       'tballs_acc', 'tballs_inacc', 'ycards', 'rcards', 'offsides',
       'saved_pen', 'missed_penalties', 'owngoals', 'degree_centrality',
       'betweenness_centrality', 'closeness_centrality', 'flow_centrality',
       'flow_success', 'betweenness2goals', 'win', 'lost', 'is_home_team',
       'minutesPlayed', 'game_duration', 'crosses_att', 'saves_made',
       'goals_conceded']]
    return final_df

input_df1 = final_df(player_net_metrics1, player_stats1)
input_df2 = final_df(player_net_metrics2, player_stats2)


# Combine both teams' player statistics into one DataFrame
player_stats = pd.concat([player_stats1, player_stats2])


# Assuming df_team1 and df_team2 have the 'Team' column that contains the team names
#team_name1 = df_team1['Team'].iloc[0]
#team_name2 = df_team2['Team'].iloc[0]

# Sidebar to select team
#team_option = st.selectbox('Select Team', [team_name1, team_name2])

# Filter players based on selected team
if team == team_name1:
    team_data = input_df1[input_df1['is_home_team'] == 1]
    # Show only players from team_name1
    #player_option = st.selectbox('Select a player', input_df1['Player'].sort_values().unique())
else:
    team_data = input_df2[input_df2['is_home_team'] == 0]
    # Show only players from team_name2
    #player_option = st.selectbox('Select a player', input_df2['Player'].sort_values().unique())


# Combine both teams' player statistics into one DataFrame
player_stats = pd.concat([player_stats1, player_stats2])

input_df = pd.concat([input_df1, input_df2])

# Assuming 'Player' is in input_df, filter data for the selected player
player_data = input_df[input_df['Player'] == selected_player]

# Drop the 'Player' column (since the model doesn't need this for prediction)
player_data_for_prediction = player_data.drop(columns=['Player'])

# Predict player rating using the model
predicted_rating = model.predict(player_data_for_prediction)

# Display the predicted rating
#st.write(f"### Rating for {player_option}: {predicted_rating[0]:.2f}")
minutes_played = player_data.loc[player_data['Player'] == selected_player, 'minutesPlayed'].values[0]
accurate_passes = player_data.loc[player_data['Player'] == selected_player, 'passes_acc'].values[0]
inaccurate_passes = player_data.loc[player_data['Player'] == selected_player, 'passes_inacc'].values[0]
shots_on_target = player_data.loc[player_data['Player'] == selected_player, 'shots_ontarget'].values[0]
shots_off_target = player_data.loc[player_data['Player'] == selected_player, 'shots_offtarget'].values[0]
shots_blocked = player_data.loc[player_data['Player'] == selected_player, 'shotsblocked'].values[0]

passing_accuracy = round((accurate_passes / (inaccurate_passes + accurate_passes)) * 100)
#st.write(f"**Goals**: {inaccurate_passes}")

col1, col2, col3 = st.columns(3)

with col1:
    ui.metric_card(title="Player Rating", content=f'{predicted_rating[0]:.2f}', description=f"{selected_player}", key="card1")
with col2:
    ui.metric_card(title="Passing Stats", content=f'{passing_accuracy}%', description=f"Total Passes: {inaccurate_passes + accurate_passes}", key="card2")
with col3:
    ui.metric_card(title="Total Shots", content=f"{shots_on_target + shots_off_target +shots_blocked}", description=f'Off Target: {shots_off_target}', key="card3")

col1, col2 = st.columns(2)

# Display player statistics once the player is selected
if selected_player:
    player_stats_selected = player_stats[player_stats['Player'] == selected_player].iloc[0]
    
    
    # Defensive stats
    defensive_stats = {
        "Metric": ["Fouls", "Interceptions", "Clearances", "Tackles", "Ground Duels Won", "Ground Duels Lost", "Aerial Duels Won", "Aerial Duels Lost"],
        "Value": [
            player_stats_selected['fouls'], 
            player_stats_selected['interceptions'], 
            player_stats_selected['clearances'], 
            player_stats_selected['tackles'], 
            player_stats_selected['grduels_w'], 
            player_stats_selected['grduels_l'], 
            player_stats_selected['aerials_w'], 
            player_stats_selected['aerials_l']
        ]
    }
    defensive_stats_df = pd.DataFrame(defensive_stats)

    # Standard stats
    standard_stats = {
        "Metric": ["Goals", "Assists", "Yellow Cards", "Red Cards", "Own Goals", "Minutes Played","Successful Dribbles","Unsuccessful Dribbles"],
        "Value": [
            player_stats_selected['goals'], 
            player_stats_selected['assists'], 
            player_stats_selected['ycards'], 
            player_stats_selected['rcards'], 
            player_stats_selected['owngoals'], 
            player_stats_selected['minutesPlayed'],
            player_stats_selected['drib_success'], 
            player_stats_selected['drib_unsuccess']
        ]
    }
    standard_stats_df = pd.DataFrame(standard_stats)

    # Display the tables in two columns
    with col1:
        st.write("#### Defensive Stats")
        st.table(defensive_stats_df)  # Static table for defensive stats

    with col2:
        st.write("#### Standard Stats")
        st.table(standard_stats_df)  # Static table for standard stats

# Combine dataframes for both teams
df = pd.concat([df_team1, df_team2], ignore_index=True)

# Filter the event data
passes_df = df[(df['Event']).isin(['Pass', 'Long Kick'])]  # Save pass data
shots_df = df[df['Event'] == 'Shot'].reset_index(drop=True)  # Save shot data

pitch = VerticalPitch(pitch_type='statsbomb')
#x = main_minerva.passmaps(passes_df, team, selected_player, pitch, 'Passes')
#with col1:
 #   x = main_minerva.passmaps(passes_df, team, selected_player, pitch, 'Passes')    
  #  print(x)

# Call the function to get different types of pass dataframes
passes_df, keypass_df, crosses_df, setpieces_df, prg_passes_df, passes_final_third_df = main_minerva.generate_pass_types(df1, df2)

# Let the user select the type of passes to visualize
pass_types = ['Passes','Progressive Passes', 'Crosses', 'Key Passes', 'Passes Entering Final 3rd']
#with col2:    
selected_pass_type = st.selectbox('Select Pass Type to Visualize:', pass_types)

# Add the passmaps visualizations in the columns
#with col2:
if selected_pass_type == 'Progressive Passes':
    st.write(f"### {selected_player} Progressive Passes")
    main_minerva.passmaps(prg_passes_df, team, selected_player, pitch, 'Progressive passes')

elif selected_pass_type == 'Passes':
    st.write(f"### {selected_player} Crosses")
    main_minerva.passmaps(passes_df, team, selected_player, pitch, 'Passes')

elif selected_pass_type == 'Crosses':
    st.write(f"### {selected_player} Crosses")
    main_minerva.passmaps(crosses_df, team, selected_player, pitch, 'Crosses')

elif selected_pass_type == 'Key Passes':
    st.write(f"### {selected_player} Key Passes")
    main_minerva.passmaps(keypass_df, team, selected_player, pitch, 'Key passes')

elif selected_pass_type == 'Passes Entering Final 3rd':
    st.write(f"### {selected_player} Passes Entering Final 3rd")
    main_minerva.passmaps(passes_final_third_df, team, selected_player, pitch, 'Final Third')

