import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as path_effects
from mplsoccer import VerticalPitch, Pitch
from matplotlib.lines import Line2D
import main_minerva
import streamlit_shadcn_ui as ui
import requests

st.set_page_config(
    page_title="Minerva App",
    page_icon="👋🏼",
    layout="centered"
)

@st.cache
def fetch_csv_from_drive(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Convert content to a pandas DataFrame
        df = pd.read_csv(url)
        return df
    else:
        st.error("Failed to fetch the CSV file from Google Drive")
        return None

# Access the secret links
drive_csv_url_1 = st.secrets["google_drive"]["csv_file_1"]
drive_csv_url_2 = st.secrets["google_drive"]["csv_file_2"]

# Fetch CSV files from Google Drive
df_team1 = fetch_csv_from_drive(drive_csv_url_1)
df_team2 = fetch_csv_from_drive(drive_csv_url_2)

# Load event data for both teams
#df_team1 = pd.read_csv('Minerva_vs_Sudeva_Minerva_data.csv')
#df_team2 = pd.read_csv('Minerva_vs_Sudeva_Sudeva_data.csv')  # Replace with the actual file for the second team

# Assuming df_team1 and df_team2 have the 'Team' column that contains the team names
team_name1 = df_team1['Team'].iloc[0]
team_name2 = df_team2['Team'].iloc[0]

# Combine dataframes for both teams
df = pd.concat([df_team1, df_team2], ignore_index=True)

# Filter the event data
passes_df = df[(df['Event']).isin(['Pass', 'Long Kick'])]  # Save pass data
#shots_df = df[df['Event'] == 'Shot'].reset_index(drop=True)  # Save shot data

st.title(f'{team_name1} vs {team_name2}\nMatch Analysis')
#st.subheader("Filter to any team/player to see all of their shots taken")

# Create two columns for the visualizations
col1, col2 = st.columns(2)

# Selectbox to choose the team
team = st.selectbox('Select a team', (team_name1, team_name2))

# avg locations viz

x = main_minerva.process_data(df_team1)
y = main_minerva.process_data(df_team2)

mins_df1 = main_minerva.mins_secs(x)
mins_df2 = main_minerva.mins_secs(y)

p = main_minerva.add_passer_recipient_columns(mins_df1, event_col='Event', outcome_col='Outcome', jersey_col='Jersey')
q = main_minerva.add_passer_recipient_columns(mins_df2, event_col='Event', outcome_col='Outcome', jersey_col='Jersey')

#st.dataframe(q)

avg_team1_all, avg_team1_def, avg_team2_all, avg_team2_def = main_minerva.avg_loc_for_teams(p, q)
#st.dataframe(avg_team1_all)        

p['Passer'] = p['Passer'].replace(-1, np.nan)
p['Recipient'] = p['Recipient'].replace(-1, np.nan)

# Convert the column to integer dtype
p['Passer'] = p['Passer'].astype('Int64')
p['Recipient'] = p['Recipient'].astype('Int64')

q['Passer'] = q['Passer'].replace(-1, np.nan)
q['Recipient'] = q['Recipient'].replace(-1, np.nan)

# Convert the column to integer dtype
q['Passer'] = q['Passer'].astype('Int64')
q['Recipient'] = q['Recipient'].astype('Int64')

with col1:
    main_minerva.visualize_passing_network(p, team_name1, col='#05b7e0')        
with col2:
    main_minerva.visualize_passing_network(q, team_name2, col='#ff5733') 

shotmap_df1 = df_team1[df_team1['Event'] == 'Shot']
shotmap_df2 = df_team2[df_team2['Event'] == 'Shot']

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
        fig, ax = plt.subplots(figsize=(16, 11))
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

# Define columns for the layout
col1, col2 = st.columns(2)

# Selectbox to choose the team
#team = st.selectbox('Select a team', (team_name1, team_name2))

# Load the appropriate dataset based on selection
if team == team_name1:
    shots_df = process_shots_df(df_team1)
    shotmap_df = shotmap_df1  # Use shotmap_df1 for shots_map function
    color = '#05b7e0'  # Customize team color for team_name1
else:
    shots_df = process_shots_df(df_team2)
    shotmap_df = shotmap_df2  # Use shotmap_df2 for shots_map function
    color = '#ff5733'  # Customize team color for team_name2

# Extract list of players from the dataset
#players = shots_df['Player'].unique()

# Selectbox to choose a player
#selected_player = st.selectbox('Select a player', players)

# Filter shots_df to include only shots from the selected player
team_shots_df = shots_df[shots_df['Team'] == team]

# Plot the zone-based shot map in the first column
with col1:
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    # Call your zone-based plot function here (if you have one)
    plot_shots_map(team_shots_df, color, ax1)
    total_shots = len(team_shots_df)  # Calculate the total shots for the player/team
    ax1.set_title('Shot zones % wise', fontsize=20)  # Show total shots in the plot title
    st.pyplot(fig1)

# Plot the shot map in the second column
with col2:
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    # Use the selected shotmap_df (either shotmap_df1 or shotmap_df2)
    shots_map(shotmap_df[shotmap_df['Team'] == team], color, ax2)
    ax2.set_title(f'Total Shots: {total_shots}', fontsize=20)  # Show total shots in the plot title
    st.pyplot(fig2)

# team stats
with col1:
    st.subheader(f'{team_name1} ' "Statistics")
    st.dataframe(main_minerva.calculate_team_stats(df_team1, df_team2))
with col2:
    st.subheader(f'{team_name2} ' "Statistics")
    st.dataframe(main_minerva.calculate_team_stats(df_team2, df_team1))


avg_loc = st.selectbox('Select a type', ('All Events Average locations', 'Defensive average locations'))

with col1:
    # Selectbox to choose the team
    #avg_loc1 = st.selectbox('Select a type', ('All Events Average locations', 'Defensive average locations'))
    # Load the appropriate dataset based on selection
    if avg_loc == 'All Events Average locations':
        main_minerva.avg_positions(avg_team1_all, col='#05b7e0', team=team_name1)
    else:
        main_minerva.avg_positions(avg_team1_def, col='#05b7e0', team=team_name1)
        
   #main_minerva.avg_positions(avg_team1_all, col='blue', team=team_name1)
with col2:
    # Selectbox to choose the team
    #avg_loc2 = st.selectbox('Select a type', ('All Events Average locations', 'Defensive average locations'))
    # Load the appropriate dataset based on selection
    if avg_loc == 'All Events Average locations':
        main_minerva.avg_positions(avg_team2_all, col='#ff5733', team=team_name2)
    else:
        main_minerva.avg_positions(avg_team2_def, col='#ff5733', team=team_name2)



st.logo("ballers_logo.png")
st.sidebar.markdown("@baller_metrics - Follow us on Instagram")
ui.link_button(text="Visit our Instagram", url="https://www.instagram.com/baller_metrics/?igsh=MnBkOWw0aG01MXd4%2Fstreamlit-shadcn-ui", key="link_btn")
       
