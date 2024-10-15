import re
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import patheffects as path_effects
from mplsoccer import VerticalPitch, Pitch
import streamlit as st

def process_data(dataframe):
    # Regular expression pattern to extract minutes and seconds
    pattern = r'(\d+):(\d+)'

    # Function to convert timestamps to "mm:ss" format
    def convert_to_mmss(timestamp):
        match = re.match(pattern, timestamp)
        if match:
            minutes, seconds = match.groups()
            return '{:02d}:{:02d}'.format(int(minutes), int(seconds))
        else:
            return None

    # Convert timestamps in 'Time' column to "mm:ss" format
    dataframe['Time'] = dataframe['Time'].apply(convert_to_mmss)

    return dataframe

# Split the 'Time' column into 'Minutes' and 'Seconds'
def mins_secs(df_test):
    # Split the 'Time' column into 'Minutes' and 'Seconds'
    df_test[['Minutes', 'Seconds']] = df_test['Time'].str.split(':', expand=True)

    # Convert 'Minutes' and 'Seconds' columns to numeric type
    df_test['Minutes'] = pd.to_numeric(df_test['Minutes'])
    df_test['Seconds'] = pd.to_numeric(df_test['Seconds'])
    return df_test

def add_passer_recipient_columns(df, event_col='Event', outcome_col='Outcome', jersey_col='Jersey'):
    # Initialize empty lists for recipients and passers
    recipient_list = []
    passer_list = []

    # Convert 'Jersey' column to numeric to handle NaN values
    df[jersey_col] = pd.to_numeric(df[jersey_col], errors='coerce')

    # Iterate through the DataFrame
    for index, row in df.iterrows():
        # Check if the 'Event' is either 'Pass', 'Long kick', or 'Head Pass' and the 'Outcome' is 'Complete'
        if row[event_col] in ['Pass', 'Long kick', 'Head Pass'] and row[outcome_col] == 'Complete':
            # Append the passer (current row's jersey number)
            passer_list.append(int(row[jersey_col]))
            # Check if there is a next row
            if index + 1 < len(df):
                # Append the recipient (next row's jersey number)
                recipient_list.append(int(df.loc[index + 1, jersey_col]))
            else:
                # If no next row, append NaN to the recipient list
                recipient_list.append(np.nan)
        else:
            # For other events, append NaN to both lists
            passer_list.append(np.nan)
            recipient_list.append(np.nan)

    # Add 'Recipient' and 'Passer' columns to the DataFrame
    df['Recipient'] = recipient_list
    df['Passer'] = passer_list

    return df

def calculate_player_stats(df):
    """
    Calculate various football statistics for each player from the provided event data.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing event data with columns such as 'Player', 'Event', 'Outcome', 'Assist', 'Key Pass', etc.

    Returns:
    pd.DataFrame: A DataFrame containing aggregated statistics for each player.
    """

    # Define the boundaries of the box
    #x_min, x_max = 18, 62
    #y_min, y_max = 102, 120
    
    # Set 'is_home' to 1 if df equals df1, else set it to 0
    #is_home_value = 1 if df.equals(df1) else 0
    
    # Group by Player to calculate statistics for each player
    player_stats = df.groupby('Player').agg(
        goals=('Event', lambda x: ((x == 'Shot') & (df.loc[x.index, 'Outcome'] == 'Goal')).sum()),  # Goals
        assists=('Assist', lambda x: x.sum()),  # Assists
        shots_ontarget=('Event', lambda x: ((x == 'Shot') & (df.loc[x.index, 'Outcome'].isin(['Goal','Saved']))).sum()),  # Shots on Target
        shots_offtarget=('Event', lambda x: ((x == 'Shot') & (df.loc[x.index, 'Outcome'] == 'Off Target')).sum()),  # Shots Off Target
        shotsblocked=('Event', lambda x: ((x == 'Shot') & (df.loc[x.index, 'Outcome'] == 'Blocked')).sum()),  # Shots Blocked
        #chances2score=('Key Pass', lambda x: x.sum()),  # Key Passes leading to chances
        drib_success=('Event', lambda x: ((x == 'Offensive Duel') & 
                                          (df.loc[x.index, 'Outcome'] == 'Won') & 
                                          (df.loc[x.index, 'Take On'] == True)).sum()),  # Successful Dribbles
        drib_unsuccess=('Event', lambda x: ((x == 'Offensive Duel') & 
                                            (df.loc[x.index, 'Outcome'] == 'Lost') & 
                                            (df.loc[x.index, 'Take On'] == True)).sum()),  # Unsuccessful Dribbles
        keypasses=('Key Pass', lambda x: x.sum()),  # Key Passes
        touches=('Event', 'count'),  # Total Touches (all events)
        passes_acc=('Event', lambda x: ((x == 'Pass') & (df.loc[x.index, 'Outcome'] == 'Complete')).sum()),  # Accurate Passes
        passes_inacc=('Event', lambda x: ((x == 'Pass') & (df.loc[x.index, 'Outcome'] == 'Incomplete')).sum()),  # Inaccurate Passes
        crosses_att=('Event', lambda x: (x == 'Cross').sum()),  # Total Crosses Attempted
        lballs_acc=('Event', lambda x: ((x == 'Long Kick') & (df.loc[x.index, 'Outcome'] == 'Complete')).sum()),  # Accurate Long Balls
        lballs_inacc=('Event', lambda x: ((x == 'Long Kick') & (df.loc[x.index, 'Outcome'] == 'Incomplete')).sum()),  # Inaccurate Long Balls
        grduels_w=('Event', lambda x: ((x.isin(['Offensive Duel', 'Defensive Duel'])) & (df.loc[x.index, 'Outcome'] == 'Won')).sum()),  # Ground Duels Won
        grduels_l=('Event', lambda x: ((x.isin(['Offensive Duel', 'Defensive Duel'])) & (df.loc[x.index, 'Outcome'] == 'Lost')).sum()),  # Ground Duels Lost
        aerials_w=('Event', lambda x: ((x == 'Air Duel') & (df.loc[x.index, 'Outcome'] == 'Won')).sum()),  # Aerial Duels Won
        aerials_l=('Event', lambda x: ((x == 'Air Duel') & (df.loc[x.index, 'Outcome'] == 'Lost')).sum()),  # Aerial Duels Lost
        fouls=('Event', lambda x: (x == 'Foul').sum()),  # Fouls
        clearances=('Clearance', lambda x: x.sum()),  # Clearances
        interceptions=('Event', lambda x: (x == 'Interception').sum()),  # Interceptions
        tackles=('Event', lambda x: ((x == 'Defensive Duel') & (df.loc[x.index, 'Outcome'] == 'Won')).sum()),  # Tackles
        dribbled_past=('Event', lambda x: ((x == 'Defensive Duel') & 
                                           (df.loc[x.index, 'Outcome'] == 'Lost') & 
                                           (df.loc[x.index, 'Take On'] == True)).sum()),  # Dribbled Past
        tballs_acc=('Event', lambda x: ((x == 'Pass') & 
                                        (df.loc[x.index, 'Outcome'] == 'Complete') & 
                                        (df.loc[x.index, 'Through Pass'] == True)).sum()),  # Accurate Through Balls
        tballs_inacc=('Event', lambda x: ((x == 'Pass') & 
                                          (df.loc[x.index, 'Outcome'] == 'Incomplete') & 
                                          (df.loc[x.index, 'Through Pass'] == True)).sum()),  # Inaccurate Through Balls
        ycards=('Outcome', lambda x: (x == 'Yellow Card').sum()),  # Yellow Cards
        rcards=('Outcome', lambda x: (x == 'Red Card').sum()),  # Red Cards
        offsides=('Event', lambda x: (x == 'Offside').sum()),  # Offsides
        saved_pen=('Event', lambda x: ((x == 'Penalty') & (df.loc[x.index, 'Outcome'] == 'Saved')).sum()),  # Penalties Saved
        missed_penalties=('Event', lambda x: ((x == 'Penalty') & (df.loc[x.index, 'Outcome'].isin(['Off Target','Saved']))).sum()),  # Missed Penalties
        owngoals=('Outcome', lambda x: (x == 'Own Goal').sum()),  # Own Goals
        win=('win', lambda x: (x == 1).sum()),  # Matches Won
        lost=('lost', lambda x: (x == 1).sum()),  # Matches Lost
        minutesPlayed=('Minutes', lambda x: x.max() - x.min()),  # Minutes Played
        game_duration=('Minutes', lambda x: df['Minutes'].max()), 
        goals_conceded=('Event', lambda x: ((x == 'Save') & (df.loc[x.index, 'Outcome'] == 'Goal')).sum()),
        saves_made=('Event', lambda x: ((x == 'Save') & (df.loc[x.index, 'Outcome'] == 'Saved')).sum())
        ).reset_index()        
    
    player_stats['assists'] = player_stats['assists'].astype(int)
    player_stats['keypasses'] = player_stats['keypasses'].astype(int)

    return player_stats


def calculate_network_metrics(df):
    # Filter the dataframe to focus only on passes (Pass, Long Kick, Head Pass)
    pass_df = df[df['Event'].isin(['Pass', 'Long Kick', 'Head Pass']) & (df['Outcome'] == 'Complete')]

    # Create a directed graph for the pass network
    G = nx.DiGraph()

    # Add edges to the graph: 'Passer' -> 'Recipient'
    for index, row in pass_df.iterrows():
        passer = row['Passer']
        recipient = row['Recipient']
        if pd.notna(passer) and pd.notna(recipient):
            G.add_edge(passer, recipient)

    # 1. Degree Centrality
    degree_centrality = nx.degree_centrality(G)

    # 2. Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(G)

    # 3. Closeness Centrality
    closeness_centrality = nx.closeness_centrality(G)

    # 4. Flow Centrality
    # Adding weight based on the number of successful passes between the same pair of players
    for u, v, data in G.edges(data=True):
        data['weight'] = pass_df[(pass_df['Passer'] == u) & (pass_df['Recipient'] == v)].shape[0]
    flow_centrality = nx.betweenness_centrality(G, weight='weight')

    # 5. Flow Success
    # Measure flow success based on pass sequences that lead to key events like 'Key Pass', 'Assist', 'Goal'
    flow_success = {}
    successful_passes = df[(df['Key Pass'] == True) | (df['Assist'] == True) | (df['Outcome'] == 'Goal')]
    for player in G.nodes():
        success_count = successful_passes[successful_passes['Passer'] == player].shape[0]
        total_passes = pass_df[pass_df['Passer'] == player].shape[0]
        flow_success[player] = success_count / total_passes if total_passes > 0 else 0

    # 6. Betweenness2Goals
    # Counting how often a player bridges passes that lead to goals
    goal_events = df[df['Outcome'] == 'Goal']
    betweenness2goals = {}
    for player in G.nodes():
        goal_related_passes = goal_events[goal_events['Passer'] == player].shape[0]
        betweenness2goals[player] = betweenness_centrality[player] * goal_related_passes

    # Create a DataFrame to combine all the metrics
    metrics_df = pd.DataFrame({
        'Jersey': list(G.nodes()),
        'degree_centrality': pd.Series(degree_centrality),
        'betweenness_centrality': pd.Series(betweenness_centrality),
        'closeness_centrality': pd.Series(closeness_centrality),
        'flow_centrality': pd.Series(flow_centrality),
        'flow_success': pd.Series(flow_success),
        'betweenness2goals': pd.Series(betweenness2goals)
    }).fillna(0)  # Fill NaN values with 0 for players who may not have contributed to certain metrics

    # Convert 'Player' (Jersey) column to integers
    metrics_df['Jersey'] = metrics_df['Jersey'].astype(int)
    return metrics_df

# Extract unique 'Player_name' and 'Jersey' pairs from event_df
def unique_jerseys(df, network_df):
    unique_jerseys = df[['Player', 'Jersey']].drop_duplicates()
    network_df = network_df.merge(unique_jerseys, on='Jersey', how='right')
    return network_df
# Define a function to count goals for a team
def calculate_goals(df):
    return df[(df['Event'] == 'Shot') & (df['Outcome'] == 'Goal')].shape[0]

# STATS TABLE #

def calculate_team_stats(df, df_opp):
    stats_data = {
        "Statistic": [],
        "Value": []
    }

    # PASSES COMPLETED / PASSES ATTEMPTED
    passes_completed = df[(df['Event'].isin(['Pass', 'Long Kick', 'Head Pass'])) & (df['Outcome'] == 'Complete')]
    passes_attempted = df[(df['Event'].isin(['Pass', 'Long Kick', 'Head Pass']))]
    passes_completed_count = passes_completed.shape[0]
    passes_attempted_count = passes_attempted.shape[0]
    pass_completion_rate = passes_completed_count / passes_attempted_count if passes_attempted_count != 0 else 0
    
    stats_data["Statistic"].append("PASSES COMPLETED / PASSES ATTEMPTED")
    stats_data["Value"].append(f"{passes_completed_count} / {passes_attempted_count}")

    stats_data["Statistic"].append("PASSING ACCURACY")
    stats_data["Value"].append("{:.2f}%".format(pass_completion_rate * 100))

    # TOTAL SHOTS
    total_shots = df[df['Event'] == 'Shot'].shape[0]
    stats_data["Statistic"].append("TOTAL SHOTS")
    stats_data["Value"].append(total_shots)

    # KEY PASSES
    key_passes = df['Key Pass'].sum()
    stats_data["Statistic"].append("KEY PASSES")
    stats_data["Value"].append(key_passes)

    # DUELS, DUELS WON / DUELS WON %
    duels = df[df['Event'].isin(['Offensive Duel', 'Defensive Duel', 'Air Duel'])]
    duels_won = duels[duels['Outcome'] == 'Won']
    duels_lost = duels[duels['Outcome'] == 'Lost']
    duels_count = duels.shape[0]
    duels_won_count = duels_won.shape[0]
    duels_lost_count = duels_lost.shape[0]
    duels_won_percentage = (duels_won_count / duels_count) * 100 if duels_count != 0 else 0
    
    stats_data["Statistic"].append("DUELS")
    stats_data["Value"].append(duels_count)

    stats_data["Statistic"].append("DUELS WON / DUELS LOST")
    stats_data["Value"].append(f"{duels_won_count} / {duels_lost_count}")

    stats_data["Statistic"].append("DUELS WON %")
    stats_data["Value"].append("{:.2f}%".format(duels_won_percentage))

    # INTERCEPTIONS
    interceptions = df[df['Event'] == 'Interception'].shape[0]
    stats_data["Statistic"].append("INTERCEPTIONS")
    stats_data["Value"].append(interceptions)

    # CORNERS
    corners = df[df['Event'] == 'Corner'].shape[0]
    stats_data["Statistic"].append("CORNERS")
    stats_data["Value"].append(corners)

    # FREE KICKS
    free_kicks = df[df['Event'] == 'Free Kick'].shape[0]
    stats_data["Statistic"].append("FREE KICKS")
    stats_data["Value"].append(free_kicks)

    # THROW-INS
    throw_ins = df[df['Event'] == 'Throw In'].shape[0]
    stats_data["Statistic"].append("THROW-INS")
    stats_data["Value"].append(throw_ins)

    # OFFSIDES
    offsides = df[df['Event'] == 'Offside'].shape[0]
    stats_data["Statistic"].append("OFFSIDES")
    stats_data["Value"].append(offsides)

    # FOULS, FOULS SUFFERED
    fouls_committed = df[df['Event'] == 'Foul'].shape[0]
    #fouls_suffered = opponent_df[opponent_df['Event'] == 'Foul'].shape[0]  # Count fouls committed by the opponent
    stats_data["Statistic"].append("FOULS")
    stats_data["Value"].append(fouls_committed)

    fouls_suffered = df_opp[df_opp['Event'] == 'Foul'].shape[0]
    stats_data["Statistic"].append("FOULS SUFFERED")
    stats_data["Value"].append(fouls_suffered)

    # YELLOW CARDS
    yellow_cards = df[df['Outcome'] == 'Yellow Card'].shape[0]
    stats_data["Statistic"].append("YELLOW CARDS")
    stats_data["Value"].append(yellow_cards)

    # RED CARDS
    red_cards = df[df['Outcome'] == 'Red Card'].shape[0]
    stats_data["Statistic"].append("RED CARDS")
    stats_data["Value"].append(red_cards)

    # Create the dataframe
    general_stats_df = pd.DataFrame(stats_data)
    return general_stats_df

############################
# SHOTS FUNCTIONS #
############################

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

def categorize_shot(row, zones):
    for zone, bounds in zones.items():
        if (bounds['X_min'] <= row['Y'] <= bounds['X_max'] and
            bounds['Y_min'] <= row['X'] <= bounds['Y_max']):
            return zone
    return np.nan  # If the shot doesn't fall into any defined zone

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

######################
# Passmaps functions #
######################
            
def passmaps(df_pass, team, player, pitch, pass_type):
        """Plot the passmap for the selected team or player."""
        
        # Ensure that a team is selected
        if not team:
            st.warning('Please select a team to view the pass map.')
            return  # Exit the function if no team is selected
        
        # Filter for the selected team
        df_pass = df_pass[df_pass["Team"] == team]
        
        # If a player is selected, filter for that player
        if player:
            df_pass = df_pass[df_pass["Player"] == player]
        
        # Check if the filtered dataframe is empty
        if len(df_pass) == 0:
            st.warning(f'No pass data available for {team} {player if player else ""}')
            return  # Exit the function if no data is available
        
        # Separate completed and incomplete passes
        pass_comp_df = df_pass[df_pass['Outcome'] == 'Complete']
        pass_incomp_df = df_pass[df_pass['Outcome'] == 'Incomplete']
        
        # Calculate pass accuracy
        pass_accuracy = round((len(pass_comp_df) / len(df_pass)) * 100)
        thresh = pass_accuracy - 70
        
        # Display pass stats
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Total Passes", value=len(df_pass))
        col2.metric(label="Complete", value=len(pass_comp_df))
        col3.metric(label="Passing accuracy", value=f'{pass_accuracy}%')#, delta= f'{thresh}%')

        # Create a new pitch for passes
        pitch = Pitch(pitch_color='#a5c771', line_color='white', positional=False, stripe=True, corner_arcs=True)
        fig_pass_map, ax_pass_map = pitch.draw(figsize=(8, 4))  # Smaller figure size

        # Plot completed passes
        pitch.arrows(pass_comp_df.X, pass_comp_df.Y, 
                    pass_comp_df.endX, pass_comp_df.endY, color="black", ax=ax_pass_map, width=1)
        pitch.scatter(pass_comp_df.X, pass_comp_df.Y, alpha=0.2, s=50, color="black", ax=ax_pass_map)  

        # Plot incomplete passes
        pitch.lines(pass_incomp_df.X, pass_incomp_df.Y,
                    pass_incomp_df.endX, pass_incomp_df.endY,
                    lw=1, color='grey', ax=ax_pass_map)
        
        pitch.scatter(pass_incomp_df.endX, pass_incomp_df.endY, s=20, color="grey", ax=ax_pass_map, marker='x')

        # Update the title dynamically
        #fig_pass_map.suptitle(f'{player if player else ""} {pass_type}', fontsize=15)
        
        # Show pass map plot
        st.pyplot(fig_pass_map)

# Generate types of passes

def generate_pass_types(df1, df2):
    # Merge the two dataframes
    merged_df = pd.concat([df1, df2])

    # Key passes (Passes and Long Kicks, Head Passes that are Key Passes)
    keypass_df = merged_df[(merged_df["Event"].isin(['Pass', 'Long Kick', 'Head Pass']) & merged_df['Key Pass'])]

    # Standard passes (excluding Long Kicks and Head Passes)
    passes_df = merged_df.loc[merged_df["Event"] == 'Pass']

    # Crosses
    crosses_df = merged_df.loc[merged_df["Event"] == 'Cross']

    # Set pieces (Free Kicks, Throw Ins, Goal Kicks, Corners)
    setpieces_df = merged_df.loc[merged_df["Event"].isin(['Free Kick', 'Throw In', 'Goal Kick', 'Corner'])]

    # Progressive passes (based on distance and a condition for being 'progressive')
    prg_df = merged_df[(merged_df['Event']).isin(['Pass', 'Long Kick'])]

    # Calculate distances from the beginning and end of the pass to a reference point (usually goal)
    prg_df['beginning'] = np.sqrt(np.square(120 - prg_df['X']) + np.square(40 - prg_df['Y']))
    prg_df['end'] = np.sqrt(np.square(120 - prg_df['endX']) + np.square(40 - prg_df['endY']))

    # Reset index to ensure clean data
    prg_df.reset_index(drop=True, inplace=True)

    # Define progressive passes: end distance is less than 75% of the beginning distance
    prg_df['progressive'] = [(prg_df['end'][x]) / (prg_df['beginning'][x]) < .75 for x in range(len(prg_df['beginning']))]

    # Filter only the progressive passes and remove any where X is greater than 100
    prg_passes_df = prg_df.loc[(prg_df['progressive'] == True) & (prg_df['X'] < 100)]

    #passes entering final third
    # Step 1: Define criteria for passes entering the final third
    passes_df['enters_final_third'] = (passes_df['X'] < 80) & (passes_df['endX'] > 80)
    # Step 2: Filter out passes entering the final third
    passes_final_third_df = passes_df[passes_df['enters_final_third']]

    #enters_final_third_df = (merged_df['X'] < 80) & (merged_df['endX'] > 80)
    # Return all the relevant DataFrames
    return passes_df, keypass_df, crosses_df, setpieces_df, prg_passes_df, passes_final_third_df
