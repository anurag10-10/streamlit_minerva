import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load the model
model = pickle.load(open('player_ratings_prediction.pickle', 'rb'))

# Load the columns from the JSON file
with open('model_columns.json', 'r') as file:
    model_columns = json.load(file)

# Load event data for both teams (replace with actual file paths)
df_team1 = pd.read_csv('PunjabFC_vs_Minerva_U15_PunjabFC_data.csv')
df_team2 = pd.read_csv('PunjabFC_vs_Minerva_U15_Minerva_data.csv')

# Define a function to count goals for a team
def calculate_goals(df):
    return df[(df['Event'] == 'Shot') & (df['Outcome'] == 'Goal')].shape[0]

# Calculate goals for each team
team1_goals = calculate_goals(df_team1)
team2_goals = calculate_goals(df_team2)

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
import re

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


# Process the data
df1 = process_data(df_team1)
df2 = process_data(df_team2)
# Split the 'Time' column into 'Minutes' and 'Seconds'
def mins_secs(df_test):
    # Split the 'Time' column into 'Minutes' and 'Seconds'
    df_test[['Minutes', 'Seconds']] = df_test['Time'].str.split(':', expand=True)

    # Convert 'Minutes' and 'Seconds' columns to numeric type
    df_test['Minutes'] = pd.to_numeric(df_test['Minutes'])
    df_test['Seconds'] = pd.to_numeric(df_test['Seconds'])
    return df_test
    
mins_df1 = mins_secs(df1)
mins_df2 = mins_secs(df2)
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
team1 = add_passer_recipient_columns(mins_df1)
team2 = add_passer_recipient_columns(mins_df2)

import pandas as pd
import numpy as np

# Load the event data CSV file into a dataframe
team1['is_home'] = 1
team2['is_home'] = 0



def calculate_player_stats(df):
    """
    Calculate various football statistics for each player from the provided event data.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing event data with columns such as 'Player', 'Event', 'Outcome', 'Assist', 'Key Pass', etc.

    Returns:
    pd.DataFrame: A DataFrame containing aggregated statistics for each player.
    """

    # Define the boundaries of the box
    x_min, x_max = 18, 62
    y_min, y_max = 102, 120
    
    # Set 'is_home' to 1 if df equals df1, else set it to 0
    is_home_value = 1 if df.equals(df1) else 0
    
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
    

    return player_stats

# Example usage:
player_stats1 = calculate_player_stats(df1)
player_stats2 = calculate_player_stats(df2)
player_stats1['is_home_team'] = 1 # home team
player_stats2['is_home_team'] = 0 # away team

# Define a function to count goals for a team
def calculate_goals(df):
    return df[(df['Event'] == 'Shot') & (df['Outcome'] == 'Goal')].shape[0]

# Calculate goals for each team
team1_goals = calculate_goals(df_team1)
team2_goals = calculate_goals(df_team2)

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

#import pandas as pd
import networkx as nx

def calculate_network_metrics(team_df, player_stats_df):
    """
    Function to calculate network metrics for a team's passes and combine with player statistics.
    
    Args:
    team_df (pd.DataFrame): Dataframe containing event data for the team (including passes).
    player_stats_df (pd.DataFrame): Dataframe containing player statistics.
    
    Returns:
    pd.DataFrame: A dataframe with calculated network metrics and player stats.
    """

    # Filter the dataframe to focus only on passes (Pass, Long Kick, Head Pass)
    pass_df = team_df[team_df['Event'].isin(['Pass', 'Long Kick', 'Head Pass']) & (team_df['Outcome'] == 'Complete')]

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

    # 4. Flow Centrality (weighted betweenness centrality)
    for u, v, data in G.edges(data=True):
        data['weight'] = pass_df[(pass_df['Passer'] == u) & (pass_df['Recipient'] == v)].shape[0]
    flow_centrality = nx.betweenness_centrality(G, weight='weight')

    # 5. Flow Success (successful passes leading to key events)
    flow_success = {}
    successful_passes = team_df[(team_df['Key Pass'] == True) | (team_df['Assist'] == True) | (team_df['Outcome'] == 'Goal')]
    for player in G.nodes():
        success_count = successful_passes[successful_passes['Passer'] == player].shape[0]
        total_passes = pass_df[pass_df['Passer'] == player].shape[0]
        flow_success[player] = success_count / total_passes if total_passes > 0 else 0

    # 6. Betweenness2Goals (betweenness related to goals)
    goal_events = team_df[team_df['Outcome'] == 'Goal']
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

    # Convert 'Jersey' column to integers
    metrics_df['Jersey'] = metrics_df['Jersey'].astype(int)

    # Extract unique 'Player' and 'Jersey' pairs from team_df
    unique_jerseys = team_df[['Player', 'Jersey']].drop_duplicates()

    # Merge the unique jersey mapping with metrics_df on 'Jersey'
    metrics_df = metrics_df.merge(unique_jerseys, on='Jersey', how='right')

    # Merge metrics_df with player_stats_df on 'Player'
    final_df = metrics_df.merge(player_stats_df, on='Player', how='left')

    # Drop unwanted columns and reorder the remaining ones
    final_df = final_df.drop(columns=['Jersey', 'Player'])
    final_df = final_df[[
        'goals', 'assists', 'shots_ontarget', 'shots_offtarget', 'shotsblocked', 
        'drib_success', 'drib_unsuccess', 'keypasses', 'touches', 'passes_acc', 
        'passes_inacc', 'lballs_acc', 'lballs_inacc', 'grduels_w', 'grduels_l', 
        'aerials_w', 'aerials_l', 'fouls', 'clearances', 'interceptions', 'tackles', 
        'dribbled_past', 'tballs_acc', 'tballs_inacc', 'ycards', 'rcards', 'offsides', 
        'saved_pen', 'missed_penalties', 'owngoals', 'degree_centrality', 
        'betweenness_centrality', 'closeness_centrality', 'flow_centrality', 
        'flow_success', 'betweenness2goals', 'win', 'lost', 'is_home_team', 
        'minutesPlayed', 'game_duration', 'crosses_att', 'saves_made', 'goals_conceded'
    ]]

    return final_df

final_metrics_df = calculate_network_metrics(team2, player_stats2)

st.dataframe(final_metrics_df)
# Combine both teams' player statistics into one DataFrame
player_stats = pd.concat([player_stats1, player_stats2])


team_name1 = df_team1['Team'].iloc[0]
team_name2 = df_team2['Team'].iloc[0]


# Sidebar to select team
team_option = st.selectbox('Select Team', [f'{team_name1}', f'{team_name2}'])

# Filter players based on selected team
if team_option == team_name1:
    team_data = final_metrics_df[final_metrics_df['is_home_team'] == 1]
else:
    team_data = final_metrics_df[final_metrics_df['is_home_team'] == 0]

# Select player based on filtered team data
player_option = st.selectbox('Select Player', final_metrics_df['Player'].unique())

# Display player statistics once the player is selected
if player_option:
    player_stats_selected = final_metrics_df[final_metrics_df['Player'] == player_option].iloc[0]

    # Display the player's stats in Streamlit
    st.write(f"### Statistics for {player_option}")
    st.write(f"**Goals**: {player_stats_selected['goals']}")
    st.write(f"**Assists**: {player_stats_selected['assists']}")
    st.write(f"**Shots on Target**: {player_stats_selected['shots_ontarget']}")
    st.write(f"**Shots Off Target**: {player_stats_selected['shots_offtarget']}")
    st.write(f"**Shots Blocked**: {player_stats_selected['shotsblocked']}")
    st.write(f"**Dribbles Successful**: {player_stats_selected['drib_success']}")
    st.write(f"**Dribbles Unsuccessful**: {player_stats_selected['drib_unsuccess']}")
    st.write(f"**Key Passes**: {player_stats_selected['keypasses']}")
    st.write(f"**Touches**: {player_stats_selected['touches']}")
    st.write(f"**Accurate Passes**: {player_stats_selected['passes_acc']}")
    st.write(f"**Inaccurate Passes**: {player_stats_selected['passes_inacc']}")
    st.write(f"**Accurate Long Balls**: {player_stats_selected['lballs_acc']}")
    st.write(f"**Inaccurate Long Balls**: {player_stats_selected['lballs_inacc']}")
    st.write(f"**Fouls**: {player_stats_selected['fouls']}")
    st.write(f"**Interceptions**: {player_stats_selected['interceptions']}")
    st.write(f"**Clearances**: {player_stats_selected['clearances']}")
    st.write(f"**Tackles**: {player_stats_selected['tackles']}")
    st.write(f"**Ground Duels Won**: {player_stats_selected['grduels_w']}")
    st.write(f"**Ground Duels Lost**: {player_stats_selected['grduels_l']}")
    st.write(f"**Aerial Duels Won**: {player_stats_selected['aerials_w']}")
    st.write(f"**Aerial Duels Lost**: {player_stats_selected['aerials_l']}")
    st.write(f"**Yellow Cards**: {player_stats_selected['ycards']}")
    st.write(f"**Red Cards**: {player_stats_selected['rcards']}")
    st.write(f"**Own Goals**: {player_stats_selected['owngoals']}")
    st.write(f"**Minutes Played**: {player_stats_selected['minutesPlayed']}")

    # Prepare the stats for prediction (reshape into 2D array and ensure same columns as in training)
    player_data = pd.DataFrame(player_stats_selected).T

    # Preprocess the player data
    #player_data_processed = preprocessor.transform(player_data)

    # Predict player rating using the model
    predicted_rating = model.predict(player_data)

    # Display the predicted rating
    st.write(f"### Predicted Rating for {player_option}: {predicted_rating[0]:.2f}")