import re
import pandas as pd
import numpy as np
import networkx as nx

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
