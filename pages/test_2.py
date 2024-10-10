import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import main_minerva

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
team_name1 = df_team1['Team'].iloc[0]
team_name2 = df_team2['Team'].iloc[0]

# Sidebar to select team
team_option = st.selectbox('Select Team', [team_name1, team_name2])

# Filter players based on selected team
if team_option == team_name1:
    team_data = input_df1[input_df1['is_home_team'] == 1]
    # Show only players from team_name1
    player_option = st.selectbox('Select a player', input_df1['Player'].sort_values().unique())
else:
    team_data = input_df2[input_df2['is_home_team'] == 0]
    # Show only players from team_name2
    player_option = st.selectbox('Select a player', input_df2['Player'].sort_values().unique())


# Combine both teams' player statistics into one DataFrame
player_stats = pd.concat([player_stats1, player_stats2])

input_df = pd.concat([input_df1, input_df2])

# Select player based on filtered team data
#player_option = st.selectbox('Select Player', final_metrics_df['Player'].unique())

col1, col2 = st.columns(2)

# Display player statistics once the player is selected
if player_option:
    player_stats_selected = player_stats[player_stats['Player'] == player_option].iloc[0]
    with col1:
        # Display the player's stats in Streamlit
        st.write(f"### Stats for {player_option}") 
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
    
    # Assuming 'Player' is in input_df, filter data for the selected player
player_data = input_df[input_df['Player'] == player_option]

# Drop the 'Player' column (since the model doesn't need this for prediction)
player_data_for_prediction = player_data.drop(columns=['Player'])

# Predict player rating using the model
predicted_rating = model.predict(player_data_for_prediction)

# Display the predicted rating
#st.write(f"### Rating for {player_option}: {predicted_rating[0]:.2f}")
with col2:
        col2.metric(label=f"### Rating for {player_option}:", value=f'{predicted_rating[0]:.2f}')