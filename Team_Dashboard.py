import json
import streamlit as st
import pandas as pd
from mplsoccer import VerticalPitch
import main_minerva
import streamlit_shadcn_ui as ui

st.set_page_config(
    page_title="Minerva App",
    page_icon="👋🏼",
    layout="centered"
)

# Load event data for both teams
df_team1 = pd.read_csv('Minerva_vs_Sudeva_Minerva_data.csv')
df_team2 = pd.read_csv('Minerva_vs_Sudeva_Sudeva_data.csv')  # Replace with the actual file for the second team

# Assuming df_team1 and df_team2 have the 'Team' column that contains the team names
team_name1 = df_team1['Team'].iloc[0]
team_name2 = df_team2['Team'].iloc[0]

# Combine dataframes for both teams
df = pd.concat([df_team1, df_team2], ignore_index=True)

# Filter the event data
passes_df = df[(df['Event']).isin(['Pass', 'Long Kick'])]  # Save pass data
shots_df = df[df['Event'] == 'Shot'].reset_index(drop=True)  # Save shot data

# Get all unique team names from both datasets
team_names = df['Team'].unique()

st.title(f'{team_name1} vs {team_name2}\nMatch Analysis')
#st.subheader("Filter to any team/player to see all of their shots taken")

# Add shot outcome options
with st.sidebar:
    st.header("Filter Options")
    #outcome_options = shots_df['Outcome'].unique()
    outcome_options = ['Goal','Off Target','Saved','Blocked']
    selected_outcomes = st.multiselect('Select Shot Outcome(s)', outcome_options, default=outcome_options)
    
    # Select team and player
    team = st.selectbox('Select a team', team_names, index=None)
    player = st.selectbox('Select a player', df[df['Team'] == team]['Player'].sort_values().unique(), index=None)

# Function to filter data based on team, player, and outcomes
def filter_data(df, team, player, outcomes):
    """Filter the data based on team, player selection, and outcomes."""
    if team:
        df = df[df['Team'] == team]
    if player:
        df = df[df['Player'] == player]
    if outcomes:
        df = df[df['Outcome'].isin(outcomes)]
    return df

# Filter the data for shots based on team, player, and outcome
filtered_shots_df = filter_data(shots_df, team, player, selected_outcomes)
off_target = len(filtered_shots_df[filtered_shots_df['Outcome'] == 'Off Target'])
# Create two columns for the visualizations
col1, col2 = st.columns(2)

# Reduce the size of the visualizations by setting a smaller figsize
with col1:
    st.subheader(f'{player if player else ""} Shotmap')
    pitch = VerticalPitch(pitch_type='statsbomb', half=False)
    fig_shot_map, ax_shot_map = pitch.draw(figsize=(6, 6))  # Smaller figure size

    def plot_shots(df, ax, pitch):
        colors = {
            'Goal': 'green',
            'Off target': 'red',
            'Saved': 'yellow',
            'Blocked': 'blue'
        }

        # Use different colors for each outcome
        for outcome in df['Outcome'].unique():
            outcome_df = df[df['Outcome'] == outcome]
            pitch.scatter(
                outcome_df.X, outcome_df.Y, marker='o', s=100,
                ax=ax,
                color=colors.get(outcome, 'black'),
                edgecolors='black',
                alpha=1, 
                zorder=2,
                label=outcome  # Adding a label for the legend
            )
        ax.legend(loc='center')  # Add a legend to distinguish outcomes
        col1, col2 = st.columns(2)
        col1.metric(label="Total Shots", value=len(filtered_shots_df))
        col2.metric(label="Off Target", value=len(filtered_shots_df[filtered_shots_df['Outcome'] == 'Off Target']))
        #ui.metric_card(title="Total Shots", content=len(filtered_shots_df), description=f'Off Target - {off_target}')#, key="card1")
    plot_shots(filtered_shots_df, ax_shot_map, pitch)

    # Show shot plot
    st.pyplot(fig_shot_map)

with col2:
    st.subheader(f'{player if player else ""} Passmap')

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
        pitch = VerticalPitch(pitch_color='#a5c771', line_color='white', positional=False, stripe=True, corner_arcs=True)
        fig_pass_map, ax_pass_map = pitch.draw(figsize=(6, 6))  # Smaller figure size

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
        fig_pass_map.suptitle(f'{team} {player if player else ""} {pass_type}', fontsize=15)
        
        # Show pass map plot
        st.pyplot(fig_pass_map)



    # Call passmaps, which now accounts for team or player
    passmaps(passes_df, team, player, pitch, 'Passes')

# team stats
with col1:
    st.subheader(f'{team_name1} ' "Statistics")
    st.dataframe(main_minerva.calculate_team_stats(df_team1, df_team2))
with col2:
    st.subheader(f'{team_name2} ' "Statistics")
    st.dataframe(main_minerva.calculate_team_stats(df_team2, df_team1))


st.logo("baller_metrics_logo_bg.png")
st.sidebar.markdown("@baller_metrics - Follow us on Instagram")
ui.link_button(text="Visit our Instagram", url="https://www.instagram.com/baller_metrics/?igsh=MnBkOWw0aG01MXd4%2Fstreamlit-shadcn-ui", key="link_btn")

