import pickle
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import Counter

# Load the pickle file
pickle_file_path = 'September_10_2024_debate.pkl'
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Convert the data to a DataFrame (assuming it's compatible)
df = pd.DataFrame(data)


def calculate_speaking_rates_for_speaker(df, speaker):
    speaker_df = df[df['speaker'] == speaker].copy()

    # Ensure the necessary columns exist before calculations
    if 'endTime' in speaker_df.columns and 'startTime' in speaker_df.columns and 'word_weight' in speaker_df.columns:
        # Calculate words per minute for each turn
        speaker_df['duration'] = speaker_df['endTime'] - speaker_df['startTime']
        speaker_df['words_per_minute'] = speaker_df['word_weight'] / speaker_df['duration'] * 60
    else:
        speaker_df['words_per_minute'] = 0  # Default if columns are missing

    # Ensure claims columns exist
    if {'claim_of_facts_extractive_supporting_quotes_claim',
        'claim_of_value_extractive_supporting_quotes_claim',
        'claim_of_policy_extractive_supporting_quotes_claim'}.issubset(speaker_df.columns):
        speaker_df['claims_per_turn'] = speaker_df[['claim_of_facts_extractive_supporting_quotes_claim',
                                                    'claim_of_value_extractive_supporting_quotes_claim',
                                                    'claim_of_policy_extractive_supporting_quotes_claim']].apply(
            lambda row: sum([bool(claim) for claim in row]), axis=1
        )
    else:
        speaker_df['claims_per_turn'] = 0  # Default if claims columns are missing

    # Calculate arguments per turn if columns exist
    argument_columns = [
        'claim_of_facts_extractive_supporting_quotes_argument',
        'claim_of_value_extractive_supporting_quotes_argument',
        'claim_of_policy_extractive_supporting_quotes_argument'
    ]

    available_argument_columns = [col for col in argument_columns if col in speaker_df.columns]
    if available_argument_columns:
        speaker_df['argument_count'] = speaker_df[available_argument_columns].apply(
            lambda row: sum([bool(arg) for arg in row]), axis=1)
    else:
        speaker_df['argument_count'] = 0

    # Calculate average arguments per topic and merge
    grouped_df = speaker_df.groupby(['Topic']).agg(avg_arguments_per_topic=('argument_count', 'mean')).reset_index()
    speaker_df = pd.merge(speaker_df, grouped_df[['Topic', 'avg_arguments_per_topic']], on='Topic', how='left')

    return speaker_df
def plot_speaker_comparison(df, speaker1, speaker2):
    # Calculate speaking rates for each speaker
    speaker1_df = calculate_speaking_rates_for_speaker(df, speaker1)
    speaker2_df = calculate_speaking_rates_for_speaker(df, speaker2)

    # Calculate the average for the specified metrics for each speaker
    avg_words_speaker1 = speaker1_df['words_per_minute'].mean()
    avg_claims_speaker1 = speaker1_df['claims_per_turn'].mean()
    avg_arguments_speaker1 = speaker1_df['avg_arguments_per_topic'].mean()

    avg_words_speaker2 = speaker2_df['words_per_minute'].mean()
    avg_claims_speaker2 = speaker2_df['claims_per_turn'].mean()
    avg_arguments_speaker2 = speaker2_df['avg_arguments_per_topic'].mean()

    # Data for the bar chart
    labels = ['Words per Minute', 'Claims per Turn', 'Arguments per Topic']
    speaker1_values = [avg_words_speaker1, avg_claims_speaker1, avg_arguments_speaker1]
    speaker2_values = [avg_words_speaker2, avg_claims_speaker2, avg_arguments_speaker2]

    # Create the plot
    fig = go.Figure(data=[
        go.Bar(name=speaker1, x=labels, y=speaker1_values, marker_color='blue'),
        go.Bar(name=speaker2, x=labels, y=speaker2_values, marker_color='red')
    ])

    # Update layout
    fig.update_layout(
        title=f'Comparison of {speaker1} and {speaker2}',
        xaxis_title='Metrics',
        yaxis_title='Average Values',
        barmode='group'
    )

    # Show the plot
    st.plotly_chart(fig)


def plot_speaking_time_distribution(df, speaker1, speaker2):
    # Calculate total speaking time for each speaker
    speaker1_df = df[df['speaker'] == speaker1]
    speaker2_df = df[df['speaker'] == speaker2]

    # Speaking time is the difference between endTime and startTime
    total_time_speaker1 = speaker1_df['endTime'].sum() - speaker1_df['startTime'].sum()
    total_time_speaker2 = speaker2_df['endTime'].sum() - speaker2_df['startTime'].sum()

    # Data for the pie chart
    labels = [speaker1, speaker2]
    values = [total_time_speaker1, total_time_speaker2]
    colors = ['blue', 'red']  # Kamala is blue, Trump is red

    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker=dict(colors=colors))])

    fig.update_layout(
        title_text=f'Average Speaking Time (seconds) Distribution between {speaker1} and {speaker2}'
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)

def plot_radar_chart_epl_comparison(df, speaker1, speaker2):
    # Define the columns for Ethos, Pathos, and Logos
    ethos_columns = ['epl_ethos_authority', 'epl_ethos_trust', 'epl_ethos_credibility', 'epl_ethos_power', 'epl_ethos_reliability']
    pathos_columns = ['epl_pathos_joy', 'epl_pathos_sadness', 'epl_pathos_disgust', 'epl_pathos_fear',
                      'epl_pathos_rage', 'epl_pathos_anticipation', 'epl_pathos_surprise', 'epl_pathos_trust']
    logos_columns = ['epl_logos_premise_strength', 'epl_logos_conclusion_strength', 'epl_logos_soundness',
                     'epl_logos_validity']

    # Filter the data for each speaker
    speaker1_df = df[df['speaker'] == speaker1]
    speaker2_df = df[df['speaker'] == speaker2]

    # Calculate the average for Ethos, Pathos, and Logos for each speaker
    ethos_avg_speaker1 = speaker1_df[ethos_columns].mean().mean()
    pathos_avg_speaker1 = speaker1_df[pathos_columns].mean().mean()
    logos_avg_speaker1 = speaker1_df[logos_columns].mean().mean()

    ethos_avg_speaker2 = speaker2_df[ethos_columns].mean().mean()
    pathos_avg_speaker2 = speaker2_df[pathos_columns].mean().mean()
    logos_avg_speaker2 = speaker2_df[logos_columns].mean().mean()

    # Data for the radar chart
    categories = ['Ethos', 'Pathos', 'Logos']

    # Create radar chart
    fig = go.Figure()

    # Add trace for speaker 1
    fig.add_trace(go.Scatterpolar(
        r=[ethos_avg_speaker1, pathos_avg_speaker1, logos_avg_speaker1],
        theta=categories,
        fill='toself',
        name=speaker1,
        marker_color='blue'
    ))

    # Add trace for speaker 2
    fig.add_trace(go.Scatterpolar(
        r=[ethos_avg_speaker2, pathos_avg_speaker2, logos_avg_speaker2],
        theta=categories,
        fill='toself',
        name=speaker2,
        marker_color='red'
    ))

    # Update the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, max(ethos_avg_speaker1, pathos_avg_speaker1, logos_avg_speaker1,
                                                        ethos_avg_speaker2, pathos_avg_speaker2, logos_avg_speaker2)])
        ),
        showlegend=True,
        title=f'Ethos, Pathos, Logos Comparison between {speaker1} and {speaker2}'
    )

    # Show the plot
    st.plotly_chart(fig)


def plot_epl_attributes_radar(df, speaker1, speaker2, category):
    # Define the attributes for Ethos, Pathos, and Logos
    categories = {
        'Ethos': ['epl_ethos_authority', 'epl_ethos_trust', 'epl_ethos_credibility', 'epl_ethos_power'],
        'Pathos': ['epl_pathos_joy', 'epl_pathos_sadness', 'epl_pathos_disgust', 'epl_pathos_fear',
                   'epl_pathos_rage', 'epl_pathos_anticipation', 'epl_pathos_surprise', 'epl_pathos_trust'],
        'Logos': ['epl_logos_premise_strength', 'epl_logos_conclusion_strength', 'epl_logos_soundness',
                  'epl_logos_validity']
    }

    # Filter the data for each speaker
    speaker1_df = df[df['speaker'] == speaker1]
    speaker2_df = df[df['speaker'] == speaker2]

    # Get the attributes based on the selected category
    selected_columns = categories[category]

    # Calculate the average for each attribute for both speakers
    speaker1_values = speaker1_df[selected_columns].mean().values
    speaker2_values = speaker2_df[selected_columns].mean().values

    # Radar chart requires a loop-back to the first category for a closed shape
    labels = selected_columns
    labels.append(labels[0])  # Append the first attribute to close the radar chart
    speaker1_values = list(speaker1_values) + [speaker1_values[0]]
    speaker2_values = list(speaker2_values) + [speaker2_values[0]]

    # Create radar chart
    fig = go.Figure()

    # Add trace for speaker 1
    fig.add_trace(go.Scatterpolar(
        r=speaker1_values,
        theta=labels,
        fill='toself',
        name=speaker1,
        marker_color='blue'
    ))

    # Add trace for speaker 2
    fig.add_trace(go.Scatterpolar(
        r=speaker2_values,
        theta=labels,
        fill='toself',
        name=speaker2,
        marker_color='red'
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False)
        ),
        showlegend=True,
        title=f'Comparison of {category} Attributes between {speaker1} and {speaker2}'
    )

    # Show the radar chart in Streamlit
    st.plotly_chart(fig)


def plot_ethos_pathos_logos_line(df, selected_category):
    # Define the columns for Ethos, Pathos, and Logos
    ethos_columns = ['epl_ethos_authority', 'epl_ethos_trust', 'epl_ethos_credibility', 'epl_ethos_power']
    pathos_columns = ['epl_pathos_joy', 'epl_pathos_sadness', 'epl_pathos_disgust', 'epl_pathos_fear',
                      'epl_pathos_rage', 'epl_pathos_anticipation', 'epl_pathos_surprise', 'epl_pathos_trust']
    logos_columns = ['epl_logos_premise_strength', 'epl_logos_conclusion_strength', 'epl_logos_soundness',
                     'epl_logos_validity']

    # Select the relevant columns based on the chosen category
    if selected_category == 'Ethos':
        selected_columns = ethos_columns
    elif selected_category == 'Pathos':
        selected_columns = pathos_columns
    else:
        selected_columns = logos_columns

    # Filter the data to include only Harris and Trump
    df_filtered = df[df['speaker'].isin(['Kamala Harris', 'Donald Trump'])].copy()

    # Calculate the average score for the selected category across turns for each speaker
    df_filtered['average_score'] = df_filtered[selected_columns].mean(axis=1)

    # Define the custom color mapping
    color_map = {'Kamala Harris': 'blue', 'Donald Trump': 'red'}

    # Create a line chart comparing Kamala Harris and Donald Trump
    fig = px.line(df_filtered, x='turn', y='average_score', color='speaker',
                  title=f'Average {selected_category} Score Over the Debate: Harris vs Trump',
                  labels={'turn': 'Turn Number', 'average_score': f'Average {selected_category} Score'},
                  color_discrete_map=color_map,  # Apply the custom color map
                  line_shape='linear')

    fig.update_layout(xaxis_title='Turn Number', yaxis_title=f'Average {selected_category} Score')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def plot_intuitive_deliberative_comparison(df, speaker1, speaker2, selected_system):
    # Filter the data for each speaker
    speaker1_df = df[df['speaker'] == speaker1].sort_values('turn')
    speaker2_df = df[df['speaker'] == speaker2].sort_values('turn')

    # Determine which system to plot
    system_column = 'system1_score' if selected_system == 'Intuitive Thinking' else 'system2_score'

    # Create the line graph
    fig = go.Figure()

    # Add trace for speaker 1 (Kamala Harris)
    fig.add_trace(go.Scatter(x=speaker1_df['turn'], y=speaker1_df[system_column],
                             mode='lines', name=speaker1, line=dict(color='blue')))

    # Add trace for speaker 2 (Donald Trump)
    fig.add_trace(go.Scatter(x=speaker2_df['turn'], y=speaker2_df[system_column],
                             mode='lines', name=speaker2, line=dict(color='red')))

    # Update layout
    fig.update_layout(
        title=f'{selected_system} Comparison Between {speaker1} and {speaker2}',
        xaxis_title='Turn Number',
        yaxis_title=f'{selected_system} Score',
        height=600,
        showlegend=True
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)

def plot_heatmap_s1_s2_vs_epl(df, speaker1, speaker2, color_scale='Viridis'):
    # Define the EPL attributes (Ethos, Pathos, Logos)
    epl_attributes = ['epl_ethos_authority', 'epl_ethos_trust', 'epl_ethos_credibility', 'epl_ethos_power',
                      'epl_pathos_joy', 'epl_pathos_sadness', 'epl_pathos_disgust', 'epl_pathos_fear',
                      'epl_pathos_rage', 'epl_pathos_anticipation', 'epl_pathos_surprise', 'epl_pathos_trust',
                      'epl_logos_premise_strength', 'epl_logos_conclusion_strength', 'epl_logos_soundness', 'epl_logos_validity']

    # Filter the DataFrame to only include the selected speaker's data
    speaker1_df = df[df['speaker'] == speaker1]
    speaker2_df = df[df['speaker'] == speaker2]

    # Select only system1, system2, and EPL attributes
    heatmap_data_speaker1 = speaker1_df[['system1_score', 'system2_score'] + epl_attributes]
    heatmap_data_speaker2 = speaker2_df[['system1_score', 'system2_score'] + epl_attributes]

    # Calculate correlations for both speakers
    correlation_speaker1 = heatmap_data_speaker1.corr().loc[epl_attributes, ['system1_score', 'system2_score']]
    correlation_speaker2 = heatmap_data_speaker2.corr().loc[epl_attributes, ['system1_score', 'system2_score']]

    # Create subplots: two columns, one for each speaker
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{speaker1} Heatmap', f'{speaker2} Heatmap'), shared_yaxes=True)

    # Heatmap for Kamala Harris
    fig.add_trace(go.Heatmap(
        z=correlation_speaker1.values,
        x=['Intuitive Thinking', 'Deliberative Thinking'],
        y=epl_attributes,
        zmin=-1,  # Minimum value for the color scale (negative correlation)
        zmax=1,   # Maximum value for the color scale (positive correlation)
        colorscale=color_scale,  # Customizable color scale
        text=correlation_speaker1.values,  # Add correlation values as text
        texttemplate="%{text:.3f}",  # Format text with two decimal places
        textfont={"size": 12},  # Font size for the text
        showscale=False  # Hide color scale for this heatmap
    ), row=1, col=1)

    # Heatmap for Donald Trump
    fig.add_trace(go.Heatmap(
        z=correlation_speaker2.values,
        x=['Intuitive Thinking', 'Deliberative Thinking'],
        y=epl_attributes,
        zmin=-1,  # Minimum value for the color scale
        zmax=1,   # Maximum value for the color scale
        colorscale=color_scale,  # Customizable color scale
        text=correlation_speaker2.values,  # Add correlation values as text
        texttemplate="%{text:.3f}",  # Format text with two decimal places
        textfont={"size": 12},  # Font size for the text
        showscale=True  # Show color scale for this heatmap
    ), row=1, col=2)

    # Update layout
    fig.update_layout(
        height=600,  # Smaller height
        width=900,  # Adjust width for side-by-side display
        title_text="Intuitive (S1) and Deliberative (S2) Thinking vs EPL Attributes",
        xaxis_title="Thinking Styles",
        yaxis_title="EPL Attributes",
        showlegend=False,
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)


def plot_intuitive_deliberative_by_topic(df, speaker1, speaker2, selected_system):
    # Group by topic and calculate the average score for each topic
    system_column = 'system1_score' if selected_system == 'Intuitive Thinking' else 'system2_score'

    # Group by Topic for each speaker
    speaker1_df = df[df['speaker'] == speaker1].groupby('Topic')[system_column].mean().reset_index()
    speaker2_df = df[df['speaker'] == speaker2].groupby('Topic')[system_column].mean().reset_index()

    # Merge dataframes for side-by-side comparison
    merged_df = pd.merge(speaker1_df, speaker2_df, on="Topic", suffixes=(f'_{speaker1}', f'_{speaker2}'))

    # Create the bar graph
    fig = go.Figure()

    # Add bar for speaker 1 (Kamala Harris)
    fig.add_trace(go.Bar(
        x=merged_df['Topic'],
        y=merged_df[f'{system_column}_{speaker1}'],
        name=speaker1,
        marker_color='blue'
    ))

    # Add bar for speaker 2 (Donald Trump)
    fig.add_trace(go.Bar(
        x=merged_df['Topic'],
        y=merged_df[f'{system_column}_{speaker2}'],
        name=speaker2,
        marker_color='red'
    ))

    # Update layout
    fig.update_layout(
        title=f'{selected_system} Thinking by Topic: {speaker1} vs {speaker2}',
        xaxis_title='Topic',
        yaxis_title=f'{selected_system} Score',
        barmode='group',  # Grouped bar chart
        height=600,
        width=1000
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)

def plot_sunbursts_side_by_side(df, color_mapping=None):
    # If no color mapping is provided, use a default mapping for Harris and Trump
    if color_mapping is None:
        color_mapping = {
            "Kamala Harris": "blue",
            "Donald Trump": "red"
        }

    # Filter the data to include only Harris and Trump
    df_filtered = df[df['speaker'].isin(['Kamala Harris', 'Donald Trump'])].copy()

    # Initialize an empty list to store sunburst data for topics
    sunburst_data_topics = []

    # Iterate through each row of the filtered dataframe for topics
    for _, row in df_filtered.iterrows():
        if isinstance(row['claim_of_facts_topics'], list):
            for topic in row['claim_of_facts_topics']:
                if topic:
                    sunburst_data_topics.append({
                        "speaker": row['speaker'],
                        "topic": topic,
                        "claim_type": "Claim of Facts"
                    })
        if isinstance(row['claim_of_value_topics'], list):
            for topic in row['claim_of_value_topics']:
                if topic:
                    sunburst_data_topics.append({
                        "speaker": row['speaker'],
                        "topic": topic,
                        "claim_type": "Claim of Value"
                    })
        if isinstance(row['claim_of_policy_topics'], list):
            for topic in row['claim_of_policy_topics']:
                if topic:
                    sunburst_data_topics.append({
                        "speaker": row['speaker'],
                        "topic": topic,
                        "claim_type": "Claim of Policy"
                    })

    # Initialize an empty list to store sunburst data for impacted groups/populations
    sunburst_data_groups = []

    # Iterate through each row of the filtered dataframe for impacted groups/populations
    for _, row in df_filtered.iterrows():
        if isinstance(row['claim_of_facts_impacted_groups_populations'], list):
            for group in row['claim_of_facts_impacted_groups_populations']:
                if group:
                    sunburst_data_groups.append({
                        "speaker": row['speaker'],
                        "group": group,
                        "claim_type": "Claim of Facts"
                    })
        if isinstance(row['claim_of_value_impacted_groups_populations'], list):
            for group in row['claim_of_value_impacted_groups_populations']:
                if group:
                    sunburst_data_groups.append({
                        "speaker": row['speaker'],
                        "group": group,
                        "claim_type": "Claim of Value"
                    })
        if isinstance(row['claim_of_policy_impacted_groups_populations'], list):
            for group in row['claim_of_policy_impacted_groups_populations']:
                if group:
                    sunburst_data_groups.append({
                        "speaker": row['speaker'],
                        "group": group,
                        "claim_type": "Claim of Policy"
                    })

    # Convert the lists of sunburst data to DataFrames
    sunburst_df_topics = pd.DataFrame(sunburst_data_topics)
    sunburst_df_groups = pd.DataFrame(sunburst_data_groups)

    # Drop rows where 'topic' or 'group' is missing
    sunburst_df_topics = sunburst_df_topics.dropna(subset=['topic'])
    sunburst_df_groups = sunburst_df_groups.dropna(subset=['group'])

    # Create the first sunburst chart for topics using Plotly Express
    fig_topics = px.sunburst(
        sunburst_df_topics,
        path=['speaker', 'claim_type', 'topic'],
        title="Topics Discussed in Presidential Debate",
        color='speaker',
        color_discrete_map=color_mapping
    )

    # Create the second sunburst chart for impacted groups using Plotly Express
    fig_groups = px.sunburst(
        sunburst_df_groups,
        path=['speaker', 'claim_type', 'group'],
        title="Impacted Groups/Populations in Presidential Debate",
        color='speaker',
        color_discrete_map=color_mapping
    )

    # Display the sunburst charts side by side in Streamlit using st.columns()
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_topics)

    with col2:
        st.plotly_chart(fig_groups)

def plot_claim_sankey(df, speaker, number_of_groups=5):
    # Filter the DataFrame for the selected speaker
    df_speaker = df[df['speaker'] == speaker].copy()

    # Ensure the impacted groups are lists
    df_speaker['claim_of_facts_impacted_groups_populations'] = df_speaker['claim_of_facts_impacted_groups_populations'].apply(lambda x: x if isinstance(x, list) else [])
    df_speaker['claim_of_value_impacted_groups_populations'] = df_speaker['claim_of_value_impacted_groups_populations'].apply(lambda x: x if isinstance(x, list) else [])
    df_speaker['claim_of_policy_impacted_groups_populations'] = df_speaker['claim_of_policy_impacted_groups_populations'].apply(lambda x: x if isinstance(x, list) else [])

    fact_to_group_counter = Counter()
    policy_to_group_counter = Counter()
    values_to_group_counter = Counter()

    # Counter for impacted groups
    for _, row in df_speaker.iterrows():
        facts = [group.title() for group in row['claim_of_facts_impacted_groups_populations']]
        values = [group.title() for group in row['claim_of_value_impacted_groups_populations']]
        policies = [group.title() for group in row['claim_of_policy_impacted_groups_populations']]

        for fact in facts:
            fact_to_group_counter[fact] += 1
        for policy in policies:
            policy_to_group_counter[policy] += 1
        for value in values:
            values_to_group_counter[value] += 1

    # Get the top impacted groups from each
    top_fact = fact_to_group_counter.most_common(number_of_groups)
    top_policy = policy_to_group_counter.most_common(number_of_groups)
    top_value = values_to_group_counter.most_common(number_of_groups)

    # Create a list of unique groups
    unique_groups = set([group for group, _ in top_fact + top_policy + top_value])
    labels = ['Claims of Fact', 'Claims of Value', 'Claims of Policy'] + list(unique_groups)

    sources = []
    targets = []
    values = []
    colors = []

    # Add Claims of Fact to the impacted groups
    for target, count in top_fact:
        sources.append(0)  # Index for 'Claims of Fact'
        targets.append(labels.index(target))
        values.append(count)
        colors.append('rgba(31, 119, 180, 0.8)')

    # Add Claims of Value to the impacted groups
    for target, count in top_value:
        sources.append(1)  # Index for 'Claims of Value'
        targets.append(labels.index(target))
        values.append(count)
        colors.append('rgba(255, 127, 14, 0.8)')

    # Add Claims of Policy to the impacted groups
    for target, count in top_policy:
        sources.append(2)  # Index for 'Claims of Policy'
        targets.append(labels.index(target))
        values.append(count)
        colors.append('rgba(44, 160, 44, 0.8)')

    # Create the Sankey chart
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 'rgba(44, 160, 44, 0.8)']
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors
        )
    )])

    fig.update_layout(title_text=f"Claims of Value/Fact/Policy to Impacted Groups - {speaker}", font_size=10)
    return fig


def plot_bias_fallacy_heatmap(df, speaker):
    # Filter data by speaker
    df_speaker = df[df['speaker'] == speaker]

    # Ensure the columns are lists
    df_speaker['topics'] = df_speaker['topics'].apply(lambda x: x if isinstance(x, list) else [])
    df_speaker['epl_logos_bias_types'] = df_speaker['epl_logos_bias_types'].apply(
        lambda x: x if isinstance(x, list) else [])
    df_speaker['epl_logos_fallacy_types'] = df_speaker['epl_logos_fallacy_types'].apply(
        lambda x: x if isinstance(x, list) else [])

    # Explode the topics column to separate each topic into its own row
    topics_df = df_speaker.explode('topics')

    # Explode the bias and fallacy columns to separate each type into its own row
    biases_df = topics_df.explode('epl_logos_bias_types')
    fallacies_df = topics_df.explode('epl_logos_fallacy_types')

    # Define the full list of biases and fallacies
    biases = ["Overconfidence Bias", "Negativity Bias", "Sunk Cost Fallacy", "Status Quo Bias", "Self-Serving Bias",
              "Hindsight Bias", "Bandwagon Effect", "Availability Heuristic", "Anchoring Bias", "Confirmation Bias"]
    fallacies = ["Begging the Question", "Equivocation", "False Cause", "Red Herring", "Circular Reasoning",
                 "Hasty Generalization", "Slippery Slope", "False Dichotomy", "Straw Man", "Ad Hominem"]

    # Count the occurrences of each bias and fallacy per topic
    bias_counts = biases_df.groupby(['topics', 'epl_logos_bias_types']).size().unstack(fill_value=0)
    fallacy_counts = fallacies_df.groupby(['topics', 'epl_logos_fallacy_types']).size().unstack(fill_value=0)

    # Ensure all biases and fallacies are included, even if they have no counts
    bias_counts = bias_counts.reindex(columns=biases, fill_value=0)
    fallacy_counts = fallacy_counts.reindex(columns=fallacies, fill_value=0)

    # Concatenate the bias and fallacy counts into a single dataframe
    bias_fallacy_counts = pd.concat([bias_counts, fallacy_counts], axis=1).fillna(0)

    # Remove biases and fallacies that have zero counts across all topics
    bias_fallacy_counts = bias_fallacy_counts.loc[:, (bias_fallacy_counts != 0).any(axis=0)]

    # Reset index for plotting
    bias_fallacy_counts = bias_fallacy_counts.reset_index()

    # Create a heatmap of the bias and fallacy counts
    fig = px.imshow(bias_fallacy_counts.set_index(['topics']).T,
                    labels=dict(x="Topics", y="Biases and Fallacies", color="Count"),
                    x=bias_fallacy_counts['topics'],
                    title=f'Heatmap of Biases and Fallacies for {speaker} by Topic')

    # Update x-axis and y-axis to make labels readable
    fig.update_xaxes(tickangle=-45, automargin=True)  # Rotate x-axis labels for readability
    fig.update_yaxes(automargin=True)  # Ensure y-axis labels don't overlap

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)


def show():
    st.write("### Overview of the Debate")

    # Extract the unique speakers from the dataset
    speakers = df['speaker'].unique()
    exclude_speakers = ["Narrator", "David Muir", "Linsey Davis"]
    filtered_speakers = [speaker for speaker in speakers if speaker not in exclude_speakers]


    # Dictionary of public figures and their bios
    figures = {
        "Donald Trump": """
            **Birthday:** June 14, 1946 (age 78 years)\n
            **Hometown:** New York, NY\n
            **Career Bio:** Donald Trump is a businessman, television personality, and the 45th president of the United States (2017–2021). Before entering politics, Trump built a real estate empire, primarily focused on high-end properties in Manhattan through his company, the Trump Organization. His business ventures extended beyond real estate, as he licensed the Trump brand for various products, from hotels to golf courses, and appeared frequently in media and entertainment.\n\n
            In 2016, Trump ran a successful presidential campaign as a Republican, focusing on a populist message that resonated with many voters, despite having no prior political or military experience. His presidency was marked by controversial policies, including tax reforms, deregulation, a focus on stricter immigration controls, and a renegotiation of trade deals. However, his administration faced significant challenges, including two impeachment trials and the COVID-19 pandemic. After leaving office in 2021, Trump continued to be a polarizing figure in U.S. politics, retaining significant influence within the Republican Party.\n
            **Fun Facts:** 
            1)  Before his presidency, Trump was widely known for hosting the reality TV show The Apprentice, where contestants competed for a job in one of his companies. The show popularized his catchphrase "You're fired!"
            2)  Over the years, Trump has made numerous cameos in films and television shows, such as Home Alone 2 and Zoolander, further solidifying his pop culture presence.
            3) Trump has appeared in WWE (World Wrestling Entertainment) events and even participated in the famous "Battle of the Billionaires" match at WrestleMania 23 in 2007. His team won, and he shaved the head of WWE owner Vince McMahon as part of the match's stipulation.
            4) Trump co-wrote The Art of the Deal in 1987, which became a bestseller and is often seen as a reflection of his business philosophy and style.

            """,
        "Kamala Harris": """
            **Birthday:** October 20, 1964 (age 59 years)\n
            **Hometown:** Oakland, CA\n
            **Career Bio:** Kamala Harris is the 49th vice president of the United States and the first woman, first Black woman, and first person of South Asian descent to hold the office. Before becoming vice president, Harris built a prominent career in law and politics. She began as a prosecutor in California, eventually becoming the District Attorney of San Francisco in 2003. Known for her focus on criminal justice reform, she later served as California's Attorney General from 2011 to 2017, where she worked on issues like consumer protection, reducing truancy, and advancing LGBTQ+ rights.\n\n
            In 2017, Harris was elected to the U.S. Senate, where she earned a reputation as a skilled questioner during committee hearings, particularly in matters of judicial nominations and the Trump administration's handling of key issues. Her progressive policy positions on healthcare, climate change, and immigration resonated with many within the Democratic Party. In 2020, Joe Biden selected Harris as his running mate, and they went on to win the election. As vice president, Harris has focused on tackling issues like voting rights, immigration reform, and spearheading efforts related to gender equity, and addressing root causes of migration from Central America.\n
            **Fun Facts:** 
            1) Harris is known to be an enthusiastic cook and has shared many of her favorite recipes, including traditional Indian dishes like masala dosa, online on platforms like YouTube.
            2) Harris has often mentioned her love for hip-hop music. During her campaign, she shared that she enjoys artists like Tupac Shakur and Snoop Dogg, reflecting her California roots.
            3) Harris joined Alpha Kappa Alpha (AKA) Sorority, Incorporated, while attending Howard University. AKA is the first historically African American sorority, and she remains an active and proud member of the organization.
            4) In addition to her political work, Harris has written multiple books, including The Truths We Hold: An American Journey, and a children’s book, Superheroes Are Everywhere, which shares inspiring stories from her life.
            
            """
    }

    # Create a dropdown to select a speaker
    selected_speaker = st.selectbox('Select a Speaker', filtered_speakers)

    # Check if the selected speaker is in the figures dictionary, and display the bio if it exists
    if selected_speaker in figures:
        st.write(f"**{selected_speaker}**")
        st.markdown(figures[selected_speaker])
    else:
        st.write(f"No detailed biography available for {selected_speaker}")

    # Display speaking rates
    # speaker_df = df[df['speaker'] == selected_speaker]
    # speaker_rates_df = calculate_speaking_rates_for_speaker(speaker_df)
    # st.write(f"#### Speaking Rates for {selected_speaker}")
    # avg_words_per_minute = speaker_rates_df['words_per_minute'].mean()
    # avg_claims_per_turn = speaker_rates_df['claims_per_turn'].mean()
    # avg_arguments_per_topic = speaker_rates_df['avg_arguments_per_topic'].mean()
    #
    # st.write(f"**Average Words per Minute**: {avg_words_per_minute:.2f}")
    # st.write(f"**Average Claims per Turn**: {avg_claims_per_turn:.2f}")
    # st.write(f"**Average Arguments per Topic**: {avg_arguments_per_topic:.2f}")

    st.write("### Speaking Rate")
    plot_speaker_comparison(df, "Kamala Harris", "Donald Trump")
    plot_speaking_time_distribution(df, "Kamala Harris", "Donald Trump")

    st.write("### Average Ethos Pathos Logos (EPL) Scores")
    plot_radar_chart_epl_comparison(df, "Kamala Harris", "Donald Trump")

    selected_category = st.selectbox('Select a Category', ['Ethos', 'Pathos', 'Logos'])
    plot_epl_attributes_radar(df, "Kamala Harris", "Donald Trump", selected_category)
    plot_ethos_pathos_logos_line(df, selected_category)

    st.write('### How Intuitive and Deliberative Thinking fare over the course of the debate?')
    selected_system = st.selectbox('Select a Thinking Style', ['Intuitive Thinking', 'Deliberative Thinking'])
    plot_intuitive_deliberative_comparison(df, "Kamala Harris", "Donald Trump", selected_system)

    st.write('### Correlation between Intuitive + Deliberative Thinking vs EPL Attributes')
    plot_heatmap_s1_s2_vs_epl(df, "Kamala Harris", "Donald Trump")

    st.write('### How Intuitive and Deliberative Thinking fare change across Topics?')
    selected_system = st.selectbox('Select a Thinking Style', ['Intuitive Thinking', 'Deliberative Thinking'], key="thinking_style")
    plot_intuitive_deliberative_by_topic(df, "Kamala Harris", "Donald Trump", selected_system)

    st.write('### What Topics and Impacted Groups were prominent among the various types of clams per candidate')
    plot_sunbursts_side_by_side(df)

    all_groups = set()
    df['claim_of_facts_impacted_groups_populations'].apply(
        lambda x: all_groups.update(x if isinstance(x, list) else []))
    df['claim_of_value_impacted_groups_populations'].apply(
        lambda x: all_groups.update(x if isinstance(x, list) else []))
    df['claim_of_policy_impacted_groups_populations'].apply(
        lambda x: all_groups.update(x if isinstance(x, list) else []))

    max_groups = len(all_groups)
    st.write("### How do claims flow into topics in the debate?")
    number_of_groups = st.slider("Select number of impacted groups to display from each claim type", min_value=1, max_value=max_groups, value=5)
    fig_harris = plot_claim_sankey(df, 'Kamala Harris', number_of_groups)
    fig_trump = plot_claim_sankey(df, 'Donald Trump', number_of_groups)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_harris)
    with col2:
        st.plotly_chart(fig_trump)

    st.write("### What kind of biases and fallacies are associated with ")
    plot_bias_fallacy_heatmap(df, 'Kamala Harris')
    plot_bias_fallacy_heatmap(df, 'Donald Trump')


# Call the function in your Streamlit app
if __name__ == "__main__":
    show()
