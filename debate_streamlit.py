import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re


# Load the pickle file
pickle_file_path = 'debate_analysis_data0-30WithRef.pkl'
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Convert the data to a DataFrame (assuming it's compatible)
df = pd.DataFrame(data)

def initialize_session_state():
    if 'df' not in st.session_state:
        st.session_state['df'] = df

# Function to calculate words and sentences in a text
def analyze_text(content):
    words = len(content.split()) if isinstance(content, str) else 0
    sentences = len(content.split('.')) if isinstance(content, str) else 0
    return words, sentences


# Function to split content into smaller paragraphs
def split_into_paragraphs(content):
    sentences = content.split('. ')
    paragraphs = []
    paragraph = ""
    for i, sentence in enumerate(sentences):
        paragraph += sentence + ". "
        if (i + 1) % 3 == 0 or i == len(sentences) - 1:
            paragraphs.append(paragraph.strip())
            paragraph = ""
    return paragraphs

def categorize_section(content):
    content_lower = content.lower()  # Normalize to lowercase for easier pattern matching

    if '?' in content_lower:
        return "Question"
    if 'i agree' in content_lower:
        return "Response"

    # Keywords or patterns for each section
    if any(word in content_lower for word in ["welcome", "cnn", "abc", "candidates", "rules"]):
        return "Introduction"
    elif any(word in content_lower for word in ["?", "question", "follow-up"]):
        return "Question"
    elif any(word in content_lower for word in ["response", "economy", "country", "tax", "taxes", "in response", "answer", "reply", "thanks"]):
        return "Response"
    elif any(word in content_lower for word in ["however", "disagree", "counter"]):
        return "Rebuttal"
    elif any(word in content_lower for word in ["additionally", "furthermore", "follow up"]):
        return "Follow-up"
    elif any(word in content_lower for word in ["conclusion", "finally", "closing statement"]):
        return "Closing Statement"
    elif any(word in content_lower for word in ["thank you"]):
        return "Outro"
    else:
        return "Uncategorized"

# Function to create a color block for a list of items (words, phrases)
def create_color_block(items, colors):
    # Ensure items are treated as a list of words, not characters
    if isinstance(items, str):
        items = items.split()  # Split string into words

    colored_items = ""
    for i, item in enumerate(items):
        color = colors[i % len(colors)]  # Cycle through the colors
        colored_items += f"<span style='background-color: {color}; padding: 4px; margin-right: 6px; border-radius: 4px;'>{item}</span>"
    return colored_items


# Function to gather all previous turns by the same speaker
def get_turns_by_speaker(df, current_turn, speaker):
    turns = []
    word_counts = []
    sentence_counts = []

    normalized_speaker = speaker.strip().lower()

    for i in range(current_turn + 1):
        if 'speaker' in df.iloc[i]:
            df_speaker = df.iloc[i]['speaker'].strip().lower()
            if df_speaker == normalized_speaker:
                content = df.iloc[i]['content'] if 'content' in df.iloc[i] else ""
                words, sentences = analyze_text(content)
                turns.append(i + 1)  # Store turn number (starting from 1)
                word_counts.append(words)
                sentence_counts.append(sentences)

    return turns, word_counts, sentence_counts


def format_speaking_time(speaking_time):
    # Convert seconds to minutes and seconds
    minutes = speaking_time // 60
    seconds = speaking_time % 60

    # Return formatted string in "M:SS" format, ensuring seconds are two digits
    return f"{int(minutes)}:{int(seconds):02d}"


def calculate_speaking_time(selected_turn):
    # Get start and end times
    start_time = selected_turn.get('startTime', 0)
    end_time = selected_turn.get('endTime', 0)

    # Calculate speaking time in seconds
    speaking_time = end_time - start_time

    # Ensure speaking time is non-negative
    if speaking_time < 0:
        speaking_time = 0

    # Return formatted speaking time (in minutes and seconds if >= 60s, otherwise just seconds)
    return format_speaking_time(speaking_time)

def display_supporting_quote(value, default_message="No supporting quote from the transcript"):
    if isinstance(value, list) and len(value) == 0:
        return default_message
    elif isinstance(value, list):
        return ', '.join(map(str, value))  # Converts list to a comma-separated string without brackets
    return value

def get_previous_speakers_data(df, current_turn, current_speaker):
    previous_speakers_data = {}

    # Loop through previous turns to gather data for other speakers
    for i in range(current_turn):
        speaker = df.iloc[i]['speaker']
        if speaker != current_speaker:
            if speaker not in previous_speakers_data:
                previous_speakers_data[speaker] = {'turns': [], 'word_counts': [], 'sentence_counts': []}

            previous_speakers_data[speaker]['turns'].append(df.iloc[i]['turn'])
            content = df.iloc[i]['content']
            words, sentences = analyze_text(content)
            previous_speakers_data[speaker]['word_counts'].append(words)
            previous_speakers_data[speaker]['sentence_counts'].append(sentences)

    return previous_speakers_data

def plot_word_count_comparison(turns, word_counts, current_speaker, previous_speakers_data=None):
    # Create a line graph for word count comparison over all turns by the current speaker
    word_fig = go.Figure()

    # Add current speaker's word count as a line with markers
    word_fig.add_trace(go.Scatter(
        x=turns, y=word_counts, mode='lines+markers', name=f'{current_speaker} Word Count', line=dict(color='blue')
    ))

    # Add previous speakers' word counts as lines and markers if data is provided
    if previous_speakers_data is not None:
        for speaker, data in previous_speakers_data.items():
            speaker_turns = data['turns']
            speaker_word_counts = data['word_counts']
            word_fig.add_trace(go.Scatter(
                x=speaker_turns, y=speaker_word_counts, mode='lines+markers', name=f'{speaker} Word Count'
            ))

    # Update layout for the figure
    word_fig.update_layout(
        title=f'Word Count Progression for {current_speaker}',
        xaxis_title='Turn Number',
        yaxis_title='Number of Words'
    )

    # Display the chart in Streamlit
    st.plotly_chart(word_fig)

def plot_sentence_count_comparison(turns, sentence_counts, current_speaker, previous_speakers_data=None):
    # Create a line graph for sentence count comparison over all turns by the current speaker
    sentence_fig = go.Figure()

    # Add current speaker's sentence count as a line with markers
    sentence_fig.add_trace(go.Scatter(
        x=turns, y=sentence_counts, mode='lines+markers', name=f'{current_speaker} Sentence Count', line=dict(color='green')
    ))

    # Add previous speakers' sentence counts as lines and markers if data is provided
    if previous_speakers_data is not None:
        for speaker, data in previous_speakers_data.items():
            speaker_turns = data['turns']
            speaker_sentence_counts = data['sentence_counts']
            sentence_fig.add_trace(go.Scatter(
                x=speaker_turns, y=speaker_sentence_counts, mode='lines+markers', name=f'{speaker} Sentence Count'
            ))

    # Update layout for the figure
    sentence_fig.update_layout(
        title=f'Sentence Count Progression for {current_speaker}',
        xaxis_title='Turn Number',
        yaxis_title='Number of Sentences'
    )

    # Display the chart in Streamlit
    st.plotly_chart(sentence_fig)



# Function to gather topics from all claim fields
def gather_topics(selected_turn):
    topics = []

    # Gather topics from facts, values, and policies
    if 'claim_of_facts_topic' in selected_turn and selected_turn['claim_of_facts_topic']:
        topics.append(selected_turn['claim_of_facts_topic'])

    if 'claim_of_value_topic' in selected_turn and selected_turn['claim_of_value_topic']:
        topics.append(selected_turn['claim_of_value_topic'])

    # For claim_of_policy_topics, ensure it's a list and extend topics with all elements
    if 'claim_of_policy_topics' in selected_turn and selected_turn['claim_of_policy_topics']:
        topics.extend(selected_turn['claim_of_policy_topics'])  # Assuming this is a list

    # Additional checks to ensure other potential topic fields are included
    if 'additional_facts_topics' in selected_turn and selected_turn['additional_facts_topics']:
        topics.extend(selected_turn['additional_facts_topics'])

    if 'additional_value_topics' in selected_turn and selected_turn['additional_value_topics']:
        topics.extend(selected_turn['additional_value_topics'])

    if 'additional_policy_topics' in selected_turn and selected_turn['additional_policy_topics']:
        topics.extend(selected_turn['additional_policy_topics'])

    return topics

def clean_topics(topics):
    if isinstance(topics, list):
        if len(topics) > 1:
            # Join all but the last topic with commas, then add 'and' before the last topic
            return ', '.join(topics[:-1]) + f", and {topics[-1]}"
        else:
            # If only one topic exists, return it as is
            return topics[0] if topics else ''
    else:
        # If it's not a list, just return the value as is
        return topics


def gather_claims_arguments_topics(selected_turn):
    data = []

    # Claims and arguments hierarchy - correct field names
    claims_hierarchy = {
        'Fact Claims': selected_turn.get('claim_of_facts_topics', []),  # Adjusted field name
        'Value Claims': selected_turn.get('claim_of_value_topics', []),  # Ensure this is correct too
        'Policy Claims': selected_turn.get('claim_of_policy_topics', [])
    }

    for claim_type, topics in claims_hierarchy.items():
        if topics:  # Check if the topics are not empty or None
            if isinstance(topics, list):
                for topic in topics:
                    data.append([claim_type, topic])
            elif isinstance(topics, str) and topics.strip():  # Ensure it's a non-empty string
                data.append([claim_type, topics])

    return data

# Function to gather impacted groups for claims and arguments
def gather_impacted_groups(selected_turn):
    data = []

    impacted_groups_hierarchy = {
        'Fact Claims': selected_turn.get('claim_of_facts_impacted_groups_populations', []),
        'Value Claims': selected_turn.get('claim_of_value_impacted_groups_populations', []),
        'Policy Claims': selected_turn.get('claim_of_policy_impacted_groups_populations', [])
    }

    for claim_type, impacted_groups in impacted_groups_hierarchy.items():
        if isinstance(impacted_groups, list):
            for group in impacted_groups:
                data.append([claim_type, group])
        elif isinstance(impacted_groups, str):
            data.append([claim_type, impacted_groups])

    return data


# Function to create sunburst for claims/arguments and impacted groups
def create_sunburst(data, title):
    labels = []
    parents = []

    for claim_type, association in data:
        labels.append(association)
        parents.append(claim_type)

        # Add the claim type to ensure all parts of hierarchy are present
        if claim_type not in labels:
            labels.append(claim_type)
            parents.append("")  # Top-level claim doesn't have a parent

    sunburst_fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        branchvalues="total",
    ))

    sunburst_fig.update_layout(title=title)

    return sunburst_fig


def gather_biases(selected_turn):
    # Gather bias types as a list
    biases = selected_turn.get('epl_logos_bias_types', [])

    # Return biases as a comma-separated string or an empty string if no biases
    return ', '.join(biases) if biases else ''


def gather_fallacies(selected_turn):
    # Gather fallacy types as a list
    fallacies = selected_turn.get('epl_logos_fallacy_types', [])

    # Return fallacies as a comma-separated string or an empty string if no fallacies
    return ', '.join(fallacies) if fallacies else ''


# Function to generate radar chart
def generate_radar_chart(category, selected_turn):
    if category == "Ethos":
        labels = ["Trust", "Power", "Authority", "Credibility", "Reliability"]
        values = [
            selected_turn['epl_ethos_trust'],
            selected_turn['epl_ethos_power'],
            selected_turn['epl_ethos_authority'],
            selected_turn['epl_ethos_credibility'],
            selected_turn['epl_ethos_reliability']
        ]
    elif category == "Pathos":
        labels = ["Anticipation", "Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Rage"]
        values = [
            selected_turn['epl_pathos_anticipation'],
            selected_turn['epl_pathos_joy'],
            selected_turn['epl_pathos_trust'],
            selected_turn['epl_pathos_fear'],
            selected_turn['epl_pathos_surprise'],
            selected_turn['epl_pathos_sadness'],
            selected_turn['epl_pathos_disgust'],
            selected_turn['epl_pathos_rage']
        ]
    else:
        labels = ["Premise Strength", "Soundness", "Conclusion Strength", "Validity"]
        values = [
            selected_turn['epl_logos_premise_strength'],
            selected_turn['epl_logos_conclusion_strength'],
            selected_turn['epl_logos_soundness'],
            selected_turn['epl_logos_validity']
        ]

    # Create radar chart
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=category
    ))

    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[100, 999], # Assuming the scores are normalized between 100 and 999
            ),
        ),
        showlegend=False,
        title=f"{category} Radar Chart"
    )

    return radar_fig

def get_thinking_scores_for_speaker(df, current_turn, current_speaker):
    # Filter the DataFrame for the current speaker and only include turns up to the selected turn
    speaker_df = df[(df['speaker'] == current_speaker) & (df['turn'] <= current_turn)]

    # Extract turns, system1 (intuitive), and system2 (deliberative) scores
    turns = speaker_df['turn'].tolist()
    intuitive_scores = speaker_df['system1_score'].tolist()
    deliberative_scores = speaker_df['system2_score'].tolist()

    return turns, intuitive_scores, deliberative_scores

def get_previous_thinking_scores(df, current_turn, current_speaker):
    previous_speakers_data = {}

    # Filter the DataFrame for all other speakers and limit to current and previous turns
    for speaker in df['speaker'].unique():
        if speaker != current_speaker:
            speaker_df = df[(df['speaker'] == speaker) & (df['turn'] <= current_turn)]
            turns = speaker_df['turn'].tolist()
            intuitive_scores = speaker_df['system1_score'].tolist()
            deliberative_scores = speaker_df['system2_score'].tolist()

            previous_speakers_data[speaker] = {
                'turns': turns,
                'intuitive_scores': intuitive_scores,
                'deliberative_scores': deliberative_scores
            }

    return previous_speakers_data



# Function to extract Intuitive and Deliberative thinking scores (formerly System 1 and System 2)
def get_thinking_scores(df, current_turn, speaker):
    turns = []
    intuitive_scores = []
    deliberative_scores = []

    # Normalize speaker name
    normalized_speaker = speaker.strip().lower()

    for i in range(current_turn + 1):
        if 'speaker' in df.iloc[i]:
            df_speaker = df.iloc[i]['speaker'].strip().lower()
            if df_speaker == normalized_speaker:
                turns.append(i + 1)  # Store turn number (starting from 1)
                intuitive_scores.append(df.iloc[i].get('system1_score', 0))  # Intuitive thinking (formerly system1_score)
                deliberative_scores.append(df.iloc[i].get('system2_score', 0))  # Deliberative thinking (formerly system2_score)

    return turns, intuitive_scores, deliberative_scores


def plot_thinking_scores(turns, intuitive_scores, deliberative_scores, speaker, previous_speakers_data=None):
    thinking_score_fig = go.Figure()

    # Plot Intuitive thinking (System 1) score line with markers for current speaker
    thinking_score_fig.add_trace(go.Scatter(
        x=turns, y=intuitive_scores, mode='lines+markers', name=f'{speaker} Intuitive Thinking',
        line=dict(color='blue')
    ))

    # Plot Deliberative thinking (System 2) score line with markers for current speaker
    thinking_score_fig.add_trace(go.Scatter(
        x=turns, y=deliberative_scores, mode='lines+markers', name=f'{speaker} Deliberative Thinking',
        line=dict(color='green')
    ))

    # Plot previous speakers' scores (lines + markers)
    if previous_speakers_data is not None:
        for prev_speaker, data in previous_speakers_data.items():
            prev_turns = data['turns']
            prev_intuitive_scores = data['intuitive_scores']
            prev_deliberative_scores = data['deliberative_scores']

            # Add Intuitive thinking for previous speaker (lines + markers)
            thinking_score_fig.add_trace(go.Scatter(
                x=prev_turns, y=prev_intuitive_scores, mode='lines+markers', name=f'{prev_speaker} Intuitive Thinking'
            ))

            # Add Deliberative thinking for previous speaker (lines + markers)
            thinking_score_fig.add_trace(go.Scatter(
                x=prev_turns, y=prev_deliberative_scores, mode='lines+markers', name=f'{prev_speaker} Deliberative Thinking'
            ))

    # Update layout of the graph
    thinking_score_fig.update_layout(
        title=f'Intuitive and Deliberative Thinking Progression for {speaker}',
        xaxis_title='Turn Number',
        yaxis_title='Score',
        showlegend=True
    )

    # Display the graph
    st.plotly_chart(thinking_score_fig)


def analyze_thinking(turn_number, system1_score, system2_score, speaker, system1_explanation,
                     system2_explanation, selected_turn):
    # Extract topics from the selected turn
    topics = selected_turn.get('topics', [])

    # Format topics for the explanation (join them if more than one topic exists)
    if topics:
        topic_str = ', '.join(topics)  # Convert topics list to string
    else:
        topic_str = "the discussion"

    # Analyze which system was dominant and provide an explanation
    if system1_score > system2_score:
        analysis = (f"Turn {turn_number+1}: {speaker} relied more on **Intuitive Thinking**, "
                    f"making quick, instinctive decisions during this turn. This suggests that {speaker} might "
                    f"have a strong familiarity with the topic(s) of **{topic_str}**, allowing them to make fast judgments.")
        analysis += f" {system1_explanation}"
    elif system2_score > system1_score:
        analysis = (f"Turn {turn_number+1}: {speaker} employed **Deliberative Thinking**, "
                    f"taking a slower and more analytical approach during this turn. This suggests that {speaker} may have "
                    f"found the topic(s) of **{topic_str}** more complex or unfamiliar, requiring deeper reflection.")
        analysis += f" {system2_explanation}"
    else:
        analysis = (
            f"Turn {turn_number+1}: {speaker} demonstrated a **balanced approach**, utilizing both Intuitive "
            f"and Deliberative Thinking equally. This could suggest that {speaker} is experienced in the topic(s) of **{topic_str}**, "
            f"but also recognizes the need for careful analysis in certain aspects.")
        analysis += f" {system1_explanation} and {system2_explanation}."

    return analysis

def analyze_biases_and_fallacies(selected_turn):
    # Extract topics from the selected turn
    topics = selected_turn.get('topics', [])

    # Format topics for readability
    if topics:
        cleaned_topics = ', '.join(topics)  # Convert topics list to string
    else:
        cleaned_topics = "the discussion"

    # Extract bias and fallacy explanations from the selected turn
    bias_expl = selected_turn.get('epl_logos_bias_expl', [])
    fallacies_expl = selected_turn.get('epl_logos_fallacies_expl', [])

    # Analyze biases
    if bias_expl:
        bias_analysis = f"Analysis: Biases Detected in the context of {cleaned_topics}:"
        for bias in bias_expl:
            bias_analysis += f"\n\n- {bias} This suggests that the speaker's reasoning related to **{cleaned_topics}** may have been influenced by cognitive shortcuts, leading to potential distortions in judgment."
    else:
        bias_analysis = f"No biases detected for the topic of **{cleaned_topics}**."

    # Analyze fallacies
    if fallacies_expl:
        fallacy_analysis = f"**Fallacies Detected in the context of {cleaned_topics}:**"
        for fallacy in fallacies_expl:
            fallacy_analysis += f"\n\n- {fallacy} This suggests that the speaker's arguments related to **{cleaned_topics}** may have logical inconsistencies, potentially weakening the argument."
    else:
        fallacy_analysis = f"Analysis: No fallacies detected for the topic of **{cleaned_topics}."

    return bias_analysis, fallacy_analysis



# Function to calculate averages for ethos, pathos, and logos for a specific turn
def calculate_turn_averages(turn_data):
    # Ethos columns
    ethos_columns = ['epl_ethos_authority', 'epl_ethos_trust', 'epl_ethos_credibility', 'epl_ethos_power']
    ethos_avg = turn_data[ethos_columns].mean()

    # Pathos columns
    pathos_columns = ['epl_pathos_joy', 'epl_pathos_sadness', 'epl_pathos_disgust', 'epl_pathos_fear',
                      'epl_pathos_rage', 'epl_pathos_anticipation', 'epl_pathos_surprise', 'epl_pathos_trust']
    pathos_avg = turn_data[pathos_columns].mean()

    # Logos columns
    logos_columns = ['epl_logos_premise_strength', 'epl_logos_conclusion_strength', 'epl_logos_soundness', 'epl_logos_validity']
    logos_avg = turn_data[logos_columns].mean()

    return ethos_avg, pathos_avg, logos_avg

# Function to generate a bar graph for a specific turn
def generate_bar_graph(ethos_avg, pathos_avg, logos_avg, speaker):
    fig = go.Figure()

    # Create the bar chart
    fig.add_trace(go.Bar(x=['Ethos', 'Pathos', 'Logos'], y=[ethos_avg, pathos_avg, logos_avg],
                         marker_color=['blue', 'green', 'red'], name='Averages'))

    # Update layout for the graph
    fig.update_layout(
        title=f'Ethos, Pathos, Logos Averages for {speaker}',
        xaxis_title='Category',
        yaxis_title='Average Value',
        yaxis=dict(range=[0, 999])
    )

    return fig

# Function to generate a radar chart for a specific turn
def generate_radar_epl_average_chart_(ethos_avg, pathos_avg, logos_avg, speaker):
    fig = go.Figure()

    # Add data to the radar chart
    fig.add_trace(go.Scatterpolar(
        r=[ethos_avg, pathos_avg, logos_avg],
        theta=['Ethos', 'Pathos', 'Logos'],
        fill='toself',
        name='Averages'
    ))

    # Update layout for the radar chart
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                range=[0, 999]
            )),
        title=f'Ethos, Pathos, Logos Radar Chart for {speaker}',
    )

    return fig

# Function to generate analysis for Ethos, Pathos, and Logos
def analyze_ethos_pathos_logos(turn_data, speaker):
    # Calculate Ethos average
    ethos_columns = ['epl_ethos_authority', 'epl_ethos_trust', 'epl_ethos_credibility', 'epl_ethos_power', 'epl_ethos_reliability']
    ethos_avg = turn_data[ethos_columns].mean()

    # Calculate Pathos average
    pathos_columns = ['epl_pathos_joy', 'epl_pathos_sadness', 'epl_pathos_disgust', 'epl_pathos_fear',
                      'epl_pathos_rage', 'epl_pathos_anticipation', 'epl_pathos_surprise', 'epl_pathos_trust']
    pathos_avg = turn_data[pathos_columns].mean()

    # Calculate Logos average
    logos_columns = ['epl_logos_premise_strength', 'epl_logos_conclusion_strength', 'epl_logos_soundness', 'epl_logos_validity']
    logos_avg = turn_data[logos_columns].mean()

    # Identify which of Ethos, Pathos, Logos has the highest average
    if ethos_avg > pathos_avg and ethos_avg > logos_avg:
        leading_category = "Ethos"
        leading_value = ethos_avg
    elif pathos_avg > ethos_avg and pathos_avg > logos_avg:
        leading_category = "Pathos"
        leading_value = pathos_avg
    else:
        leading_category = "Logos"
        leading_value = logos_avg

    # Gather topics from the selected turn
    topics = gather_topics(turn_data)
    topics_cleaned = clean_topics(topics)

    # Initial analysis based on the highest category
    analysis = (f"{speaker} has a higher average {leading_category} ({leading_value:.2f}) compared to the other two "
                f"categories. This suggests that {speaker} is particularly strong in {leading_category.lower()}, "
                f"indicating a focus on ")

    # If there's no topics
    topics_display = topics_cleaned if topics_cleaned else "discussion"

    if leading_category == "Ethos":
        analysis += f"credibility, authority, and trustworthiness in **{topics_display}**."
    elif leading_category == "Pathos":
        analysis += f"emotional appeal, connecting with the audience on a personal level in **{topics_display}**."
    else:
        analysis += f"logical consistency, structure, and reasoning in **{topics_display}**."

    # Now let's analyze Ethos with explanations
    analysis += "\n\n### Ethos Analysis:\n"
    ethos_explanations = {
        'Authority': turn_data.get('epl_ethos_authority_expl', 'No explanation provided.'),
        'Trust': turn_data.get('epl_ethos_trust_expl', 'No explanation provided.'),
        'Credibility': turn_data.get('epl_ethos_credibility_expl', 'No explanation provided.'),
        'Reliability': turn_data.get('epl_ethos_reliability_expl','No explanation provided.'),
        'Power': turn_data.get('epl_ethos_power_expl', 'No explanation provided.')
    }

    for trait, explanation in ethos_explanations.items():
        analysis += f"- **{trait}**: {explanation}\n"

    # Now let's analyze Pathos with explanations
    analysis += "\n### Pathos Analysis:\n"
    pathos_explanations = {
        'Joy': turn_data.get('epl_pathos_joy_expl', 'No explanation provided.'),
        'Sadness': turn_data.get('epl_pathos_sadness_expl', 'No explanation provided.'),
        'Disgust': turn_data.get('epl_pathos_disgust_expl', 'No explanation provided.'),
        'Fear': turn_data.get('epl_pathos_fear_expl', 'No explanation provided.'),
        'Rage': turn_data.get('epl_pathos_rage_expl', 'No explanation provided.'),
        'Anticipation': turn_data.get('epl_pathos_anticipation_expl', 'No explanation provided.'),
        'Surprise': turn_data.get('epl_pathos_surprise_expl', 'No explanation provided.'),
        'Trust': turn_data.get('epl_pathos_trust_expl', 'No explanation provided.')
    }

    for emotion, explanation in pathos_explanations.items():
        analysis += f"- **{emotion}**: {explanation}\n"

    # Now let's analyze Logos with explanations
    analysis += "\n### Logos Analysis:\n"
    logos_explanations = {
        'Premise Strength': turn_data.get('epl_logos_premise_strength_expl', 'No explanation provided.'),
        'Conclusion Strength': turn_data.get('epl_logos_conclusion_strength_expl', 'No explanation provided.'),
        'Soundness': turn_data.get('epl_logos_soundness_expl', 'No explanation provided.'),
        'Validity': turn_data.get('epl_logos_validity_expl', 'No explanation provided.')
    }

    for logic, explanation in logos_explanations.items():
        analysis += f"- **{logic}**: {explanation}\n"

    return analysis


# Function to gather relevancy, clarity, and topics for each turn
def get_relevancy_clarity_topics(df):
    topics_list = []
    relevancy_scores = []
    clarity_scores = []

    for i in range(len(df)):
        turn_data = df.iloc[i]

        # Gather topics
        topics = gather_topics(turn_data)

        if topics:  # Only consider turns that have topics
            for topic in topics:
                topics_list.append(topic)  # Append each topic
                relevancy_scores.append(turn_data.get('relevancy', 0))
                clarity_scores.append(turn_data.get('clarity', 0))

    return topics_list, relevancy_scores, clarity_scores


# Function to plot a heatmap for Relevancy and Clarity by Topic
def plot_heatmap_relevancy_clarity_to_topics(topics, relevancy_scores, clarity_scores):
    # Create a DataFrame for the heatmap
    heatmap_df = pd.DataFrame({
        'Topic': topics,
        'Relevancy': relevancy_scores,
        'Clarity': clarity_scores
    })

    # Pivot the data for the heatmap
    relevancy_pivot = heatmap_df.pivot_table(index='Topic', values='Relevancy', aggfunc='mean').reset_index()
    clarity_pivot = heatmap_df.pivot_table(index='Topic', values='Clarity', aggfunc='mean').reset_index()

    # Merge the two pivot tables on 'Topic'
    merged_pivot = pd.merge(relevancy_pivot, clarity_pivot, on='Topic', how='outer')

    # Use Plotly's heatmap to visualize
    fig = px.imshow(merged_pivot[['Relevancy', 'Clarity']].T,
                    labels=dict(x="Topic", y="Metric", color="Score"),
                    x=merged_pivot['Topic'], y=['Relevancy', 'Clarity'],
                    aspect='auto')

    fig.update_layout(title="Relevancy and Clarity Heatmap by Topic")

    # Display the heatmap
    st.plotly_chart(fig)

# Function to create the quadrant graph with highlighting based on relevancy_score
def create_quadrant_graph(relevancy_score):
    # Create figure
    fig = go.Figure()

    # Add quadrant lines
    fig.add_shape(type="line",
                  x0=-1, y0=0, x1=1, y1=0,
                  line=dict(color="Black", width=2))
    fig.add_shape(type="line",
                  x0=0, y0=-1, x1=0, y1=1,
                  line=dict(color="Black", width=2))

    # Define the default color for the quadrants
    colors = {
        'Right Subject, Right Verb': 'rgba(144,238,144,0.5)',  # light green
        'Wrong Subject, Right Verb': 'rgba(255,165,0,0.5)',    # light orange
        'Right Subject, Wrong Verb': 'rgba(135,206,250,0.5)',  # light blue
        'Wrong Subject, Wrong Verb': 'rgba(255,182,193,0.5)'   # light pink
    }

    # Determine the quadrant to highlight based on relevancy_score
    if relevancy_score == [1, 1]:
        highlighted_quadrant = 'Right Subject, Right Verb'
    elif relevancy_score == [-1, 1]:
        highlighted_quadrant = 'Wrong Subject, Right Verb'
    elif relevancy_score == [1, -1]:
        highlighted_quadrant = 'Right Subject, Wrong Verb'
    elif relevancy_score == [-1, -1]:
        highlighted_quadrant = 'Wrong Subject, Wrong Verb'
    else:
        highlighted_quadrant = None

    # Highlight the relevant quadrant
    if highlighted_quadrant == 'Right Subject, Right Verb':
        fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                      fillcolor=colors['Right Subject, Right Verb'],
                      line=dict(color="Black", width=1))
    elif highlighted_quadrant == 'Wrong Subject, Right Verb':
        fig.add_shape(type="rect", x0=-1, y0=0, x1=0, y1=1,
                      fillcolor=colors['Wrong Subject, Right Verb'],
                      line=dict(color="Black", width=1))
    elif highlighted_quadrant == 'Right Subject, Wrong Verb':
        fig.add_shape(type="rect", x0=0, y0=-1, x1=1, y1=0,
                      fillcolor=colors['Right Subject, Wrong Verb'],
                      line=dict(color="Black", width=1))
    elif highlighted_quadrant == 'Wrong Subject, Wrong Verb':
        fig.add_shape(type="rect", x0=-1, y0=-1, x1=0, y1=0,
                      fillcolor=colors['Wrong Subject, Wrong Verb'],
                      line=dict(color="Black", width=1))

    # Add annotations for quadrants
    fig.add_annotation(x=0.5, y=0.5,
                       text="Right Subject, Right Verb",
                       showarrow=False,
                       font=dict(size=12),
                       align="center")
    fig.add_annotation(x=-0.5, y=0.5,
                       text="Wrong Subject, Right Verb",
                       showarrow=False,
                       font=dict(size=12),
                       align="center")
    fig.add_annotation(x=0.5, y=-0.5,
                       text="Right Subject, Wrong Verb",
                       showarrow=False,
                       font=dict(size=12),
                       align="center")
    fig.add_annotation(x=-0.5, y=-0.5,
                       text="Wrong Subject, Wrong Verb",
                       showarrow=False,
                       font=dict(size=12),
                       align="center")

    # Set axis properties
    fig.update_xaxes(range=[-1, 1], zeroline=False)
    fig.update_yaxes(range=[-1, 1], zeroline=False)

    # Update layout
    fig.update_layout(title="Relevancy: Subject-Verb Quadrant Graph",
                      xaxis_title="Subject",
                      yaxis_title="Verb",
                      showlegend=False)

    return fig

# Main Streamlit display function
def show():
    initialize_session_state()

    # Streamlit app title
    st.title('Interactive Transcript')

    # Table of Contents
    st.markdown('## Table of Contents')
    st.markdown("""
    - [Turn Analysis](#turn-analysis)
       - [Rhetorical Weight](#rhetorical-weight)
         -   [# of Words per Turn](#number-of-words)
         -   [# of Sentences per Turn](#number-of-sentences)
         -   [Claims and Arguments](#claims-and-arguments)
         -   [Topics and Impacted group](#topics-and-impacted-groups)
    - [Speaker Analysis](#speaker-analysis)
       - [Intuitive and Deliberative (System) Thinking](#system-thinking)
       - [Biases and Fallacies](#distortions)
    - [Content Analysis](#content-analysis)
       - [Ethos Pathos Logos](#ethos-pathos-logos)
       - [Clarity and Relevancy](#clarity-relevancy)


    """)

    st.write('## Turn Analysis')
    st.write('### Rhetorical Weight')

    # Slider for selecting the turn number
    turn_number_display = st.slider('Select a Turn Number:', min_value=1, max_value=len(df), step=1)

    # Extract turn-specific data
    turn_number = turn_number_display - 1
    selected_turn = df.iloc[turn_number]

    # Extract content (text) for this turn
    content = selected_turn['content'] if 'content' in selected_turn else ""

    # Analyze the number of words and sentences for this content
    words, sentences = analyze_text(content)

    # Speaker information, with only the first letter capitalized
    speaker = selected_turn['speaker'] if 'speaker' in selected_turn else "Unknown"
    current_speaker = selected_turn['speaker'] if 'speaker' in selected_turn else "Unknown"


    # Gather previous turns by the same speaker
    turns, word_counts, sentence_counts = get_turns_by_speaker(df, turn_number, current_speaker)


    # Claims and arguments counts
    claims_count = {
        'Fact Claims': len(selected_turn['claim_of_facts_extractive_supporting_quotes_claim']) if 'claim_of_facts_extractive_supporting_quotes_claim' in selected_turn else 0,
        'Value Claims': len(selected_turn['claim_of_value_extractive_supporting_quotes_claim']) if 'claim_of_value_extractive_supporting_quotes_claim' in selected_turn else 0,
        'Policy Claims': len(selected_turn['claim_of_policy_extractive_supporting_quotes_claim']) if 'claim_of_policy_extractive_supporting_quotes_claim' in selected_turn else 0
    }

    arguments_count = {
        'Fact Arguments': len(selected_turn['claim_of_facts_extractive_supporting_quotes_argument']) if 'claim_of_facts_extractive_supporting_quotes_argument' in selected_turn else 0,
        'Value Arguments': len(selected_turn['claim_of_value_extractive_supporting_quotes_argument']) if 'claim_of_value_extractive_supporting_quotes_argument' in selected_turn else 0,
        'Policy Arguments': len(selected_turn['claim_of_policy_extractive_supporting_quotes_argument']) if 'claim_of_policy_extractive_supporting_quotes_argument' in selected_turn else 0
    }

    # Convert the dictionaries into lists for plotting
    claim_types = list(claims_count.keys())
    claim_values = list(claims_count.values())

    argument_types = list(arguments_count.keys())
    argument_values = list(arguments_count.values())


    # Display Turn Information
    st.write(f"### Turn {turn_number_display}")
    st.write(f"### Speaker: {speaker}")

    speaking_time = calculate_speaking_time(selected_turn)
    st.write(f"**Speaking Time:** {speaking_time}")

    # Split content into paragraphs
    paragraphs = split_into_paragraphs(content)

    # Categorize the content if it's Introduction, Questions, Responses, Rebuttals, Follow-ups, Closing Statements, Outro
    section = categorize_section(content)

    st.write(f"### Section: {section}")

    # Display content in paragraphs
    st.write("### Content:")
    for paragraph in paragraphs:
        st.write(paragraph)

    # Words and Sentences section
    st.write("### Number of Words")
    st.write(f"{words}")

    speaker_view = st.selectbox('View:', ['Current Speaker Only', 'Current and Other Speakers'])
    # Get turns and word/sentence counts for the current speaker
    turns, word_counts, sentence_counts = get_turns_by_speaker(df, turn_number, current_speaker)

    # Get word/sentence counts for previous speakers
    previous_speakers_data = get_previous_speakers_data(df, turn_number, current_speaker)

    # Plot Word Count Comparison based on the selected view
    if speaker_view == 'Current Speaker Only':
        plot_word_count_comparison(turns, word_counts, current_speaker, previous_speakers_data=None)
    else:
        plot_word_count_comparison(turns, word_counts, current_speaker, previous_speakers_data)

    # Analysis for Words with differences and immediate previous turn only
    if len(word_counts) > 1:
        word_diff = word_counts[-1] - word_counts[-2]
        if word_diff > 0:
            st.write(
                f"In the current turn {turns[-1]}, {current_speaker} said {word_counts[-1]} words, which is an increase of {word_diff} words from the previous turn {turns[-2]}, where they said {word_counts[-2]} words.")
        elif word_diff < 0:
            st.write(
                f"In the current turn {turns[-1]}, {current_speaker} said {word_counts[-1]} words, which is a decrease of {abs(word_diff)} words from the previous turn {turns[-2]}, where they said {word_counts[-2]} words.")
        else:
            st.write(
                f"In the current turn {turns[-1]}, {current_speaker} said {word_counts[-1]} words, which is the same as the previous turn {turns[-2]}.")

    st.write("### Number of Sentences")
    st.write(f"{sentences}")

    if speaker_view == 'Current Speaker Only':
        plot_sentence_count_comparison(turns, sentence_counts, current_speaker, previous_speakers_data=None)
    else:
        plot_sentence_count_comparison(turns, sentence_counts, current_speaker, previous_speakers_data)


    # Analysis for Sentences with differences and immediate previous turn only
    if len(sentence_counts) > 1:
        sentence_diff = sentence_counts[-1] - sentence_counts[-2]
        if sentence_diff > 0:
            st.write(
                f"In the current turn {turns[-1]}, {current_speaker} said {sentence_counts[-1]} sentences, which is an increase of {sentence_diff} sentences from the previous turn {turns[-2]}, where they said {sentence_counts[-2]} sentences.")
        elif sentence_diff < 0:
            st.write(
                f"In the current turn {turns[-1]}, {current_speaker} said {sentence_counts[-1]} sentences, which is a decrease of {abs(sentence_diff)} sentences from the previous turn {turns[-2]}, where they said {sentence_counts[-2]} sentences.")
        else:
            st.write(
                f"In the current turn {turns[-1]}, {current_speaker} said {sentence_counts[-1]} sentences, which is the same as the previous turn {turns[-2]}.")

    # Claims and Arguments summary
    st.write('### Claims and Arguments')
    st.write(f"{speaker} gave {sum(claim_values)} claims and {sum(argument_values)} arguments in turn {turn_number_display}.")

    # Check if there are any non-blank claims or arguments
    claim_values = [len(selected_turn.get('claim_of_facts_extractive_supporting_quotes_claim', [])),
                    len(selected_turn.get('claim_of_value_extractive_supporting_quotes_claim', [])),
                    len(selected_turn.get('claim_of_policy_extractive_supporting_quotes_claim', []))]

    argument_values = [len(selected_turn.get('claim_of_facts_extractive_supporting_quotes_argument', [])),
                       len(selected_turn.get('claim_of_value_extractive_supporting_quotes_argument', [])),
                       len(selected_turn.get('claim_of_policy_extractive_supporting_quotes_argument', []))]

    # Show topics and impacted groups regardless of claims or arguments
    st.write("### Topics and Impacted Groups")
    # Extract topics from the DataFrame
    topics = set()
    impacted_groups = set()

    # Extract topics from the DataFrame
    if 'topics' in selected_turn and selected_turn['topics']:
        # Convert topics to a list if it's not already a list
        topics.update(
            selected_turn['topics'] if isinstance(selected_turn['topics'], list) else [selected_turn['topics']])

    # Extract impacted groups using previous logic
    if 'claim_of_facts_topic' in selected_turn and selected_turn['claim_of_facts_topic']:
        impacted_groups.update(selected_turn['claim_of_facts_topic'])
    if 'claim_of_value_topic' in selected_turn and selected_turn['claim_of_value_topic']:
        impacted_groups.update(selected_turn['claim_of_value_topic'])
    if 'claim_of_policy_topics' in selected_turn and selected_turn['claim_of_policy_topics']:
        impacted_groups.update(selected_turn['claim_of_policy_topics'])

    if 'claim_of_facts_impacted_groups_populations' in selected_turn and selected_turn[
        'claim_of_facts_impacted_groups_populations']:
        impacted_groups.update(selected_turn['claim_of_facts_impacted_groups_populations'])
    if 'claim_of_value_impacted_groups_populations' in selected_turn and selected_turn[
        'claim_of_value_impacted_groups_populations']:
        impacted_groups.update(selected_turn['claim_of_value_impacted_groups_populations'])
    if 'claim_of_policy_impacted_groups_populations' in selected_turn and selected_turn[
        'claim_of_policy_impacted_groups_populations']:
        impacted_groups.update(selected_turn['claim_of_policy_impacted_groups_populations'])

    if topics or impacted_groups:
        # Display topics
        if topics:
            colors = ['#FF69B4', '#87CEFA', '#90EE90']
            st.markdown(f"**Topics**: {create_color_block(list(topics), colors)}", unsafe_allow_html=True)

        # Display impacted groups
        if impacted_groups:
            colors = ['#FFA07A', '#DDA0DD', '#ADD8E6']
            st.markdown(f"**Impacted Groups**: {create_color_block(list(impacted_groups), colors)}",
                        unsafe_allow_html=True)
    else:
        st.write("No topics or impacted groups.")

    # Check if there are any non-blank claims or arguments before showing graphs
    if any(claim_values) or any(argument_values):
        # Create two columns for side-by-side bar charts
        col1, col2 = st.columns(2)

        # Claims Bar Chart using Plotly
        if any(claim_values):
            with col1:
                st.write("### Claims")
                claims_fig = go.Figure([go.Bar(x=claim_types, y=claim_values, marker_color='blue')])
                claims_fig.update_layout(title='Types of Claims', xaxis_title='Claim Type', yaxis_title='Count')
                st.plotly_chart(claims_fig, use_container_width=True)

        # Arguments Bar Chart using Plotly
        if any(argument_values):
            with col2:
                st.write("### Arguments")
                arguments_fig = go.Figure([go.Bar(x=argument_types, y=argument_values, marker_color='green')])
                arguments_fig.update_layout(title='Types of Arguments', xaxis_title='Argument Type', yaxis_title='Count')
                st.plotly_chart(arguments_fig, use_container_width=True)

        # Gather claims/arguments topics
        claims_arguments_data = gather_claims_arguments_topics(selected_turn)

        # Gather impacted groups data
        impacted_groups_data = gather_impacted_groups(selected_turn)

        # Create sunburst for claims/arguments topics
        claims_arguments_sunburst = create_sunburst(claims_arguments_data, "Claims/Arguments Topics")

        # Create sunburst for impacted groups
        impacted_groups_sunburst = create_sunburst(impacted_groups_data, "Impacted Groups")

        # Display sunbursts side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(claims_arguments_sunburst)
        with col2:
            st.plotly_chart(impacted_groups_sunburst)


    st.markdown("[Back to Table of Contents](#table-of-contents)")
    # -------------------- Speaking Analysis Section --------------------

    st.write("## Speaker Analysis")
    st.write(f"### Turn {turn_number_display}")
    st.write(f"### Speaker: {speaker}")
    # Show the unannotated content for Speaker Analysis
    unannotated_content = selected_turn['content'] if 'content' in selected_turn else ""
    paragraphs_unannotated = split_into_paragraphs(unannotated_content)

    st.write("### Content:")
    for paragraph in paragraphs_unannotated:
        st.write(paragraph)

    st.write("### System Thinking")
    st.markdown("""
        **Intuitive (System 1) thinking** is our brain's fast, automatic, and intuitive way of thinking. It handles routine decisions and reactions without much conscious effort.

        **Examples:**
        - **Driving on a familiar route:** When you drive the same route to work every day, you often find yourself on "autopilot." You don't have to think hard about every turn or traffic light; it just happens naturally.
        - **Recognizing a friend's face:** You instantly recognize your friend's face in a crowd without having to consciously analyze each feature.
        - **Catching a ball:** When someone throws a ball to you, you instinctively move your hand to catch it without thinking about the physics involved.

        **Deliberative (System 2) thinking** is our brain's slow, deliberate, and analytical way of thinking. It kicks in when we need to solve complex problems, make important decisions, or learn something new.

        **Examples:**
        - **Solving a math problem:** When you need to solve a difficult math equation, you focus, break down the problem, and work through it step by step.
        - **Planning a vacation:** When organizing a trip, you consider different destinations, compare prices, and make detailed arrangements, which requires careful thought and research.
        - **Learning a new language:** When learning a new language, you study grammar rules, memorize vocabulary, and practice speaking, all of which require concentrated effort and time.
        """)

    st.markdown("#### How do Intuitive and Deliberative thinking fare over the course of the debate?")
    # Dropdown for selecting whether to show only the current speaker or all speakers
    speaker_view = st.selectbox('View Thinking Scores:', ['Current Speaker Only', 'Current and Other Speakers'],
                                key='thinking_scores_view')

    # Get turns and thinking scores for the current speaker
    turns, intuitive_scores, deliberative_scores = get_thinking_scores_for_speaker(df, turn_number + 1, current_speaker)

    # Get thinking scores for previous speakers
    previous_speakers_data = get_previous_thinking_scores(df, turn_number + 1, current_speaker)

    # Plot Thinking Scores Comparison based on the selected view
    if speaker_view == 'Current Speaker Only':
        plot_thinking_scores(turns, intuitive_scores, deliberative_scores, current_speaker, previous_speakers_data=None)
    else:
        plot_thinking_scores(turns, intuitive_scores, deliberative_scores, current_speaker, previous_speakers_data)

    # Get system1_score and system2_score for this turn
    system1_score = selected_turn.get('system1_score', 0)
    system2_score = selected_turn.get('system2_score', 0)

    # Get additional information for the analysis
    topics = gather_topics(selected_turn)
    topics = clean_topics(topics)
    speaker = selected_turn.get('speaker', 'Unknown Speaker')
    system1_explanation = selected_turn.get('system1_explanation',
                                            'System 1 reflects quick, automatic judgments and decisions.')
    system2_explanation = selected_turn.get('system2_explanation',
                                            'System 2 reflects slow, analytical, and reasoned decision-making.')

    analysis = analyze_thinking(turn_number, system1_score, system2_score, speaker, system1_explanation,
                                system2_explanation, selected_turn)
    st.write(analysis)

    st.write("### Distortions")
    st.markdown("""
        **Biases** are systematic patterns of deviation from norm or rationality in judgment, which often affect decisions and judgments.

        | **Bias**              | **Description**                                                                                                                                                  | **Example**                                                                                                  |
        |-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
        | Overconfidence Bias   | Overestimating one's own abilities or the accuracy of one's predictions.                                                                                         | Believing you can complete a complex project in a much shorter time than is realistic.                       |
        | Negativity Bias       | Giving more weight to negative experiences or information than positive ones.                                                                                     | Remembering a single criticism more strongly than multiple compliments.                                       |
        | Sunk Cost Fallacy     | Continuing an endeavor once an investment in money, effort, or time has been made, even if it’s no longer beneficial.                                             | Staying in a failing business because a lot of money has already been invested.                               |
        | Status Quo Bias       | Preferring things to stay the same by doing nothing or by sticking with a decision made previously.                                                               | Using the same software for years because it's familiar, despite better options being available.              |
        | Self-Serving Bias     | Attributing positive events to one’s own character but attributing negative events to external factors.                                                           | Attributing good grades to intelligence but blaming poor grades on bad teaching.                              |
        | Hindsight Bias        | Believing, after an event has occurred, that one would have predicted or expected the outcome.                                                                    | Claiming "I knew it all along" after a sports team wins a match.                                              |
        | Bandwagon Effect      | Adopting a belief or behavior because it seems popular or because others are doing it.                                                                            | Starting to wear a certain style of clothing because it's currently trendy.                                   |
        | Availability Heuristic| Overestimating the importance of information that is readily available.                                                                                          | Believing that plane crashes are more common than they are because they are highly publicized.                |
        | Anchoring Bias        | Relying too heavily on the first piece of information encountered when making decisions.                                                                          | Focusing on the first price mentioned in a negotiation, which sets the standard for the rest of the negotiation.|
        | Confirmation Bias     | Searching for, interpreting, and remembering information that confirms one’s preconceptions.                                                                      | Only reading news articles that align with one’s political beliefs.                                           |

        **Fallacies** are errors in reasoning that undermine the logic of an argument.

        | **Fallacy**             | **Description**                                                                                                    | **Example**                                                                                                  |
        |-------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
        | Begging the Question    | Assuming the truth of the conclusion of an argument in the premise.                                                | "We must trust the news because it always tells the truth."                                                  |
        | Equivocation            | Using an ambiguous term in more than one sense, thus making an argument misleading.                                | "A feather is light. What is light cannot be dark. Therefore, a feather cannot be dark."                     |
        | False Cause             | Presuming that a real or perceived relationship between things means that one is the cause of the other.           | "Every time my brother uses hairspray, it rains. My brother's hairspray causes rain."                        |
        | Red Herring             | Introducing an irrelevant topic to divert attention from the original issue.                                       | "Why worry about the environment when there are so many other problems in the world?"                        |
        | Circular Reasoning      | The reasoner begins with what they are trying to end with.                                                         | "The Bible is true because it says so in the Bible."                                                         |
        | Hasty Generalization    | Making a broad generalization based on a small sample.                                                             | "My two friends who are engineers are bad at socializing, so all engineers must be socially awkward."        |
        | Slippery Slope          | Asserting that a relatively small first step leads to a chain of related events culminating in some significant impact. | "If we allow students to redo their assignments, soon they'll expect to redo every task."                    |
        | False Dichotomy         | Presenting two options as the only possibilities when others exist.                                                | "You're either with us or against us."                                                                       |
        | Straw Man               | Misrepresenting someone’s argument to make it easier to attack.                                                   | "Person A: We should improve the public healthcare system. Person B: Person A wants free healthcare for everyone, which is unrealistic." |
        | Ad Hominem              | Attacking the person making the argument rather than the argument itself.                                          | "You can't trust John's opinion on financial issues because he's terrible at managing his own money."        |
        """)

    st.markdown("#### What kinds of biases and fallacies are associated with certain topics?")
    # Gather topics, biases, and fallacies for the current turn
    biases = gather_biases(selected_turn)
    fallacies = gather_fallacies(selected_turn)

    # Get the text for bias and fallacy explanations if available
    epl_logos_bias_text = display_supporting_quote(selected_turn.get('epl_logos_bias_text', []))
    epl_logos_fallacy_text = display_supporting_quote(selected_turn.get('epl_logos_fallacy_text', []))

    # Biases and fallacies for the selected turn
    bias_explanation = f"Supporting quote: {epl_logos_bias_text}"
    fallacy_explanation = f"Supporting quote: {epl_logos_fallacy_text}"

    # Provide an analysis for biases and fallacies with context of the topics
    bias_analysis, fallacy_analysis = analyze_biases_and_fallacies(selected_turn)
    st.write(f"Biases for turn {turn_number+1}: {biases}")
    st.write(bias_explanation)
    st.write(bias_analysis)

    st.write(f"Fallacies for turn {turn_number+1}: {fallacies}")
    st.write(fallacy_explanation)
    st.write(fallacy_analysis)

    # Back to Table of Contents link
    st.markdown("[Back to Table of Contents](#table-of-contents)")

    # -------------------- Content Analysis Section --------------------
    st.write("## Content Analysis")
    st.write(f"### Turn {turn_number_display}")
    st.write(f"### Speaker: {speaker}")

    turn_score = selected_turn.get('turn_score', None)
    cumulative_score = selected_turn.get('cumulative_score', None)

    st.write(f"#### Score for Turn {turn_number + 1}: +{turn_score} points")
    st.write(f"#### Cumulative Score up to Turn {turn_number + 1}: {cumulative_score} points")


    # Show the unannotated content for Speaker Analysis
    unannotated_content = selected_turn['content'] if 'content' in selected_turn else ""
    paragraphs_unannotated = split_into_paragraphs(unannotated_content)

    st.write("### Content:")
    for paragraph in paragraphs_unannotated:
        st.write(paragraph)

    st.write("### What Goes Into the Scoring?")
    st.markdown("""
    For each turn, the speaker will get a total score. A higher score represents a more effective speaker who excels in making their message clear, emotionally impactful, and logically sound. Such a speaker is likely to be more persuasive and engaging, maintaining the audience’s attention while delivering arguments that are credible, relevant, and well-structured.
    The score is based on several key factors that evaluate the quality and effectiveness of the speaker's rhetoric. A higher score indicates that the speaker performed better in the following areas:
    
    - **Appeal (Ethos, Pathos, Logos)**:
        - **Ethos**: This refers to the speaker's credibility and trustworthiness. A higher score in Ethos means the speaker is perceived as more reliable, knowledgeable, and authoritative on the subject.
        - **Pathos**: This evaluates the emotional appeal to the audience. A higher score in Pathos suggests that the speaker was able to connect emotionally with the audience, evoking emotions like trust, empathy, or even fear and excitement.
        - **Logos**: This measures the logical consistency and persuasiveness of the speaker's arguments. A higher score in Logos indicates that the speaker's reasoning is sound, based on clear evidence and logical structure.

    - **Clarity**:
        Clarity in communication refers to the quality of being easily understood. Clear communication ensures that the message is conveyed accurately and without ambiguity. It involves the use of precise language, logical structuring of ideas, and avoiding unnecessary complexity. It's either listed as Weak (0), Average (1), or Strong (2)

    - **Relevancy**:
        This measures how closely the speaker's content relates to the topic at hand using the right subject and verb. 

    """)

    st.write("### Ethos, Pathos, Logos")
    # Dropdown to select between Ethos, Pathos, and Logos
    category = st.selectbox("Select Category for Analysis", ["Ethos", "Pathos", "Logos"])

    # Generate and display the radar chart based on the category
    radar_fig = generate_radar_chart(category, selected_turn)
    st.plotly_chart(radar_fig)

    # Calculate averages for ethos, pathos, and logos for this turn
    ethos_avg, pathos_avg, logos_avg = calculate_turn_averages(selected_turn)

    # Generate and display the bar graph
    fig = generate_bar_graph(ethos_avg, pathos_avg, logos_avg, speaker)

    # Generate and display the radar chart
    radar_fig = generate_radar_epl_average_chart_(ethos_avg, pathos_avg, logos_avg, speaker)

    # Display the charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.plotly_chart(radar_fig, use_container_width=True)

    # Perform ethos, pathos, logos analysis
    st.write(f"### Ethos Pathos Logos Analysis for Turn {turn_number_display}: {speaker}")
    analysis = analyze_ethos_pathos_logos(selected_turn, speaker)
    st.markdown(analysis)

    st.write("### Clarity and Relevancy")

    relevancy_score = selected_turn.get('relevancy', 0)

    clarity_score = selected_turn.get('clarity', 0)
    if isinstance(clarity_score, list):
        clarity_score = clarity_score[0]

    # Extract explanations for relevancy and clarity
    relevancy_expl = selected_turn.get('relevancy_expl', 'No explanation provided.')
    clarity_expl = selected_turn.get('clarity_expl', 'No explanation provided.')

    # Display clarity score and explanation
    relevancy_fig = create_quadrant_graph(relevancy_score)
    st.plotly_chart(relevancy_fig)

    # Display relevancy score and explanation
    st.write(f"**Relevancy Score:** {relevancy_score}")
    st.write(f"**Relevancy Explanation:** Relevancy measures how relevant the turn is to the previous two turns. {relevancy_expl}")


    st.write(f"**Clarity Score:** {clarity_score}")
    st.write(f"**Clarity Explanation:** {clarity_expl}")

# Run the Streamlit display
if __name__ == "__main__":
    show()
