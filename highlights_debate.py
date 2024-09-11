import pickle
import streamlit as st
import pandas as pd
import io
from nltk import Tree
import spacy
import benepar

# Load the Spacy model with benepar
# nlp = spacy.load("en_core_web_md")
# nlp.add_pipe("benepar", config={"model": "benepar_en3"})

# syntactic_categories = {
#     "S": "Sentence",
#     "NP": "Noun Phrase",
#     "VP": "Verb Phrase",
#     "PP": "Prepositional Phrase",
#     "ADJP": "Adjective Phrase",
#     "ADVP": "Adverb Phrase",
#     "SBAR": "Subordinate Clause",
#     "SBARQ": "Question introduced by a wh-word or wh-phrase",
#     "SINV": "Inverted Declarative Sentence",
#     "SQ": "Inverted Yes/No Question",
#     "WHNP": "Wh-Noun Phrase",
#     "WHADJP": "Wh-Adjective Phrase",
#     "WHADVP": "Wh-Adverb Phrase",
#     "WHPP": "Wh-Prepositional Phrase",
#     "CC": "Coordinating Conjunction",
#     "DT": "Determiner",
#     "JJ": "Adjective",
#     "JJR": "Adjective, Comparative",
#     "JJS": "Adjective, Superlative",
#     "NN": "Noun, Singular or Mass",
#     "NNS": "Noun, Plural",
#     "NNP": "Proper Noun, Singular",
#     "NNPS": "Proper Noun, Plural",
#     "PRP": "Pronoun, Personal",
#     "PRP$": "Pronoun, Possessive",
#     "RB": "Adverb",
#     "RBR": "Adverb, Comparative",
#     "RBS": "Adverb, Superlative",
#     "VB": "Verb, Base Form",
#     "VBD": "Verb, Past Tense",
#     "VBG": "Verb, Gerund or Present Participle",
#     "VBN": "Verb, Past Participle",
#     "VBP": "Verb, Non-3rd Person Singular Present",
#     "VBZ": "Verb, 3rd Person Singular Present",
#     "IN": "Preposition or Subordinating Conjunction",
#     "MD": "Modal",
#     "CD": "Cardinal Number",
#     "EX": "Existential There",
#     "FW": "Foreign Word",
#     "LS": "List Item Marker",
#     "PDT": "Predeterminer",
#     "POS": "Possessive Ending",
#     "RP": "Particle",
#     "SYM": "Symbol",
#     "TO": "To",
#     "UH": "Interjection",
#     "WDT": "Wh-Determiner",
#     "WP": "Wh-Pronoun",
#     "WP$": "Wh-Possessive Pronoun",
#     "WRB": "Wh-Adverb",
#     ".": "Sentence-final punctuation (period, question mark, exclamation mark)",
#     ",": "Comma",
#     ":": "Colon or ellipsis",
#     "-LRB-": "Left round bracket",
#     "-RRB-": "Right round bracket",
#     "`": "Opening quotation mark",
#     "''": "Closing quotation mark",
#     "#": "Number sign",
#     "$": "Dollar sign",
#     "HYPH": "Hyphen",
#     "NFP": "Non-final punctuation",
#     "SQ": "Inverted Yes/No Question",
#     "SBARQ": "Question introduced by a wh-word or wh-phrase",
#     "FRAG": "Fragment",
#     "INTJ": "Interjection",
#     "LST": "List marker",
#     "NAC": "Not a Constituent",
#     "NX": "Head of a complex noun phrase",
#     "PRN": "Parenthetical",
#     "PRT": "Particle",
#     "QP": "Quantifier Phrase",
#     "RRC": "Reduced Relative Clause",
#     "UCP": "Unlike Coordinated Phrase",
#     "X": "Unknown or unclassified",
#     "SINV": "Inverted Declarative Sentence",
#     "NX": "Complex Noun Phrase (inside an NP)",
#     "CONJP": "Conjunction Phrase (e.g., 'either/or')",
#     "LST": "List Marker (used in listings)",
#     "PRN": "Parenthetical",
#     "RRC": "Reduced Relative Clause",
#     "X": "Unknown or Unclassified Constituent",
#     "ROOT": "Root of the parse tree",
#     "TOP": "Top node of a parse tree",
#     "WHPP": "Wh-Prepositional Phrase",
#     "NX": "Complex Noun Phrase",
#     "TYPO": "Typo",
#     "EDITED": "Edited",
#     "META": "Meta Statement (e.g., 'I'd say...')",
#     "CODE": "Code",
#     "EMBED": "Embedded Sentence (used for embedding other categories)",
#     "NML": "Noun Phrase without Det/Quantifier",
#     "PDT": "Pre-determiner (e.g., 'both', 'all')",
#     "INTJ": "Interjection (e.g., 'oh', 'wow')",
#     "LST": "List item marker (e.g., '1.', 'a.')",
#     "NAC": "Not a Constituent",
#     "NML": "Nominal Modifier",
#     "NP-TMP": "Temporal Noun Phrase",
#     "RRC": "Reduced Relative Clause",
#     "UCP": "Unlike Coordinated Phrase",
#     "VP-TTL": "Title Verb Phrase",
#     "XS": "Unknown Syntactic Category",
# }

# def pretty_print_tree(parse_tree):
#     output = io.StringIO()
#     parse_tree.pretty_print(stream=output)
#     return output.getvalue()

# # Function to parse a sentence and return a tree structure
# def parse_sentence(sentence):
#     doc = nlp(sentence)
#     sent = list(doc.sents)[0]  # Assuming we're dealing with a single sentence
#     tree = sent._.parse_string
#     return tree

# def display_parse_tree(tree_str):
#     tree = Tree.fromstring(tree_str)
#     st.text(pretty_print_tree(tree))

# def extract_syntactic_categories(tree_str):
#     """Extract syntactic categories and their corresponding words from a parse tree."""
#     tree = Tree.fromstring(tree_str)
#     extracted = []

#     for subtree in tree.subtrees():
#         label = subtree.label()
#         words = " ".join(subtree.leaves())
#         full_category = syntactic_categories.get(label, label)  # Get friendly name if available
#         extracted.append((f"{label} ({full_category})", words))

#     return extracted

# # Function to display the syntactic categories in a table
# def display_syntactic_table(tree_str):
#     categories = extract_syntactic_categories(tree_str)
#     df = pd.DataFrame(categories, columns=["Syntactic Category (Label and Full Name)", "Words"])
#     st.write("##### Syntactic Categories and Phrases")
#     st.dataframe(df)

# Load the pickle file
pickle_file_path = 'September_10_2024_debate.pkl'
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Convert the data to a DataFrame (assuming it's compatible)
df = pd.DataFrame(data)

# Filter data for specific speakers (Trump and Harris)
filtered_data = df[df['speaker'].isin(['Donald Trump', 'Kamala Harris'])]



# Function to clean and combine text and track impacted groups and turn numbers
def clean_and_combine_text(facts_list, policy_list, value_list, impacted_groups_facts, impacted_groups_policy,
                           impacted_groups_value, turn_numbers):
    combined_items = []
    item_types = []  # To store the type of each item (fact, policy, value)
    impacted_groups = []  # To store the impacted groups
    combined_turn_numbers = []  # To store turn numbers

    def flatten_and_clean(items, item_type, impacted_group_list, turns):
        cleaned_items = []
        for idx, item in enumerate(items):
            if isinstance(item, list):
                for subitem in item:
                    if subitem.strip():
                        cleaned_items.append(subitem.strip())
                        item_types.append(item_type)
                        impacted_groups.append(impacted_group_list[idx] if idx < len(impacted_group_list) else None)
                        combined_turn_numbers.append(turns[idx] if idx < len(turns) else None)
            elif isinstance(item, str) and item.strip():
                cleaned_items.append(item.strip())
                item_types.append(item_type)
                impacted_groups.append(impacted_group_list[idx] if idx < len(impacted_group_list) else None)
                combined_turn_numbers.append(turns[idx] if idx < len(turns) else None)
        return cleaned_items

    combined_items.extend(flatten_and_clean(facts_list, 'Fact', impacted_groups_facts, turn_numbers))
    combined_items.extend(flatten_and_clean(policy_list, 'Policy', impacted_groups_policy, turn_numbers))
    combined_items.extend(flatten_and_clean(value_list, 'Value', impacted_groups_value, turn_numbers))

    return combined_items, item_types, impacted_groups, combined_turn_numbers

# Helper function to display impacted groups as color-coded blocks with dark colors
def display_impacted_groups(groups):
    dark_colors = ['#2e3b4e', '#3b4f61', '#4e617b', '#614870', '#773b4a', '#8b2c2a', '#2c4b30', '#3e5159']

    if groups:
        formatted_groups = ' '.join([
            f"<span style='background-color:{dark_colors[i % len(dark_colors)]}; color: white; padding: 5px; border-radius: 5px;'>{group}</span>"
            for i, group in enumerate(groups)
        ])
        st.markdown(f"**Impacted Group:** {formatted_groups}", unsafe_allow_html=True)
    else:
        st.write("No impacted groups specified")

# Helper function to display Ethos, Pathos, Logos explanations (only the explanations)
def display_epl_explanations(row, selected_claim_idx):
    def clean_explanation(explanation):
        if isinstance(explanation, list):
            return ', '.join(explanation) if explanation else 'N/A'
        return explanation if explanation else 'N/A'

    st.write("#### Ethos, Pathos, Logos Analysis:")

    # Display Ethos explanations
    st.write("**Ethos**:")
    st.write(f"- Trust: {row['epl_ethos_trust_expl'][selected_claim_idx]}")
    st.write(f"- Power: {row['epl_ethos_power_expl'][selected_claim_idx]}")
    st.write(f"- Authority: {row['epl_ethos_authority_expl'][selected_claim_idx]}")
    st.write(f"- Credibility: {row['epl_ethos_credibility_expl'][selected_claim_idx]}")
    st.write(f"- Reliability: {row['epl_ethos_reliability_expl'][selected_claim_idx]}")

    # Display Pathos explanations
    st.write("**Pathos**:")
    st.write(f"- Anticipation: {row['epl_pathos_anticipation_expl'][selected_claim_idx]}")
    st.write(f"- Joy: {row['epl_pathos_joy_expl'][selected_claim_idx]}")
    st.write(f"- Trust: {row['epl_pathos_trust_expl'][selected_claim_idx]}")
    st.write(f"- Fear: {row['epl_pathos_fear_expl'][selected_claim_idx]}")
    st.write(f"- Surprise: {row['epl_pathos_surprise_expl'][selected_claim_idx]}")
    st.write(f"- Sadness: {row['epl_pathos_sadness_expl'][selected_claim_idx]}")
    st.write(f"- Disgust: {row['epl_pathos_disgust_expl'][selected_claim_idx]}")
    st.write(f"- Rage: {row['epl_pathos_rage_expl'][selected_claim_idx]}")

    # Display Logos explanations
    st.write("**Logos**:")
    st.write(f"- Premise: {row['epl_logos_premise_expl'][selected_claim_idx]}")
    st.write(f"- Conclusion: {row['epl_logos_conclusion_expl'][selected_claim_idx]}")
    st.write(f"- Soundness: {row['epl_logos_soundness_expl'][selected_claim_idx]}")
    st.write(f"- Validity: {row['epl_logos_validity_expl'][selected_claim_idx]}")

    # Display Bias and Fallacy explanations
    st.write("#### Bias and Fallacies Analysis:")
    st.write(f"- Bias Type(s): {clean_explanation(row.get('epl_logos_bias_types', [[]])[selected_claim_idx])}")
    st.write(f"- Bias Explanation: {clean_explanation(row.get('epl_logos_bias_expl', [''])[selected_claim_idx])}")
    st.write(f"- Fallacy Type(s): {clean_explanation(row.get('epl_logos_fallacy_types', [[]])[selected_claim_idx])}")
    st.write(f"- Fallacy Explanation: {clean_explanation(row.get('epl_logos_fallacy_expl', [''])[selected_claim_idx])}")

    # Display Intuitive and Deliberative explanations
    st.write("#### Intuitive Thinking and Deliberative Thinking")
    st.write(f"- Intuitive Thinking: {clean_explanation(row.get('system1_explanation', [''])[selected_claim_idx])}")
    st.write(f"- Deliberative Thinking: {clean_explanation(row.get('system2_explanation', [''])[selected_claim_idx])}")

def show():
    st.title("Phases")

    # Ensure the dataset is group by topics
    if 'Phase' not in df.columns:
        st.error("The 'Phase' field is missing in the dataset.")
        return

    # Group by 'Phase' and count occurrences (if needed)
    grouped_phases = df.groupby('Phase').size().reset_index(name='count')

    # Dropdown for selecting a phase
    phase_list = grouped_phases['Phase'].unique()
    selected_phase = st.selectbox('Select a Phase:', phase_list)

    # Filter the data based on the selected phase
    filtered_data_by_phase = df[df['Phase'] == selected_phase]
    speakers = filtered_data_by_phase['speaker'].unique()
    grouped_by_speaker = filtered_data_by_phase.groupby('speaker')

    # Sort the filtered data by turn numbers
    sorted_phase_data = filtered_data_by_phase.sort_values(by='turn')

    for speaker, speaker_data in grouped_by_speaker:
        st.write(f"### Speaker: {speaker}")

        # Sort the speaker data by turn number
        sorted_speaker_data = speaker_data.sort_values(by='turn')

        # Display the selected phase content with proper numbering
        for idx, row in sorted_speaker_data.iterrows():
            turn_number = row['turn']
            content = row['content']
            st.markdown(f"**{selected_phase} {idx + 1}** | **Turn {turn_number}**: {content}")

        st.write("---")

    st.write("---")

    st.title("Topics")

    # Ensure the dataset is grouped by topics
    if 'Topic' not in df.columns:
        st.error("The 'Topic' field is missing in the dataset.")
        return

    # Group by 'Topic' and count occurrences
    grouped_topics = df.groupby('Topic').size().reset_index(name='count')

    # Dropdown for selecting topic category
    if grouped_topics.empty:
        st.warning("No topics found in the dataset.")
        return

    topic_list = grouped_topics['Topic'].unique()
    selected_topic = st.selectbox('Select a Topic:', topic_list)


    # Second dropdown for claims and arguments
    selected_option = st.selectbox(
        'Select Claims or Arguments:',
        ['Claims', 'Arguments']
    )

    # Function to sort items by turn numbers
    def sort_by_turns(items, types, impacted_groups, turn_numbers):
        combined_data = list(zip(turn_numbers, items, types, impacted_groups))
        combined_data.sort(key=lambda x: x[0])  # Sort by turn number
        sorted_turn_numbers, sorted_items, sorted_types, sorted_groups = zip(*combined_data)
        return sorted_items, sorted_types, sorted_groups, sorted_turn_numbers

    # Process Claims section
    if selected_option == 'Claims':
        claim_columns = [
            'claim_of_facts_extractive_supporting_quotes_claim',
            'claim_of_policy_extractive_supporting_quotes_claim',
            'claim_of_value_extractive_supporting_quotes_claim',
            'claim_of_facts_impacted_groups_populations',
            'claim_of_policy_impacted_groups_populations',
            'claim_of_value_impacted_groups_populations',
            'turn',

            # Ethos columns
            'epl_ethos_power_expl',
            'epl_ethos_authority_expl',
            'epl_ethos_trust_expl',
            'epl_ethos_credibility_expl',
            'epl_ethos_reliability_expl',

            # Pathos columns
            'epl_pathos_anticipation_expl',
            'epl_pathos_joy_expl',
            'epl_pathos_trust_expl',
            'epl_pathos_fear_expl',
            'epl_pathos_surprise_expl',
            'epl_pathos_sadness_expl',
            'epl_pathos_disgust_expl',
            'epl_pathos_rage_expl',

            # Logos columns
            'epl_logos_premise_expl',
            'epl_logos_conclusion_expl',
            'epl_logos_soundness_expl',
            'epl_logos_validity_expl',

            # Bias and fallacy columns
            'epl_logos_bias_types',
            'epl_logos_bias_expl',
            'epl_logos_fallacy_types',
            'epl_logos_fallacy_expl',

            # System 1 and 2 columns
            'system1_explanation',
            'system2_explanation'
        ]

        claims_data = filtered_data[
            (filtered_data['Topic'].apply(lambda x: selected_topic in x)) &
            (filtered_data[claim_columns].notnull().any(axis=1))
            ][['speaker'] + claim_columns]

        grouped_claims = claims_data.groupby('speaker').agg({
            'claim_of_facts_extractive_supporting_quotes_claim': list,
            'claim_of_policy_extractive_supporting_quotes_claim': list,
            'claim_of_value_extractive_supporting_quotes_claim': list,
            'claim_of_facts_impacted_groups_populations': list,
            'claim_of_policy_impacted_groups_populations': list,
            'claim_of_value_impacted_groups_populations': list,
            'turn': list,

            # Ethos columns
            'epl_ethos_power_expl': list,
            'epl_ethos_authority_expl': list,
            'epl_ethos_trust_expl': list,
            'epl_ethos_credibility_expl': list,
            'epl_ethos_reliability_expl': list,

            # Pathos columns
            'epl_pathos_anticipation_expl': list,
            'epl_pathos_joy_expl': list,
            'epl_pathos_trust_expl': list,
            'epl_pathos_fear_expl': list,
            'epl_pathos_surprise_expl': list,
            'epl_pathos_sadness_expl': list,
            'epl_pathos_disgust_expl': list,
            'epl_pathos_rage_expl': list,

            # Logos columns
            'epl_logos_premise_expl': list,
            'epl_logos_conclusion_expl': list,
            'epl_logos_soundness_expl': list,
            'epl_logos_validity_expl': list,

            # Bias and fallacy columns
            'epl_logos_bias_types': list,
            'epl_logos_bias_expl': list,
            'epl_logos_fallacy_types': list,
            'epl_logos_fallacy_expl': list,

            # System 1 and 2 columns
            'system1_explanation': list,
            'system2_explanation': list
        }).reset_index()

        for _, row in grouped_claims.iterrows():
            st.write(f"**Speaker: {row['speaker']}**")

            combined_claims, claim_types, impacted_groups, turn_numbers = clean_and_combine_text(
                row['claim_of_facts_extractive_supporting_quotes_claim'],
                row['claim_of_policy_extractive_supporting_quotes_claim'],
                row['claim_of_value_extractive_supporting_quotes_claim'],
                row['claim_of_facts_impacted_groups_populations'],
                row['claim_of_policy_impacted_groups_populations'],
                row['claim_of_value_impacted_groups_populations'],
                row['turn']
            )

            # Sort the claims by turn number
            combined_claims, claim_types, impacted_groups, turn_numbers = sort_by_turns(
                combined_claims, claim_types, impacted_groups, turn_numbers
            )

            max_length = min(len(combined_claims), len(turn_numbers))
            for i in range(max_length):
                st.markdown(f"**Claim {i + 1}**  |  **Turn {turn_numbers[i]}**  :  {combined_claims[i]}")

            claims_dropdown = st.selectbox(
                f"Select a Claim to Focus on for {row['speaker']}:",
                [f"Claim {i + 1}" for i in range(len(combined_claims))],
                key=f"{row['speaker']}_claims"
            )

            selected_claim_idx = int(claims_dropdown.split()[-1]) - 1
            st.write(f"**Selected Claim**: {combined_claims[selected_claim_idx]}")
            st.write(f"**Claim Type**: {claim_types[selected_claim_idx]}")
            st.write(f"**Turn Number**: {turn_numbers[selected_claim_idx]}")
            display_impacted_groups(impacted_groups[selected_claim_idx])
            display_epl_explanations(row, selected_claim_idx)
            # parsed_tree = parse_sentence(combined_claims[selected_claim_idx])
            # st.write("#### Parsed Tree:")
            # display_parse_tree(parsed_tree)
            # display_syntactic_table(parsed_tree)
            st.write("---")

    # Process Arguments section (similar logic to Claims)
    if selected_option == 'Arguments':
        argument_columns = [
            'claim_of_facts_extractive_supporting_quotes_argument',
            'claim_of_policy_extractive_supporting_quotes_argument',
            'claim_of_value_extractive_supporting_quotes_argument',
            'claim_of_facts_impacted_groups_populations',
            'claim_of_policy_impacted_groups_populations',
            'claim_of_value_impacted_groups_populations',
            'turn',

            # Ethos columns
            'epl_ethos_power_expl',
            'epl_ethos_authority_expl',
            'epl_ethos_trust_expl',
            'epl_ethos_credibility_expl',
            'epl_ethos_reliability_expl',

            # Pathos columns
            'epl_pathos_anticipation_expl',
            'epl_pathos_joy_expl',
            'epl_pathos_trust_expl',
            'epl_pathos_fear_expl',
            'epl_pathos_surprise_expl',
            'epl_pathos_sadness_expl',
            'epl_pathos_disgust_expl',
            'epl_pathos_rage_expl',

            # Logos columns
            'epl_logos_premise_expl',
            'epl_logos_conclusion_expl',
            'epl_logos_soundness_expl',
            'epl_logos_validity_expl',

            # Bias and fallacy columns
            'epl_logos_bias_types',
            'epl_logos_bias_expl',
            'epl_logos_fallacy_types',
            'epl_logos_fallacy_expl',

            # System 1 and 2 columns
            'system1_explanation',
            'system2_explanation'
        ]

        arguments_data = filtered_data[
            (filtered_data['Topic'].apply(lambda x: selected_topic in x)) &
            (filtered_data[argument_columns].notnull().any(axis=1))
            ][['speaker'] + argument_columns]

        grouped_arguments = arguments_data.groupby('speaker').agg({
            'claim_of_facts_extractive_supporting_quotes_argument': list,
            'claim_of_policy_extractive_supporting_quotes_argument': list,
            'claim_of_value_extractive_supporting_quotes_argument': list,
            'claim_of_facts_impacted_groups_populations': list,
            'claim_of_policy_impacted_groups_populations': list,
            'claim_of_value_impacted_groups_populations': list,
            'turn': list,

            # Ethos columns
            'epl_ethos_power_expl': list,
            'epl_ethos_authority_expl': list,
            'epl_ethos_trust_expl': list,
            'epl_ethos_credibility_expl': list,
            'epl_ethos_reliability_expl': list,

            # Pathos columns
            'epl_pathos_anticipation_expl': list,
            'epl_pathos_joy_expl': list,
            'epl_pathos_trust_expl': list,
            'epl_pathos_fear_expl': list,
            'epl_pathos_surprise_expl': list,
            'epl_pathos_sadness_expl': list,
            'epl_pathos_disgust_expl': list,
            'epl_pathos_rage_expl': list,

            # Logos columns
            'epl_logos_premise_expl': list,
            'epl_logos_conclusion_expl': list,
            'epl_logos_soundness_expl': list,
            'epl_logos_validity_expl': list,

            # Bias and fallacy columns
            'epl_logos_bias_types': list,
            'epl_logos_bias_expl': list,
            'epl_logos_fallacy_types': list,
            'epl_logos_fallacy_expl': list,

            # System 1 and 2 columns
            'system1_explanation': list,
            'system2_explanation': list
        }).reset_index()

        for _, row in grouped_arguments.iterrows():
            st.write(f"**Speaker: {row['speaker']}**")

            combined_arguments, argument_types, impacted_groups, turn_numbers = clean_and_combine_text(
                row['claim_of_facts_extractive_supporting_quotes_argument'],
                row['claim_of_policy_extractive_supporting_quotes_argument'],
                row['claim_of_value_extractive_supporting_quotes_argument'],
                row['claim_of_facts_impacted_groups_populations'],
                row['claim_of_policy_impacted_groups_populations'],
                row['claim_of_value_impacted_groups_populations'],
                row['turn']
            )

            # Sort the arguments by turn number
            combined_arguments, argument_types, impacted_groups, turn_numbers = sort_by_turns(
                combined_arguments, argument_types, impacted_groups, turn_numbers
            )

            max_length = min(len(combined_arguments), len(turn_numbers))
            for i in range(max_length):
                st.markdown(f"**Argument {i + 1}**  |  **Turn {turn_numbers[i]}**  :  {combined_arguments[i]}")

            arguments_dropdown = st.selectbox(
                f"Select an Argument to Focus on for {row['speaker']}:",
                [f"Argument {i + 1}" for i in range(len(combined_arguments))],
                key=f"{row['speaker']}_arguments"
            )

            selected_argument_idx = int(arguments_dropdown.split()[-1]) - 1
            st.write(f"**Selected Argument**: {combined_arguments[selected_argument_idx]}")
            st.write(f"**Argument Type**: {argument_types[selected_argument_idx]}")
            st.write(f"**Turn Number**: {turn_numbers[selected_argument_idx]}")
            display_impacted_groups(impacted_groups[selected_argument_idx])
            display_epl_explanations(row, selected_argument_idx)
            # parsed_tree = parse_sentence(combined_arguments[selected_argument_idx])
            # st.write("#### Parsed Tree:")
            # display_parse_tree(parsed_tree)
            # display_syntactic_table(parsed_tree)
            st.write("---")

if __name__ == '__main__':
    show()
