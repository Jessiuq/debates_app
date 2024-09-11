import pickle
import streamlit as st
import pandas as pd

# Load the pickle file
pickle_file_path = 'September_10_2024_debate.pkl'
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Convert the data to a DataFrame (assuming it's compatible)
df = pd.DataFrame(data)

def calculate_speaking_rates(df):
    # Calculate words per minute for each turn
    df['duration'] = df['endTime'] - df['startTime']
    df['words_per_minute'] = df['word_weight'] / df['duration'] * 60

    # Calculate claims per turn
    df['claims_per_turn'] = df[['claim_of_facts_extractive_supporting_quotes_claim',
                                'claim_of_value_extractive_supporting_quotes_claim',
                                'claim_of_policy_extractive_supporting_quotes_claim']].apply(lambda row: sum([bool(claim) for claim in row]), axis=1)

    argument_columns = [
        'claim_of_facts_extractive_supporting_quotes_argument',
        'claim_of_value_extractive_supporting_quotes_argument',
        'claim_of_policy_extractive_supporting_quotes_argument'
    ]

    # Filter out columns that exist in the DataFrame
    available_argument_columns = [col for col in argument_columns if col in df.columns]

    # If argument columns exist, count the number of arguments for each row
    if available_argument_columns:
        df['argument_count'] = df[available_argument_columns].apply(lambda row: sum([bool(arg) for arg in row]), axis=1)
    else:
        df['argument_count'] = 0  # Default to 0 arguments if columns aren't found

    # Group by 'Topic' and 'speaker', then calculate the average number of arguments per topic
    grouped_df = df.groupby(['Topic', 'speaker']).agg(
        avg_arguments_per_topic=('argument_count', 'mean')
    ).reset_index()

    # Merge the calculated 'avg_arguments_per_topic' back into the original DataFrame for the speaker
    df = pd.merge(df, grouped_df[['Topic', 'speaker', 'avg_arguments_per_topic']], on=['Topic', 'speaker'], how='left')

    return df


def show():
    st.write("### Overview of the Debate")

    # Extract the unique speakers from the dataset
    speakers = df['speaker'].unique()
    exclude_speakers = ["Narrator"]
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
            
            """,
        "David Muir": """
            **Birthday:** November 8, 1973 (age 50 years)\n
            **Hometown:** Syracuse, NY\n
            **Career Bio:** David Muir is a prominent American journalist and the anchor of ABC World News Tonight, one of the most-watched evening news programs in the United States. Born in 1973, Muir began his career in journalism as a local news reporter and anchor, quickly gaining recognition for his work. He joined ABC News in 2003 as a correspondent and became a regular fixture on the network, reporting from global hotspots and covering major stories such as Hurricane Katrina, the Haiti earthquake, and the Arab Spring. His award-winning reporting helped him rise through the ranks, leading to his appointment as the anchor of ABC's flagship news program in 2014.\n\n
            Muir is known for his incisive interviewing style and his ability to connect with viewers during major news events. He has won multiple Emmy and Edward R. Murrow awards for his work, particularly for his international reporting. In addition to anchoring, Muir also serves as a co-anchor of 20/20, ABC's long-running newsmagazine program. Under his leadership, World News Tonight has maintained its position as one of the top-rated evening news broadcasts in the U.S.\n
            **Fun Facts**: 
            1)  Muir is fluent in Spanish, which has been an asset in his reporting, particularly when covering stories in Latin America and other Spanish-speaking regions.
            2) Muir has reported from over 50 countries around the globe, covering stories from war zones to natural disasters, and often immerses himself in local cultures while on assignment.
            3) As a teenager, Muir wrote letters to local news anchors for career advice and even created his own mock newscasts at home.
            4) Muir is known to prioritize fitness despite his demanding schedule. He has been seen running in Central Park and often makes time for workouts to stay healthy and energized for his busy career.

            """,
        "Linsey Davis": """
            **Birthday:** October 21, 1977 (age 46 years)\n
            **Hometown:** South Jersey, NJ\n
            **Career Bio:** Linsey Davis is an accomplished American journalist, currently serving as an anchor for ABC News Live Prime and weekend anchor for World News Tonight. She is also a correspondent for major ABC programs like Good Morning America and 20/20. Davis began her journalism career in local news, working as a reporter at several TV stations before joining ABC News in 2007. Since then, she has covered significant national and international events, including presidential elections, mass shootings, and natural disasters.\n\n 
            Davis has earned widespread recognition for her in-depth reporting and impactful interviews with key political figures and newsmakers. Linsey Davis has also become a notable voice in broadcast journalism, known for her thoughtful reporting and commitment to addressing social justice issues.\n
            **Fun Facts:**
            1) In addition to her journalism career, she is also a bestselling author of children’s books, such as The World is Awake and One Big Heart, which focus on positivity and inclusion.
            2) In her earlier years, Davis was a track and field star, excelling in sprinting events. Her athletic background instilled a strong sense of discipline and perseverance that she carries into her journalism career.
            3) Davis has won several Emmy Awards for her outstanding journalism and coverage of significant events, including her reporting on major national stories and in-depth investigative pieces.
            4) Lindsey Davis has spoken openly about how her Christian faith plays a significant role in her life and career. Her book One Big Heart was inspired by her faith and the belief in celebrating everyone’s uniqueness.

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
    speaker_df = df[df['speaker'] == selected_speaker]
    speaker_rates_df = calculate_speaking_rates(speaker_df)
    st.write(f"#### Speaking Rates for {selected_speaker}")
    avg_words_per_minute = speaker_rates_df['words_per_minute'].mean()
    avg_claims_per_turn = speaker_rates_df['claims_per_turn'].mean()
    avg_arguments_per_topic = speaker_rates_df['avg_arguments_per_topic'].mean()

    st.write(f"**Average Words per Minute**: {avg_words_per_minute:.2f}")
    st.write(f"**Average Claims per Turn**: {avg_claims_per_turn:.2f}")
    st.write(f"**Average Arguments per Topic**: {avg_arguments_per_topic:.2f}")
# Call the function in your Streamlit app
if __name__ == "__main__":
    show()
