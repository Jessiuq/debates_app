from streamlit_navigation_bar import st_navbar
import streamlit as st
import debate_streamlit
import highlights_debate
import debate_summary

page = st_navbar(["Summary", "Interactive Transcript", "Highlights"])

if page == "Summary":
    debate_summary.show()

elif page == "Interactive Transcript":
    debate_streamlit.show()

elif page == "Highlights":
    highlights_debate.show()