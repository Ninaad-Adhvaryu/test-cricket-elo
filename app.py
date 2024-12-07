import streamlit as st
import pandas as pd

# Load CSV files
upsets_df = pd.read_csv("./greatest_upsets.csv", encoding="utf-8")
no_1_days_df = pd.read_csv("./days_at_no_1_corrected.csv", encoding="utf-8")

# Title of the app
st.title("Interactive ELO Analysis")

# Create tabs for navigation
tab1, tab2, tab3 = st.tabs(["ELO Rating Graph", "Greatest Upsets", "Days at No. #1"])

# Tab 1: ELO Rating Graph
with tab1:
    st.subheader("ELO Rating Graph")
    # Read the HTML file
    with open("./ELO_Rating_Plot.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    st.components.v1.html(html_content, height=600)  # Embed the HTML graph

# Tab 2: Greatest Upsets
with tab2:
    st.subheader("Top 5 Greatest Upsets")
    st.dataframe(upsets_df)

# Tab 3: Days Spent at No. #1
with tab3:
    st.subheader("Days Spent at No. #1 by Each Team")
    st.dataframe(no_1_days_df)
