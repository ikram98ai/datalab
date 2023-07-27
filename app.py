import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from utils.preprocessing import proc_data

def get_data():
    file = st.file_uploader("Upload your csv here")
    if not file:
        file="./train.csv"
    df = pd.read_csv(file)
    return df

    
def show_table(df,proc_df):
    original,modified = st.tabs(["Modified","Original"])
    with modified:
        st.dataframe(proc_df,height=210)
    with original:
        st.dataframe(df,height=210)
    return proc_df


def main_config():
    st.set_page_config(
    page_title="Data Lab",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


def main():
    main_config()
    st.header("Data")
    df = get_data()
    proc_df = proc_data(df)
    with st.container():
        stats,table = st.columns([2,3])
        with stats:
            st.subheader("Statistics")
            show_table(df.describe(),proc_df.describe())
        with table:
            st.subheader("Table")
            show_table(df,proc_df)
    
    with st.container():
        st.subheader("Prepocessing")



if __name__ == '__main__':
    main()