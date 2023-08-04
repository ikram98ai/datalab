from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
from utils.preprocessing import proc_data, normalize, transform_stats, select_target
from utils.pipeline import MLDataPipeline
def main_config():
    st.set_page_config(
    page_title="Data Lab",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

def get_data():
    with st. sidebar:
        file = st.file_uploader("Upload your csv here")
    if not file:
        file="./train.csv"
    df = pd.read_csv(file)
    return df


def get_X_y(df):
    with st.sidebar:
        st.subheader("Prepocessing")
        target = st.selectbox("Select Target",df.columns)
        if not target: target = "Survived"
        return select_target(df,target)
    
def normalize_data(proc_df):
    with st.sidebar:
        norm = st.selectbox("Normalize",["standard","min_max",
                     "max_abs","robust"])
        return normalize(proc_df,norm)

def show_table(df):
    data,stats = st.tabs(["Data","Statistics"])
    with data:
        st.dataframe(df,height=210)
    with stats:
        st.dataframe(transform_stats(df),height=210)

def report_pred(X,y):
    pipeline = MLDataPipeline(X,y)
    model = RandomForestClassifier(random_state=42)
    pipeline.train_model(model)
    evaluation_report = pipeline.evaluate_model(model)

    st.header("Prediction Report")
    for k,v in evaluation_report.items():
        st.write(k,v)

def main():
    main_config()
    st.header("Data")
    df = get_data()
    X,y = get_X_y(df)
    proc_df = proc_data(X)
    norm_df = normalize_data(proc_df)
    with st.container():
        show_table(norm_df)
        report_pred(norm_df,y)
    
    


if __name__ == '__main__':
    main()