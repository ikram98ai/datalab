from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
from utils.preprocessing import get_model, proc_data, normalize, transform_stats, get_X_y,model_classes
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


def select_target(df):
    with st.sidebar:
        st.subheader("Prepocessing")
        target = st.selectbox("Select Target",df.columns)
        return get_X_y(df,target)
    
def normalize_data(proc_df):
    with st.sidebar:
        norm = st.selectbox("Normalize",["standard","min_max",
                     "max_abs","robust"])
        return normalize(proc_df,norm)

def select_model():
    with st.sidebar:
        st.subheader("Model")
        model_name = st.selectbox("Select Model",model_classes.keys())
        return model_name
    
def show_table(df):
    st.header("Data")
    data,stats = st.tabs(["Data","Statistics"])
    with data:
        st.dataframe(df,height=210)
    with stats:
        st.dataframe(transform_stats(df),height=210)

def show_charts(df):
    st.header("Visualization")
    xcol, ycol = st.columns(2)
    x = xcol.selectbox("Select X column",df.columns)
    y = ycol.selectbox("Select Y column",df.columns)
    line, area, bar = st.tabs(["Line Chart", "Area Chart","Bar Chart"])
    line.line_chart(df,x=x,y=y)
    area.area_chart(df,x=x,y=y)
    bar.bar_chart(df,x=x,y=y)

def report_pred(X,y,model_name):
    st.header("Prediction Report")
    pipeline = MLDataPipeline(X,y)
    model = get_model(model_name)
    pipeline.train_model(model)
    evaluation_report = pipeline.evaluate_model(model)
    for k,v in evaluation_report.items():
        st.write(k,v)

def main():
    main_config()
    df = get_data()
    X,y = select_target(df)
    proc_df = proc_data(X)
    norm_df = normalize_data(proc_df)
    model_name = select_model()
    with st.container():
        show_table(norm_df)
        show_charts(proc_df)
        report_pred(norm_df,y,model_name)
    
    


if __name__ == '__main__':
    main()