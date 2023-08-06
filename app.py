import streamlit as st
import pandas as pd
from utils import proc_data, normalize, transform_stats
from utils import MLDataPipeline,classification_models,regression_models

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
    with st.sidebar:
        file = st.file_uploader("Upload your csv here")
    if not file:
        file="./titanic.csv"
    df = pd.read_csv(file)

    with st.sidebar:
        st.subheader("Prepocessing")
        target = st.selectbox("Select Target",df.columns)
        X = df.drop(target, axis=1)
        y = df[target]

    return X,y
    
def normalize_data(df):
    with st.sidebar:
        norm = st.selectbox("Normalize",["standard","min_max",
                     "max_abs","robust"])
        return normalize(df,norm)

def show_table(df):
    st.header("Data")
    data,stats = st.tabs(["Data","Statistics"])
    with data:
        st.dataframe(df,height=210)
    with stats:
        st.dataframe(transform_stats(df),height=210)

def show_charts(X,y):
    st.header("Visualization")
    df =X.copy()
    df[y.name] = y
    ycol, xcol  = st.columns(2)
    y = ycol.selectbox("Select Y column",df.columns)
    x = xcol.selectbox("Select X column",df.columns)
    line, area, bar = st.tabs(["Line Chart", "Area Chart","Bar Chart"])
    line.line_chart(df,x=x,y=y)
    area.area_chart(df,x=x,y=y)
    bar.bar_chart(df,x=x,y=y)

def drop_columns(df):
    with st.sidebar:
        columns = st.multiselect("Drop Columns",df.columns)
    return df.drop(columns, axis=1)


def show_pred_report(X,y):
    st.header("Prediction Report")
    pipeline = MLDataPipeline(X,y)
    model_name = st.selectbox("Select Model",classification_models.keys())
    if pipeline.is_classification:
        model = classification_models[model_name]()
    else:
        model = regression_models[model_name]()
    pipeline.train_model(model)
    evaluation_report = pipeline.evaluate_model(model)
    for k,v in evaluation_report.items():
        st.write(k,v)

def main():
    main_config()
    X,y = get_data()
    proc_df = proc_data(X)
    norm_df = normalize_data(proc_df)
    df = drop_columns(norm_df)

    with st.container():
        show_table(df)
        show_charts(df,y)
        show_pred_report(df,y)
    
    


if __name__ == '__main__':
    main()