import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from FuzzyCMeans import FuzzyCMeans
from MCFCM import MCFCM
from FCMT2I import FCMT2I
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from statistics import mean
import pickle
import base64

def download_model(model, filename):
    output_model = pickle.dumps(model)
    b64 = base64.b64encode(output_model).decode()
    href = f'<a href="data:file/output_model;base64,{b64}" download={filename}>Download {filename} File</a>'
    st.markdown(href, unsafe_allow_html=True)

st.title("""
FUZZY CLUSTERING APPLICATION
This App Perform Fuzzy Clustering Algorithm 
""")
st.write('---')

st.sidebar.header('MENU')

st.sidebar.subheader('1. Upload Data')
data_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if data_file is not None:
    df_full = pd.read_csv(data_file)
    st.header("Dataset")
    st.dataframe(df_full)

    st.sidebar.subheader('2. Explore Data Analysis')
    pr = ProfileReport(df_full, explorative=True)
    if st.sidebar.button('Generate Profiling Report'):
        st.header("Explore Data Analysis")
        st_profile_report(pr)

    st.sidebar.subheader('3. Data Preprocessing')
    columns = list(df_full.columns)
    cols = list(df_full.columns)
    all = 'All'
    cols.append(all)
    features = st.sidebar.multiselect("Choose Columns to Cluster", cols)
    if "All" in features:
        features = columns
    label = st.sidebar.selectbox("Choose label column", columns)

    st.sidebar.subheader('4. Fuzzy Clustering Model')
    train_type = ['YES', 'NO']
    evaluate = st.sidebar.selectbox("Include Evaluation", train_type)
    menu = ['FCM', 'FCMT2I', 'MCFCM']
    choice = st.sidebar.selectbox("Choose Clustering Algorithm", menu)
    K = st.sidebar.slider('Number of clusters', int(1), int(30), int(1))
    epsilon = 1e-4

    if evaluate == 'YES':
        ratio = st.sidebar.slider('Choose train/test split: ', float(0), float(1), float(0.8))
        l = st.sidebar.slider('Choose number of iterations: ', int(1), int(100), int(1))
        df_full1 = df_full.sample(frac=ratio)
        df_full2 = df_full.drop(df_full1.index)

        original_labels1 = list(map(str, df_full1[label]))

        original_labels2 = list(map(str, df_full2[label]))

        df1 = df_full1.dropna()  # drop all row that have NA in any attribute
        df1 = df1[features]

        df2 = df_full2.dropna()  # drop all row that have NA in any attribute
        df2 = df2[features]

        if len(features) > 0:
            df1 = pd.get_dummies(df1, dtype=float)  # Convert categorical variable into dummy/indicator variables.
            df2 = pd.get_dummies(df2, dtype=float)  # Convert categorical variable into dummy/indicator variables.

        data_frame1 = np.asarray(df1, dtype=float)
        data_frame2 = np.asarray(df2, dtype=float)

        N = len(data_frame1)

        if choice == "FCM":
            m = st.sidebar.slider('Choose m', int(1), int(10), int(2))

            if st.sidebar.button('Predict'):
                rand = []
                db = []
                ch = []
                sh = []
                for i in range(l):
                    my_clustering = FuzzyCMeans(m, data_frame1, N, K, epsilon)
                    my_cluster_centers, my_labels = my_clustering.fit()

                    pred = my_clustering.predict(my_cluster_centers, data_frame2)
                    randIndex = rand_score(original_labels2, pred.T[0])

                    db_score = davies_bouldin_score(data_frame1, my_labels)
                    ch_score = calinski_harabasz_score(data_frame1, my_labels)
                    s_score = silhouette_score(data_frame1, my_labels)

                    rand.append(randIndex)
                    db.append(db_score)
                    ch.append(ch_score)
                    sh.append(s_score)

                st.write('---')
                st.header('Performance Metrics')
                st.subheader("Rand Score")
                st.write(mean(rand))
                st.subheader("Davies Bouldin Score")
                st.write(mean(db))
                st.subheader("Calinski Harabasz Score")
                st.write(mean(ch))
                st.subheader("Silhouette Score")
                st.write(mean(sh))
                st.write('---')

        elif choice == "MCFCM":
            m_l = st.sidebar.slider('m1', float(1), float(10), float(2))
            m_u = st.sidebar.slider('m2', float(1), float(10), float(2))
            fmSmall = st.sidebar.slider('fm(Small)', float(0), float(1), float(0))
            uVery = st.sidebar.slider('uVery', float(0), float(1), float(0))
            uMore = st.sidebar.slider('uMore', float(0), float(1), float(0))
            length = st.sidebar.slider('length', int(2), int(4), int(2))

            if st.sidebar.button('Predict'):
                rand = []
                db = []
                ch = []
                sh = []
                for i in range(l):
                    my_clustering = MCFCM(m_l, m_u, data_frame1, N, K, fmSmall, uVery, uMore, length, epsilon)
                    my_cluster_centers, my_labels = my_clustering.fit()

                    pred = my_clustering.predict(my_cluster_centers, data_frame2)
                    randIndex = rand_score(original_labels2, pred.T[0])

                    db_score = davies_bouldin_score(data_frame1, my_labels)
                    ch_score = calinski_harabasz_score(data_frame1, my_labels)
                    s_score = silhouette_score(data_frame1, my_labels)

                    rand.append(randIndex)
                    db.append(db_score)
                    ch.append(ch_score)
                    sh.append(s_score)

                st.write('---')
                st.header('Performance Metrics')
                st.subheader("Rand Score")
                st.write(mean(rand))
                st.subheader("Davies Bouldin Score")
                st.write(mean(db))
                st.subheader("Calinski Harabasz Score")
                st.write(mean(ch))
                st.subheader("Silhouette Score")
                st.write(mean(sh))
                st.write('---')

        elif choice == "FCMT2I":
            m_l = st.sidebar.slider('m1', float(1), float(10), float(2))
            m_u = st.sidebar.slider('m2', float(1), float(10), float(2))
            m = st.sidebar.slider('Choose m', int(1), int(10), int(2))

            if st.sidebar.button('Predict'):
                rand = []
                db = []
                ch = []
                sh = []
                for i in range(l):
                    my_clustering = FCMT2I(data_frame1, np.random.rand(K, len(features)) * 12, m, m_l, m_u)
                    my_cluster_centers, my_labels = my_clustering.fit()

                    pred = my_clustering.predict(my_cluster_centers, data_frame2)
                    randIndex = rand_score(original_labels2, pred.T[0])

                    db_score = davies_bouldin_score(data_frame1, my_labels)
                    ch_score = calinski_harabasz_score(data_frame1, my_labels)
                    s_score = silhouette_score(data_frame1, my_labels)

                    rand.append(randIndex)
                    db.append(db_score)
                    ch.append(ch_score)
                    sh.append(s_score)

                st.write('---')
                st.header('Performance Metrics')
                st.subheader("Rand Score")
                st.write(mean(rand))
                st.subheader("Davies Bouldin Score")
                st.write(mean(db))
                st.subheader("Calinski Harabasz Score")
                st.write(mean(ch))
                st.subheader("Silhouette Score")
                st.write(mean(sh))
                st.write('---')

    else:
        filename1 = 'center.pkl'
        filename2 = 'label.pkl'

        df = df_full.dropna()  # drop all row that have NA in any attribute
        df = df[features]

        if len(features) > 0:
            df = pd.get_dummies(df, dtype=float)  # Convert categorical variable into dummy/indicator variables.

        data_frame = np.asarray(df, dtype=float)
        N = len(data_frame)

        if choice == "FCM":
            m = st.sidebar.slider('Choose m', int(1), int(10), int(2))

            if st.sidebar.button('RUN'):
                my_clustering = FuzzyCMeans(m, data_frame, N, K, epsilon)
                my_cluster_centers, my_labels = my_clustering.fit()

                st.write('---')
                st.header('Model')
                st.subheader("Centers")
                download_model(my_cluster_centers, filename1)
                st.subheader("Labels")
                download_model(my_labels, filename2)
                st.write('---')

        elif choice == "MCFCM":
            m_l = st.sidebar.slider('m1', float(1), float(10), float(2))
            m_u = st.sidebar.slider('m2', float(1), float(10), float(2))
            fmSmall = st.sidebar.slider('fm(Small)', float(0), float(1), float(0))
            uVery = st.sidebar.slider('uVery', float(0), float(1), float(0))
            uMore = st.sidebar.slider('uMore', float(0), float(1), float(0))
            length = st.sidebar.slider('length', int(2), int(4), int(2))

            if st.sidebar.button('RUN'):
                my_clustering = MCFCM(m_l, m_u, data_frame, N, K, fmSmall, uVery, uMore, length, epsilon)
                my_cluster_centers, my_labels = my_clustering.fit()

                st.write('---')
                st.header('Model')
                st.subheader("Centers")
                download_model(my_cluster_centers, filename1)
                st.subheader("Labels")
                download_model(my_labels, filename2)
                st.write('---')

        elif choice == "FCMT2I":
            m_l = st.sidebar.slider('m1', float(1), float(10), float(2))
            m_u = st.sidebar.slider('m2', float(1), float(10), float(2))
            m = st.sidebar.slider('Choose m', int(1), int(10), int(2))

            if st.sidebar.button('RUN'):
                my_clustering = FCMT2I(data_frame, np.random.rand(K, len(features)) * 12, m, m_l, m_u)
                my_cluster_centers, my_labels = my_clustering.fit()

                st.write('---')
                st.header('Model')
                st.subheader("Centers")
                download_model(my_cluster_centers, filename1)
                st.subheader("Labels")
                download_model(my_labels, filename2)
                st.write('---')

