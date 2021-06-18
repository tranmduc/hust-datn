import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from Algorithm.FuzzyCMeans import FuzzyCMeans
from Algorithm.MCFCM import MCFCM
from Algorithm.FCMT2I import FCMT2I
from plot import scatter_, histogram_, box_
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
This App Performs Fuzzy Clustering Algorithm 
""")
st.write('---')

st.sidebar.header('MENU')
st.sidebar.subheader('1. Upload Data')
data_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
if data_file is not None:
    df_full = pd.read_csv(data_file)
    columns = list(df_full.columns)
    st.header("Dataset")
    st.dataframe(df_full)
    st.write('---')

    st.sidebar.subheader('2. Exploratory Data Analysis')
    dropID = st.sidebar.selectbox("Does your dataset contains ID field?", ['YES', 'NO'])
    if dropID == 'YES':
        df_full.drop(columns[0], axis=1, inplace=True)
        del columns[0]
    st.header("Exploratory Data Analysis")
    st.subheader("Summary")
    st.write(df_full.describe())

    label_cols = ['None']
    label_cols = label_cols + columns

    st.sidebar.subheader("Visualization")
    x = st.sidebar.selectbox("Choose X", columns)
    y = st.sidebar.selectbox("Choose Y", columns)
    c = st.sidebar.selectbox("Choose Categorical", label_cols)
    bins = st.sidebar.slider('Choose number of bins: ', int(2), int(20), int(10))

    if st.sidebar.button('Plot'):
        scatter_fig = scatter_(df_full, x, y, c)
        histogram = histogram_(df_full, x, c, bins)
        st.subheader('Scatter Plot')
        st.write(scatter_fig)
        st.subheader('Histogram')
        st.write(histogram)

        if c != 'None':
            box_fig = box_(df_full, x, c)
            st.subheader('Box Plot')
            st.write(box_fig)

    st.sidebar.subheader('3. Data Preprocessing')
    dropNA = st.sidebar.selectbox("Do you want to drop all row that have NA in any attribute", ['YES', 'NO'])
    if dropNA =='YES':
        df_full = df_full.dropna()

    label = st.sidebar.selectbox("Choose Categorical Label column", label_cols)
    regression_label = st.sidebar.selectbox("Choose Regression Label column", label_cols)

    cols = ['All']
    cols = cols + columns
    cols = [col for col in cols if (col != label and col != regression_label)]

    features = st.sidebar.multiselect("Choose Columns to Cluster", cols)
    if "All" in features:
        features = [col for col in cols if col not in features]

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

        original_labels1 = []
        original_labels2 = []
        if label != 'None':
            original_labels1 = list(map(str, df_full1[label]))
            original_labels2 = list(map(str, df_full2[label]))

        df1 = df_full1[features]
        df2 = df_full2[features]

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

                    db_score = davies_bouldin_score(data_frame1, my_labels)
                    ch_score = calinski_harabasz_score(data_frame1, my_labels)
                    s_score = silhouette_score(data_frame1, my_labels)
                    db.append(db_score)
                    ch.append(ch_score)
                    sh.append(s_score)

                    if label != 'None':
                        randIndex = rand_score(original_labels2, pred.T[0])
                        rand.append(randIndex)

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

                    db_score = davies_bouldin_score(data_frame1, my_labels)
                    ch_score = calinski_harabasz_score(data_frame1, my_labels)
                    s_score = silhouette_score(data_frame1, my_labels)
                    db.append(db_score)
                    ch.append(ch_score)
                    sh.append(s_score)

                    if label != 'None':
                        randIndex = rand_score(original_labels2, pred.T[0])
                        rand.append(randIndex)

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

                    db_score = davies_bouldin_score(data_frame1, my_labels)
                    ch_score = calinski_harabasz_score(data_frame1, my_labels)
                    s_score = silhouette_score(data_frame1, my_labels)
                    db.append(db_score)
                    ch.append(ch_score)
                    sh.append(s_score)

                    if label != 'None':
                        randIndex = rand_score(original_labels2, pred.T[0])
                        rand.append(randIndex)

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

        df = df_full[features]

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
                if regression_label != 'None':
                    df_full['clusters'] = my_labels
                    labels = []
                    for i in range(0, K):
                        tmp = df_full[df_full['clusters'] == i]
                        tmp = tmp[regression_label].mean()
                        labels.append(tmp)
                    st.subheader("Labels")
                    download_model(labels, filename2)
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
                if regression_label != 'None':
                    df_full['clusters'] = my_labels
                    labels = []
                    for i in range(0, K):
                        tmp = df_full[df_full['clusters'] == i]
                        tmp = tmp[regression_label].mean()
                        labels.append(tmp)
                    st.subheader("Labels")
                    download_model(labels, filename2)
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
                if regression_label != 'None':
                    df_full['clusters'] = my_labels
                    labels = []
                    for i in range(0, K):
                        tmp = df_full[df_full['clusters'] == i]
                        tmp = tmp[regression_label].mean()
                        labels.append(tmp)
                    st.subheader("Labels")
                    download_model(labels, filename2)
                st.write('---')

