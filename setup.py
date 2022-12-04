import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Aplikasi Data Mining", page_icon=":computer:", layout="wide")

st.subheader("**M. Ilham Anggis Bangkit Pamungkas**")
st.subheader("**200411100197**")

st.title("Penambangan Data :notebook:")

st.write("Klasifikasi Spesies Penguin")



tab1, tab2, tab3, tab4 = st.tabs(["Deskripsi Data", "Preprocessing", "Modelling", "Implementasi"])

with tab1:
    st.header("Deskripsi Data")
    st.markdown("[Link Data](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv)")
    penguin_raw = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    st.write(penguin_raw)
    st.subheader("Penjelasan :")
    st.caption("species")
    st.write("""
        Pada kolom species akan diinputkan jenis dari penguin yang dimaksud agar
        nanti dapat menjadi patokan atau output yang dihasilkan dari klasifikasi
    """)
    st.caption("island")
    st.write("""
        Island adalah daerah tempat tinggal dari penguin yang akan diinputkan ke dalam klasifikasi, 
        memiliki 3 inputan yaitu Torgersen, Biscoe, dan Dream
    """)
    st.caption("bill_length_mm")
    st.write("""
        Panjang paruh penguin yang diukur dari ujung belakang ke ujung depan paruh diukur
        dalam satuan milimeter(mm)
        dan akan dimasukkan sebagai inputan klasifikasi
    """)
    st.caption("depth_length_mm")
    st.write("""
        Panjang paruh penguin yang diukur dari ujung atas ke ujung bawah paruh diukur
        dalam satuan milimeter(mm)
        dan akan dimasukkan sebagai inputan klasifikasi
    """)
    st.caption("flipper_length_mm")
    st.write("""
        Ukuran dari sirip penguin sebagai inputan klasifikasi dalam satuan milimeter(mm)
    """)
    st.caption("body_mass_g")
    st.write("""
        Massa pada tubuh penguin sebagai inputan dalam satuan gram(g)
    """)
    st.caption("sex")
    st.write("""
        Jenis kelamin yang dimiliki oleh penguin, yang inputannya hanya 2 yaitu male(jantan) dan female(betina)
    """)

with tab2:
    st.header("Preprocessing")
    st.write("""
         Preprocessing adalah teknik penambangan data yang digunakan untuk mengubah data mentah dalam format yang berguna dan efisien.
    """)
    penguin_raw = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    x = penguin_raw.iloc[:,2:-1]

    st.write("Sebelum dinormalisasi")
    st.write(x.head(10))

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data = min_max_scaler.fit_transform(x)
    dataset = pd.DataFrame({'bill_length_mm':data[:,0],'bill_depth_mm':data[:,1],'flipper_length_mm':data[:,2],'body_mass_g':data[:,3]})

    st.write("Setelah dinormalisasi")
    st.write(dataset.head(10))
with tab3:
    st.header("Modelling")

with tab4:
    st.header("Implementasi")

    def user_input_features():
        island = st.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.selectbox('Sex',('male','female'))
        bill_length_mm = st.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        
        features = pd.DataFrame(data, index=[0])
        return features
        
    input_df = user_input_features()

    penguins_raw = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
    penguins = penguins_raw.drop(columns=['species'], axis=1)
    df = pd.concat([input_df,penguins],axis=0)

    encode = ['sex','island']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]
    df = df[:1]

    st.subheader("Hasil :")
    st.write(df)

    load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)

    st.subheader('Prediksi')
    penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
    st.write(penguins_species[prediction])

    st.subheader('Probabilitas Prediksi')
    st.write(prediction_proba)
