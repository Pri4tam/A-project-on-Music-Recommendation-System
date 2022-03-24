
import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

st.title('Model Deployment: Music Recommendation System')

st.sidebar.header('User Input Parameters')

def user_input_features():
    track_title = st.text_input("enter Song Title","Type here...")
    
    if(st.button('Submit')):
        result = track_title.title()
        st.success(result)
    
    data = {'track_title':track_title}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features() 
st.subheader('User Input parameters')
st.write(df)

final = pd.read_csv('final.csv')
metadata = pd.read_csv('metadata.csv')
merged_meta = metadata.merge(final, on='track_id', how='inner')
merged_meta_final = merged_meta[["track_id","album_title","artist_name","genre","track.11 interest_x","track.14 listens_x","track_title"]]
final = shuffle(final)
X = final.loc[[i for i in range(0, 100)]]
Y = final.loc[[i for i in range(100, final.shape[0])]]
X = shuffle(X)
Y = shuffle(Y)

metadata = metadata.set_index('track_id')
merged_meta_final = merged_meta_final.set_index('track_id')
kmeans = KMeans(n_clusters=5)
temp_final_df = Y[:]
temp_final_df.reset_index(drop=True, inplace=True)
temp_final_df = temp_final_df.set_index('track_id', drop=False)

def fit(df, algo, flag=0):
    if flag:
        algo.fit(df)
    else:
         algo.partial_fit(df)          
    df['label'] = algo.labels_
    return (df, algo)

def predict(t, Y):
    y_pred = t[1].predict(Y)
    mode = pd.Series(y_pred).mode()
    return t[0][t[0]['label'] == mode.loc[0]]

def recommend(recommendations, meta, Y):
    dat = []
    for i in Y['track_id']:
        dat.append(i)
    genre_mode = meta.loc[dat]['genre'].mode()
    artist_mode = meta.loc[dat]['artist_name'].mode()
    title_mode = meta.loc[dat]['album_title'].mode()
    return meta[meta['genre'] == genre_mode.iloc[0]], meta[meta['artist_name'] == artist_mode.iloc[0]], meta.loc[recommendations['track_id']],meta[meta['album_title'] == title_mode.iloc[0]]

def searchSong(searchString, meta, Y):
    dat = []
    filtered_meta = meta[meta['track_title'].str.contains(searchString, case=False) == True]
    filtered_meta.sort_values('track.14 listens_x')
    intersect_meta = Y.loc[Y.index.intersection(filtered_meta.index)]
    for i in intersect_meta['track_id']:
        dat.append(i)
    genre_mode = meta.loc[dat]['genre'].mode()
    artist_mode = meta.loc[dat]['artist_name'].mode()
    print(genre_mode.shape[0])
    print(artist_mode.shape[0])
   
    return 'No Song available' if (genre_mode.shape[0] == 0) else (meta[meta['genre'] == genre_mode.iloc[0]]), 'No Song available' if (artist_mode.shape[0] == 0) else (meta[meta['artist_name'] == artist_mode.iloc[0]]), filtered_meta.head(10)
    

t = fit(X, kmeans, 1)
recommendations = predict(t, Y)
filtered_meta = merged_meta_final[merged_meta_final['track_title'].str.contains(df.iloc[0,0], case=False) == True]
output1 = recommend(recommendations, merged_meta_final, Y)

genre_default,artist_default = output1[0],output1[1]

print(df.iloc[0,0])
#filtered_meta = merged_meta_final[merged_meta_final['track_title'].str.contains(df.iloc[0,0], case=False) == True]
output2 = searchSong(df.iloc[0,0],filtered_meta, temp_final_df)

genre_r, artist_r, search_r = output2[0],output2[1],output2[2]
st.subheader('Recommended Song')
print(search_r.head())
st.write(search_r.head())

st.subheader('Recommended Song by artist')
if isinstance(artist_r, pd.DataFrame):
    st.write(artist_r.head())
    print(artist_r.head())
else:
    st.write(artist_default.head())
    
st.subheader('Recommended Song by genre')
if isinstance(genre_r, pd.DataFrame):
    st.write(genre_r.head())
    print(genre_r.head())
else:
    st.write(genre_default.head())







