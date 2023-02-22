# For loading data and data manipulation
import pandas as pd
# For mathematical computations
import numpy as np
# For model building
from sklearn.cluster import KMeans
# For finding the optimal number of k
from sklearn.metrics import silhouette_score
# For evaluating performance of the model
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector as connection
import warnings
warnings.filterwarnings('ignore')

movie_series_cluster_df =[]
# Create function which will allot cluster to a movie 
cluster = []




def run_model():
    print('Starting the Model......')
    # MySQL DB Connection
    try:
        mydb = connection.connect(host="localhost", database = 'movie_series_rec',user="root", passwd="root",use_pure=True)
        movie_series_query = "Select * from movie_series_data;"
        movie_series_df = pd.read_sql(movie_series_query, mydb)
        mv_srs_user_query = "Select * from movie_series_userid;"
        mv_srs_user_df = pd.read_sql(mv_srs_user_query, mydb)
        print("Data from DB loaded...")
        mydb.close()
    except Exception as e:
        mydb.close()
        print(str(e))

    #Removing Trailing Spaces from Movie/Series Titles
    movie_series_df['TITLE'] = movie_series_df['TITLE'].str.strip()

    # Check if the missing values has been removed
    movie_series_df.isna().sum()

    movie_series_df[['RATED']] = movie_series_df[['RATED']].fillna(value='Unrated')
    movie_series_df[['POSTER']] = movie_series_df[['POSTER']].fillna(value='Unavailable')
    movie_series_df[['ACTORS']] = movie_series_df[['ACTORS']].fillna(value='Unknown')
    movie_series_df.isna().sum()

    # Delete missing values
    movie_series_df.dropna(inplace= True)
    movie_series_df.isna().sum()

    # Drop the duplicated values as it is less than 10%
    movie_series_df.drop_duplicates(inplace = True)
    movie_series_df.shape

    movie_series_df.TITLE_ID.duplicated().sum()

    movie_series_df = movie_series_df.drop_duplicates(subset=['TITLE_ID'], keep=False)
    movie_series_df.shape

    movie_series_df = movie_series_df.drop_duplicates(subset=['TITLE', 'TYPE', 'IMDB_SCORE'], keep=False)
    movie_series_df.shape

    # Captalize RATED column 
    movie_series_df['RATED'] = movie_series_df['RATED'].str.upper()


    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['OPEN', 'UNRATED', 'G', 'TV-G', 'LIVRE', 'E', 'BTL', 'S', 'K-3'], 'G'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['R', 'TV-MA', 'M-PG' 'NC-17', 'AO', 'X', 'O.AL.'], 'R'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['TV-Y7-FV', 'TV-Y7'], 'TV-Y7'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['MA-13', 'TV-13', '13'], 'MA-13'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['M', '15'], 'M'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['MA-17', 'M-PG', '16'], 'MA-17'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['NOT RATED', 'UNRATED', 'Unrated', 'Not Rated'], 'PG'))


    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['M', 'MA-13', 'UNAVAILABLE'], 'PG-13'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['NC-17', 'MA-17'], 'R'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['GP', 'E10+', '12','M/PG', 'TV-PG', 'TV-14'], 'PG'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['TV-Y7','TV-Y'], 'G'))
    movie_series_df[['RATED']] = movie_series_df[['RATED']].replace(dict.fromkeys(['APPROVED','PASSED'], 'G'))
   

    # Merge movie and movie_uid to get User ID for the movie dataset
    mv_srs_user_df = pd.merge(mv_srs_user_df, movie_series_df)
    mv_srs_user_sample_df_1 = mv_srs_user_df.query('1 <= USER_ID <= 100')
    mv_srs_user_sample_df_2 = mv_srs_user_df.query('1000 <= USER_ID <= 1100')
    mv_srs_user_sample_df_3 = mv_srs_user_df.query('2000 <= USER_ID <= 2100')
    mv_srs_user_sample_df_4 = mv_srs_user_df.query('3000 <= USER_ID <= 3100')
    mv_srs_user_sample_df_5 = mv_srs_user_df.query('4000 <= USER_ID <= 4100')
    mv_srs_user_sample_df_6 = mv_srs_user_df.query('5000 <= USER_ID <= 5100')
    mv_srs_user_sample_df_7 = mv_srs_user_df.query('6000 <= USER_ID <= 6100')
    mv_srs_user_sample_df_8 = mv_srs_user_df.query('7000 <= USER_ID <= 7100')
    mv_srs_user_sample_df_9 = mv_srs_user_df.query('8000 <= USER_ID <= 8100')
    mv_srs_user_sample_df_10 = mv_srs_user_df.query('9000 <= USER_ID <= 9100')
    mv_srs_user_sample_df_11 = mv_srs_user_df.query('9900 <= USER_ID <= 10000')


    # List of your dataframes
    pdList = [mv_srs_user_sample_df_1, mv_srs_user_sample_df_2, mv_srs_user_sample_df_3, mv_srs_user_sample_df_4, mv_srs_user_sample_df_5,
              mv_srs_user_sample_df_6, mv_srs_user_sample_df_7, mv_srs_user_sample_df_8, mv_srs_user_sample_df_9, mv_srs_user_sample_df_10,
              mv_srs_user_sample_df_11] 
    mv_srs_user_sample_df = pd.concat(pdList)
    mv_srs_user_sample_df.head()

    mv_srs_user_sample_df

    # movies_df.columns

    mv_srs_user_sample_df.columns

    mv_srs_user_sample_df = mv_srs_user_sample_df.drop(['POSTER', 'ACTORS'], axis = 1)

    # Splitting Genres into different columns. Here we just create columns and put there initial value as 0
    genres = mv_srs_user_sample_df.GENRE
    a = list()
    for i in genres:
        genre = i
        a.append(genre.split(','))
    a = pd.DataFrame(a)   
    b = a[0].unique()
    for i in b:
        mv_srs_user_sample_df[i] = 0
    mv_srs_user_sample_df.head(2)

    # Assign 1 to all the columns which are present in the Genres
    for i in b:
        mv_srs_user_sample_df.loc[mv_srs_user_sample_df['GENRE'].str.contains(i), i] = 1

    mv_srs_user_sample_df.sample(10)

    # Capitalize the column names
    mv_srs_user_sample_df.columns = map(lambda x: str(x).upper(), mv_srs_user_sample_df.columns)
    columns_to_cluster = ['USER_ID', 'TITLE_ID', 'DRAMA', 'CRIME', 'COMEDY', 'HORROR', 'ACTION', 'SHORT', 'ADVENTURE', 'FANTASY', 'ANIMATION', 'BIOGRAPHY', 'DOCUMENTARY', 'MYSTERY', 'THRILLER', 'WESTERN', 'ADULT', 'SCI-FI', 'FAMILY', 'ROMANCE',
    'FILM-NOIR', 'MUSIC', 'MUSICAL', 'GAME-SHOW', 'SPORT', 'WAR', 'HISTORY', 'NEWS', 'REALITY-TV', 'TALK-SHOW']

    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    # Currently using MinMaxScaler, due to minor inertia values in KMeans
    ss = RobustScaler()

    mv_srs_user_sample_df_scaled = ss.fit_transform(mv_srs_user_sample_df[columns_to_cluster])
    # print("Base de músicas: ", songs_scaled[0,:])
    mv_srs_user_sample_df_scaled

    columns_to_cluster_scaled = ['USER_ID_SCALED', 'TITLE_ID_SCALED', 'DRAMA_SCALED', 'CRIME_SCALED', 'COMEDY_SCALED', 'HORROR_SCALED', 'ACTION_SCALED', 'SHORT_SCALED',
                                 'ADVENTURE_SCALED', 'FANTASY_SCALED', 'ANIMATION_SCALED', 'BIOGRAPHY_SCALED', 'DOCUMENTARY_SCALED', 'MYSTERY_SCALED', 'THRILLER_SCALED', 
                                 'WESTERN_SCALED', 'ADULT_SCALED', 'SCI-FI_SCALED', 'FAMILY_SCALED', 'ROMANCE_SCALED', 'FILM-NOIR_SCALED', 'MUSIC_SCALED', 'MUSICAL_SCALED', 
                                 'GAME-SHOW_SCALED', 'SPORT_SCALED', 'WAR_SCALED', 'HISTORY_SCALED', 'NEWS_SCALED', 'REALITY-TV_SCALED', 'TALK-SHOW_SCALED']

    mv_srs_user_sample_df_scaled_2 = pd.DataFrame(mv_srs_user_sample_df_scaled, columns=columns_to_cluster_scaled)
    print("Data Processing Completed...")
    # Commented out IPython magic to ensure Python compatibility.
    # %%time
    # # Model building and training
    model = KMeans(n_clusters = 9)
    model.fit(mv_srs_user_sample_df_scaled_2)
    print("KMeans Processing Completed...")

    # Creating an extra column in data for storing the cluster values
    mv_srs_user_sample_df['CLUSTER'] = model.labels_
    
    # The Cluster which occurs maximum number of times is alloted to the movie  
    mv_srs_user_sample_df.groupby("TITLE_ID").apply(lambda x: allot_cluster(x))

    cluster_1 = pd.DataFrame(cluster)

    # Rename the columns
    cluster_1.rename(columns = {0:'TITLE_ID',1:'CLUSTER'},inplace=True)

    # Merge the merged movie and series dataset and the cluster dataset
    global movie_series_cluster_df
    movie_series_cluster_df = pd.merge(cluster_1 , movie_series_df , how='outer', on='TITLE_ID')
    movie_series_cluster_df.sample(10)

    # The null values were the movies we deleted while merging the file  
    movie_series_cluster_df.isnull().sum()

    movie_series_cluster_df.dropna(inplace= True)
    movie_series_cluster_df.isnull().sum()

    movie_series_cluster_df.shape

    movie_series_cluster_df
    movie_series_cluster_df

    print("Execution Complete......")

def allot_cluster(group):
    a = pd.DataFrame(group)
    b = pd.DataFrame(a['CLUSTER'].value_counts())
    c = a.index 
    d = [a['TITLE_ID'][c[0]],int(b.idxmax())]
    cluster.append(d)

def get_recommendation_by_title(x):
    global movie_series_cluster_df
    print("Starting the get_recommendation_by_title......")
    movie_series_cluster_df = pd.DataFrame(movie_series_cluster_df)

    movie_series_temp_df = movie_series_cluster_df.loc[(movie_series_cluster_df['TITLE'] == x)]
    movie_series_temp_df = movie_series_temp_df.reset_index();
    print(movie_series_temp_df.GENRE)
    movie_series_temp_df.TITLE_ID[0]
    l = int(movie_series_temp_df.TITLE_ID[0])
    l = movie_series_cluster_df['CLUSTER'][movie_series_cluster_df.TITLE_ID == l]
    i = movie_series_cluster_df.CLUSTER == int(l)
    k = movie_series_cluster_df['TITLE'][i].sample(n = 20)
    df = pd.DataFrame(k)
    df = df.reset_index(drop = False)
    df.columns =['INDEX', 'TITLE']
    listTitleID = df.TITLE

    d = pd.DataFrame()

    for x in listTitleID:
        recommendation_df = movie_series_cluster_df.loc[movie_series_cluster_df['TITLE'] == x]
        temp = pd.DataFrame(
            {  
                'TITLE_ID': recommendation_df.TITLE_ID,
                'TITLE': recommendation_df.TITLE,
                'IMDB_SCORE': recommendation_df.IMDB_SCORE,
                'TYPE': recommendation_df.TYPE,
                'RATED': recommendation_df.RATED,
                'GENRE': recommendation_df.GENRE,
                'POSTER': recommendation_df.POSTER,
                'YEAR': recommendation_df.YEAR,
                'LANGUAGES': recommendation_df.LANGUAGES,
            }
        )

        d = pd.concat([d, temp])
    d = d.reset_index(drop = True)
    return d
