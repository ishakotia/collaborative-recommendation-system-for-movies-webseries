Requirements before execution
----------------------------------
Install Python 3 using homebrew (brew install python) or by manually installing the package from https://www.python.org.

python -m pip install -U pip

python -m pip install mysql-connector-python

pip install pandas

pip install numpy

pip install -U scikit-learn

pip install scikit-metrics

pip install -U matplotlib

pip install seaborn

pip install yellowbrick

pip install mlxtend 

MySQL DB installation: https://dev.mysql.com/doc/refman/8.0/en/installing.html

MySQL Workbench Installation: https://dev.mysql.com/downloads/workbench/

IMPORTANT: Before Importing "movie_series_rec_movie_series_data.sql" and "movie_series_rec_movie_series_userid.sql", Please create schema with a name "movie_series_rec"

For importing the sql files into MySQL DB: https://dev.mysql.com/doc/workbench/en/wb-admin-export-import-management.html

Above step is crucial before running the jupyter notebook files or Running the Application Server.

----------------------------------------------------------------------------------------------------------------------------

Movie_WebSeries_Rec__Apriori.ipynb and Movie_WebSeries_Rec_Kmeans_Cluster.ipynb can be run in Jupyter Notebook. Before running, please update MySQL username and password in both the files, currently it is set "root"


----------------------------------------------------------------------------------------------------------------------------

K-Mean-Flask-App Server

- Python 3 is required to run the app

- MySQL is required to be installed and loaded with both tables

- k_means.py needs to be updated to provide MySQL username and password, currently it is set "root" for both

- After getting into the path of K-Mean-Flask-App, run command "python3 main.py" in the terminal/Command prompt
 
- App will start on port 5000 in around 60 seconds.

- Once the app is started, open the browser and log on to http://localhost:5000

- This open a UI which has search bar in which movie tilte or series title can be submitted. For e.g. movie: "Jumanji: Welcome to the Jungle"  or series: "The Boys"

- Press Submit button and recommendation will be shown. Press submit again if recommendation doesn't seem good, it will then get refreshed.

- Recommendations contain Poster, Title, IMDB scores, Type of content, Genre and Year.