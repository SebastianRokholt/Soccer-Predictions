# Soccer Predictions
### Predicting the outcome of matches for the Norwegian women's soccer championships (Toppserien)
This repository contains the code for a Data Science project about predicting the outcome of soccer matches, including a simple web application built with Flask.


## The Data
The dataset was provided through the course material for INF161: Data Science at the University of Bergen, and was 
sourced from [FBRef](https://fbref.com/en/comps/185/history/Toppserien-Seasons). The main dataset contains statistics on Toppserien 
soccer matches in the years 2017 - 2019. In addition, the repository contains an overview of the planned fixtures (games) for the 2020 season, 
which was used for the final predictions at the end of the project. These predictions were submitted to the INF161 course's [Kaggle contest](https://www.kaggle.com/c/inf161-innforing-i-data-science-2021/overview). 

## The Process
**Part 1: Preparations** <br>
Data inspection, cleaning and wrangling. Feature engineering and some simple visualisations.<br>
**Part 2: Modelling** <br>
Exploratory data analysis and machine learning modelling. Evaluations and predictions on the 2020 data.<br>
**Part 3: Implementation** <br>
Creating a simple Flask application for the model, where a user can enter two Toppserien teams and receive a predicted match outcome.<br>


## Repository Content
* [raw data]() - The raw data files.
* [Part 1: preparations.ipynb]() - Cleaning the raw data files. 
* [Prepared Data]() - The prepared data files after running [preparations.ipynb]()
* [Part 2: analysis-and-modelling.ipynb]() - Exploratory data analysis, preprocessing and modelling.
* [Part 3: soccer-predictions-website]() - Files for running the Flask web application.


## Setup
1) Clone repository. To run the Flask application, download `soccer-predictions-website` and `model.pkl` as a minimum.
2) Make sure that Python 3.6 or newer is installed. I recommend running the .ipynb files in Jupyter Notebook or Google Colab. 
3) run `python -m pip install -r 'requirements.txt'` in the terminal to install dependencies.
4) run `python app.py` to run the web application locally at port 8080.
