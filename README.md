# Disaster Response Pipeline Project
This repository contains the files for a web app that utilizes ETL and ML Pipelines to enable an emergency worker to input a message and get classification results in several categories. The web app will also display visualizations of the data.

### Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.

## Instructions <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


## Project Motivation<a name="motivation"></a>
The goal of the project is to classify the disaster messages into categories to enable disaster response groups to perform more efficiently. Messages and category data from [Figure Eight](https://appen.com/) was used to build a model for an API that classifies disaster messages. Through a web app, the user can input a new message and get classification results in several categories. The web app also displays visualizations for all of the Figure Eight disaster response data, grouping messages into genres and categories.

## File Descriptions <a name="files"></a>

An ETL pipeline was developed to extract data from the messages and categories files, transform the data for NLP, and load the data into a SQL Lite database. These files are located in the "Data" folder. 
    Data folder contents:
        -disaster_categories.csv
        -disaster_messages.csv
        -process_data.py

A Machine Learning pipeline was developed to incorporate Count Vectorizer, Tfidf Transformer, and a Multi Output Classifier and utilize Gridsearch to find the best performing parameters and output a pickle file. These are located in the "models" folder.
    Models folder contents:
        -train_classifier.py
        -classifier.pkl
        
A web app was developed to enable the user to classify new messages, incorporating the machine learning model from the Machine Learning pipeline. The app outputs categories for the submitted message, as well as descriptive charts. The files for the web app are located in the "app" folder.
    App folder contents:
        -run.py
        -go.html
        -master.html

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to [Figure Eight](https://appen.com/) for the data and [Udacity] (https://udacity.com/) for the training. Otherwise, feel free to use the code here as you would like! 
