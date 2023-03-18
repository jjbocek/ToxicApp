# A Toxic Comment Identifier App

## Data Science Notebooks
All Notebooks used for this project are located in the Final_Project Folder
* Toxic_App_Main.ipynb - The Main Notebook for the project.
* Toxic_App_Exploratory_Data_cleaning.ipynb - The notebook used for data exploratory and cleaning.
* Toxic_App_Doc2Vec.ipynb - The notebook used for anything related to using the Doc2Vec data transformation.
* Toxic_App_Rochhio_Classifer.ipynb - The notebook used for using the Rocchio_Classifier model.
  * This notebook was only used to tune the Doc2Vec models. 
* Toxic_App_Tuning_&_Evaluation_funcs.ipynb - The notebook used for model tuning and evaluation.

## A Toxic Comment Identifier App
* We developed a very simple command-line application that trains the best 3 models we evaluated and puts them into an ensemble model for the application to use. 
* The application will prompt the user for a comment and then classify it using the chosen model.
* This application resides in the Toxic_App/ directory and is called toxic_app.py.  It can simply be run by running python toxicApp.py.
