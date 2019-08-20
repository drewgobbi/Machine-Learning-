This subdirectory contains all of the code used in the project.

### Data Cleaning and Exploratory Data Analysis (EDA)
This script cleans the source datasets downloaded from FRED, linearly interpolates missing values, normalizes data and stores all of the time series in dataframes. 

It also contains some exploratory data analysis to test for difference and trend stationarity in each series. 

### Training and Testing Files 

All other files represent end to end routines for the remaining EDA, neuron configuration, LSTM training and testing, validation, and back testing. The files are labeled according their FRED designation and then diff or undiff denotes whether the model trains on a differenced or levels time series. 
