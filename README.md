# Aviation Accidents Analysis - Phase 1 Project
## 1. Project Overview

Your company is expanding in to new industries to diversify its portfolio. Specifically, they are interested in purchasing and operating airplanes for commercial and private enterprises, but do not know anything about the potential risks of aircraft. You are charged with determining which aircraft are the lowest risk for the company to start this new business endeavor. You must then translate your findings into actionable insights that the head of the new aviation division can use to help decide which aircraft to purchase.

## 2. Defining the Question
**Problem Statement :** *Determine which aircraft are the lowest risk for the company to purchase and operate, providing actionable insights for decision-making.*

### Understanding the Context and Data Relevance
From our dataset, there are a few columns that can be used to determine the risk factor of each Make/Model.
To determine risk, we can use variables such as:
* Incident Frequency: Number of incidents per aircraft make/model.
* Severity of Incidents: Categorized by total fatalities, serious injuries, minor injuries, and uninjured passengers.
* Aircraft Damage: Types of damage recorded (e.g., destroyed, substantial, minor).
* Phase of Flight: Common phases during which incidents occur (e.g., landing, takeoff).
* Weather Conditions: Incidents correlated with different weather conditions.

### Data Understanding

In this project, we will be working with a [Kaggle dataset](https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses/data) from the National Transportation Safety Board that includes aviation accident data from 1962 to 2023 about civil aviation accidents and selected incidents in the United States and international waters.

The data is contained is two separate CSV files: 
1. `AviationData.csv`: each record represents data about an accident that occurred with attributes such ad date, Location etc...
2. `USState_Codes.csv`: the state name and its associated abbreviation

To answer our question, relevant columns include model, make, number of accidents/incidents and severity

## 3. Reading the data
## 4. Tidying the Dataset
### **Question 1:** *What is the distribution of incidents per make/model of aircraft?*
Our question is to identify which aircraft has the lowest risk.

To determine which aircraft make/model combination have the lowest risk, we need to calculate the incident frequency for each and take the lowest value. 
The `Make` and `Model` columns contain 63 and 92 missing values respectively, which represents 0.07% & 0.1% of the total records respectively. For this 
Part, I am choosing to drop the null values as it ensures that the data remains complete without significantly impacting the dataset size.

I thought about mapping the `Model` column to the `Make` column to extrapolate and fill `Models` where the `Makes` match. 
However, it is important to note that aircrafts can be the same make and different models. Therefore, this would have introduced a bias. 
### **Question 2:** *What is the Severity of Injuries per Make/Model ?*
### **Question 3:** *What is the distribution of aircraft damage per Make/Model?*
### **Question 4:** *Distribution of which Phase of Flight Accidents/incidents commomly occur?*
### **Question 5:** *What is the distribution of Weather Conditions when Incidents occur?*
### **Question 6:** *What is the Risk Score of each Make/Model Combination?*

```python


```python
#plot Make/Model Vs ERCS score 

# Creating a copy to work for this analysis
aviation_filtered1 = aviation_filtered.copy()

# Combine Make and Model columns into a single column
aviation_filtered1['Make_Model'] = aviation_filtered['Make'] + ' ' + aviation_filtered['Model']

# Group by the combined Make/Model column and calculate the mean ERCS
make_model_ercs = aviation_filtered1.groupby('Make_Model')['ERCS.Score'].mean().reset_index()

# Sort the values for better visualization
make_model_ercs = make_model_ercs.sort_values(by='ERCS.Score').head(20)

# Plot the results
plt.figure(figsize=(14, 8))
sns.barplot(x='ERCS.Score', y='Make_Model', data=make_model_ercs, palette='viridis')
plt.title('Mean ERCS Score per Make/Model')
plt.xlabel('Mean ERCS')
plt.ylabel('Make/Model')
plt.xlim(0,5)
plt.show()
```


    
![png](aviation-accident-analysis_files/aviation-accident-analysis_98_0.png)
    



## 6. Exploratory Data Analysis
## 7. Conclusions
1. Most aircraft Accidents/Incidents occurred during VMC(Visual Meteorological Conditions).
2. Most aircraft incidents/accidents per Make/Model with lower risk have similar incident frequencies and risk scores making it difficult to select just one or two.
3. Most aircrafts sustain substantial damage in the event of an accident.

## 8. Recommendations
1. Given that VMC is recommended for flying, further investigation should be carried out as to what else could be contributing to this high accidents rate.
2. Conduct further research into the lower risk category of Make/Model combinations and find other considerations e.g.,initial cost of investment.
3. Conduct further investigation as to whether there's correllation between aircraft damage and manufacture or maintenance procedures.

## 9. Data Visualization
Dashboards are a powerful tool to facilitate data-driven decision making. For this project, [You can access the interactive dashboard here.](https://public.tableau.com/app/profile/vallary.banda/viz/aviation-accident-analysis/Aviation-dashboard)