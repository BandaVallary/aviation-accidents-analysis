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

## 9. Data Visualization
Dashboards are a powerful tool to facilitate data-driven decision making. For this project, 

[You can access the interactive dashboard here.](https://public.tableau.com/app/profile/vallary.banda/viz/aviation-accident-analysis/Aviation-dashboard)