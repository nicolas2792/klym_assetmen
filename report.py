import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import path_all
import pandas as pd
import plotly.express as px
import scipy.stats as stats


st.title('Klym Datascience Assement')

st.header('1. Data Analysis')

st.title('Streamlit Analysis Report')

# Introduction
st.write("""
### Completeness

First, let's assess the completeness of the dataset by checking for missing values.
""")

# Load data
path = path_all()
car_df = pd.read_csv(path + r'\data\raw\vehicles.csv')
crashes_df = pd.read_csv(path + r'\data\raw\crashes_poverty.csv', sep=';')
counties_df = pd.read_csv(path + r'\data\raw\counties.csv', sep=';')

# Calculate missing values
nas_df = car_df[['odometer', 'vin', 'condition']].isna().sum() / car_df.shape[0]
nas_df = nas_df.reset_index()
nas_df.columns = ['Variable', 'Percentage missing Values']
st.table(nas_df)

# Explanation of missing values
st.write("""
For every variable, there is at least a 10% of missing values. In some cases, it goes up to 40%. Additionally, these variables rely heavily on the person who published the information, making it challenging to obtain accurate data.
""")

# Odometer Analysis
st.write("""
### Odometer
""")

# Scatter plot
fig = px.scatter(car_df[car_df['price'] < car_df['price'].quantile(0.99)], x='odometer', y='price')
st.plotly_chart(fig)

st.write("""
The scatter plot shows a relationship between the price and the odometer, indicating that price tends to decrease as odometer value increases. This relationship is expected as odometer value represents the usage of the car.
""")

# Histogram
fig = px.histogram(car_df[car_df['odometer'] < car_df['odometer'].quantile(0.9)], x='odometer')
st.plotly_chart(fig)

st.write("""
The histogram illustrates that odometer values are concentrated in the lower range (0-30 thousand miles), which explains the concentration seen in the scatter plot.
""")

# Correlation analysis
df_apoyo = car_df.dropna(subset='odometer')
corrspr = stats.spearmanr(df_apoyo['price'], df_apoyo['odometer'])
corrper = stats.pearsonr(df_apoyo['price'], df_apoyo['odometer'])

st.header('P-values for Spearman and Pearson Correlations')
st.write(f"Correlation value Spearman: {round(corrspr.pvalue, 4)}")
st.write(f"Correlation value Pearson: {round(corrper.pvalue, 4)}")

st.write("""
According to the p-values from the Pearson and Spearman tests, there is a significant correlation between price and odometer. To handle missing values and prevent similar issues in the future, I propose converting the odometer variable into categories.
""")

# Odometer categories
bins = [-np.inf, 0, 30000, 60000, 90000, 120000, 150000, np.inf]
car_df['odometer'] = car_df['odometer'].fillna(-1)
labels = ['no_data', '0-30ml', '31-60ml', '61-90ml', '91-120ml', '121mil-150mil', '+151ml']
car_df['odometer_cat'] = pd.cut(car_df['odometer'], bins=bins, labels=labels)
df_apoyo = car_df['odometer_cat'].value_counts(normalize=True).reset_index()
fig = px.bar(df_apoyo, 'odometer_cat', 'proportion')
st.plotly_chart(fig)

st.write("""
Building a categorical variable with 30k-mile intervals is a common practice since it aligns with typical car usage patterns.
""")

# Condition Analysis
st.write("""
### Condition
""")

st.write("""
The condition variable has a significant number of missing values (43%). Assessing the relationship between condition and price reveals no substantial differences across condition levels. Given the high percentage of missing values and the subjective nature of the condition variable, it's challenging to accurately impute missing values.
""")

# VIN Analysis
st.write("""
### VIN
""")

st.write(f"""
VIN, which serves as a unique identifier for each car, has a high percentage of missing values (40%). Additionally, some VINs are duplicated, providing redundant information. Due to these issues, further analysis of the VIN variable is not recommended.
""")

# Redundancy Analysis
st.header('Redundant Information')

st.write("""
After conducting correlation tests between variables, no evidence of redundancy was found. The table below displays the results of the correlation analysis.
""")
df_anova = pd.read_csv(path + r'\data\interim\cor_cat.csv')
st.table(df_anova)

df_cor = pd.read_csv(path + r'\data\interim\cor_num.csv')
st.table(df_cor)

st.write("""
The absence of significant correlations between independent variables indicates that redundancy is not a concern. However, due to the dimensionality of some categorical variables, not all variables will be included in the model.
""")

# Model Building
st.header('Model Building')

st.write("""
### Proposed Model
The proposed model is a Hubert regression, selected using an AutoML framework. The choice of this model aims to address outliers effectively. However, the focus should be on variable selection rather than the model itself. Given the challenges posed by the dimensionality of certain variables, model interpretability is prioritized.
""")

# Metrics
st.header('Metrics')

st.write("""
For model evaluation, standard regression metrics such as R-squared, MAE, RMSE are used. Additionally, a custom metric related to the explanation of the problem is implemented.
""")

image = open(path + r"\reports\output.png", 'rb').read()
st.image(image, caption='Metrics', use_column_width=True)

image = open(path + r"\reports\output_2.png", 'rb').read()
st.image(image, caption='Metrics', use_column_width=True)

st.write("""
### Formula
The proposed metric follows the formula:

\[ \text{Maximize:} \ (price_{pred} - price_{real}) + ((price_{pred}/price_{real})-1) * (price_{pred} - price_{real}) \]
""")

# Presentation to Stakeholders
st.header('Presentation to Stakeholders')

st.write("""
To present the model to business stakeholders, emphasis will be placed on explaining the significance of the selected variables and their impact on the model's performance. Demonstrations showcasing the model's functionality with real-time data will be conducted to ensure stakeholder understanding.
""")

image = open(path + r"\reports\output_fet.png", 'rb').read()
st.image(image, caption='Metrics', use_column_width=True)


# Model Monitoring
st.header('Model Monitoring')

st.write("""
Model monitoring is an ongoing process involving regular assessments of model performance. Monitoring includes comparing predicted prices against real prices, defining profit targets, and analyzing data and concept drifts. Any significant deviations from expected performance will trigger model retraining.
""")