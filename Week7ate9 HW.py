#!/usr/bin/env python
# coding: utf-8

# #### Question 1:
# 
# The Simple Linear Regression (SLR) model mathematically specifies a linear relationship between a predictor X and an outcome Y as:
# 
# Y = (beta0) + (beta1)X + 𝜀
# 
# Where:
# 
# - beta0 is the intercept (the value of Y when X = 0)
# - beta1 is the slope
# - 𝜀 is is a random error term, assumed to be normally distributed with mean zero and variance s^2
# 
# Statistically, this specification implies that for any given value of X and Y, values are sampled from a normal distribution centered at A + BX with variance s^2.

# In[1]:


# Question 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Setting the parameters for the simple linear regression model
beta_0 = 2    # Intercept
beta_1 = 0.5  # Slope
sigma = 1     # Standard deviation of the error term

# Generate data for the predictor variable X
np.random.seed(0)
X = np.linspace(0, 10, 50)  # 50 evenly spaced values from 0 to 10

# Calculate the true regression line values for Y (without error)
Y_true = beta_0 + beta_1 * X

# Add normally distributed error to simulate observed Y values
errors = np.random.normal(0, sigma, X.shape)
Y_observed = Y_true + errors

# Plot the true regression line and observed values
plt.figure(figsize=(10, 6))
plt.plot(X, Y_true, label="True regression line (no error)", color="blue", linestyle="--")
plt.scatter(X, Y_observed, color="red", alpha=0.7, label="Observed values (with error)")

# Visualize a normal distribution centered around a specific point on the regression line
X_point = 5
Y_point = beta_0 + beta_1 * X_point
x_vals = np.linspace(Y_point - 3*sigma, Y_point + 3*sigma, 100)
plt.plot([X_point]*len(x_vals), x_vals, 'k--', alpha=0.4)  # Vertical line at X=5
plt.plot(x_vals * 0 + X_point, norm.pdf(x_vals, Y_point, sigma) * 3 + X_point, 'k', alpha=0.4)
plt.fill_betweenx(x_vals, X_point - 0.15, X_point + 0.15, color="gray", alpha=0.2, label="Normal distribution (Y | X=5)")

# Labeling the plot
plt.xlabel("Predictor Variable X")
plt.ylabel("Outcome Variable Y")
plt.legend()
plt.title("Simple Linear Regression Model - True Line vs Observed Values with Error")
plt.grid(True)
plt.show()


# In[2]:


# Question 2

# Re-import necessary libraries and recreate the data since the environment was reset
import numpy as np
import pandas as pd

# Set the parameters
beta_0 = 2     # Intercept
beta_1 = 0.5   # Slope
sigma = 1      # Standard deviation of the error term

# Generate data for predictor variable X
np.random.seed(0)
X = np.linspace(0, 10, 50)  # 50 evenly spaced values from 0 to 10

# Calculate the true regression line values for Y (without error)
Y_true = beta_0 + beta_1 * X

# Add normally distributed error to simulate observed Y values
errors = np.random.normal(0, sigma, X.shape)
Y_observed = Y_true + errors

# Combine X and Y_observed into a DataFrame
df = pd.DataFrame({'x': X, 'Y': Y_observed})

# Display the first few rows of the DataFrame
df.head()


# In[3]:


# Question 2

import statsmodels.formula.api as smf  # what is this library for?
import plotly.express as px  # this is a ploting library

# what are the following two steps doing?
model_data_specification = smf.ols("Y~x", data=df) 
fitted_model = model_data_specification.fit() 

# what do each of the following provide?
fitted_model.summary()  # simple explanation? 
fitted_model.summary().tables[1]  # simple explanation?
fitted_model.params  # simple explanation?
fitted_model.params.values  # simple explanation?
fitted_model.rsquared  # simple explanation?

# what two things does this add onto the figure?
df['Data'] = 'Data' # hack to add data to legend 
fig = px.scatter(df, x='x',  y='Y', color='Data', 
                 trendline='ols', title='Y vs. x')

# This is essentially what above `trendline='ols'` does
fig.add_scatter(x=df['x'], y=fitted_model.fittedvalues,
                line=dict(color='blue'), name="trendline='ols'")

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# ##### Chat for questions 1 and 2:
# 
# I "hit the plan limit" for the free version of chatgpt, so I couldn't ask it for a summary, but here is the link:
# 
# https://chatgpt.com/share/6723fde7-c3d8-800c-a40c-f05488357fa3

# #### Question 3:
# 
# The key difference is that the theoretical line is derived from the model's assumptions (intercept and slope), while the fitted line reflects the actual data, which includes noise and variability. This illustrates the concept of random sampling variation, as different samples will yield different fitted lines around the theoretical line.

# #### Question 4:
# 
# - fitted_model.summary().tables[1]: outputs the estimated parameters, including the coefficients for the intercept (beta0) and the predictor variable, along with statistics like standard errors, t-values, and p-values.
# 
# - fitted_model.params or fitted_model.params.values: these give you the estimated coefficients which are needed to calculate the fitted values.
# 
# - fitted_model.fittedvalues: This is the final outcome of applying the regression equation using the estimated parameters on the observed predictor values. Each fitted value corresponds to a predicted Y based on the regression model applied to the respective X value.
# 
# Overall, the fitted values are derived directly from the parameters obtained in the regression analysis.

# #### Question 5: 
# 
# OLS chooses the line that minimizes the sum of the squared residuals. This way, positive and negative residuals don't cancel each other out. This approach helps ensure a robust and reliable estimation of the relationship between variables while emphasizing the importance of larger errors.

# #### Question 6: 
# 
# In summary, the expression can be interpreted as the proportion of variation in the outcome variable Y explained by the regression model because it "starts" with the total variation in Y (SST), then it substracts the portion of variation that is not captured by the model (SSR). The resulting value represents how much of the total variability is accounted for by the fitted model, essentially quantifying the effectiveness of the regression in explaining the outcome variable Y.
# 
# The accuracy of a regression model refers to how closely the predicted values match the actual observed values. R^2 serves as a direct measure of this relationship. A high R^2 suggests that the model's predictions are closely aligned with the actual values of Y. This indicates that the independent variables included in the model effectively capture the underlying relationship with Y. A low R^2 suggests that the model fails to explain a significant portion of the variability in Y. This indicates that either the model is poorly specified (e.g., missing important predictors) or that the relationship between the predictors and Y is weak.

# In[4]:


# Question 7

import pandas as pd
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots

# This data shows the relationship between the amount of fertilizer used and crop yield
data = {'Amount of Fertilizer (kg) (x)': [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 
                                          2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 
                                          4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 
                                          6.4, 6.6, 6.8, 7, 7.2, 7.4, 7.6, 7.8, 8, 
                                          8.2, 8.4, 8.6, 8.8,9, 9.2, 9.4, 9.6],
        'Crop Yield (tons) (Y)': [18.7, 16.9, 16.1, 13.4, 48.4, 51.9, 31.8, 51.3, 
                                  63.9, 50.6, 58.7, 82.4, 66.7, 81.2, 96.5, 112.2, 
                                  132.5, 119.8, 127.7, 136.3, 148.5, 169.4, 177.9, 
                                  186.7, 198.1, 215.7, 230.7, 250.4, 258. , 267.8, 
                                  320.4, 302. , 307.2, 331.5, 375.3, 403.4, 393.5,
                                  434.9, 431.9, 451.1, 491.2, 546.8, 546.4, 558.9]}
df = pd.DataFrame(data)
fig1 = px.scatter(df, x='Amount of Fertilizer (kg) (x)', y='Crop Yield (tons) (Y)',
                  trendline='ols', title='Crop Yield vs. Amount of Fertilizer')

# Perform linear regression using scipy.stats
slope, intercept, r_value, p_value, std_err = \
    stats.linregress(df['Amount of Fertilizer (kg) (x)'], df['Crop Yield (tons) (Y)'])
# Predict the values and calculate residuals
y_hat = intercept + slope * df['Amount of Fertilizer (kg) (x)']
residuals = df['Crop Yield (tons) (Y)'] - y_hat
df['Residuals'] = residuals
fig2 = px.histogram(df, x='Residuals', nbins=10, title='Histogram of Residuals',
                    labels={'Residuals': 'Residuals'})

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Crop Yield vs. Amount of Fertilizer', 
                                    'Histogram of Residuals'))
for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)
fig.update_layout(title='Scatter Plot and Histogram of Residuals',
    xaxis_title='Amount of Fertilizer (kg)', yaxis_title='Crop Yield (tons)',
    xaxis2_title='Residuals', yaxis2_title='Frequency', showlegend=False)

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# #### Question 7:
# 
# Assumption 1:
# 
# The relationship between the independent variable (amount of fertilizer) and the dependent variable (crop yield) is linear. Although this seems true at a glance, the curve of points seems to point to an exponential relation rather than a linear relation, but only at a very small degree.
# 
# Assumption 2:
# 
# The residuals should be normally distributed. As you can see, they are not.

# ##### Chat for Q 3-7:
# 
# I am very sorry but I deleted the chat accidentaly after completing this part, as I thought I had already given the chat summary and link. Please forgive me 🙏

# #### Question 8:
# 
# The null hypothesis of "no linear association (on average)" in terms of the Simple Linear Regression model parameter is:
# 
# 
# H0: beta_1 = 0
# 
# where:
# 
# - beta_1 = 0 indicates no linear association between the predictor X and the outcome Y on average, meaning that variations in X do not predict any change in Y.
# - If beta_1 not equal 0, this would imply the presence of a linear association between X and Y, with the direction and magnitude of this relationship determined by the value of beta_1.

# In[5]:


# Question 8:

import seaborn as sns
import statsmodels.formula.api as smf

# The "Classic" Old Faithful Geyser dataset
old_faithful = sns.load_dataset('geyser')

linear_for_specification = 'duration ~ waiting'
model = smf.ols(linear_for_specification, data=old_faithful)
fitted_model = model.fit()
fitted_model.summary()


# #### Question 8: cont.
# 
# 1. p-value for waiting coefficient (0.000)
# 
# The p-value for the waiting coefficient is below the standard significance threshold of 0.05. This means we reject the null hypothesis (𝐻0:𝛽1=0), providing strong evidence of a statistically significant linear association between waiting time and eruption duration.
# 
# 2. Coefficient Interpretation:
# 
# Slope (waiting): 
# 
# 0.0756
# 
# This indicates that for each additional minute of waiting time, the eruption duration is predicted to increase by 0.0756 minutes. The confidence interval [0.071, 0.080] for this slope does not include zero, reinforcing that the effect is statistically significant.
# 
# 3. F-statistic and p-value (Prob (F-statistic): 8.13e-100)
# 
# The F-statistic is extremely high with an associated p-value near zero, indicating that the overall model significantly improves our understanding of eruption duration compared to a model with no predictors.

# In[6]:


# Question 9:

import plotly.express as px
import statsmodels.formula.api as smf


short_wait_limit = 62 # 64 # 66 #
short_wait = old_faithful.waiting < short_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[short_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (<"+str(short_wait_limit)+")", 
                 trendline='ols')

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# #### Question 9:
# 
# The slope estimate of 0.0069 indicates that for each additional minute of waiting time, the duration of the eruption increases by only 0.0069 minutes (or about 0.4 seconds).
# The associated p-value is 0.238, which is not statistically significant. The confidence interval for the slope [-0.005, 0.019] includes zero, which also suggests that we cannot confidently claim a linear relationship between waiting time and duration within this subset.

# In[7]:


# Question 10:

import plotly.express as px

long_wait_limit = 71
long_wait = old_faithful.waiting > long_wait_limit

print(smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1])

# Create a scatter plot with a linear regression trendline
fig = px.scatter(old_faithful[long_wait], x='waiting', y='duration', 
                 title="Old Faithful Geyser Eruptions for short wait times (>"+str(long_wait_limit)+")", 
                 trendline='ols')
fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[8]:


# Question 10, part 1: create fitted Simple Linear Regression models for boostrap samples and collect and 
# visualize the bootstrapped sampling distribution of the fitted slope coefficients of the fitted models

# this is the visualization of coefficients part

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf

# Define the number of bootstrap samples
n_bootstrap = 1000

# Initialize a list to store slope coefficients from each bootstrap sample
bootstrap_slopes = []

# Filter data for long wait times
long_wait_limit = 71
long_wait_data = old_faithful[old_faithful['waiting'] > long_wait_limit]

# Generate bootstrap samples and fit a linear model to each sample
for _ in range(n_bootstrap):
    # Create a bootstrap sample from the filtered data
    bootstrap_sample = long_wait_data.sample(frac=1, replace=True)
    
    # Fit the linear model
    model = smf.ols('duration ~ waiting', data=bootstrap_sample).fit()
    
    # Append the slope coefficient to the list
    bootstrap_slopes.append(model.params['waiting'])

# Convert to a DataFrame for easier plotting
bootstrap_df = pd.DataFrame(bootstrap_slopes, columns=['Slope'])

# Plot the distribution of bootstrap slope coefficients
fig = px.histogram(
    bootstrap_df, x='Slope', nbins=30,
    title="Bootstrapped Sampling Distribution of the Slope Coefficients",
    labels={'Slope': 'Slope Coefficient'}
)

# Add a vertical line for the mean slope value
mean_slope = np.mean(bootstrap_slopes)
fig.add_vline(x=mean_slope, line_dash="dash", line_color="red", 
              annotation_text="Mean Slope", annotation_position="top right")

# Show the plot
fig.show(renderer="png")  # Use "png" renderer for compatibility with GitHub and MarkUs


# In[9]:


# Question 10, part 1: create fitted Simple Linear Regression models for boostrap samples and collect and 
# visualize the bootstrapped sampling distribution of the fitted slope coefficients of the fitted models

# this is the linear regression models:

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf

# Define the number of bootstrap samples
n_bootstrap = 1000

# Initialize lists to store slope coefficients, intercepts, and fitted models
bootstrap_slopes = []
bootstrap_intercepts = []
bootstrap_models = []

# Filter data for long wait times
long_wait_limit = 71
long_wait_data = old_faithful[old_faithful['waiting'] > long_wait_limit]

# Create a figure for plotting all regression lines
fig = go.Figure()

# Generate bootstrap samples and fit a linear model to each sample
for i in range(n_bootstrap):
    # Create a bootstrap sample from the filtered data
    bootstrap_sample = long_wait_data.sample(frac=1, replace=True)
    
    # Fit the linear model
    model = smf.ols('duration ~ waiting', data=bootstrap_sample).fit()
    
    # Store the fitted model
    bootstrap_models.append(model)
    
    # Store the slope and intercept of the model
    bootstrap_slopes.append(model.params['waiting'])
    bootstrap_intercepts.append(model.params['Intercept'])
    
    # Create a regression line for the current bootstrap sample
    x_vals = np.linspace(long_wait_data['waiting'].min(), long_wait_data['waiting'].max(), 100)
    y_vals = model.params['Intercept'] + model.params['waiting'] * x_vals
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(width=0.5, color='blue'), showlegend=False))

# Add the scatter plot of the original data
fig.add_trace(go.Scatter(x=long_wait_data['waiting'], y=long_wait_data['duration'], mode='markers', 
                         name='Original Data', marker=dict(color='black', opacity=0.6, size=5)))

# Add title and labels
fig.update_layout(
    title="Regression Lines for Bootstrapped Samples of Long Wait Times (> 71 minutes)",
    xaxis_title="Waiting Time (minutes)",
    yaxis_title="Duration (minutes)",
    showlegend=True
)

# Show the plot
fig.show(renderer="png")  # Use "png" renderer for compatibility with GitHub and MarkUs

# Optional: Inspect some of the fitted models' parameters
print("First model's slope:", bootstrap_slopes[0])
print("First model's intercept:", bootstrap_intercepts[0])



# In[10]:


# Question 10, part 2: simulate samples (of size n=160) from a Simple Linear Regression model that uses, along with the 
# values of waiting for to create simuations of and use these collect and visualize the sampling distribution of the 
# fitted slope coefficient under a null hypothesis assumption of "no linear association (on average)"

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf

# Parameters under the null hypothesis
beta_0 = 1.65
beta_1 = 0
sigma = 0.37
n_simulations = 1000  # Number of simulations

# Use the waiting values from the long-wait subset as predictor values
long_wait_limit = 71
long_wait_data = old_faithful[old_faithful['waiting'] > long_wait_limit]
waiting_values = long_wait_data['waiting'].values

# Storage for simulated slopes
simulated_slopes = []

# Perform simulations
for _ in range(n_simulations):
    # Generate the response variable 'duration' under the null hypothesis
    # duration = beta_0 + beta_1 * waiting + random noise
    simulated_duration = beta_0 + beta_1 * waiting_values + np.random.normal(0, sigma, size=len(waiting_values))
    
    # Create a DataFrame for this simulated dataset
    simulated_data = pd.DataFrame({'waiting': waiting_values, 'duration': simulated_duration})
    
    # Fit a linear model and store the slope coefficient
    model = smf.ols('duration ~ waiting', data=simulated_data).fit()
    simulated_slopes.append(model.params['waiting'])

# Convert simulated slopes to a DataFrame for plotting
simulated_slopes_df = pd.DataFrame(simulated_slopes, columns=['Slope'])

# Plot the distribution of simulated slope coefficients
fig = px.histogram(
    simulated_slopes_df, x='Slope', nbins=30,
    title="Sampling Distribution of the Slope Coefficients Under Null Hypothesis (No Association)",
    labels={'Slope': 'Slope Coefficient'}
)

# Add a vertical line at the mean of the simulated slopes (expected to be close to 0)
mean_slope = np.mean(simulated_slopes)
fig.add_vline(x=mean_slope, line_dash="dash", line_color="red", 
              annotation_text="Mean Slope", annotation_position="top right")

# Show the plot
fig.show(renderer="png")  # Use "png" renderer for compatibility with GitHub and MarkUs

# Optional: Display summary statistics for the simulated slopes
print("Mean of simulated slopes:", mean_slope)
print("Standard deviation of simulated slopes:", np.std(simulated_slopes))


# In[11]:


# Question 10, part 3: report if 0 is contained within a 95% bootstrapped confidence interval

# Calculate the 95% confidence interval using the percentile method
lower_bound = np.percentile(simulated_slopes, 2.5)
upper_bound = np.percentile(simulated_slopes, 97.5)

# Check if zero is contained within this interval
contains_zero = lower_bound <= 0 <= upper_bound

# Report the confidence interval and whether zero is within it
print(f"95% Bootstrapped Confidence Interval for the Slope: [{lower_bound:.4f}, {upper_bound:.4f}]")
print(f"Does the confidence interval contain zero? {'Yes' if contains_zero else 'No'}")


# In[12]:


# Question 10, part 3: simulating the p-value

import numpy as np
import statsmodels.formula.api as smf

# Set parameters
n_permutations = 1000

# Observed slope
observed_model = smf.ols('duration ~ waiting', data=long_wait_data).fit()
observed_slope = observed_model.params['waiting']

# Storage for permuted slopes
permuted_slopes = []

# Perform permutations
for _ in range(n_permutations):
    # Shuffle the 'duration' values to break any association with 'waiting'
    permuted_duration = np.random.permutation(long_wait_data['duration'].values)
    permuted_data = long_wait_data.assign(duration=permuted_duration)
    
    # Fit a linear model to the permuted data
    permuted_model = smf.ols('duration ~ waiting', data=permuted_data).fit()
    permuted_slopes.append(permuted_model.params['waiting'])

# Calculate the p-value as the proportion of permuted slopes as extreme or more extreme than observed slope
extreme_slopes = [slope for slope in permuted_slopes if abs(slope) >= abs(observed_slope)]
p_value = len(extreme_slopes) / n_permutations

# Report the p-value
print(f"Simulated p-value: {p_value:.4f}")


# In[13]:


smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1]


# #### Question 10:
# 
# According to this code that chatgpt gave me, this 95% confidence interval contains zero. This means we cannot say with 95% confidence that there exists an association between the variables "waiting" and"duration". This also suggests the slope of "waiting" in the simple linear regression model is zero.
# 
# the simulated p-value does not match "smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().summary().tables[1]".

# ##### Chat summaries questions 8-10
# 
# summary:
# 
# Certainly! Here’s a summary of our conversation:
# 
# Null Hypothesis in Simple Linear Regression: We discussed formulating a null hypothesis of "no linear association" for a Simple Linear Regression model in terms of the slope coefficient 
# 𝛽
# 1
# β 
# 1
# ​
#  . The null hypothesis is that 
# 𝛽
# 1
# =
# 0
# β 
# 1
# ​
#  =0, meaning changes in the predictor (e.g., waiting time) do not affect the response variable (e.g., duration).
# 
# Old Faithful Geyser Data Analysis:
# 
# We examined the relationship between waiting (time between eruptions) and duration (length of eruptions) in the Old Faithful dataset.
# Fitting a linear model to the full dataset showed a significant positive association, with an R-squared value of 0.811, indicating that 81.1% of the variability in duration could be explained by waiting.
# Analysis for Short Wait Times:
# 
# When analyzing only short wait times, the slope coefficient was not significant, suggesting little evidence of a relationship in this subset.
# Analysis for Long Wait Times:
# 
# We filtered for long wait times and fit a model specifically for that subset. This model's slope was positive and significant, indicating a strong association between waiting and duration for long waits.
# Bootstrap Confidence Interval for Slope:
# 
# We created a bootstrap sampling distribution of the slope coefficients to see the variability and calculate a 95% confidence interval.
# We checked if zero was contained within this interval to assess if the observed association was statistically significant.
# Simulation Under the Null Hypothesis:
# 
# To further investigate, we simulated samples from a model with 
# 𝛽
# 1
# =
# 0
# β 
# 1
# ​
#  =0 (no association) to understand the expected distribution of the slope under the null hypothesis.
# We visualized the bootstrapped sampling distribution to check if the observed slope was within the range expected under no association.
# Simulated P-value with Permutation Test:
# 
# Finally, we performed a permutation test by repeatedly shuffling duration values to break any association with waiting. We fit a linear model to each permutation and calculated the proportion of slopes as extreme as the observed slope. This proportion served as a simulated p-value for testing the null hypothesis.
# In sum, we explored both statistical and simulation-based methods to assess the relationship between waiting and duration in different subsets of the Old Faithful dataset, gaining insights into when and how this association is statistically meaningful.
# 
# 
# link: https://chatgpt.com/share/672b8c0b-5770-800c-bcb8-c316becc44b6

# #### Question 11:
# 
# The shift to an indicator variable changes the focus from describing a continuous effect to comparing two distinct categories. This model may be preferable if the relationship between waiting and duration has natural breaks or if a threshold effect better captures the data patterns. It sacrifices the nuanced view of a continuous relationship in exchange for a simpler, potentially more interpretable comparison between groups.
# 
# 
# 
# - smf.ols('duration ~ waiting', data=old_faithful)
# this command initializes an Ordinary Least Squares (OLS) regression model with duration as the response variable and waiting as the predictor.
# 
# - smf.ols('duration ~ waiting', data=old_faithful[short_wait])
# The code smf.ols('duration ~ waiting', data=old_faithful[short_wait]) is a Simple Linear Regression model specification that focuses only on the subset of the data where the waiting time is classified as "short."
# 
# - smf.ols('duration ~ waiting', data=old_faithful[long_wait])
# The code smf.ols('duration ~ waiting', data=old_faithful[long_wait]) sets up a Simple Linear Regression model that focuses only on the subset of the Old Faithful dataset where the waiting time is classified as "long."
# 
# The data provides strong evidence that there is a statistically significant difference in eruption duration between "short" and "long" wait times in the Old Faithful dataset, with long wait times associated with considerably longer eruptions on average.

# In[17]:


# Question 11

from IPython.display import display

display(smf.ols('duration ~ C(kind, Treatment(reference="short"))', data=old_faithful).fit().summary().tables[1])

fig = px.box(old_faithful, x='kind', y='duration', 
             title='duration ~ kind',
             category_orders={'kind': ['short', 'long']})
fig.show(renderer = "png") # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[16]:


# Question 12

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats
import numpy as np

model_residuals = {
    '<br>Model 1:<br>All Data using slope': smf.ols('duration ~ waiting', data=old_faithful).fit().resid,
    '<br>Model 2:<br>Short Wait Data': smf.ols('duration ~ waiting', data=old_faithful[short_wait]).fit().resid,
    '<br>Model 3:<br>Long Wait Data': smf.ols('duration ~ waiting', data=old_faithful[long_wait]).fit().resid,
    '<br>Model 4:<br>All Data using indicator': smf.ols('duration ~ C(kind, Treatment(reference="short"))', data=old_faithful).fit().resid
}

fig = make_subplots(rows=2, cols=2, subplot_titles=list(model_residuals.keys()))
for i, (title, resid) in enumerate(model_residuals.items()):

    if i == 1:  # Apply different bins only to the second histogram (index 1)
        bin_size = dict(start=-1.9, end=1.9, size=0.2)
    else:
        bin_size = dict(start=-1.95, end=1.95, size=0.3)

    fig.add_trace(go.Histogram(x=resid, name=title, xbins=bin_size, histnorm='probability density'), 
                  row=int(i/2)+1, col=(i%2)+1)
    fig.update_xaxes(title_text="n="+str(len(resid)), row=int(i/2)+1, col=(i%2)+1)    
    
    normal_range = np.arange(-3*resid.std(),3*resid.std(),0.01)
    fig.add_trace(go.Scatter(x=normal_range, mode='lines', opacity=0.5,
                             y=stats.norm(loc=0, scale=resid.std()).pdf(normal_range),
                             line=dict(color='black', dash='dot', width=2),
                             name='Normal Distribution<br>(99.7% of its area)'), 
                  row=int(i/2)+1, col=(i%2)+1)
    
fig.update_layout(title_text='Histograms of Residuals from Different Models')
fig.update_xaxes(range=[-2,2])
fig.show(renderer = "png") # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# Hi - Just fyi I do not have the energy to finish questions 12-14 at the moment, but I will be sure to finish them when I can!
# thanks for understanding :)

# In[ ]:




