#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Replace with the raw URL of your CSV file
url = 'https://raw.githubusercontent.com/pointOfive/stat130chat130/refs/heads/main/CP/CSCS_data_anon.csv'

# Read the CSV file
data = pd.read_csv(url, low_memory=False)

# Display the entire dataset for these two columns
pd.set_option('display.max_rows', None)  # Remove row limit for display


# In[2]:


subset = data[[
    "CONNECTION_activities_hug_p3m", "CONNECTION_activities_kissed_p3m", "CONNECTION_activities_sex_p3m",
    "LONELY_direct"
]]

subset.dropna().describe()


# In[3]:


# Subset the columns you want to analyze
subset = data[[
    "CONNECTION_activities_hug_p3m", "CONNECTION_activities_kissed_p3m", "CONNECTION_activities_sex_p3m",
    "LONELY_direct"
]]

# Remove rows where "presented but no response" appears in the 'LONELY_direct' column
subset_cleaned = subset[subset != 'Presented but no response'].dropna()

# Display the first few rows and data types
for column in subset_cleaned.columns:
    print(f"Unique values in column '{column}':")
    print(subset_cleaned[column].unique())
    print("\n")
    
subset_cleaned.describe()


# In[4]:


# encoding ordinal variables
ordinal_map1 = {
    'Daily or almost daily': 7, 'Not in the past three months': 1, 'Weekly' : 5, 'Monthly' : 3,
 'A few times a month' : 4, 'Less than monthly': 2, 'A few times a week' : 6
}

# Apply the mapping to the relevant columns
subset_cleaned["CONNECTION_activities_hug_p3m"] = subset_cleaned["CONNECTION_activities_hug_p3m"].map(ordinal_map1)
subset_cleaned["CONNECTION_activities_kissed_p3m"] = subset_cleaned["CONNECTION_activities_kissed_p3m"].map(ordinal_map1)
subset_cleaned["CONNECTION_activities_sex_p3m"] = subset_cleaned["CONNECTION_activities_sex_p3m"].map(ordinal_map1)

# Encoding ordinal variables
ordinal_map2 = {
    'Occasionally or a moderate amount of time (e.g. 3-4 days)' : 4,
 'Rarely (e.g. less than 1 day)' : 2, 'All of the time (e.g. 5-7 days)]' : 5,
 'Some or a little of the time (e.g. 1-2 days)' : 3,
 'None of the time (e.g., 0 days)' : 1
}

# Apply the mapping to the relevant columns
subset_cleaned["LONELY_direct"] = subset_cleaned["LONELY_direct"].map(ordinal_map2)

# Check the transformation
subset_cleaned.head()
subset_cleaned.describe()


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

# Define the columns to plot
columns = ["CONNECTION_activities_hug_p3m", "CONNECTION_activities_kissed_p3m", 
           "CONNECTION_activities_sex_p3m", "LONELY_direct"]

# Set up the plotting area
for col in columns:
    # Count the frequencies of each unique value in the column
    value_counts = subset_cleaned[col].value_counts().sort_index()

    # Create a bar plot
    sns.barplot(x=value_counts.index, y=value_counts.values, palette="Blues_d")

    # Add labels and title
    plt.title(f"Frequency of {col} Values", fontsize=14)
    plt.xlabel(f"{col} (Categories)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()



# In[6]:


# Calculate Spearman's correlation
correlation_matrix = subset_cleaned.corr(method='spearman')
correlation_matrix


# In[15]:


import numpy as np
from itertools import combinations

# Function to compute Goodman and Kruskal's Gamma
def compute_gamma(contingency_table):
    concordant = 0
    discordant = 0
    table = contingency_table.to_numpy()
    row_indices, col_indices = np.indices(table.shape)

    # Pairwise combinations of cells
    for (i1, j1), (i2, j2) in combinations(zip(row_indices.ravel(), col_indices.ravel()), 2):
        if (i1 < i2 and j1 < j2) or (i1 > i2 and j1 > j2):  # Concordant pairs
            concordant += table[i1, j1] * table[i2, j2]
        elif (i1 < i2 and j1 > j2) or (i1 > i2 and j1 < j2):  # Discordant pairs
            discordant += table[i1, j1] * table[i2, j2]

    # Gamma calculation
    return (concordant - discordant) / (concordant + discordant) if concordant + discordant > 0 else 0

# Bootstrapping function
def bootstrap_gamma(data, x_var, y_var, n_iterations=10000, random_state=42):
    np.random.seed(random_state)  # Ensure reproducibility
    gammas = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        resampled_data = data.sample(frac=1, replace=True)
        
        # Create a contingency table
        contingency_table = pd.crosstab(resampled_data[x_var], resampled_data[y_var])
        
        # Calculate Gamma
        gamma = compute_gamma(contingency_table)
        gammas.append(gamma)
    
    # Compute the mean and confidence intervals
    gamma_mean = np.mean(gammas)
    ci_lower, ci_upper = np.percentile(gammas, [2.5, 97.5])
    
    return gamma_mean, ci_lower, ci_upper

# Apply bootstrapping for each connection variable
connection_vars = [
    "CONNECTION_activities_hug_p3m",
    "CONNECTION_activities_kissed_p3m",
    "CONNECTION_activities_sex_p3m"
]

bootstrap_results = {}
for var in connection_vars:
    gamma_mean, ci_lower, ci_upper = bootstrap_gamma(subset_cleaned, var, "LONELY_direct")
    bootstrap_results[var] = {
        "Mean Gamma": gamma_mean,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper
    }

# Display the results
for var, stats in bootstrap_results.items():
    print(f"Variable: {var}")
    print(f"Mean Gamma: {stats['Mean Gamma']:.3f}")
    print(f"95% Confidence Interval: ({stats['95% CI Lower']:.3f}, {stats['95% CI Upper']:.3f})")
    print("-" * 30)
import numpy as np
from itertools import combinations

# Function to compute Goodman and Kruskal's Gamma
def compute_gamma(contingency_table):
    concordant = 0
    discordant = 0
    table = contingency_table.to_numpy()
    row_indices, col_indices = np.indices(table.shape)

    # Pairwise combinations of cells
    for (i1, j1), (i2, j2) in combinations(zip(row_indices.ravel(), col_indices.ravel()), 2):
        if (i1 < i2 and j1 < j2) or (i1 > i2 and j1 > j2):  # Concordant pairs
            concordant += table[i1, j1] * table[i2, j2]
        elif (i1 < i2 and j1 > j2) or (i1 > i2 and j1 < j2):  # Discordant pairs
            discordant += table[i1, j1] * table[i2, j2]

    # Gamma calculation
    return (concordant - discordant) / (concordant + discordant) if concordant + discordant > 0 else 0

# Bootstrapping function
def bootstrap_gamma(data, x_var, y_var, n_iterations=10000, random_state=42):
    np.random.seed(random_state)  # Ensure reproducibility
    gammas = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        resampled_data = data.sample(frac=1, replace=True)
        
        # Create a contingency table
        contingency_table = pd.crosstab(resampled_data[x_var], resampled_data[y_var])
        
        # Calculate Gamma
        gamma = compute_gamma(contingency_table)
        gammas.append(gamma)
    
    # Compute the mean and confidence intervals
    gamma_mean = np.mean(gammas)
    ci_lower, ci_upper = np.percentile(gammas, [2.5, 97.5])
    
    return gamma_mean, ci_lower, ci_upper

# Apply bootstrapping for each connection variable
connection_vars = [
    "CONNECTION_activities_hug_p3m",
    "CONNECTION_activities_kissed_p3m",
    "CONNECTION_activities_sex_p3m"
]

bootstrap_results = {}
for var in connection_vars:
    gamma_mean, ci_lower, ci_upper = bootstrap_gamma(subset_cleaned, var, "LONELY_direct")
    bootstrap_results[var] = {
        "Mean Gamma": gamma_mean,
        "95% CI Lower": ci_lower,
        "95% CI Upper": ci_upper
    }

# Display the results
for var, stats in bootstrap_results.items():
    print(f"Variable: {var}")
    print(f"Mean Gamma: {stats['Mean Gamma']:.3f}")
    print(f"95% Confidence Interval: ({stats['95% CI Lower']:.3f}, {stats['95% CI Upper']:.3f})")
    print("-" * 30)


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot the bootstrap results
def plot_bootstrap_gamma(gammas, variable_name, ci_lower, ci_upper):
    plt.figure(figsize=(10, 6))
    sns.histplot(gammas, bins=30, kde=True, color='skyblue', edgecolor='black')
    
    # Add vertical lines for the confidence intervals
    plt.axvline(ci_lower, color='red', linestyle='--', label=f'95% CI Lower: {ci_lower:.3f}')
    plt.axvline(ci_upper, color='red', linestyle='--', label=f'95% CI Upper: {ci_upper:.3f}')
    
    # Add vertical line for the mean
    mean_gamma = np.mean(gammas)
    plt.axvline(mean_gamma, color='green', linestyle='-', label=f'Mean Gamma: {mean_gamma:.3f}')
    
    # Titles and labels
    plt.title(f'Bootstrap Distribution of Gamma for {variable_name}', fontsize=14)
    plt.xlabel('Gamma', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

# Apply the plot function for each connection variable
for var in connection_vars:
    # Resample to get bootstrap samples
    gammas = []
    for _ in range(10000):
        resampled_data = subset_cleaned.sample(frac=1, replace=True)
        contingency_table = pd.crosstab(resampled_data[var], resampled_data["LONELY_direct"])
        gamma = compute_gamma(contingency_table)
        gammas.append(gamma)
    
    # Calculate confidence intervals
    ci_lower, ci_upper = np.percentile(gammas, [2.5, 97.5])
    
    # Plot
    plot_bootstrap_gamma(gammas, var, ci_lower, ci_upper)


# In[14]:


import numpy as np
from itertools import combinations

def calculate_gamma(data, x_var, y_var):
    """
    Calculate Goodman and Kruskal's Gamma for two ordinal variables.
    Includes a breakdown of concordant and discordant pairs.
    """
    # Extract values for the two variables
    x = data[x_var].values
    y = data[y_var].values
    
    concordant = 0
    discordant = 0
    total_pairs = 0

    # Generate all unique pairs of indices
    for (i, j) in combinations(range(len(x)), 2):
        # Compare the values for the two variables
        x_diff = np.sign(x[i] - x[j])  # Sign of the difference for x
        y_diff = np.sign(y[i] - y[j])  # Sign of the difference for y
        
        # Only consider pairs that are not tied on both x and y
        if x_diff != 0 or y_diff != 0:
            total_pairs += 1
            if x_diff == y_diff:
                concordant += 1
            else:
                discordant += 1

    # Calculate Gamma
    gamma = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else None

    # Print breakdown
    print(f"Total pairs: {total_pairs}")
    print(f"Concordant pairs: {concordant}")
    print(f"Discordant pairs: {discordant}")
    if gamma is not None:
        print(f"Goodman and Kruskal's Gamma: {gamma:.3f}")
    else:
        print("Gamma cannot be calculated (no concordant or discordant pairs).")

    return gamma

# Example usage with your dataset
connection_vars = [
    "CONNECTION_activities_hug_p3m",
    "CONNECTION_activities_kissed_p3m",
    "CONNECTION_activities_sex_p3m"
]

for var in connection_vars:
    print(f"\nCalculating Gamma for {var} vs LONELY_direct:")
    calculate_gamma(subset_cleaned, var, "LONELY_direct")

