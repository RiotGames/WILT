import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('cross_data.csv', index_col=0)

# Calculate averages
df['Average'] = df.mean(axis=1)
column_averages = df.mean(axis=0)
df.loc['Average'] = column_averages

df.loc['Average', 'Average'] = None

# Define font sizes
title_fontsize = 24
tick_label_fontsize = 18
label_fontsize = 16
annotation_fontsize = 18

# Set up the matplotlib figure
plt.figure(figsize=(14, 12)) 

# Create the heatmap and capture the Axes object
ax = sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.2f',
                cbar_kws={'label': 'Value'},
                annot_kws={'size': annotation_fontsize},
                linewidths=.5, linecolor='gray')

# Get the number of rows and columns
num_rows, num_cols = df.shape

# Draw a thicker horizontal line before the last row (averages)
ax.axhline(num_rows - 1, color='black', linewidth=2)

# Draw a thicker vertical line before the last column (averages)
ax.axvline(num_cols - 1, color='black', linewidth=2)

# Set titles and labels
plt.title('Model Performance with Swapped Tests', fontsize=title_fontsize)
plt.xlabel('Model used for tests', fontsize=label_fontsize)
plt.ylabel('Model used for inference', fontsize=label_fontsize)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=tick_label_fontsize)
plt.yticks(rotation=0, fontsize=tick_label_fontsize)

# Improve layout
plt.tight_layout()

# Save the heatmap
plt.savefig('model_comparison_heatmap.png', dpi=300)

print("Heatmap with averages has been saved as 'model_comparison_heatmap.png'")
