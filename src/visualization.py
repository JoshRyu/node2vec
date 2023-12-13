import pandas as pd
import matplotlib.pyplot as plt

# Load result set
data_pagerank = pd.read_csv('../results/csv/cora_deepwalk.csv')
# Filter the data for p = 0.25 and q = 0.25
filtered_pagerank = data_pagerank[(data_pagerank['p'] == 1.0) & (data_pagerank['q'] == 1.0)]

# Calculate the average and top values for the filtered data
average_scores_pagerank = filtered_pagerank[['macro_f1_score', 'micro_f1_score']].mean()
top_scores_pagerank = filtered_pagerank[['macro_f1_score', 'micro_f1_score']].max()

# Adjust the y-axis scale for the filtered data
y_min_pagerank = filtered_pagerank[['macro_f1_score', 'micro_f1_score']].min().min() - 0.01
y_max_pagerank = top_scores_pagerank.max() + 0.01

# Add an iteration column for the filtered data
filtered_pagerank['iteration'] = filtered_pagerank.index - filtered_pagerank.index.min() + 1

# Plot with the adjusted y-axis scale and annotations for the filtered data
plt.figure(figsize=(12, 6))
plt.plot(filtered_pagerank['iteration'], filtered_pagerank['macro_f1_score'], label='Macro F1 Score', marker='o', color='blue')
plt.plot(filtered_pagerank['iteration'], filtered_pagerank['micro_f1_score'], label='Micro F1 Score', marker='^', color='green')

# Annotations for average and top values
plt.axhline(average_scores_pagerank['macro_f1_score'], color='blue', linestyle='--')
plt.text(len(filtered_pagerank), average_scores_pagerank['macro_f1_score'], f'  Average: {average_scores_pagerank["macro_f1_score"]:.4f}', va='bottom', color='blue')
plt.axhline(top_scores_pagerank['macro_f1_score'], color='blue', linestyle=':')
plt.text(len(filtered_pagerank), top_scores_pagerank['macro_f1_score'], f'  Top: {top_scores_pagerank["macro_f1_score"]:.4f}', va='bottom', color='blue')
plt.axhline(average_scores_pagerank['micro_f1_score'], color='green', linestyle='--')
plt.text(len(filtered_pagerank), average_scores_pagerank['micro_f1_score'], f'  Average: {average_scores_pagerank["micro_f1_score"]:.4f}', va='bottom', color='green')
plt.axhline(top_scores_pagerank['micro_f1_score'], color='green', linestyle=':')
plt.text(len(filtered_pagerank), top_scores_pagerank['micro_f1_score'], f'  Top: {top_scores_pagerank["micro_f1_score"]:.4f}', va='bottom', color='green')

# Set the y-axis limits
plt.ylim(y_min_pagerank, y_max_pagerank)

# Title, X-Axis, Y-Axis
plt.title('DeepWalk: Macro F1 Score')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
