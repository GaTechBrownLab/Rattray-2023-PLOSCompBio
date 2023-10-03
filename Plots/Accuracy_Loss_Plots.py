import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
runs = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']
# Define models
models = ['ResNet-50 (Baseline)', 'VGG-19', 'MobileNetV2', 'Xception']
# Define test and validation accuracies
test_acc = [90.73, 86.20, 93.85, 88.10, 91.96, 83.92, 79.75, 86.87, 82.76, 87.92, 53.39, 58.63, 46.51, 49.78, 56.88, 88.36, 85.97, 90.98, 86.84, 89.75]
val_acc = [92.93, 88.12, 96.47, 90.18, 94.02, 86.96, 82.29, 89.52, 85.12, 90.03, 54.40, 60.71, 47.98, 51.23, 58.96, 90.25, 87.81, 92.13, 88.72, 91.88]

# Create an extended list of models for the dataset
models_extended_acc = np.repeat(models, len(test_acc)//len(models)).tolist() * 2




# Combine test and validation accuracies and generate a label for each
acc = test_acc + val_acc
labels = ['Test']*len(test_acc) + ['Validation']*len(val_acc)

# Generate a dataframe for seaborn
df_acc = pd.DataFrame({
    'Model': models_extended_acc,
    'Accuracy': acc,
    'Type': labels
})

# Sort by maximum accuracy
sorted_idx = df_acc.groupby('Model')['Accuracy'].transform(max).sort_values(ascending=False).index
df_acc = df_acc.reindex(sorted_idx).reset_index(drop=True)

# Repeat models names for loss data
models_extended_loss = models * len(runs)

# Define loss data
val_loss = [0.251, 0.448, 1.853, 0.300, 0.354, 0.516, 1.645, 0.335, 0.175, 0.382, 2.112, 0.267, 0.285, 0.470, 1.983, 0.318, 0.214, 0.422, 1.724, 0.279]
test_loss = [0.278, 0.510, 2.043, 0.330, 0.312, 0.563, 1.918, 0.362, 0.202, 0.452, 2.189, 0.295, 0.312, 0.532, 2.070, 0.353, 0.251, 0.479, 1.987, 0.309]

# Create DataFrame for Validation and Test loss data
val_df = pd.DataFrame({'Model': models_extended_loss, 'Loss': val_loss, 'Type': ['Validation']*len(val_loss)})
test_df = pd.DataFrame({'Model': models_extended_loss, 'Loss': test_loss, 'Type': ['Test']*len(test_loss)})

# Combine DataFrames
df_loss = pd.concat([val_df, test_df], ignore_index=True)

df_loss = pd.concat([val_df, test_df], ignore_index=True)

# Sort by loss in ascending order
sorted_idx = df_loss.groupby('Model')['Loss'].transform(min).sort_values().index
df_loss = df_loss.reindex(sorted_idx).reset_index(drop=True)


# Create figure with two subplots: one for accuracy, one for loss
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Accuracy plot
sns.stripplot(x='Model', y='Accuracy', hue='Type', data=df_acc[df_acc['Type'] == 'Validation'], dodge=True, palette='Set2', ax=ax1, hue_order=['Validation', 'Test'], size=8, linewidth=1, marker='o')
sns.stripplot(x='Model', y='Accuracy', hue='Type', data=df_acc[df_acc['Type'] == 'Test'], dodge=True, palette='Set2', ax=ax1, hue_order=['Validation', 'Test'], size=8, linewidth=1, marker='^')
sns.despine(ax=ax1)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Performance on Validation and Test Data')
ax1.set_ylim(35, 100)  # Adjust y-axis limits to reduce whitespace
ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

# For y-axis of Accuracy plot
ax1.set_ylabel('Accuracy (%)')
ax1.yaxis.label.set_size(20)  # Set size to 20
ax1.tick_params(axis='y', labelsize=14)  # Increase y-axis tick labels to size 14

# For y-axis of Loss plot
ax2.tick_params(axis='y', labelsize=14)
ax2.set_ylabel('Cross-Entropy Loss')
ax2.yaxis.label.set_size(20)  # Set size to 20

# Loss plot
sns.stripplot(x='Model', y='Loss', hue='Type', data=df_loss[df_loss['Type'] == 'Validation'], dodge=True, palette='Set2', ax=ax2, hue_order=['Validation', 'Test'], size=8, linewidth=1, marker='o')
sns.stripplot(x='Model', y='Loss', hue='Type', data=df_loss[df_loss['Type'] == 'Test'], dodge=True, palette='Set2', ax=ax2, hue_order=['Validation', 'Test'], size=8, linewidth=1, marker='^')
sns.despine(ax=ax2)
ax2.set_title('Model Loss on Validation and Test Data')
ax2.set_ylabel('Cross-Entropy Loss')
ax2.set_ylim(0, np.max(val_loss+test_loss) + 0.1)  # Set y-axis limits
ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, fontsize=20, fontweight='bold', va='top', ha='right')

# For title of Accuracy and Loss plots
ax1.set_title('Model Performance on Validation and Test Data', fontsize=20)  # Set title font size to 20
ax2.set_title('Model Loss on Validation and Test Data', fontsize=20)  # Set title font size to 20

for ax in [ax1, ax2]:
    ax.set_xlabel('Model')
    ax.xaxis.label.set_size(20)
    ax.tick_params(axis='x', labelsize=14)# Set size to 20

# Manually specify color and marker for the legend
validation_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumseagreen', markersize=10)
test_marker = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='darkorange', markersize=10)

# Move legend to right side
ax1.legend(handles=[validation_marker, test_marker], labels=['Validation', 'Test'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
ax2.legend(handles=[validation_marker, test_marker], labels=['Validation', 'Test'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)

sns.despine(ax=ax2)
ax1.grid(False)
ax2.grid(False)

plt.tight_layout()
fig.savefig('model_performance.pdf', dpi=300, bbox_inches='tight')

plt.show()