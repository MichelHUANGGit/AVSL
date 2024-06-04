import matplotlib.pyplot as plt

# Data for the plot
methods = [
    "Baseline ResNet50 (frozen, finetuned FC)",
    "AVSL ResNet50 with ContrastiveLoss",
    "AVSL ResNet50 (paper params)",
    "AVSL ResNet50 (tuned params)",
    "AVSL EfficientNetSmall",
    "AVSL EfficientNetLarge"
]

accuracies = [63, 82, 85, 86, 88, 89]

# Create the bar plot
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(methods, accuracies, color='skyblue')

# Add text annotations for each bar
for bar in bars:
    width = bar.get_width()
    ax.text(width - 2, bar.get_y() + bar.get_height()/2, f'{width}%', 
            ha='center', va='center', color='white', fontsize=10, fontweight='bold')

# Set labels and title
ax.set_xlabel('Accuracy (%)')
ax.set_title('Leaderboard Accuracy (Public)')
ax.set_xlim(0, 100)

# Show the plot
plt.tight_layout()
plt.show()
