import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

CSV_PATH = "outputs/results/results_summary.csv"
OUTPUT_DIR = "outputs/results/plots"
FILE_NAME = "vit_freezing_analysis_n10.png"

# Define the logical order of freezing modes for the X-axis
ORDER = [
    "freeze_none", 
    "freeze_patch", 
    "freeze_patch_0_2", 
    "freeze_patch_0_5", 
    "freeze_patch_0_11"
]

def main():
    # 1. Load Data
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    # 2. Filter for ViT only (just in case ResNet is in there)
    df_vit = df[df['model'] == 'vit_b_16'].copy()
    
    # 3. Create Output Directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. Setup Plot Style (Publication Quality)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.figure(figsize=(10, 6))

    # 5. Plot the MEAN Trend Line with ERROR BANDS (Standard Deviation)
    # Seaborn does this automatically with errorbar='sd'
    sns.lineplot(
        data=df_vit,
        x='freeze_mode',
        y='test_acc',
        sort=False,       # Prevent auto-sorting, we use custom order below
        marker='o',       # Add dots for the means
        markersize=10,
        linewidth=3,
        errorbar='sd',    # Draws the band for Standard Deviation
        label='Mean ± Std Dev'
    )

    # 6. Overlay Individual Seed Points (The "Raw Data")
    # We use stripplot to show the black dots for each seed
    sns.stripplot(
        data=df_vit,
        x='freeze_mode',
        y='test_acc',
        color='black',
        size=6,
        alpha=0.6,        # Transparency so overlapping dots are visible
        jitter=0.05,      # Slight random noise on X-axis to separate dots
        zorder=3          # Force dots to sit ON TOP of the line
    )

    # 7. Formatting the Axes
    # Manually force the X-axis order to ensure it follows our logical progression
    plt.gca().set_xticks(range(len(ORDER)))
    plt.gca().set_xticklabels(ORDER, rotation=30, ha="right")
    
    plt.title(f"ViT-B/16 Freezing Strategy (N={df_vit['seed'].nunique()} Seeds)", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Test Accuracy (%)", fontsize=14)
    plt.xlabel("Freezing Strategy", fontsize=14)
    plt.ylim(55, 85)  # Set Y-axis range to focus on the data (adjust if needed)
    
    # Clean up legend
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    # 8. Save
    save_path = os.path.join(OUTPUT_DIR, FILE_NAME)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
