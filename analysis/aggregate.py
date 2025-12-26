import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set publication style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def main():
    csv_path = "outputs/results/results_summary.csv"
    if not os.path.exists(csv_path):
        print("Results file not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Define orders
    vit_order = ["freeze_none", "freeze_patch", "freeze_patch_0_2", 
                 "freeze_patch_0_5", "freeze_patch_0_11"]
    resnet_order = ["freeze_none", "freeze_0", "freeze_0_1", 
                    "freeze_0_1_2", "freeze_0_1_2_3", "freeze_0_1_2_3_4"]

    out_dir = "outputs/results/final_analysis"
    os.makedirs(out_dir, exist_ok=True)

    #   1. Generate Stats CSV (Mean +/- Std) 
    stats = df.groupby(['model', 'freeze_mode'])['test_acc'].agg(['mean', 'std', 'count']).reset_index()
    stats.to_csv(f"{out_dir}/aggregated_stats.csv", index=False)
    print("Stats saved to aggregated_stats.csv")

    #   2. Plot ViT Aggregate (Line plot with Error Band) 
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df[df['model'] == 'vit_b_16'],
        x='freeze_mode',
        y='test_acc',
        hue='model',
        style='model',
        markers=True,
        dashes=False,
        errorbar='sd',  # Shows Standard Deviation band automatically
        sort=False      # Don't auto-sort, use our manual order logic below if needed
    )
    # Fix X-axis order manually since lineplot can be tricky with categories
    plt.gca().set_xticks(range(len(vit_order)))
    plt.gca().set_xticklabels(vit_order, rotation=45)
    
    plt.title("ViT-B/16: Freezing Strategy (Mean ± Std over 10 Seeds)")
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Freezing Strategy")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/vit_aggregate_trend.png", dpi=300)
    plt.close()

    # 3. Plot ResNet Aggregate 
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df[df['model'] == 'resnet50'],
        x='freeze_mode',
        y='test_acc',
        errorbar='sd',
        markers=True
    )
    plt.gca().set_xticks(range(len(resnet_order)))
    plt.gca().set_xticklabels(resnet_order, rotation=45)
    
    plt.title("ResNet50: Freezing Strategy (Mean ± Std over 10 Seeds)")
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Freezing Strategy")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/resnet_aggregate_trend.png", dpi=300)
    plt.close()
    
    print(f"Aggregate plots saved to {out_dir}")

if __name__ == "__main__":
    main()
