import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set publication-quality theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_seed(csv_path, seed, out_dir):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # Filter for this specific seed
    df_seed = df[df['seed'] == seed].copy()
    
    if df_seed.empty:
        print(f"No data found for seed {seed}")
        return

    # 2. Define Order (Important for 'Bell Shape' visualization)
    vit_order = ["freeze_none", "freeze_patch", "freeze_patch_0_2", 
                 "freeze_patch_0_5", "freeze_patch_0_11"]
    
    resnet_order = ["freeze_none", "freeze_0", "freeze_0_1", 
                    "freeze_0_1_2", "freeze_0_1_2_3", "freeze_0_1_2_3_4"]
    
    # 3. Plot ViT
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df_seed[df_seed['model'] == 'vit_b_16'],
        x='freeze_mode', 
        y='test_acc',
        order=vit_order,
        palette="viridis",  # Modern, colorblind-friendly
        edgecolor="black"   # Sharp edges for publication
    )
    plt.title(f"ViT-B/16 Freezing Strategy (Seed {seed})")
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Freezing Strategy")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/seed{seed}_vit_barplot.png", dpi=300)
    plt.close()



    # 4. Plot ResNet
    resnet_data = df_seed[df_seed['model'] == 'resnet50']

    if not resnet_data.empty:
        plt.figure(figsize=(8, 6))
        sns.barplot(
            data=df_seed[df_seed['model'] == 'resnet50'],
            x='freeze_mode', 
            y='test_acc',
            order=resnet_order,
            palette="magma",
            edgecolor="black"
        )
        plt.title(f"ResNet50 Freezing Strategy (Seed {seed})")
        plt.ylabel("Test Accuracy (%)")
        plt.xlabel("Freezing Strategy")
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/seed{seed}_resnet_barplot.png", dpi=300)
        plt.close()
        
        print(f"Plots saved to {out_dir} for seed {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    
    # Path to your results file
    csv_file = "outputs/results/results_summary.csv"
    output_dir = "outputs/results/plots"
    
    if os.path.exists(csv_file):
        plot_seed(csv_file, args.seed, output_dir)
    else:
        print(f"Results file not found: {csv_file}")
