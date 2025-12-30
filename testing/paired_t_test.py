from scipy import stats
import numpy as np

# SAMPLE DATA (SEEDS 1, 8, 16, 24, 32, 40, 48, 56, 64, 72) FOR FREEZE_NONE AND FREEZE_5 (0% AND 50%)

freeze_none = np.array([75, 75, 74, 72, 72, 70, 67, 67, 73, 71])
freeze_50 = np.array([68, 73, 73, 77, 76, 70, 73, 73, 75, 74])

# PAIRED T-TEST
t_stat, p_value = stats.ttest_rel(freeze_none, freeze_50)

# Effect size (Cohen's d for paired samples)
differences = freeze_none - freeze_50
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)
cohens_d = mean_diff / std_diff

print(f"Mean freeze-none: {np.mean(freeze_none):.2f}%")
print(f"Mean freeze-50%: {np.mean(freeze_50):.2f}%")
print(f"Mean difference: {mean_diff:.2f}%")
print(f"Paired t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.4f}")
