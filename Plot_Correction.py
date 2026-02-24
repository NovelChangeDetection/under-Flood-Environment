import numpy as np
from prettytable import PrettyTable


def plot_Table():
    eval1 = np.load('Eval_All_Steps.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC', 'Overall Accuracy', 'Mean IOU', 'Separated Kappa']
    Classifier = ['TERMS', 'Unet', 'TRSANet', 'GA-CNN', 'DDUNet++', 'SCA-MCA-E-ADDUNet++-UAV-EWGOA']
    value1 = eval1[4, :, 4:]

    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    Table.add_column(Classifier[1], value1[5, :])
    Table.add_column(Classifier[2], value1[6, :])
    Table.add_column(Classifier[3], value1[7, :])
    Table.add_column(Classifier[4], value1[8, :])
    Table.add_column(Classifier[5], value1[9, :])

    print('-------------------------------------------------- Table Method Comparison',
          '--------------------------------------------------')
    print(Table)


an = 0
if an ==  1:
    import numpy as np
    from scipy import stats

    # Example: Cost values from multiple runs (replace with real data)
    uav_ewgoa = np.array([0.78, 0.80, 0.75, 0.77, 0.76])
    wgoa = np.array([1.25, 1.28, 1.26, 1.27, 1.26])


    # Descriptive statistics
    def describe(data):
        return {
            "Best": np.min(data),
            "Worst": np.max(data),
            "Mean": np.mean(data),
            "Median": np.median(data),
            "Std": np.std(data, ddof=1)
        }


    print("UAV-EWGOA:", describe(uav_ewgoa))
    print("WGOA:", describe(wgoa))

    # Two-sample t-test (one-tailed)
    t_stat, p_val = stats.ttest_ind(uav_ewgoa, wgoa, equal_var=False)
    p_val_one_tailed = p_val / 2 if t_stat < 0 else 1 - (p_val / 2)
    print(f"T-test p-value (one-tailed): {p_val_one_tailed}")

    # 95% confidence interval for UAV-EWGOA
    mean = np.mean(uav_ewgoa)
    sem = stats.sem(uav_ewgoa)
    ci = stats.t.interval(0.95, len(uav_ewgoa) - 1, loc=mean, scale=sem)
    print(f"95% CI for UAV-EWGOA mean: {ci}")

an = 0
if an == 1:
    import pandas as pd
    import matplotlib.pyplot as plt

    # ==== Data (Replace with your actual Figure 7 numbers) ====
    data = {
        "Steps": [50, 100, 150, 200, 250],
        "IoU (WO)": [69.36, 82.75, 79.56, 78.55, 82.07],
        "IoU (W)": [89.77, 95.04, 90.68, 91.05, 90.14],
        "Accuracy (WO)": [91.12, 93.05, 92.14, 91.86, 93.14],
        "Accuracy (W)": [97.84, 98.12, 97.55, 97.78, 97.63],
        "Dice (WO)": [82.15, 85.74, 83.24, 82.05, 85.21],
        "Dice (W)": [94.22, 96.44, 94.85, 94.92, 94.54],
        "F1 (WO)": [81.55, 85.02, 82.91, 81.88, 84.66],
        "F1 (W)": [93.94, 96.12, 94.54, 94.63, 94.24]
    }

    # ==== Create DataFrame ====
    df = pd.DataFrame(data)

    # ==== Print Table ====
    print("\n=== Figure 7 Table ===\n")
    print(df.to_string(index=False))

    # ==== Save Table to Excel & LaTeX ====
    df.to_excel("Figure7_Table.xlsx", index=False)
    with open("Figure7_Table.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))

    # ==== Line Plots for Each Metric ====
    metrics = ["IoU", "Accuracy", "Dice", "F1"]
    plt.figure(figsize=(10, 6))

    for metric in metrics:
        plt.plot(df["Steps"], df[f"{metric} (WO)"], marker='o', linestyle='--', label=f"{metric} - WO")
        plt.plot(df["Steps"], df[f"{metric} (W)"], marker='o', linestyle='-', label=f"{metric} - W")

    plt.xlabel("Steps per Epoch")
    plt.ylabel("Score (%)")
    plt.title("With vs Without Optimization Across Metrics")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

an = 1
if an == 1:
    # Metric labels
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC', 'Overall Accuracy', 'Mean IOU', 'Separated Kappa']
    Graph_Term = [0, 1, 2, 3, 7, 12, 14, 15, 16]

    # Load evaluation data
    Eval = np.load('Eval_All_Steps.npy', allow_pickle=True)  # Shape: (3, len(Batch_size), 5, num_metrics)
    Compare = ['TERMS', 'Without', 'With']
    steps = ['50', '100', '150', '200', '250']

    for m in range(Eval.shape[0]):
        value1 = Eval[m, :, 4:]

        Table = PrettyTable()
        Table.add_column(Compare[0], Terms)
        Table.add_column(Compare[1], value1[0, :])
        Table.add_column(Compare[2], value1[4, :])

        print('-------------- Steps per epoch - ',steps[m],
              ' --------------')
        print(Table)

# if __name__ == '__main__':
#     plot_Table()