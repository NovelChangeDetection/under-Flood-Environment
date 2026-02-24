import numpy as np
import pylab
from matplotlib import pyplot as plt
from prettytable import PrettyTable


def plot_results_conv():
    conv = np.load('Fitness.npy', allow_pickle=True)

    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Algorithm = ['GBOA-SCA-MCA-E-ADDUNet++', 'FOA-SCA-MCA-E-ADDUNet++', 'AZOA-SCA-MCA-E-ADDUNet++', 'WGOA-SCA-MCA-E-ADDUNet++', 'UAV-EWGOA-SCA-MCA-E-ADDUNet++']
    color = ['Deeppink', 'm', 'r', 'y', 'k']
    markerfacecolor = ['red', 'green', 'cyan', 'magenta', 'black']
    Value = np.zeros((conv.shape[0], 5))
    for j in range(conv.shape[0]):
        Value[j, 0] = np.min(conv[j, :])
        Value[j, 1] = np.max(conv[j, :])
        Value[j, 2] = np.mean(conv[j, :])
        Value[j, 3] = np.median(conv[j, :])
        Value[j, 4] = np.std(conv[j, :])

    Table = PrettyTable()
    Table.add_column("Statistical", Statistics)
    for j in range(len(Algorithm)):
        Table.add_column(Algorithm[j], Value[j, :])
    print('----------------------------- Statistical Analysis ----------------')
    print(Table)

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence')
    iteration = np.arange(conv.shape[1])
    for m in range(conv.shape[0]):
        plt.plot(iteration, conv[m, :], color=color[m], linewidth=3, marker='*',
                 markerfacecolor=markerfacecolor[m], markersize=12, label=Algorithm[m])
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    path = "./Results/Conv.png"
    plt.savefig(path)
    plt.show(block=False)
    plt.pause(2)
    plt.close()



def Plot_Seg_Results():
    Eval = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC', 'Overall Accuracy', 'Mean IOU', 'Separated Kappa']
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN']  # , 'STD']
    Full = ['TERMS', 'GBOA-SCA-MCA-E-ADDUNet++', 'FOA-SCA-MCA-E-ADDUNet++', 'AZOA-SCA-MCA-E-ADDUNet++', 'WGOA-SCA-MCA-E-ADDUNet++', 'UAV-EWGOA-SCA-MCA-E-ADDUNet++',
            'Unet', 'TRSANet', 'GA-CNN', 'DDUNet++', 'SCA-MCA-E-ADDUNet++-UAV-EWGOA']
    Graph_terms = [0, 1, 2, 3, 7, 12, 14, 15, 16]
    # Graph_terms = [0, 1, 2, 5, 6, 8, 14, 15, 16]
    stats = np.zeros((len(Graph_terms), Eval.shape[-3] + 1, 5))  # (METRICS, ALGORITHM, STATS)
    Eval_all = Eval
    for k in range(len(Graph_terms)):
        for j in range(Eval_all.shape[-3] + 1):
            if j < Eval_all.shape[-3]:
                stats[k, j, 0] = np.max(Eval_all[j][:, Graph_terms[k] + 4])
                stats[k, j, 1] = np.min(Eval_all[j][:, Graph_terms[k] + 4])
                stats[k, j, 2] = np.mean(Eval_all[j][:, Graph_terms[k] + 4])
                stats[k, j, 3] = np.median(Eval_all[j][:, Graph_terms[k] + 4])
                stats[k, j, 4] = np.std(Eval_all[j][:, Graph_terms[k] + 4])

        alg_prop = stats[k, 4, :]
        stats[k, 9, :] = alg_prop

        # Algorithm comparision
        Alg_Val = stats[k, :5, :-1]
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(Alg_Val.shape[-1])
        X = x
        colour = ['Deeppink', 'sienna', 'green', 'blue', 'teal']
        bar_width = 0.15
        offsets = [0.00, 0.15, 0.30, 0.45, 0.60]
        bars = []
        for i in range(5):
            bar_group = ax.bar(X + offsets[i], Alg_Val[i, :], color=colour[i], edgecolor='w', width=bar_width)
            bars.append(bar_group)
            highlight_map = {
                0: [0, 4],  # BEST → 1st and 5th
                1: [1],  # WORST → 2nd
                2: [2],  # MEAN → 3rd
                3: [3],  # MEDIAN → 4th
            }
            for j in range(len(X)):  # Loop over statistics (BEST, WORST, etc.)
                bar_height = Alg_Val[i, j]
                x_center = X[j] + offsets[i]
                # y_top = bar_height + 10  # 1.5
                y_top = bar_height * 1.20
                label_text = Full[i + 1]
                if i in highlight_map.get(j, []):  # Check if this algorithm should be labeled in this stat

                    ax.text(x_center + 0.125, y_top + 1.25, label_text, ha='center', va='bottom',  # 1.5
                            fontsize=10, fontweight='bold')
                    ax.hlines(y=y_top, xmin=x_center, xmax=x_center + 0.21,
                              color=colour[i], linewidth=2.5)
                    ax.vlines(x=x_center, ymin=bar_height, ymax=y_top,
                              color='k', linestyle='dotted', linewidth=1.5)  # color=colour[i]

        ax.set_xticks(X + (bar_width * 2))
        ax.set_xticklabels(Statistics, fontsize=10, fontweight='bold')
        ax.set_xlabel('Statistical Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel(Terms[Graph_terms[k]], fontsize=12, fontweight='bold')
        plt.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.yticks(np.arange(0, np.max(Alg_Val) + 7, (np.max(Alg_Val) + 2) / 10))
        ax.spines['top'].set_color('lightgray')
        ax.spines['top'].set_linewidth(0.0)
        ax.spines['right'].set_color('lightgray')
        ax.spines['right'].set_linewidth(0.0)
        plt.tight_layout()
        plt.savefig(f"./Results/Seg_{Terms[Graph_terms[k]]}_Alg.png")
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Algorithm comparision of Statistical Analysis vs ' + Terms[Graph_terms[k]])
        plt.show()

        # Method comparision
        Mtd_Val = stats[k, 5:, :-1]
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(Mtd_Val.shape[-1])
        X = x
        colour = ['coral', 'lightgreen', 'steelblue', 'peru', 'olive']
        bar_width = 0.15
        offsets = [0.00, 0.15, 0.30, 0.45, 0.60]
        bars = []
        for i in range(5):
            bar_group = ax.bar(X + offsets[i], Mtd_Val[i, :], color=colour[i], edgecolor='w', width=bar_width)
            bars.append(bar_group)

            highlight_map = {
                0: [0, 4],  # BEST → 1st and 5th
                1: [1],  # WORST → 2nd
                2: [2],  # MEAN → 3rd
                3: [3],  # MEDIAN → 4th
            }
            for j in range(len(X)):  # Loop over statistics (BEST, WORST, etc.)
                bar_height = Mtd_Val[i, j]
                x_center = X[j] + offsets[i]
                y_top = bar_height * 1.20
                # y_top = bar_height + 10  # 1.5
                label_text = Full[i + 6]
                if i in highlight_map.get(j, []):
                    ax.text(x_center + 0.125, y_top + 1.25, label_text, ha='center', va='bottom',  # 1.5
                            fontsize=10, fontweight='bold')
                    ax.hlines(y=y_top, xmin=x_center, xmax=x_center + 0.21,
                              color=colour[i], linewidth=2.5)
                    ax.vlines(x=x_center, ymin=bar_height, ymax=y_top,
                              color='k', linestyle='dotted', linewidth=1.5)  # color=colour[i]
        ax.set_xticks(X + (bar_width * 2))
        ax.set_xticklabels(Statistics, fontsize=10, fontweight='bold')
        ax.set_xlabel('Statistical Analysis', fontsize=12, fontweight='bold')
        ax.set_ylabel(Terms[Graph_terms[k]], fontsize=12, fontweight='bold')
        plt.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.yticks(np.arange(0, np.max(Alg_Val) + 7, (np.max(Alg_Val) + 2) / 10))
        ax.spines['top'].set_color('lightgray')
        ax.spines['top'].set_linewidth(0.0)
        ax.spines['right'].set_color('lightgray')
        ax.spines['right'].set_linewidth(0.0)
        plt.tight_layout()
        plt.savefig(f"./Results/Seg_{Terms[Graph_terms[k]]}_Mtd.png")
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Method comparision of Statistical Analysis vs ' + Terms[Graph_terms[k]])
        plt.show()


def Plot_obj():
    # Metric labels
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC', 'Overall Accuracy', 'Mean IOU', 'Separated Kappa']
    Graph_Term = [0, 1, 2, 3, 7, 12, 14, 15, 16]

    # Load evaluation data
    Eval = np.load('Eval_All_Steps.npy', allow_pickle=True)# Shape: (3, len(Batch_size), 5, num_metrics)
    Batch_size = ['50', '100', '150', '200', '250']

    for m in range(len(Graph_Term)):
        metric_index = Graph_Term[m] + 4
        Graph = Eval[:, :, metric_index]  # Shape: (3, len(Batch_size))

        bar_width = 0.35
        group_spacing = 1.0
        index = np.arange(len(Batch_size)) * (bar_width * 3 + group_spacing)

        plt.figure(figsize=(10, 6))

        # Plot bars
        bars1 = plt.bar(index, Graph[:, 0], bar_width, label='SCA-MCA-E-DDUNet++', color='cornflowerblue')
        bars2 = plt.bar(index + bar_width + 0.05, Graph[:, 4], bar_width, label='SCA-MCA-E-ADDUNet++-UAV-EWGOA', color='coral')

        # Add text labels above bars, rotated
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}',
                     ha='center', va='bottom', rotation=90, fontsize=10)
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}',
                     ha='center', va='bottom', rotation=90, fontsize=10)

        # Center xticks between the two bars
        xtick_positions = index + (bar_width + 0.05) / 2
        plt.xticks(xtick_positions, Batch_size)

        # Axis and label formatting
        plt.xlabel('Steps per epoch')
        plt.ylabel(Terms[Graph_Term[m]])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=3, frameon=False)

        # Style the plot
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        plt.yticks([])

        # Vertical lines between groups
        ax.axvline(x=index[0] - group_spacing / 2, color='gray', linestyle='-', linewidth=1)
        for i in range(len(Batch_size)):
            xpos = index[i] + 3 * bar_width + 0.10
            ax.axvline(x=xpos, color='gray', linestyle='-', linewidth=1)

        plt.tight_layout()
        plt.savefig(f"./Results/Steps_{Terms[Graph_Term[m]]}_Method.png", bbox_inches='tight')
        plt.show(block=False)
        plt.pause(2)
        plt.close()


if __name__ == '__main__':
    Plot_obj()
    plot_results_conv()
    Plot_Seg_Results()
