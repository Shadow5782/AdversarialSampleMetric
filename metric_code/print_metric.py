# Description: Function to plot metric results to a graph/table
# Author: Johannes Geier
# Date: 19.12.2024

import matplotlib.pyplot as plt
import pickle
import numpy as np

# Print metric results as a graph with a table
def print_graph_and_table(filepath: str, filename: str):
    # Open results list from pickle file
    with open(filepath + "/" + filename,"rb") as file:
        results = pickle.load(file)
    
    # Plot robustness vs perturbation curve
    plt.plot(results[0],results[2],"o-r")
    plt.ylim(top=1, bottom=0)
    plt.ylabel("Robustness Score")
    plt.xlabel("Perturbation")
    plt.title("Robustness vs Perturbation Curve")
    plt.grid()

    plt.savefig(filepath + "/metricCurve.png")
    
    plt.clf()

    # Plot loss vs perturbation curve
    plt.plot(results[0],results[1],"o-b")
    plt.ylabel("Loss")
    plt.xlabel("Perturbation")
    plt.title("Loss vs Perturbation Curve")
    plt.grid()

    plt.savefig(filepath + "/lossCurve.png")
    
    plt.clf()

    # Plot table with exact scores
    # Prepare numbers and text for table
    rows = ("Loss Score", "Robustness Score", "White Box Loss", "White Box Error Rate", "White Box Loss Untargeted", "White Box Error Rate Untargeted", "White Box Loss Targeted", "White Box Error Rate Targeted", "Grey Box Loss", "Grey Box Error Rate", "Grey Box Loss Untargeted", "Grey Box Error Rate Untargeted", "Grey Box Loss Targeted", "Grey Box Error Rate Targeted", "Black Box Loss", "Black Box Error Rate", "Black Box Loss Untargeted", "Black Box Error Rate Untargeted", "Black Box Loss Targeted", "Black Box Error Rate Targeted")
    columns = [f"\u03B4: {i:.3}" for i in results[0]]

    cell_text = []
    for row in results[1:]:
        cell_text.append([f"{i:.5}" for i in row])

    rcolors = plt.cm.BuPu(np.full(len(rows), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(columns), 0.1))

    plt.figure(figsize=(15,7),tight_layout={'pad':1})
    
    # Plot table
    table = plt.table(cellText=cell_text, colLabels=columns, colColours=ccolors, rowLabels=rows, rowColours=rcolors, loc="center")
    table.scale(1, 1.5)
    
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)

    plt.xticks([])
    plt.title('Values of the robustness analysis')
    plt.draw()
    # Save plot
    plt.savefig(filepath + "/ScoreTable.png")

def print_combined_graphs(paths: list[str], filename: str, labels: list[str]):
    if len(paths) != len(labels):
        raise Exception("Violation: len(paths) != len(labels) != len(filenames)")

    colors = ["b", "g", "r", "c", "m", "y", "k"]

    if len(colors) < len(paths):
        raise Exception("More colors needed, adjust manually in code: \"print_metric.py\" line 71")

    for i in range(0, len(paths)):
        with open(paths[i] + "/" + filename, "rb") as file:
            results = pickle.load(file)
            plt.plot(results[0], results[2], f"o-{colors[i]}",label=labels[i])
    # plt.ylim(top=1, bottom=0)
    plt.ylabel("Robustness Score")
    plt.xlabel("Perturbation")
    plt.title("Robustness vs Perturbation Curve")
    plt.legend()
    plt.grid()

    plt.savefig("metricCurve.png")

    plt.clf()

    for i in range(0, len(paths)):
        with open(paths[i] + "/" + filename, "rb") as file:
            results = pickle.load(file)
            plt.plot(results[0], results[1], f"o-{colors[i]}",label=labels[i])
    plt.ylabel("Loss")
    plt.xlabel("Perturbation")
    plt.title("Loss vs Perturbation Curve")
    plt.legend()
    plt.grid()

    plt.savefig("lossCurve.png")

if __name__ == "__main__":
    # Testing
    # print_graph_and_table("2_Code/attacks/standardNet", "metric_outputs.pkl")
    paths = [
        "metric_results/advTrainCW",
        "metric_results/advTrainDF",
        "metric_results/advTrainPGD",
        "metric_results/advTrainPGDandCW",
        "metric_results/standardNet"
    ]
    filename = "metric_outputs.pkl"
    labels = ["CW Adv Train", "DeepFool Adv Train", "PGD Adv Train", "PGD & CW Adv Train", "Standard"]
    print_combined_graphs(paths,filename,labels)