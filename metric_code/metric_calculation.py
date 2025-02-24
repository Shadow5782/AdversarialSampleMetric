# Description: Functions to calculate the proposed robustness metric
# Author: Johannes Geier
# Date: 12.02.2025

import cw_attack
import pgd_attack
import dataset
import print_metric
import torch
from torch.utils.data import DataLoader
import pickle
import os

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def calc_loss_and_errorrate(model: torch.nn.Module, adv_samp_path: str) -> tuple[float,float]:
    with torch.no_grad():
        # Create Dataset for evaluation
        data = dataset.CIFARDataset(adv_samp_path)
        dataloader = DataLoader(data,1024,False)

        # Evaluate loss and error rate
        loss = torch.tensor(0.0, device=device)
        error_rate = torch.tensor(0.0, device=device)
        for images, true_lables, _ in dataloader:
            images = images.to(device)
            true_lables = true_lables.to(device)
            outputs = model(images)
            error_rate += outputs.shape[0] - (outputs.argmax(dim=1) == true_lables).sum()
            loss += torch.nn.functional.cross_entropy(outputs, true_lables, reduction="sum")
        loss /= data.amount_datapoints
        error_rate /= data.amount_datapoints

    return loss.item(), error_rate.item()

# Folder for storing the metric results
task_name = "attacks/advTrainDF"

# Folder to the model which is going to be attacked 
white_box_model_path = "evaluation_models/advTrainDF/DFWhiteBox.pth"
grey_box_model_path = "evaluation_models/advTrainDF/DFGreyBox.pth"
black_box_model_path = "evaluation_models/advTrainDF/DFBlackBox.pth"

# Folders with samples to be used as starting points for adversarial sample creation
targeted_data_path = "data/targeted_data"
untargeted_data_path = "data/untargeted_data"

# Set to true if you want to print the results as graph
print_metric_to_png = True

# Define steps for allowed perturbations
distance_limit_max = 1
distance_limit_min = 0
distance_steps = 11

# Limit the amount of iterations (Timout Value)
max_iterations = 50

# Prepare Model
# White Box
from evaluation_models.WhiteBox_Model import Net as WhiteBox
wb_model = WhiteBox()
wb_model.load_state_dict(torch.load(white_box_model_path, weights_only=True, map_location=device))
wb_model = wb_model.to(device)
wb_model.eval()

# Grey Box
from evaluation_models.GreyBox_Model import SubstituteNet as GreyBox
gb_model = GreyBox()
gb_model.load_state_dict(torch.load(grey_box_model_path, weights_only=True, map_location=device))
gb_model = gb_model.to(device)
gb_model.eval()

# Black Box
from evaluation_models.BlackBox_Model import SubstituteNet as BlackBox
bb_model = BlackBox()
bb_model.load_state_dict(torch.load(black_box_model_path, weights_only=True, map_location=device))
bb_model = bb_model.to(device)
bb_model.eval()

# Prepare storage
running_distance_metric = []
running_loss_score = []
running_robustness_score = []

running_wb_loss = []
running_wb_errorrate = []
running_wb_loss_untarg = []
running_wb_errorrate_untarg = []
running_wb_loss_targ = []
running_wb_errorrate_targ = []

running_gb_loss = []
running_gb_errorrate = []
running_gb_loss_untarg = []
running_gb_errorrate_untarg = []
running_gb_loss_targ = []
running_gb_errorrate_targ = []

running_bb_loss = []
running_bb_errorrate = []
running_bb_loss_untarg = []
running_bb_errorrate_untarg = []
running_bb_loss_targ = []
running_bb_errorrate_targ = []

# Create directories
if not os.path.isdir(task_name):
    os.mkdir(task_name)
if not os.path.isdir(task_name + "/wb_untarg"):
    os.mkdir(task_name + "/wb_untarg")
if not os.path.isdir(task_name + "/wb_targ"):
    os.mkdir(task_name + "/wb_targ")
if not os.path.isdir(task_name + "/gb_untarg"):
    os.mkdir(task_name + "/gb_untarg")
if not os.path.isdir(task_name + "/gb_targ"):
    os.mkdir(task_name + "/gb_targ")
if not os.path.isdir(task_name + "/bb_untarg"):
    os.mkdir(task_name + "/bb_untarg")
if not os.path.isdir(task_name + "/bb_targ"):
    os.mkdir(task_name + "/bb_targ")

# Loop through all allowed perturbation-limits
print("Started evaluation")
for i in range(0,distance_steps):
    # Calculate current distance limitation
    eps = distance_limit_min + i * distance_limit_max/(distance_steps-1)
    running_distance_metric.append(eps)
    print(f"Run {i + 1} with eps: {eps}")
    
    # Run Attacks
    print("Started white box untargeted attack")
    _ = pgd_attack.run_pgd_attack(wb_model, untargeted_data_path, task_name + "/wb_untarg" + f"/eps_{eps:.2}", eps, 2/255, max_iterations)
    eval_output_loss, eval_output_errorrate = calc_loss_and_errorrate(wb_model,task_name + "/wb_untarg" + f"/eps_{eps:.2}")
    running_wb_loss_untarg.append(eval_output_loss)
    running_wb_errorrate_untarg.append(eval_output_errorrate)

    print("Started white box targeted attack")
    _ = cw_attack.run_cw_attack(wb_model, targeted_data_path, task_name + "/wb_targ" + f"/eps_{eps:.2}", 1, 0, max_iterations, 0.1, eps)
    eval_output_loss, eval_output_errorrate = calc_loss_and_errorrate(wb_model,task_name + "/wb_targ" + f"/eps_{eps:.2}")
    running_wb_loss_targ.append(eval_output_loss)
    running_wb_errorrate_targ.append(eval_output_errorrate)

    print("Started grey box untargeted attack")
    _ = pgd_attack.run_pgd_attack(gb_model, untargeted_data_path, task_name + "/gb_untarg" + f"/eps_{eps:.2}", eps, 2/255, max_iterations)
    eval_output_loss, eval_output_errorrate = calc_loss_and_errorrate(wb_model,task_name + "/gb_untarg" + f"/eps_{eps:.2}")
    running_gb_loss_untarg.append(eval_output_loss)
    running_gb_errorrate_untarg.append(eval_output_errorrate)

    print("Started grey box targeted attack")
    _ = cw_attack.run_cw_attack(gb_model, targeted_data_path, task_name + "/gb_targ" + f"/eps_{eps:.2}", 1, 0, max_iterations, 0.1, eps)
    eval_output_loss, eval_output_errorrate = calc_loss_and_errorrate(wb_model,task_name + "/gb_targ" + f"/eps_{eps:.2}")
    running_gb_loss_targ.append(eval_output_loss)
    running_gb_errorrate_targ.append(eval_output_errorrate)

    print("Started black box untargeted attack")
    _ = pgd_attack.run_pgd_attack(bb_model, untargeted_data_path, task_name + "/bb_untarg" + f"/eps_{eps:.2}", eps, 2/255, max_iterations)
    eval_output_loss, eval_output_errorrate = calc_loss_and_errorrate(wb_model,task_name + "/bb_untarg" + f"/eps_{eps:.2}")
    running_bb_loss_untarg.append(eval_output_loss)
    running_bb_errorrate_untarg.append(eval_output_errorrate)

    print("Started black box targeted attack")
    _ = cw_attack.run_cw_attack(bb_model, targeted_data_path, task_name + "/bb_targ" + f"/eps_{eps:.2}", 1, 0, max_iterations, 0.1, eps)
    eval_output_loss, eval_output_errorrate = calc_loss_and_errorrate(wb_model,task_name + "/bb_targ" + f"/eps_{eps:.2}")
    running_bb_loss_targ.append(eval_output_loss)
    running_bb_errorrate_targ.append(eval_output_errorrate)

    # Evaluate Robustness
    running_wb_loss.append((0.8 * running_wb_loss_untarg[-1] + 1.0 * running_wb_loss_targ[-1])/(0.8+1.0))
    running_wb_errorrate.append((0.8 * running_wb_errorrate_untarg[-1] + 1.0 * running_wb_errorrate_targ[-1])/(0.8+1.0))
    running_gb_loss.append((0.8 * running_gb_loss_untarg[-1] + 1.0 * running_gb_loss_targ[-1])/(0.8+1.0))
    running_gb_errorrate.append((0.8 * running_gb_errorrate_untarg[-1] + 1.0 * running_gb_errorrate_targ[-1])/(0.8+1.0))
    running_bb_loss.append((0.8 * running_bb_loss_untarg[-1] + 1.0 * running_bb_loss_targ[-1])/(0.8+1.0))
    running_bb_errorrate.append((0.8 * running_bb_errorrate_untarg[-1] + 1.0 * running_bb_errorrate_targ[-1])/(0.8+1.0))

    running_loss_score.append((0.8 * running_wb_loss[-1] + 0.9 * running_gb_loss[-1] + 1.0 * running_bb_loss[-1])/(0.8+0.9+1.0))
    running_robustness_score.append((0.8 * running_wb_errorrate[-1] + 0.9 * running_gb_errorrate[-1] + 1.0 * running_bb_errorrate[-1])/(0.8+0.9+1.0))
    print("Scores calculated")

print("Evaluation finished")

# Save results
if not os.path.isdir(task_name):
    os.mkdir(task_name)
with open(task_name + "/metric_outputs.pkl","wb") as output_file:
    pickle.dump([
        running_distance_metric,
        running_loss_score,
        running_robustness_score,
        running_wb_loss,
        running_wb_errorrate,
        running_wb_loss_untarg,
        running_wb_errorrate_untarg,
        running_wb_loss_targ,
        running_wb_errorrate_targ,
        running_gb_loss,
        running_gb_errorrate,
        running_gb_loss_untarg,
        running_gb_errorrate_untarg,
        running_gb_loss_targ,
        running_gb_errorrate_targ,
        running_bb_loss,
        running_bb_errorrate,
        running_bb_loss_untarg,
        running_bb_errorrate_untarg,
        running_bb_loss_targ,
        running_bb_errorrate_targ
    ], output_file)

print("Outputs saved")

# Print Results
if print_metric_to_png:
    print_metric.print_graph_and_table(task_name, "metric_outputs.pkl")
    print("Metric Printed")