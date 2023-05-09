from nuscenes import NuScenes
import json
import math
import numpy as np
import cv2
from pyquaternion import Quaternion

from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedFuture
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction.helper import angle_of_rotation, angle_diff

import matplotlib.pyplot as plt



def prediction_ade_in_time(pred_t):
    print("before subtraction: ", pred_t)
    for i, row in enumerate(pred_t):
        pred_t[i] = row - np.where(row != 0, gt_t[i], 0)
    print("after: ", pred_t)
    pred_t = np.squeeze(pred_t)
    pred_t_masked = np.ma.masked_equal(pred_t, 0)
    error = np.linalg.norm(pred_t_masked, axis=(0))
    error = np.linalg.norm(error, axis=(1))
    return error

# Load Data
DATASETPATH = '../../../v1.0-trainval_full/'
JSONPATH =  '../../results/Nuscenes/Autobot_joint_C5_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_roadLanes_test_s1/autobot_preds.json'
nusc = NuScenes('v1.0-trainval', dataroot=DATASETPATH)
print("Dataset Loaded")

f = open(JSONPATH)
predicted_trajectories = json.load(f)
print("Predicted Trajectories loaded")

# Set up true state plotting
helper = PredictHelper(nusc)
static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedFuture(helper, seconds_of_future=0)
mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

# can change this frame based on how interesting scene is
#sample_token = '7626dde27d604ac28a0240bdd54eba7a'
sample_token = predicted_trajectories[1000]['sample']

instance_count = 0
for pred in predicted_trajectories:
    if pred['sample'] == sample_token and pred['instance'] != 'None':
        instance_count += 1

# iterate to beginning of scene
while (nusc.get('sample', sample_token)['prev'] != ''):
    sample_token = nusc.get('sample', sample_token)['prev']

my_sample = nusc.get('sample', sample_token)

my_annotation_token = my_sample['anns'][0]
my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)
instance_token = my_annotation_metadata['instance_token']

fig_list = []
pred_t = np.zeros((2, 10, 60, 2))
gt_t = np.zeros((60, 2))
prob_t = np.zeros((2, 10))

# iterates over frames (samples)
for i in range(39):
    if ((i+2)%4 == 0):
        time_idx = (i-2)//4
        fig, ax = plt.subplots(figsize=(8, 6))
        # render true future of agents
        im = ax.imshow(mtp_input_representation.make_input_representation(instance_token, sample_token), label='Real Trajectory')
        sample_annotation = helper.get_sample_annotation(instance_token, sample_token)
        x_ego, y_ego = sample_annotation['translation'][:2]
        gt_t[i] = x_ego, y_ego
        yaw = quaternion_yaw(Quaternion(sample_annotation['rotation']))      
        
        # render predicted trajectory of agents
        for pred in predicted_trajectories:
            if (pred['sample'] == sample_token and pred['instance'] == instance_token):
                top_pred_indices = [np.argmax(pred['probabilities']), np.argpartition(pred['probabilities'], -2)[-2]]
        
                for k, pred_idx in enumerate(top_pred_indices):
                    x_list = []
                    y_list = []
                    for j, pos in enumerate(pred['prediction'][pred_idx]):
                        x_agent, y_agent = pos[0], pos[1]
                        x_pixel = (x_agent - x_ego) * 10
                        y_pixel = - (y_agent - y_ego) * 10
                        row_pixel = int(250 + x_pixel)
                        column_pixel = int(400 + y_pixel)
                        pred_t[k, time_idx, time_idx+j] = x_pixel, y_pixel
                        #prob_t[k, time_idx] = pred['probabilities'][pred_idx]
                        x_list.append(row_pixel), y_list.append(column_pixel)
                        
                    ax.plot(x_list, y_list, color=(1, 1, 148/255), lw=5, alpha=0.8, zorder=2)
                    ax.plot(x_list, y_list, color='black', lw=1.5, linestyle='--', dashes=(2.5, 0.5), zorder=3)
                    
                    x_arrow_start = x_list[-2]
                    y_arrow_start = y_list[-2]
                    x_end = x_list[-1]
                    y_end = y_list[-1]
                    
                    # Add an arrow annotation at the end of the line
                    arrow_props = dict(arrowstyle='->', color='black', mutation_scale=10)
                    ax.annotate("", xy=(x_end, y_end), xytext=(x_arrow_start, y_arrow_start), arrowprops=arrow_props)
                    
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # append combined image to full scene list
        fig_list.append(fig)
        plt.close()
    sample_token = nusc.get('sample', sample_token)['next']

ade_list = prediction_ade_in_time(pred_t[0])
print("average error for each time step!", ade_list)

for i in range(len(fig_list)):
        fig_list[i].savefig(f'plot_final_time={2*i + 1}.png')
    


    
    
