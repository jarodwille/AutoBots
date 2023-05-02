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
    pred_t_masked = np.ma.masked_equal(pred_t, 0)
    return np.linalg.norm(pred_t_masked, axis=1)

# Load Data
DATASETPATH = '../../../v1.0-trainval_full/'
JSONPATH =  '../../results/Nuscenes/Autobot_joint_C10_H128_E2_D2_TXH384_NH16_EW40_KLW20_NormLoss_roadLanes_test_s1/autobot_preds.json'
nusc = NuScenes('v1.0-trainval', dataroot=DATASETPATH)
print("Dataset Loaded")

f = open(JSONPATH)
predicted_trajectories = json.load(f)
print("Predicted Trajectories loaded")

# Set up true state plotting
helper = PredictHelper(nusc)
static_layer_rasterizer = StaticLayerRasterizer(helper)
agent_rasterizer = AgentBoxesWithFadedFuture(helper, seconds_of_future=2)
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
pred_t = np.zeros((40, 60, 2))

# iterates over frames (samples)
for i in range(39):
    if (i%4 == 0):
        fig, ax = plt.subplots(figsize=(8, 6))
        # render true future of agents
        im = ax.imshow(mtp_input_representation.make_input_representation(instance_token, sample_token), label='Real Trajectory')
        sample_annotation = helper.get_sample_annotation(instance_token, sample_token)
        x_ego, y_ego = sample_annotation['translation'][:2]      
        
        # render predicted trajectory of agents
        for pred in predicted_trajectories:
            if (pred['sample'] == sample_token and pred['instance'] == instance_token):
                top_pred_idx = np.argmax(pred['probabilities'])
                x_list = []
                y_list = []
                for j, pos in enumerate(pred['prediction'][top_pred_idx]):
                    x_agent, y_agent = pos[0], pos[1]
                    x_pixel = (x_agent - x_ego) * 10
                    y_pixel = - (y_agent - y_ego) * 10
                    row_pixel = int(250 + x_pixel)
                    column_pixel = int(400 + y_pixel)
                    pred_t[i, i+j] = x_pixel, y_pixel
                    ax.scatter(row_pixel, column_pixel, color='black', label='Predictions', s=25, zorder=3)
                    x_list.append(row_pixel), y_list.append(column_pixel)
                    if (j > 5): break
                ax.plot(x_list, y_list, color='limegreen', lw=3, zorder=2)
                ax.set_title('Prediction vs Real Trajectory in Agent Frame', fontsize=16)
                ax.tick_params(axis='both', which='both', length=0)
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
        ade_list = prediction_ade_in_time(pred_t)
        # append combined image to full scene list          
        fig_list.append(fig)
        plt.close()
    sample_token = nusc.get('sample', sample_token)['next']
    

for i in range(len(fig_list)):
        fig_list[i].savefig(f'plot_{i}.jpeg')
    


    
    
