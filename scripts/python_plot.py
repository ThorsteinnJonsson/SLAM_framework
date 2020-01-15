import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt



def get_translation(filename):
  poses = np.loadtxt(filename)
  translation = poses[:,[3,7,11]]  
  return translation



if __name__ == "__main__":

  sequence = "00"
  # slam_translation = get_translation("results/" + sequence + ".txt")
  slam_translation = get_translation("tmp/positions.txt")
  gt_translation = get_translation("dataset/ground_truth_poses/poses/" + sequence + ".txt")
  
  mpl.rcParams['legend.fontsize'] = 10
  
  fig_topview = plt.figure()
  plt.plot(slam_translation[:,0], slam_translation[:,2], label="SLAM")
  plt.plot(gt_translation[:,0], gt_translation[:,2], label="GT") 
  plt.axis('equal')
  plt.legend()
  plt.show()
 
  
  # fig = plt.figure()
  # ax = fig.gca(projection='3d')
  # ax.plot(slam_translation[:,0], slam_translation[:,2], label="SLAM")
  # ax.plot(gt_translation[:,0], gt_translation[:,2], label="GT") 
  # # ax.view_init(elev=90, azim=0)
  # ax.axis('equal')
  # ax.legend()
  # plt.show()