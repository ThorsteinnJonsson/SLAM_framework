import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt



def get_translation(filename):
  gt_poses = np.loadtxt(filename)
  gt_translation = gt_poses[:,[3,7,11]]  
  return gt_translation



if __name__ == "__main__":

  sequence = "05"
  # slam_translation = get_translation("results/" + sequence + ".txt")
  slam_translation = get_translation("tmp/positions.txt")
  gt_translation = get_translation("dataset/ground_truth_poses/poses/" + sequence + ".txt")

  mpl.rcParams['legend.fontsize'] = 10
  fig = plt.figure()
  plt.plot(slam_translation[:,0], slam_translation[:,2], label="SLAM")
  plt.plot(gt_translation[:,0], gt_translation[:,2], label="GT")
  plt.axis('equal')
  plt.legend()
  plt.show()