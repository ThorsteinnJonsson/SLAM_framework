# SLAM framework

SLAM framework based on [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).
Works with monocular, stereo and RGB-D cameras.

This project was written for fun as a way to self-study about SLAM. Much of it comes from ORB-SLAM and is still a work in progress. 


## Building
Building the project can be done with 
```bash
sh scripts/build.sh Release
```

For development it is generally recommended to use 
```bash
sh scripts/build.sh RelWithDebInfo
```

## Examples
There are currently two examples implemented with the KITTI dataset, one for monocular SLAM and the other for stereo SLAM. Both cpp-files can be found in the examples folder.

Running the examples can be done by executing the relevant binary in the build folder and passing in the config file and path to the kitti dataset sequence (replace XX with the number of the sequence) as follows.

```bash
./build/main_stereo config/kitti_config_stereo.json path-to-kitti/sequences/XX
```

VScode build tasks and launch configurations are also included in the .vscode folder.

## Dependencies
- Eigen3
- OpenCV
- [g2o](https://github.com/RainerKuemmerle/g2o) (included in third_party folder)
- [DBoW2](https://github.com/dorian3d/DBoW2) (included in third_party folder)
- [json](https://github.com/nlohmann/json) ((included in third_party folder)
