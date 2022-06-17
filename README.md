# UV-SLAM
Official page of *"UV-SLAM: Unconstrained Line-based SLAM Using Vanishing Points for Structural Mapping"*, which is published in IEEE RA-L with ICRA'22 presentation option.
For more details, please refer to: https://doi.org/10.1109/LRA.2022.3140816

[![journal](https://img.shields.io/badge/RA_L-9672726-4b44ce.svg)](https://ieeexplore.ieee.org/abstract/document/9672726)
[![arxiv](https://img.shields.io/badge/arXiv-2112.13515-B31B1B.svg)](https://arxiv.org/abs/2112.13515)
[![video](https://img.shields.io/badge/YouTube-B31B1B.svg)](https://youtu.be/jyUphjBxAnM)


<center>

| [ALVIO](https://link.springer.com/chapter/10.1007/978-981-16-4803-8_19)  | UV-SLAM |
| :---: | :---: |
| <img src="https://user-images.githubusercontent.com/42729711/143393636-48d80da1-189f-4ea4-9860-eb1914eddafa.png">  | <img src="https://user-images.githubusercontent.com/42729711/143393647-ec49dab0-b2e0-4c77-831a-03a819125a7f.png">  |

</center>

## Results
Mapping results for *MH_05_difficult* in the [EuRoC datasets](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
All mapping results for the EuRoC datasets is available in [here](https://github.com/url-kaist/UV-SLAM/tree/main/supplement/mapping_result.pdf)

<center>

| [ALVIO](https://link.springer.com/chapter/10.1007/978-981-16-4803-8_19) | [Previous work](https://ieeexplore.ieee.org/document/9560911) | UV-SLAM |
| :---: | :---: | :---: |
| <img src="https://user-images.githubusercontent.com/42729711/143398005-afce16e2-c3dc-4c3b-af6e-adade9a45d56.png">  | <img src="https://user-images.githubusercontent.com/42729711/143397993-edb67494-b00c-47e1-8591-532cf0c4cc46.png">  |  <img src="https://user-images.githubusercontent.com/42729711/143398028-9cf349f8-510e-4709-9859-4ff752b47f13.png">  |

</center>

## Installation
### 1. Prerequisites
-  Ubuntu 18.04
- [ROS melodic](http://wiki.ros.org/ROS/Installation)
- OpenCV 3.2.0 (under 3.4.1)
- [Ceres Solver](http://ceres-solver.org/)

### 2. Build
```
cd ~/<your_workspace>/src
git clone --recursive https://github.com/url-kaist/UV-SLAM.git
cd ..
catkin_make
source ~/<your_workspace>/devel/setup.bash
```
### 3. Trouble shooting
- If you have installed [VINS-mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) before, remove the common packages.
Example: ```benchmark_publisher```, ```camera_model```, etc

## Run on EuRoC datasets
Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).
Open three terminals and launch the vins_estimator, rviz, and play the bag file, respectively.
```
roslaunch uv_slam euroc.launch
roslaunch uv_slam vins_rviz.launch
rosbag play <dataset>
```

## Citation
If you use the algorithm in an academic context, please cite the following publication:
```
@article{lim2022uv,
  title={UV-SLAM: Unconstrained Line-based SLAM Using Vanishing Points for Structural Mapping},
  author={Lim, Hyunjun and Jeon, Jinwoo and Myung, Hyun},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  publisher={IEEE},
  volume={7},
  number={2},
  pages={1518-1525},
  doi={10.1109/LRA.2022.3140816}
}
```
```
@inproceedings{lim2021avoiding,
  title={Avoiding Degeneracy for Monocular Visual {SLAM} with Point and Line Features},
  author={Lim, Hyunjun and Kim, Yeeun and Jung, Kwangik and Hu, Sumin and Myung, Hyun},
  booktitle={Proc. IEEE International Conference on Robotics and Automation (ICRA)},
  pages={11675--11681},
  year={2021}
}
```

## Acknowledgements
We use [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) as our baseline code. Thanks Dr. Qin Tong, Prof. Shen etc very much. For line feature extraction, we use [ELSED](https://github.com/iago-suarez/ELSED). For vanishing point extraction, we use [J-linkage](http://www.diegm.uniud.it/fusiello/demo/jlk/) and [2-line exhaustive searching method](https://github.com/xiaohulugo/VanishingPointDetection).
This work was financially supported in part by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-00230, development of real·virtual environmental analysis based adaptive interaction technology) and in part by the Defense Challengeable Future Technology Program of Agency for Defense Development, Republic of Korea. The students are supported by Korea Ministry of Land, Infrastructure and Transport (MOLIT) as "Innovative Talent Education Program for Smart City" and BK21 FOUR.


## Licence
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.
We are still working on improving the code reliability.
For any technical issues, please contact Hyunjun Lim (tp02134@kaist.ac.kr).
