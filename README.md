# Auto Fidget Detection
Source codes for paper "Automatic Detection of Self-Adaptors for Psychological Distress"

This repo is free for academic use only.

If this work helps your research, pleaese cite the following paper:
(Preferred)
```
Lin, W., Orton, I., Liu, M., & Mahmoud, M. (2020, April). Automatic detection of self-adaptors for psychological distress. In 2020 15th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2020). IEEE.
```
or:
```
Lin, W., Orton, I., Li, Q., Pavarini, G., & Mahmoud, M. (2020). Looking At The Body: Automatic Analysis of Body Gestures and Self-Adaptors in Psychological Distress. arXiv preprint arXiv:2007.15815.
```
## Current Progress
1. Release the source code for using our pre-trained model to detect fidgeting has been released. [Done]
2. Release the code for labeling and training.
3. Release the dataset and benchmark

## Usage
### Guidance
#### 1. Download demo data
This demo video is part of Rhythmic Fidgeting Dataset used in [1].

Reference:
```
Mahmoud, M., Morency, L. P., & Robinson, P. (2013, December). Automatic multimodal descriptors of rhythmic body movement. In Proceedings of the 15th ACM on International conference on multimodal interaction (pp. 429-436). ACM.
```

#### 2. Run
```
python main.py
```

### Customizing
1. Change the video config in "utility/base_config.py"
2. Change hyperparameters in each file (still working to make it easier, but this is already a good configuration to start with)
3. Code to run openpose is given in "run_openpose.py", modify to fit your need.
