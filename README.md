# 3D Gaussian Splatting + Material Point Method (MPM)
Reimplementation of the [PhysGaussian](https://xpandora.github.io/PhysGaussian/) paper. Final project of the advanced computer graphics course at Brown. 

<!-- *Notes*:  -->
Our implementation is built upon [1].

We might clean the code a bit more and add more details for setup, usage, etc. in the future. 
<!-- 
## Setup
Please refer to the [PhysGaussian's repo](https://xpandora.github.io/PhysGaussian/) for environment setup instructions.

## Data
Some example 3D Gaussian models are in the  `models` folder. You can train more examples by yourself, or get some pre-trained models from [here](https://github.com/graphdeco-inria/gaussian-splatting).

- If you are on Linux, please follow the [3D Gaussian Splatting's instructions](https://github.com/graphdeco-inria/gaussian-splatting) to install and run the viewer.

- If you are on Windows, modify the below command to view the scene or object
    ```
    .\viewers\bin\SIBR_gaussianViewer_app -m models\mic --iteration 3000
    ```

## Usage
To run the current version of this code, modify the below command
```
python main.py --config_path configs/lego.json --output_path outputs/lego_debug 
```
To save the intermediate gaussian cloud and visualize the ellipsoids, 
```
python main.py --config_path configs/lego.json --output_path outputs/lego_debug --save_pcd --save_pcd_interval 10
``` -->

## Results

<!-- ### Elastic Lego -->
<!-- ![elastic-lego](outputs/lego_elastic/simulated.gif) -->
[Check here](https://drive.google.com/drive/folders/1KrnXDgvJyW1S_XX-lWqzcMKJfepvAn99?usp=drive_link)

[Presentation Slides](https://docs.google.com/presentation/d/1Q6cslEOO2gODz8dcMjzbe5Li7vIFzYUTRP-gWgDarpc/edit?usp=sharing)

<!-- ## Extra Features -->

## References
[1] Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 1-14.

[2] Xie, T., Zong, Z., Qiu, Y., Li, X., Feng, Y., Yang, Y., & Jiang, C. (2023). [Physgaussian](https://github.com/XPandora/PhysGaussian/): Physics-integrated 3d gaussians for generative dynamics. CoRR abs/2311.12198 (2023).

[3] [SIGGRAPH 2016 MPM Course](https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf)

[4] [MPM Implementation in Nvidia WARP](https://github.com/zeshunzong/warp-mpm)

### *Extra Feature References*

[5] Hu, Yuanming, et al. "Difftaichi: Differentiable programming for physical simulation." arXiv preprint arXiv:1910.00935 (2019).

[6] Li, Xuan, et al. "Pac-nerf: Physics augmented continuum neural radiance fields for geometry-agnostic system identification." arXiv preprint arXiv:2303.05512 (2023).

[7] Zhong, Licheng, et al. "Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians." arXiv preprint arXiv:2403.09434 (2024).
