# 3D Gaussian Splatting + Material Point Method (MPM)
Reimplementation of the [PhysGaussian](https://xpandora.github.io/PhysGaussian/) paper. Final project of the advanced computer graphics course at Brown. 

*Remark*: our codes are build upon [1].

## Setup
Please refer to the [PhysGaussian's repo](https://xpandora.github.io/PhysGaussian/) for environment setup instructions.

## Data
Some example Gaussian models are in the  `models` folder. You can train more examples by yourself, or get some pre-trained models from [here](https://github.com/graphdeco-inria/gaussian-splatting).

- If you are on Linux, please follow the [3D Gaussian Splatting's instructions](https://github.com/graphdeco-inria/gaussian-splatting) to install and run the viewer.

- If you are on Windows, modify the below command to view the scene/object
    ```
    .\viewers\bin\SIBR_gaussianViewer_app -m models\mic --iteration 3000
    ```

## Usage
To run the current version of this code, modify the below command
```
python main.py --config_path configs/ficus.json --output_path output/ficus_sand_gravity
```

## Method
MPM
- [Material point method after 25 years: Theory, implementation, and applications](https://www.sciencedirect.com/science/article/abs/pii/S0065215619300146) [2020]
- The material point method: a continuum-based particle method for extreme loading cases. X Zhang, Z Chen, Y Liu. [2016]
## Results

### Naive Translation
![naive-translation](output/mic/simulated.gif)

## Extra Features

## References
[1] Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4), 1-14.

[2] Xie, T., Zong, Z., Qiu, Y., Li, X., Feng, Y., Yang, Y., & Jiang, C. (2023). Physgaussian: Physics-integrated 3d gaussians for generative dynamics. CoRR abs/2311.12198 (2023).
