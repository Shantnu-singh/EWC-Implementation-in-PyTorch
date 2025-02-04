# EWC Implementation in PyTorch

## Overview
Elastic Weight Consolidation (EWC) is a continual learning technique that **mitigates catastrophic forgetting** when training deep learning models on sequential tasks. This repository provides a PyTorch implementation of EWC, along with results demonstrating its effectiveness in preserving knowledge while learning new tasks.

## Approach
Our implementation of EWC follows the original paper by [Kirkpatrick et al. (2017)]((https://www.pnas.org/content/114/13/3521)
). The key idea is to apply a quadratic regularization term that prevents significant updates to the parameters that are crucial for previously learned tasks. The importance of each parameter is estimated using the Fisher Information Matrix (FIM). This allows the model to retain knowledge from past tasks while adapting to new ones.

### **Steps in Our Implementation:**
1. **Train the model on the first task** and save the learned weights.
2. **Compute the Fisher Information Matrix (FIM)** to estimate parameter importance.
3. **Train the model on a new task**, adding a regularization term that constrains changes to important parameters.
4. **Evaluate the performance** on both old and new tasks to measure knowledge retention and adaptability.

## Installation
To use this repository, clone it and install the required dependencies:

```bash
git clone https://github.com/your-username/EWC-PyTorch.git
cd EWC-PyTorch
pip install -r requirements.txt
```

## Usage
Run the main training script with:
```bash
python train.py --taskA <dataset_A> --taskB <dataset_B>
```



## Results
Results with EWC demonstrate that the model retains performance on Task A while successfully learning Task B. The trade-off between stability and plasticity is controlled by the `lambda` parameter.

For detailed results, check the Jupyter Notebook: [Results with EWC.ipynb](./Results%20with%20EWC.ipynb).

## References
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). *Overcoming catastrophic forgetting in neural networks.* Proceedings of the National Academy of Sciences, 114(13), 3521-3526. [Link](https://www.pnas.org/content/114/13/3521)

## Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests.
