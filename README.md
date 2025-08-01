
# Efficient Adaptive Testing via Gradient Path Matching




### Introduction

	Adaptive testing is widely adopted in AI-driven educational assessment systems (e.g., GRE), where the goal is to select an optimal subset of questions from a large question bank to accurately estimate an examinee's ability. A fundamental challenge is that : optimal question subsets are inherently personalized and solving for them is NP-hard. Recently, it has been framed as a gradient matching problem: aligning gradients between selected subsets and the full question set across the entire ability space. However, such global alignment is computationally expensive and difficult to scale. In this work, we propose GPM (Gradient Path Matching), a novel framework that instead aligns gradients along possible optimization paths toward the final estimate. By leveraging intermediate gradients as supervision, GPM learns an explicit and generalizable selection algorithm from large-scale data. 


### Requirements and Dependencies

Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```


### Dataset Preparation

1. **Format**: The dataset should be in JSON format, with each record containing input features and corresponding labels.
2. **Example**: A sample dataset is provided in `data/assist2009.json`.
3. **Configuration**: Update `utils/configuration.py` to specify the dataset path and other relevant settings.


### Configuration

The key configurations for model training are defined in `utils/configuration.py`. 

- **`--model`**: Specifies the model type.
- **`--lr`**: Learning rate for the model.
- **`--meta_lr`**: Learning rate for meta-parameters.
- **`--inner_lr`**: Learning rate for the inner optimization loop.
- **`--policy_lr`**: Learning rate for the policy network.
- **`--dataset`**: Dataset to use for training (e.g., 'assist2009', 'eedi-1', 'exam').
- **`--neptune`**: Flag to enable logging to Neptune for experiment tracking (optional).
To customize, edit the parameters directly in the file or use command-line arguments.

### Training

Run the following command to start training:

```bash
python train.py
```
