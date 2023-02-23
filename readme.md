#  NHGNN-DTA
NHGNN-DTA: A Node-adaptive Hybrid Graph Neural Network for Interpretable Drug-target Binding Affinity Prediction

Code is being updated.



## File list

- Data：It contains the data files for training.
- Vocab: It contains the vocab files for protein and drug tokenizer.
- Code: It contains code files.



## Run code

Use the main_pretraining.py in the Code folder to pre-train model. 

You can customize the parameters by modifying the file, including the save path of the dataset, training super parameters and model weights.

```python
python main_pretraining.py
```

After pre-training, the main.py file is used for training. And the path of model parameter file needs to be setting.

```python
python main.py
```

## Cold drug/target/drug-target split
Use the split.py in the Code folder to split dataset in cold setting

The name of the dataset can be set to "davis" or "kiba" by "dataset_name"
The random seed can be set by "SEED"
```python
python split.py --dataset_name davis --SEED 42
```
Then you will get the training, validation and test data sets of the three cold start settings corresponding to the data set.


## Requirement

numpy~=1.20.3
rdkit~=2021.09.2
networkx~=2.6.3
pandas~=1.3.4
torch~=1.12.1
scikit-learn~=0.24.2
scipy~=1.7.1
tqdm~=4.62.3
