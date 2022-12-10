#  NHGNN-DTA

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

After pre-training, the main.py file is used for training. And the path of model parameter file in the file needs to be setting.

```python
python main.py
```



## Requirement

numpy~=1.20.3
rdkit~=2021.09.2
networkx~=2.6.3
pandas~=1.3.4
torch~=1.12.1
scikit-learn~=0.24.2
scipy~=1.7.1
tqdm~=4.62.3
