## Data Preparation
Following the detail descriptions [here](datasets/readme.md).
## Pre-Training and Downstream forecasting
The scripts for reproduction of ST-ReP and two simple baselines HL and Ridge are presented as follows.
### PEMS04
```python
python run.py --config_file PEMS04.json --modelid STReP --device cuda:0
python run.py --config_file PEMS04.json --modelid HL
python run.py --config_file PEMS04.json --modelid Ridge 
```
### PEMS08
```python
python run.py --config_file PEMS08.json --modelid STReP --device cuda:0
python run.py --config_file PEMS08.json --modelid HL
python run.py --config_file PEMS08.json --modelid Ridge 
```
### Temperature
```python
python run.py --config_file temperature2016.json --modelid STReP --device cuda:0
python run.py --config_file temperature2016.json --modelid HL
python run.py --config_file temperature2016.json --modelid Ridge 
```
### Humidity
```python
python run.py --config_file humidity2016.json --modelid STReP --device cuda:0
python run.py --config_file humidity2016.json --modelid HL
python run.py --config_file humidity2016.json --modelid Ridge 
```
### SDWPF
```python
python run.py --config_file SDWPF.json --modelid STReP --device cuda:0
python run.py --config_file SDWPF.json --modelid HL
python run.py --config_file SDWPF.json --modelid Ridge 
```
### CA
```python
python run.py --config_file ca.json --modelid STReP --device cuda:0
python run.py --config_file ca.json --modelid HL
python run.py --config_file ca.json --modelid Ridge 
```

Results will be saved in the `results` folder.