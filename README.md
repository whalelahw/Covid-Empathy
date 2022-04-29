# Understanding Empathetic Shifts During the COVID-19 Pandemic in Online Mental Health Communities

This course project is built on the code of "A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support".

## Introduction

Empathy is an integral aspect of human connection, moreover shared experiences form this emotional capability. The global pandemic created a unity unique throughout history, a shared global experience which may be a happenstance of a millennia. This study builds upon an existing empathy framework aiming to analyze different forms of empathetic responses in online communities, when expressing support for individuals dealing with COVID-19 stressors. By training an empathy detector with domain adaptation and comparing the expressed empathy in COVID and non-COVID contexts in active Online Mental Health Communities (OMHCs), we aim to analyze the difference in empathy expression patterns between posts from COVID-related vs COVID-unrelated contexts as well as the community impact. By analyzing 16-month data from the Pushshift Reddit dataset, we found there are significantly higher levels of empathy expression via emotional reactions and interpretations in COVID contexts, while less are expressed via exploration. Additionally, this positive empathetic shift has contributed to greater prosociality reflected by positive feedback from both the support seekers and community members.


## Quickstart

### 1. Prerequisites

Our framework can be compiled on Python 3 environments. The modules used in our code can be installed using:
```
$ pip install -r requirements.txt
```


### 2. Prepare training dataset
A sample raw input data file is available in [dataset/sample_input_ER.csv](dataset/sample_input_ER.csv). This file (and other raw input files in the [dataset](dataset) folder) can be converted into a format that is recognized by the model using with following command:
```
$ python3 src/process_data.py --input_path dataset/sample_input_ER.csv --output_path dataset/sample_input_model_ER.csv
```

### 3. Training the detector model
For training our model on the sample input data, run the following command:
```
$ python3 src/train.py \
	--train_path=dataset/sample_input_model_ER.csv \
	--lr=2e-5 \
	--batch_size=32 \
	--lambda_EI=1.0 \
	--lambda_RE=0.5 \
	--save_model \
	--save_model_path=output/sample_ER.pth \
	--checkpoint_num 875
```

**Note:** You may need to create an `output` folder in the main directory before running this command.

### 4. Testing the detector model and detect empathy on our data
For testing our model on the sample test input, run the following command:
```
$ python3 src/test.py \
	--input_path dataset/sample_test_input.csv \
	--output_path dataset/sample_test_output.csv \
	--ER_model_path output/sample_ER.pth \
	--IP_model_path output/sample_IP.pth \
	--EX_model_path output/sample_EX.pth
```
This will save the prediction in a output file, you can further use [eval.py](eval.py) to get metrics like accuracy and F1.

### 5. Unsupervised Domain Adaptation
As we mentioned in the paper, we do unsupervised domain adaptation by mask language modeling on unsupervised data:
```
$ python3 src/uda.py
```
It will automaticly save the checkpoints, which can be used in step 3. All hyperparameters are specified in [src/uda.py](src/uda.py).


## Collected Data and Annotation

We provide one month of collected data in [data/input_21_03.csv](data/input_21_03.csv), the output of the empathy detector is [data/output_21_03.csv](data/output_21_03.csv).

We also provide the 450 annotated collected examples in [data/OurAnnotations.csv](data/OurAnnotations.csv).

## Training Arguments

The training script accepts the following arguments: 

Argument | Type | Default value | Description
---------|------|---------------|------------
lr | `float` | `2e-5` | learning rate
lambda_EI | `float` | `0.5` | weight of empathy identification loss 
lambda_RE |  `float` | `0.5` | weight of rationale extraction loss
dropout |  `float` | `0.1` | dropout
max_len | `int` | `64` | maximum sequence length
batch_size | `int` | `32` | batch size
epochs | `int` | `4` | number of epochs
seed_val | `int` | `12` | seed value
train_path | `str` | `""` | path to input training data
dev_path | `str` | `""` | path to input validation data
test_path | `str` | `""` | path to input test data
do_validation | `boolean` | `False` | If set True, compute results on the validation data
do_test | `boolean` | `False` | If set True, compute results on the test data
save_model | `boolean` | `False` | If set True, save the trained model  
save_model_path | `str` | `""` | path to save model

## Acknowledgement

The training data and detector code is from this paper

```bash
@inproceedings{sharma2020empathy,
    title={A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support},
    author={Sharma, Ashish and Miner, Adam S and Atkins, David C and Althoff, Tim},
    year={2020},
    booktitle={EMNLP}
}
```

