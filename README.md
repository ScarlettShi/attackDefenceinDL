

#Attack and Defense of Deep Learning Models in the Field of Web Attack Detection

https://github.com/ScarlettShi/attackDefenceinDL



## Table of Contents

- [datasets](#datasets)
- [config](#config)
- [stages](#stages)
- [training](#training)


## datasets
original datasets are available in the *origin/clean* folder. 
- **allnewv2**: Allnewv2 is an open internet dataset comprising a total of 84,603 parsed HTTP request texts.
- **online**: The online dataset consists of data sampled from the live network, which has undergone de-identification processing, totaling 118,718 entries.

The poisoning datasets are available in the *attack/poison* folder.
generate_poison_train_data.py:Generate poisoned datasets using different attack methods.
The defence datasets are available in the  *defence/fitu* folder.

## config

The configuration file  *config/apple.yaml*  is used to set the parameters for the training process, including the model, optimizer, learning rate, and other hyperparameters.
- **dataset**:The experimental dataset.
- **stage**:attack or defense.
- **model_network**:The model used in the experiment.
- **attack_mode**:The poisoning method used in the attack phase.
- **if_eda**:Whether to use EDA.when the stage is defence,if_eda is True,EDA will be used.
- **eda_rate**:The rate of EDA.
- **ft_area**:The fine-tuning area.in_area or out_area.
- **fitu_rate**: when the stage is defence, fitu_rate=1, means the defense training set size is 1% of the attack
training set size. 
- **beta**:The weighting coefficient for the loss function.
- **alpha**:Scaling factor that ensures both parts of the loss function are on the same magnitude.

Other basic parameters for model training:
- **input_len**:The maximum length of the input sequence.
- **hidden_size**:The hidden size of the model.
- **optimizer**:The optimizer used in the experiment.
- **lr**:The learning rate used in the experiment.
- **batch_size**:The batch size used in the experiment.
- **epochs**:The number of epochs used in the experiment.
- **device**:The device used in the experiment.
- **bert_type**:The type of BERT model, can be adjusted by downloading the desired BERT family model and modifying this configuration.
- **vocab_size**:The size of the vocabulary.

## stages
- **origin**:The clean data is used to train the model.
- **attack**:the poisoning data is used to train the model.
- **defense**:The defense method is used to train the model.

## training
- **generate_poison_train_data.py**:Generate poisoned datasets using different attack methods.
- **generate_poison.py**:Various poisoning functions have been defined.
- **create_model.py\bert_trainer**:The neural network architecture has been defined.
- **train.py**:The main training file.

If you have any questions or suggestions, please contact me,e-mail: ljshistat@163.com
