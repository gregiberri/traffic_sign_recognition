# Traffic light recognition
### Requirements
The required packages can be found in *config/env_files/traffic_sign_recognition_env.yml*. 
Dependencies could be installed by running:
> conda env create -f config/env_files/traffic_sign_recognition_env.yml

### Configuration
The experiments are run according to configurations. The config files for those can be found in 
*config/config_files*.
Configurations can be based on each other. This way the code will use the parameters of the specified 
base config and only the newly specified parameters will be overwritten.
 
The base config file is *base.yaml*. A hpo example can be found in *base_hpo.yaml*
which is based on *base.yaml* and does hyperparameter optimization only on the specified parameters.
An example for test based on *base.yaml* can be found in *test.yaml*.

### Arguments
The code should be run with arguments: 

--id_tag specifies the name under the config where the results will be saved \
--config specifies the config name to use (eg. config "base" for *config/config_files/base.yaml*)\
--mode can be 'train', 'val', 'test' or 'hpo' 

### Required data
The required data's path should be specified inside the config file like:
> data: \
  &emsp; params: \
  &emsp; dataset_path: '/home/data/R-jpg' \

During train, val and hpo the files should be under their class subdirectory (eg. */home/data/R-jpg/stop_table*). \
During test the files should all be in the specified directory.  

### Saving and loading experiment
The save folder for the experiment outputs can be set in the config file like:
> id: "base"\
  env: \
  &emsp; result_dir: 'results'

All the experiment will be saved under the given results dir: {result_dir}/{config_id}/{id_tag arg}
1. tensorboard files
2. train and val metric csv
3. the best model
4. confusion matrices and by class metrics

If the result dir already exists and contains a model file then the experiment will automatically resume
(either resume the training or use the trained model for inference.)

### Usage
##### Training
To train the model use:
> python run.py --config base --mode train

#### Eval
For eval the  results dir ({result_dir}/{config_id}/{id_tag arg}) should contain a model as 
*model_best.pth.tar*. During eval the validation files will be inferenced and the metrics will be calculated.
> python run.py --config base --mode val

#### Test
For test the  results dir ({result_dir}/{config_id}/{id_tag arg}) should contain a model as 
*model_best.pth.tar*. During test the predictions will be saved along with the filepaths in a csv file.
> python run.py --config base_test --mode test

#### HPO
For hpo use:
> python run.py --config base_hpo --mode hpo