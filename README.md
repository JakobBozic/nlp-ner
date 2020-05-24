### Environment setup
The easiest way to run the code is to create a conda environment, all the requirements are listed in requirements.yaml

In order to train a model or evaluate a dataset (in not too long time) you will also need CUDA toolkit 10.1.

### Reproducing results from report
All the code that was used to train and evaluate models is in src directory.
Since training takes quite a while you can download some pretrained models from https://drive.google.com/open?id=1V1Mfn0jKHPEmuvbk9kCqjxAabZFjE_0Y .
In the drive folder, there are also training datasets, although you can generate them yourself, since it does not take too long.

Most of the src files contain some (absolute) paths, you will need to re-configure those.
The easiest way is to replace all occurrences of "/home/jakob/PycharmProjects/nlp-ner2" with the path to your repository. 
(in Pycharm press ctrl+shit+a, type replace and select Replace in Path)

##### Preparing data
Download zip from drive and extract data directory to root of your project.
The train/test/full datasets in proper format are already generated, if you wish to re-generate them run sentiCorefDs.py and ssj500kDs.py

##### Training models
Download zip from drive and extract models directory to root of your project.
In the provided directory, there are already trained best models for both datasets.
There are also weights for ELMO which you need if you want to use ELMO for training.
Once you have ELMO weights in correct directory (check elmo_slo.py) you can run train_model.py to train models.
Currently this will train (and save) all of the models, to train a particular one specify desired parameters at the bottom of train_model.py
Models are saved in directory specified in models.py

##### Evaluating models
All the models are evaluated after training and results are saved in same directory as model.
To re-evaluate a model, you can use eval_dataset.py, although its currently set-up for cross-evaluation and you will have to modify it a bit.
If you decied to re-evaluate models from google drive, you will first have to copy the contents from [model]/vocab/orig_labels.txt to [model]/vocab/labels.txt, 
and if you wish to then cross-evaluate the same model, you will have to do the opposite. 
(This is because datasets use different labels for words which are not named entities) 

#### Cross-evaluation
To evaluate a model trained on one dataset on the other dataset, use eval_dataset.py.
Set the flag on top of the file as you wish (optionally specify other model (model_dir = ...)) and simply run the file,
it will print results in console.

#### Using the model
If you wish to use the model on some text, use run_model.py
Specify the model and text (words only, no delimiters) and run the file.


