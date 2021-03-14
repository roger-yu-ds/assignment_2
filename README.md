# assignment_2

This project is assignment 2 of the MDSI (Master of Data Science and 
Innovation) degree, subject ADSI (Advanced Data Science for Innovation), at 
UTS. There are two parts to this project; 
1. Modelling: this constitutes cleaning the data, applying tranformations and 
fitting a model, including all the hyperparameter tuning.
   
2. Deployment: Creating a docker image and deploying the image to Heroku

## Modelling

All model artefacts are saved into the `/models` directory. Each of the 
experimental models will have a prefix, e.g. `2_pytorch`. The best 
performing model will be duplicated and named `model`. Accordingly, the other
artefacts that support `model` will be named `pipeline.sav` and 
`label_encoder.sav`.

## Deployment

The deployment is done using the `/Dockerfile` and the `heroku/yml`, which is 
sent to Heroku.

The files/directories required for production are:
* `/app`: which contains the code to run the FastAPI app
* `/models`:
  - `/model.torch`: the serialised neural network PyTorch model
  - `/label_encoder.sav`: used to convert the prediction (numerical) into the 
    beer style (string)
  - `/pipe.sav`: used to process the inputs into a format compatible with the 
    model
* `/Dockerfile`: the instructions to build the container
* `/requirements.txt`: the list of packages to be installed in the container
* `/heroku.yml`: the instructions for Heroku to run Docker 
* `/src`: contains the source code
    
## Steps to deploy to Heroku
1. Create a new app in the GUI, https://dashboard.heroku.com/new-app
1. Install the Heroku cli
1. Login from cli
1. Create a Heroku git remote `heroku git:remote -a app-name`
1. Set the stack type to be container type `heroku set:stack container`
1. Push to Heroku `git push heroku branch-name`


## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
