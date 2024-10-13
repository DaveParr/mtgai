mtgai
==============================

MTGAI is a collection of experiments with Magic: The Gathering data and AI tools.

[streamlit-app-2024-10-12-15-10-18.webm](https://github.com/user-attachments/assets/a3f963fc-cab0-460e-9faf-a52c38de30c2)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── vectorstore    <- Directory for vector store data.
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
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── app            <- Directory for the application code.
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_raw.py
    │   │   ├── make_procesed.py 
    │   │   └── make_vectorstore.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── .env            <- Environment configuration file THAT YOU MUST MAKE MAUNALLY
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


## Make the .env

You need to start a new file in the root and write the relevant settings for the project.
It looks like this:

```bash
OPENAI_API_KEY=sk-...
LANGCHAIN_PROJECT="mtgai"
LANGCHAIN_API_KEY='ls__...'
LANGCHAIN_TRACING_V2=True
```

`OPENAI_API_KEY` is the key for the OpenAI API. You can get it from the OpenAI website. It's likely that you should use a `sk-proj` "project" key. If you want to use a different key (such as a key linked to an "organisation" starting with `sk-org`), this should work, though you may also need to have set the `OPENAI_ORG_ID` environment variable.

The `LANGCHAIN_*` variables are for the Langchain Langsmith observability platform, and are optional. You can get them from the Langchain website. The `LANGCHAIN_PROJECT` is the project name, and the `LANGCHAIN_API_KEY` is the API key. The `LANGCHAIN_TRACING_V2` is a boolean that determines whether to use the new tracing system.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
