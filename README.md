# CS510 Project Advanced News Search Engine

## Introduction

This is a webpage application where you can search news of your interests with easy to use keyword search and simple filters.

There are two parts of the project: 1) the backend automated tools to automatically classify the category of the news given the headlines, and automated tools to extract location given news url. 2) the frontend application to search news in the news database.

## Dataset

The demo frontend has two news dataset for use. [news_dataset.json](news_dataset.json) provides a large news dataset with over 148k news articles with category information but no location. [news_dataset_short_loc.json](news_dataset_short_loc.json) provides a relatively small dataset with 1k news articles and with category and location information retrieved in advance.

We will use the short dataset in the demo. To change dataset, go to [flask/app/bm25.py](flask/app/bm25.py) and change `file_path` in the first line of the function.

## How to use the application

### 1) Frontend application

You can deploy our app locally using flask. You should have Python3 installed in your computer. Here is the procedure:

#### Step 1: Create virtual environment

```
cd flask
python -m venv .venv
```

#### Step 2: Activate virtual environment

Linux/MacOS:

```
source .venv/bin/activate 
```

Windows cmd

```
.venv\Scripts\activate.bat
```

Windows PowerShell

```
.venv\Scripts\Activate.ps1
```

#### Step 3: Install required packages

```
pip install flask numpy tqdm rank_bm25 textblob
```

#### Step 4: Run the application

```
python -m flask run
```

When you see prompt with `Running on http://127.0.0.1:5000`, open your browser and visit the link `http://127.0.0.1:5000`. You may now be able to use the application!

### 2) Backend Automated Tools

See [tools/demo.ipynb](tools/demo.ipynb) for interactions with automated tools. You need to have the following packages installed:

```
pytorch
pytorch_pretrained_bert
nltk
bs4
```

You also need to download the model before interacting with category classification tools. You may download the model from [here](https://drive.google.com/file/d/1iMcSRt5wRJfl6eVxUh4bsa741G7XF6Qi/view?usp=sharing). 

After you download the model, change the `model_path` in [tools/utils.py](tools/utils.py) to the file you have downloaded.