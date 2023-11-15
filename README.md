# Simple Chatbot Development from Website Data

## Introduction

This project focuses on developing a simple Chatbot from website data. The goal is to create a system that can answer questions based on information gathered from a website.

## Environment Setup

To set up the environment for this project, we use conda and Python 3.9. The project also requires the OpenAI and Streamlit packages. You can create a new conda environment from the requirements.txt file using the following commands:

```bash
conda create --name <env> --file requirements.txt
conda activate <env>
```

**NOTE**: Please remember to replace `<env>` with the name of the conda environment you want to use.

## How it works

### Set OpenAI API Key

First, you need to set your OpenAI API key in the `keys.py` file as follows:

```python
OPENAI_API_KEY = "your-api-key"
```

Replace "your-api-key" with your actual OpenAI API key.

### Get and save data from Website

Next, you can run the Python file named `embedding` with the `--url` argument followed by the URL of the website you want to extract information from by running this command:

```bash
python embedding --url <url>
```

Where `<url>` is the URL of the website you want to extract information from.

The data will be saved in the following directories:

- `text`: This directory stores the raw data.
- `processed`: This directory stores the embedded data in CSV format.

### Start the Chatbot

Finally, you can start the chatbot by running the following command:

```bash
streamlit run chatbot.py
```
