# SummaRAGizer

SummaRAGizer is a Python web application that leverages AI to summarize text documents. Utilizing the power of LangChain and OpenAI's GPT-4, it provides users with a concise and coherent summary of uploaded text documents.

## Objective

The goal of this application is to create a user-friendly interface where users can upload text documents and receive a summarized version. The application is designed to handle documents of various lengths and topics, ensuring a high-quality summary is generated for each submission.

## Features

- Python-based web application with an upload endpoint.
- Integration of LangChain and OpenAI's GPT-4 for AI-powered text summarization.
- Accepts `.txt` file formats for summarization.
- Error handling for file uploads, API limits, and other exceptions.

## Installation

You will need an [OpenAI API key](https://platform.openai.com/account/api-keys).
To set up and run SummaRAGizer on your local machine, follow these steps:

1. Clone the Repository:
```bash
git clone https://github.com/cmagganas/summaRAGizer.git
cd summaRAGizer
pip install -r requirements.txt
export OPENAI_API_KEY="sk-123..."
python app.py
```

2. Usage

    Once the application is running, navigate to http://localhost:8000/ in your web browser. You will be greeted with a simple interface for uploading your text documents.

3. Uploading Documents

    Click on the 'Upload' button.
    Select a .txt file from your device.
    Click 'Submit' to upload the file.
    The summarized content will be displayed on the screen.

## Demo Video

[Demo: Summarizing Uploaded Docs](https://www.loom.com/share/7f46586938764b72b17ff35b1b06b97c)