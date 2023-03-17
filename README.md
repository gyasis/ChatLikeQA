# ChatLikeQA

## Overview

This repository provides a question and answer system based on uploaded documents. The system is designed to help users quickly find answers to their questions by searching through a large collection of documents.

## Requirement

- Python 3.x
- Flask
- openai
- chromadb
- langchain

To install langchain, you can run the command `pip install langchain`. For the other dependencies, you can install them using `pip install <dependency-name>`.

## Usage

To use the system, follow these steps:

1. Install the required dependencies (see `Requirements` section).
2. Clone the repository to your local machine.
3. Upload your documents to the `documents/` directory.
4. Start the Flask server by running `python app.py`.
5. Open your web browser and navigate to `http://localhost:5000`.
6. Enter your question into the search bar and hit enter.
7. The system will return the most relevant documents based on your query.

## Features

- Full-text search: the system uses Elasticsearch to enable full-text search across all uploaded documents.
- Question and answer system: the system uses a question and answer system to help users find answers to their questions.

## API instructions

To use the API, you will need to set up a license key. Follow these steps to set up the API license key:

1. Go to [OPENAI](https://platform.openai.com/docs/introduction) and create an account.
2. Generate a new API key and copy it.
3. Create a file in your project called `.env`.
4. Inside the file, define a variable to hold the API key (e.g. `API_KEY = your-api-key`).
5. In your main code file, reference the environment variable (e.g. `api_key = os.environ['API_KEY']`).

Once you have set up the license key, you can make a GET request to the API endpoint at `http://localhost:5000/api/search` with the following parameters:

Please note that you should never include your API key directly in your code or in any files that will be pushed to GitHub, as this can compromise the security of your application. Instead, always store the API key as an environment variable, as described above.

## Reference

This project is based on the information provided in the Medium article "[Chat with Document(s) using OpenAI ChatGPT API and Text Embedding]" by Sung Kim. You can read the article at [[web address on Medium](https://medium.com/dev-genius/chat-with-document-s-using-openai-chatgpt-api-and-text-embedding-6a0ce3dc8bc8)].

## Author

This project was created by Gyasi Sutton.

## License

This project is licensed under the terms of the MIT license. See `license.txt` for more information.
