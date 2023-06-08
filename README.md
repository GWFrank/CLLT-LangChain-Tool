# CLLT-LangChain-Tool

## Introduction

This is the final project of [Computational Linguistics and Linguistic Theories(LING 8505)](https://lopentu.github.io/cllt2023/).

We created several [langchain](https://github.com/hwchase17/langchain) tools to retrieve information from a corpus of posts in PTT, a popular online forum in Taiwan, and crawl data from news websites.

## Data Source

The corpus of posts is provided by the lecturer of the course, and the news data is crawled from [PTS](https://news.pts.org.tw/) and [TTV](https://news.ttv.com.tw/category/%E6%94%BF%E6%B2%BB).

## Dependency Installation

We developed and test the tools with Python 3.9. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

Before importing our tools or using our interfaces, you should index the corpus first:

```python
python index.py
```

If you want to interact with the tools, you can use either the command line interface or the web interface. We recommend using the command line interface because it uses conversational agent, which is easier to use in a chat setting.

### Environment Variables

Before using both interfaces, you need to set the following environment variables in a `.env` file:

```plaintext
WEAVIATE_ADMIN_PASS=<your Weaviate admin password>
WEAVIATE_URL=<your Weaviate URL>
OPENAI_API_KEY=<your OpenAI API key>
```

### Command Line Interface

To use the command line interface, run the following command:

```bash
python cli_conversation.py
```

### Web Interface

To use the web interface, run the following command:

```bash
python web.py
```

A web page should be automatically opened in your default browser.
