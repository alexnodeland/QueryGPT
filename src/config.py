import os

"""
This file contains all the configuration variables for the project.
"""

# OpenAI Configuration
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
QUERY_MODEL = "gpt-4"
QUERY_TEMPERATURE = 0.9
PARSER_MODEL = "gpt-3.5-turbo"
PARSER_TEMPERATURE = 0.0

# Prompt Configuration
EXAMPLE_PATH = "../data/example_query.csv"
QUERY_LANGUAGE = "Cypher"

# File Configuration
QUERIES_PATH = "../data/queries.csv"
OUTPUT_PATH = "../data/output.csv"

# Processing Configuration
CHUNK_SIZE = 2
