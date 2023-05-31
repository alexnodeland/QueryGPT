import os

"""
This file contains all the configuration variables for the project.
"""

# OpenAI Configuration
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4"
TEMPERATURE = 0.9

# Prompt Configuration
EXAMPLE_PATH = "example_query.csv"
QUERY_LANGUAGE = "Cypher"

# File Configuration
QUERIES_PATH = "queries.csv"
OUTPUT_PATH = "output.csv"

# Processing Configuration
CHUNK_SIZE = 2