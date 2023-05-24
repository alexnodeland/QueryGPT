import os
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from config import MODEL, TEMPERATURE, EXAMPLE_PATH, QUERY_LANGUAGE, QUERIES_PATH, CHUNK_SIZE

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

# Build prompt
template =  "You are a bot that optimizes {query_language} queries, improving given queries to be as efficient as possible. " + \
    "For each query, provide an optimized version and any relevant notes about the optimization process. " + \
    "The output should be stored in a CSV file with the following columns: 'query_optimized', and 'note'. " + \
    "Your output should include only the CSV file. It should not include any other comments or text. " + \
    "Do not shorten or concatenate your answer with a '...'. Share the output in its complete entirety."

system_message_prompt = SystemMessagePromptTemplate.from_template(template)

example_human = SystemMessagePromptTemplate.from_template("{example_input}", additional_kwargs={"name": "example_user"})
example_ai = SystemMessagePromptTemplate.from_template("{example_output}", additional_kwargs={"name": "example_assistant"})

human_template = "{queries}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])

# Load prompt example
example = pd.read_csv(EXAMPLE_PATH)
example_query = str(example['query'][0])
example_query_optimized = str(example['query_optimized'][0])
example_note = str(example['note'][0])

# Set prompt variables
example_input = '"' + example_query + '"'
example_output = '"' + example_query_optimized + '","' + example_note + '"'

# Load queries from CSV
queries_df = pd.read_csv(QUERIES_PATH)
queries = queries_df['query'].tolist()
files = queries_df['file'].tolist()


# Chunk queries into groups of CHUNK_SIZE
query_chunks = [queries[i:i + CHUNK_SIZE] for i in range(0, len(queries), CHUNK_SIZE)]
file_chunks = [files[i:i + CHUNK_SIZE] for i in range(0, len(files), CHUNK_SIZE)]

for query_chunk, file_chunk in zip(query_chunks, file_chunks):
    # Format queries as a line delimited list
    queries_formatted = "\n".join(query_chunk)
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    output = chain.run(query_language=QUERY_LANGUAGE, example_input=example_input, example_output=example_output, queries=queries_formatted)
    print(output)
