import os
import pandas as pd
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from config import MODEL, TEMPERATURE, EXAMPLE_PATH, QUERY_LANGUAGE, QUERIES_PATH, CHUNK_SIZE

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

# Build prompt
template = "As an AI assistant, your task is to optimize database queries. Given a {query_language} query, " + \
    "your job is to provide an optimized version of the query and any relevant notes about the optimization process. " + \
    "Please structure your response as a JSON object, following this format: " + \
    "{ 'query_optimized': 'Your optimized query here', 'note': 'Your notes about the optimization process here' }. " + \
    "Ensure your response is complete and does not include any extraneous comments or text. " + \
    "Avoid shortening or truncating your answer. Share the output in its complete form."

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
output = chain.run(query_language=QUERY_LANGUAGE, example_input=example_input, example_output=example_output, queries=queries_formatted, format_instructions=format_instructions)
print(output)