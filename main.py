import os
import pandas as pd
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from typing import List
import json
from config import QUERY_MODEL, QUERY_TEMPERATURE, EXAMPLE_PATH, QUERY_LANGUAGE, QUERIES_PATH, CHUNK_SIZE, OUTPUT_PATH, OPENAI_API_KEY

print("Starting script...")
chat = ChatOpenAI(model=QUERY_MODEL, temperature=QUERY_TEMPERATURE)
print("Loaded model successfully")
# Define your desired data structure.
class OptimizedQuery(BaseModel):
    query_optimized: str = Field(description="optimized database query")
    note: str = Field(description="notes about optimization process")
class OptimizedQueries(BaseModel):
    queries: List[OptimizedQuery] = Field(description="list of optimized database queries")
# Set up a parser + inject instructions into the prompt template.
print("Setting up parser...")
parser = PydanticOutputParser(pydantic_object=OptimizedQueries)
format_instructions = parser.get_format_instructions()
print("Parser set up successfully, format instructions: \n\n", format_instructions)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
print("\nFixing parser set up successfully")

# Build prompt
template = (
    "As an AI assistant, your task is to optimize database queries. Given a {query_language} query, "
    "your job is to provide an optimized version of the query and any relevant notes about the optimization process. "
    "Ensure your response is complete and does not include any extraneous comments or text. "
    "Avoid shortening or truncating your answer. Share the output in its complete form."
    "\n\n{format_instructions}"
)
print("Building prompt...")
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

example_human = SystemMessagePromptTemplate.from_template("{example_input}", additional_kwargs={"name": "example_user"})
example_ai = SystemMessagePromptTemplate.from_template("{example_output}", additional_kwargs={"name": "example_assistant"})

human_template = "{queries}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
print("Prompt built successfully, prompt: \n\n", chat_prompt)
# Load prompt example
print("\nLoading prompt example...")
example = pd.read_csv(EXAMPLE_PATH)
example_query_a = str(example['query'][0])
example_query_optimized_a = str(example['query_optimized'][0])
example_note_a = str(example['note'][0])
example_query_b = str(example['query'][0])
example_query_optimized_b = str(example['query_optimized'][0])
example_note_b = str(example['note'][0])

# Set prompt variables
print("Setting prompt variables...")
example_input = '"'+ example_query_a +'"\n\n' + \
    '"'+ example_query_b +'"'
example_output = json.dumps({
    "queries": [
        {"query_optimized": example_query_optimized_a, "note": example_note_a},
        {"query_optimized": example_query_optimized_b, "note": example_note_b}
    ]
})


# Load queries from CSV
print("Loading queries...")
queries_df = pd.read_csv(QUERIES_PATH)
queries = queries_df['query'].tolist()
files = queries_df['file'].tolist()

# Create an empty DataFrame to store the results
output_df = pd.DataFrame(columns=["query", "file", "query_optimized", "note"])

# Chunk queries into groups of CHUNK_SIZE
print("Chunking queries...")
query_chunks = [queries[i:i + CHUNK_SIZE] for i in range(0, len(queries), CHUNK_SIZE)]
file_chunks = [files[i:i + CHUNK_SIZE] for i in range(0, len(files), CHUNK_SIZE)]

for query_chunk, file_chunk in zip(query_chunks, file_chunks):
    # Format queries as a line delimited list
    queries_formatted = "\n\n".join(query_chunk)
    print("Formatted queries successfully: \n\n", queries_formatted)
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    print("\nCreated chain successfully, running queries...")
    output = chain.run(query_language=QUERY_LANGUAGE, format_instructions=format_instructions, example_input=example_input, example_output=example_output, queries=queries_formatted)
    print("Ran queries successfully, parsing output...\n")
    try:
        output_parsed = parser.parse(output)
        print("\nParsed output successfully")
    except Exception as e:
        print(e)
        output_parsed = fixing_parser.parse(output)
        print("\nParsed output with fixer")
    print('\n', output_parsed, '\n')
    # For each result in the output, append a new row to the DataFrame
    for query, file, output_parsed in zip(query_chunk, file_chunk, output_parsed.queries):
        query_optimized = output_parsed.query_optimized
        note = output_parsed.note
        new_row = pd.DataFrame({
            "query": [query],
            "file": [file],
            "query_optimized": [query_optimized],
            "note": [note]
        })
        output_df = pd.concat([output_df, new_row], ignore_index=True)


# Save the DataFrame to a CSV file
output_df.to_csv(OUTPUT_PATH, index=False)