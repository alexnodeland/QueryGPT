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
from pydantic import BaseModel, Field, validator
from config import MODEL, TEMPERATURE, EXAMPLE_PATH, QUERY_LANGUAGE, QUERIES_PATH, CHUNK_SIZE

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

# Define your desired data structure.
class OptimizedQueries(BaseModel):
    query_optimized: str = Field(description="optimized database query")
    note: str = Field(description="notes about optimization process")
# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=OptimizedQueries)
format_instructions = parser.get_format_instructions()
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())

# Build prompt
template = (
    "As an AI assistant, your task is to optimize database queries. Given a {query_language} query, "
    "your job is to provide an optimized version of the query and any relevant notes about the optimization process. "
    "Ensure your response is complete and does not include any extraneous comments or text. "
    "Avoid shortening or truncating your answer. Share the output in its complete form."
    "\n\n{format_instructions}"
)

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
    output = chain.run(query_language=QUERY_LANGUAGE, format_instructions=format_instructions, example_input=example_input, example_output=example_output, queries=queries_formatted)
    try:
        output_parsed = parser.parse(output)
    except Exception as e:
        print(e)
        output_parsed = fixing_parser.parse(output)
    print(output_parsed)