import json
from typing import List
from pydantic import BaseModel, Field
import pandas as pd
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from config import (
    QUERY_MODEL, 
    QUERY_TEMPERATURE, 
    PARSER_MODEL, 
    PARSER_TEMPERATURE, 
    EXAMPLE_PATH, 
    QUERY_LANGUAGE,
    PROMPT_TEMPLATE, 
    QUERIES_PATH, 
    CHUNK_SIZE, 
    OUTPUT_PATH, 
    OPENAI_API_KEY
)

def initializeLLMs(query_model, query_temperature, parser_model, parser_temperature):
    """
    Initialize the language models for query and parser.

    Parameters:
    query_model (str): The model to use for the query.
    query_temperature (float): The temperature to use for the query.
    parser_model (str): The model to use for the parser.
    parser_temperature (float): The temperature to use for the parser.

    Returns:
    tuple: A tuple containing the query and parser language models.
    """
    query_llm = ChatOpenAI(model=QUERY_MODEL, temperature=QUERY_TEMPERATURE)
#   parser_llm = ChatOpenAI(model=PARSER_MODEL, temperature=PARSER_TEMPERATURE)
    parser_llm = ChatOpenAI()
    return query_llm, parser_llm

# Define your desired data structure.
class OptimizedQuery(BaseModel):
    query_optimized: str = Field(description="optimized database query")
    note: str = Field(description="notes about optimization process")
class OptimizedQueries(BaseModel):
    queries: List[OptimizedQuery] = Field(description="list of optimized database queries")

def configureParsers(parser_llm):
    """
    Configure the parsers.

    Parameters:
    parser_llm (ChatOpenAI): The language model to use for the parser.

    Returns:
    tuple: A tuple containing the parser, fixing parser, and format instructions.
    """
    print("Setting up parser...")
    parser = PydanticOutputParser(pydantic_object=OptimizedQueries)

    format_instructions = parser.get_format_instructions()
    print("Parser set up successfully, format instructions: \n\n", format_instructions)
    
#   fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=parser_llm)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
    print("\nFixing parser set up successfully")

    return parser, fixing_parser, format_instructions

def buildPrompt(prompt_template):
    """
    Build the chat prompt.

    Parameters:
    template (str): The template for the system message prompt.
    example_input (str): The example input for the chat prompt.
    example_output (str): The example output for the chat prompt.
    queries (str): The queries for the human message prompt.

    Returns:
    ChatPromptTemplate: The chat prompt template.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
    example_human = SystemMessagePromptTemplate.from_template("{example_input}", additional_kwargs={"name": "example_user"})
    example_ai = SystemMessagePromptTemplate.from_template("{example_output}", additional_kwargs={"name": "example_assistant"})
    human_message_prompt = HumanMessagePromptTemplate.from_template("{queries}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
    return chat_prompt

def loadExample(example_path):
    """
    Load the example from a CSV file.

    Parameters:
    example_path (str): The path to the CSV file.

    Returns:
    tuple: A tuple containing the example input and output.
    """
    """
    # Load prompt example
    print("\nLoading prompt example...")
    example = pd.read_csv(example_path)

    # Set prompt variables
    print("Setting prompt variables...")
    example_input = '\n\n'.join(example['query'].apply(str).tolist())
    example_output = example[['query_optimized', 'note']].to_json(orient='records')
    """

    print("\nLoading prompt example...")
    example = pd.read_csv(example_path)
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

    return example_input, example_output

def loadQueries(queries_path):
    """
    Load the queries from a CSV file.

    Parameters:
    queries_path (str): The path to the CSV file.

    Returns:
    tuple: A tuple containing the query chunks and file chunks.
    """
    # Load queries from CSV
    print("Loading queries...")
    queries_df = pd.read_csv(queries_path)
    queries = queries_df['query'].tolist()
    files = queries_df['file'].tolist()

    # Chunk queries into groups of CHUNK_SIZE
    print("Chunking queries...")
    query_chunks = [queries[i:i + CHUNK_SIZE] for i in range(0, len(queries), CHUNK_SIZE)]
    file_chunks = [files[i:i + CHUNK_SIZE] for i in range(0, len(files), CHUNK_SIZE)]
    
    return query_chunks, file_chunks

def process_query_chunk(query_llm, chat_prompt, query_chunk, format_instructions, example_input, example_output):
    """
    Process a chunk of queries.

    Parameters:
    query_llm (ChatOpenAI): The language model to use for the query.
    chat_prompt (ChatPromptTemplate): The chat prompt template.
    query_chunk (list): The chunk of queries to process.
    format_instructions (str): The format instructions for the parser.
    example_input (str): The example input for the chat prompt.
    example_output (str): The example output for the chat prompt.

    Returns:
    str: The output from the language model.
    """
    queries_formatted = "\n\n".join(query_chunk)
    chain = LLMChain(llm=query_llm, prompt=chat_prompt)
    output = chain.run(query_language=QUERY_LANGUAGE, format_instructions=format_instructions, example_input=example_input, example_output=example_output, queries=queries_formatted)
    return output

def parse_output(output, parser, fixing_parser):
    """
    Parse the output from the language model.

    Parameters:
    output (str): The output from the language model.
    parser (PydanticOutputParser): The parser.
    fixing_parser (OutputFixingParser): The fixing parser.

    Returns:
    OptimizedQueries: The parsed output.
    """
    try:
        output_parsed = parser.parse(output)
    except Exception as e:
        output_parsed = fixing_parser.parse(output)
    return output_parsed

def append_to_output_df(output_df, query_chunk, file_chunk, output_parsed):
    """
    Append the parsed output to the output DataFrame.

    Parameters:
    output_df (pd.DataFrame): The output DataFrame.
    query_chunk (list): The chunk of queries.
    file_chunk (list): The chunk of files.
    output_parsed (OptimizedQueries): The parsed output.

    Returns:
    pd.DataFrame: The updated output DataFrame.
    """
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
    return output_df

def process_all_query_chunks(query_llm, chat_prompt, query_chunks, file_chunks, format_instructions, example_input, example_output, parser, fixing_parser):
    """
    Process all query chunks.

    Parameters:
    query_llm (ChatOpenAI): The language model to use for the query.
    chat_prompt (ChatPromptTemplate): The chat prompt template.
    query_chunks (list): The list of query chunks.
    file_chunks (list): The list of file chunks.
    format_instructions (str): The format instructions for the parser.
    example_input (str): The example input for the chat prompt.
    example_output (str): The example output for the chat prompt.
    parser (PydanticOutputParser): The parser.
    fixing_parser (OutputFixingParser): The fixing parser.

    Returns:
    pd.DataFrame: The output DataFrame.
    """
    output_df = pd.DataFrame(columns=["query", "file", "query_optimized", "note"])
    for query_chunk, file_chunk in zip(query_chunks, file_chunks):
        output = process_query_chunk(query_llm, chat_prompt, query_chunk, format_instructions, example_input, example_output)
        output_parsed = parse_output(output, parser, fixing_parser)
        output_df = append_to_output_df(output_df, query_chunk, file_chunk, output_parsed)
    return output_df

def save_output(output_df, output_path):
    """
    Save the output DataFrame to a CSV file.

    Parameters:
    output_df (pd.DataFrame): The output DataFrame.
    output_path (str): The path to the CSV file.
    """
    output_df.to_csv(output_path, index=False)

def main():
    print("Starting script...")
    query_llm, parser_llm = initializeLLMs(QUERY_MODEL, QUERY_TEMPERATURE, PARSER_MODEL, PARSER_TEMPERATURE)
    print("Loaded model successfully")

    parser, fixing_parser, format_instructions = configureParsers(parser_llm)

    example_input, example_output = loadExample(EXAMPLE_PATH)

    query_chunks, file_chunks = loadQueries(QUERIES_PATH)

    chat_prompt = buildPrompt(PROMPT_TEMPLATE)

    output_df = process_all_query_chunks(query_llm, chat_prompt, query_chunks, file_chunks, format_instructions, example_input, example_output, parser, fixing_parser)

    save_output(output_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()