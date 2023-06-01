import json
import pandas as pd
from langchain import LLMChain

def loadExample(example_path):
    """
    Load the example from a CSV file.

    Parameters:
    example_path (str): The path to the CSV file.

    Returns:
    tuple: A tuple containing the example input and output. 
    """

    print("\nLoading prompt example...")
    example = pd.read_csv(example_path)
#   example_input = '\n\n'.join(example['query'].apply(str).tolist())
#   example_output = example[['query_optimized', 'note']].to_json(orient='records')
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

def loadQueries(queries_path, chunk_size):
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
    query_chunks = [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size)]
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    
    return query_chunks, file_chunks

def process_query_chunk(query_llm, query_language, chat_prompt, query_chunk, format_instructions, example_input, example_output):
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
    output = chain.run(query_language=query_language, format_instructions=format_instructions, example_input=example_input, example_output=example_output, queries=queries_formatted)
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

def process_all_query_chunks(query_llm, query_language, chat_prompt, query_chunks, file_chunks, format_instructions, example_input, example_output, parser, fixing_parser):
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
        output = process_query_chunk(query_llm, query_language, chat_prompt, query_chunk, format_instructions, example_input, example_output)
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