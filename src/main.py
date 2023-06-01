from langchain.chat_models import ChatOpenAI

from query_processing import loadExample, loadQueries, process_all_query_chunks, save_output
from prompt import buildPrompt
from parsers import configureParsers
from config import (
    QUERY_MODEL, 
    QUERY_TEMPERATURE, 
    PARSER_MODEL, 
    PARSER_TEMPERATURE, 
    EXAMPLE_PATH, 
    QUERY_LANGUAGE, 
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

def main():
    print("Starting script...")
    query_llm, parser_llm = initializeLLMs(QUERY_MODEL, QUERY_TEMPERATURE, PARSER_MODEL, PARSER_TEMPERATURE)
    print("Loaded model successfully")

    parser, fixing_parser, format_instructions = configureParsers(parser_llm)

    example_input, example_output = loadExample(EXAMPLE_PATH)

    query_chunks, file_chunks = loadQueries(QUERIES_PATH, CHUNK_SIZE)

    chat_prompt = buildPrompt()

    output_df = process_all_query_chunks(query_llm, QUERY_LANGUAGE, chat_prompt, query_chunks, file_chunks, format_instructions, example_input, example_output, parser, fixing_parser)

    save_output(output_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()