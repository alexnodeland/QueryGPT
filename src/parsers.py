from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chat_models import ChatOpenAI

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
    
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=parser_llm)
    print("\nFixing parser set up successfully")

    return parser, fixing_parser, format_instructions