from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

PROMPT_TEMPLATE = (
    "As an AI assistant, your task is to optimize database queries. Given a {query_language} query, "
    "your job is to provide an optimized version of the query and any relevant notes about the optimization process. "
    "Ensure your response is complete and does not include any extraneous comments or text. "
    "Avoid shortening or truncating your answer. Share the output in its complete form."
    "\n\n{format_instructions}"
)

def buildPrompt(prompt_template=PROMPT_TEMPLATE):
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
