from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

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
