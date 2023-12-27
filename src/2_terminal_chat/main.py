# Import necessary modules and classes from the langchain library and others
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
    ConversationSummaryMemory,
)
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Create an instance of the ChatOpenAI class, specifying the model and verbosity
chat = ChatOpenAI(model_name="gpt-3.5-turbo", verbose=True)

# Set up a conversation memory mechanism, which can be useful for maintaining context
# Uncomment the 'chat_memory' line to use a file-based chat history
memory = ConversationSummaryMemory(
    memory_key="messages",
    return_messages=True,
    # chat_memory=FileChatMessageHistory("src/2_terminal_chat/messages.json"),
    llm=chat,
)

# Define a prompt template for the chatbot, incorporating the conversation history and user input
prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

# Create a LangChain (LLMChain) with the specified language model, prompt, and memory
chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
)

# Main loop to continuously take user input and generate responses
while True:
    content = input(">> ")
    result = chain({"content": content})

    print(result["text"])
