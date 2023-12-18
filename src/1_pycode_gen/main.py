# Import necessary libraries
import argparse
from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Load environment variables from .env file
load_dotenv()

# Initialize argument parser
parser = argparse.ArgumentParser()

# Add arguments for task and language with default values
parser.add_argument("--task", type=str, default="return a list of numbers")
parser.add_argument("--language", type=str, default="python")

# Parse the arguments
args = parser.parse_args()

# Initialize OpenAI with a specific model
llm = OpenAI(model_name="gpt-3.5-turbo-1106")

# Define a code prompt template
code_prompt = PromptTemplate(
    template="Write a very short {language} function that {task}.",
    input_variables=["language", "task"],
)

# Define a test prompt template
test_prompt = PromptTemplate(
    template="Write a test for the following {language} function:\n{code}",
    input_variables=["language", "code"],
)

# Define a chain for generating code
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code",
)

# Define a chain for generating test
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test",
)

# Define a sequential chain that includes both code and test chains
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"],
)

# Generate code using the code chain and print it
result = chain(
    {
        "language": args.language,
        "task": args.task,
    }
)

print(result["code"])
print("------------------")
print(result["test"])
