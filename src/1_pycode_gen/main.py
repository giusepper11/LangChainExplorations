from dotenv import load_dotenv
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()


llm = OpenAI(model_name="gpt-3.5-turbo-1106")

code_prompt = PromptTemplate(
    template="Write a very short {language} function that {task}.",
    input_variables=["language", "task"],
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
)


result = code_chain({
    "language": "python",
    "task": "prints 'Hello World!'"
})

print(result["text"])