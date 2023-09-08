from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser


class CommaSeparatedListOutputParser(BaseOutputParser):
    def parse(self, text: str):
        return text.strip().split(", ")


template = """You are a helpful assistant who specializes in social marketing on LinkedIn.
A user will pass in a subject, and you should generate LinkedIn post no more than 250 characters long on the subject provided by the user.
ONLY return a generated post, and nothing more."""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(
    llm=ChatOpenAI(openai_api_key="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"),
    prompt=chat_prompt,
    # output_parser=CommaSeparatedListOutputParser()
)
answer = chain.run(input())

print(answer)


asbcs