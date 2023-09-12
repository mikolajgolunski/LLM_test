from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain


def create(llm, educational_context, audience, topic, question):
    template = ("You are a friendly but fair reviewer of responses to the educational free response questions. "
                "You will review an answer to the question in a following context:\n"
                f"1. The audience is: {audience}.\n"
                f"2. The answer should be connected to the topic: {topic}.\n"
                f"3. The educational context should be: {educational_context}.\n"
                f"4. The question is as follows: '{question}'\n\n"
                "Work one step at a time.\n"
                "You should describe positive and negative values of an answer and rate it on a regular US academic "
                "grading scale (A-F). After giving feedback you should end a conversation without proposing any "
                "additional assistance or asking anything more.\n"
                "At any point do not state the educational context you are working with.\n"
                "At first, state the question without any changes, than wait for a response."
                ).format(audience=audience, educational_context=educational_context, topic=topic, question=question)

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt
    )

    return chain
