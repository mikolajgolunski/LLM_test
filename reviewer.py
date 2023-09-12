from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain


def evaluate(llm, educational_context, audience, topic, question):
    template = ("You are a harsh reviewer of educational free response questions. You will review a question in a following context:\n"
                f"1. The audience for the question should be: {audience}.\n"
                f"2. The question should be connected to the topic: {topic}.\n"
                f"3. The educational context should be: {educational_context}.\n"
                "4. The created question should consisting of two parts. First part should "
                "be an introduction and context for the topic of a question. Second part should be a question "
                "itself. The question should be highly relevant, connected to the introduction and go beyond just facts.\n\n"
                "Check if question relates to each part of the given context. If it does not regard it as a bad question. Work one step at a time.\n"
                "If you think the question is good and is of high standards say 'OK'. Otherwise say 'NO'. Argument your answer."
                ).format(audience=audience, educational_context=educational_context, topic=topic)

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(
        llm=llm,
        prompt=chat_prompt
    )
    answer = chain.run(question)

    return answer
