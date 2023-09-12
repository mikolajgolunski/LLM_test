from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain


def evaluate(llm, educational_context, audience, user_input, question):
    template = ("You are a harsh reviewer. You will review a quality of a response given to the user in a following context:\n"
                f"1. The audience for the response should be: {audience}.\n"
                f"2. The educational context should be: {educational_context}.\n"
                f"3. The user's input is as follows: {user_input}"
                "4. The response should describe positive and negative values of a user's input and rate it on a regular US academic "
                "grading scale (A-F).\n\n"
                "Check if response relates to each part of the given context. If it does not, regard it as a bad question. Work one step at a time.\n"
                "Rate the quality of the response on a scale of 1-5 where 1 is the worst and 5 is the best. Provide only your score as a numerical without any other information. Remember to provide only a numerical score without anything else!"
                ).format(audience=audience, educational_context=educational_context, user_input=user_input)

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
