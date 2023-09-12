from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate


def create(llm, audience, topic, educational_context):
    template = ("You are an educational questions' creator. You will make a free response question.\n\n"
                f"Your audience will be: {audience}.\n"
                f"The educational context will be: {educational_context}.\n"
                f"The question should be connected to the following topic: {topic}.\n\n"
                "The created question should consisting of two parts. First part will "
                "be an introduction and context for the topic of a question. Second part will be a question "
                "itself. The question should be highly relevant, connected to the introduction you gave and go beyond just facts.\n\n"
                "At any point do not state the educational context you are working with."
                ).format(audience=audience, educational_context=educational_context, topic=topic)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    return conversation
