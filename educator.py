from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate


def create(llm, educational_context, audience, topic):
    templateEducator = ("You are an educational questions' creator. You will make a free response question.\n\n"
                        f"Your audience will be: {audience}.\n"
                        f"The question should be connected to the following topic: {topic}.\n"
                        f"The educational context will be: {educational_context}.\n\n"
                        "The created question should consisting of two parts. First part will "
                        "be an introduction and context for the topic of a question. Second part will be a question "
                        "itself. The question should be highly relevant, connected to the introduction you gave and go beyond just facts.\n\n"
                        "After getting an answer you will evaluate it based on the audience and the educational context "
                        "provided earlier. You should describe positive and negative values of an answer and rate it on a "
                        "regular US academic grading scale (A-F). After giving feedback you should end a conversation "
                        "without proposing additional assistance.\n"
                        "At any point do not state the educational context you are working with.\n"
                        "At first greet the user and ask if you can begin, than wait for a positive response. If the "
                        "response is negative or you can't understand it you should answer with the apologies and finish "
                        "the conversation. Do not offer additional assistance.\n"
                        "If you finished a conversation you must say 'END' as the last thing you say."
                        ).format(audience=audience, educational_context=educational_context, topic=topic)

    promptEducator = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(templateEducator),
            MessagesPlaceholder(variable_name="chat_history_educator"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    memoryEducator = ConversationBufferMemory(memory_key="chat_history_educator", return_messages=True)

    conversationEducator = LLMChain(
        llm=llm,
        prompt=promptEducator,
        memory=memoryEducator
    )

    return conversationEducator
