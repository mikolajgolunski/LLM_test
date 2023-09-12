from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate


def create(llm, audience, topic):
    templateStudent = (f"You are a {audience}. You are interested in {topic}. Do not mention you are an AI language "
                       f"model. Answer the questions in a concise fashion just how the {audience} would do. "
                       f"Do not ask questions. You should answer poorly providing only minimal viable answer."
                       ).format(audience=audience, topic=topic)

    promptStudent = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(templateStudent),
            MessagesPlaceholder(variable_name="chat_history_student"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    memoryStudent = ConversationBufferMemory(memory_key="chat_history_student", return_messages=True)

    conversationStudent = LLMChain(
        llm=llm,
        prompt=promptStudent,
        memory=memoryStudent
    )

    return conversationStudent
