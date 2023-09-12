import time
import configparser

from langchain.chat_models import ChatOpenAI

import question_generator
import question_reviewer
import reviewer
import student

config = configparser.ConfigParser()
config.read("main.ini")

audience = config["DEFAULTS"]["audience"]
educational_context = config["DEFAULTS"]["educational_context"]
topic = config["DEFAULTS"]["topic"]

my_chatLLM = ChatOpenAI(openai_api_key=config["API"]["OpenAI_API_key"], temperature=1.2)

conversationQuestionGenerator = question_generator.create(llm=my_chatLLM, audience=audience, topic=topic, educational_context=educational_context)
review_result = ""
loop_counter = 0
REVIEW_LOOPS_MAX = 3
while True:
    print("\n\n")
    print(review_result)
    question = conversationQuestionGenerator({"question": review_result})
    print("\nProposed question\n", 10*"-", "\n", question["text"], "\n", 10*"-", "\n")
    reviewers = {"yes": 0, "no": 0, "explanations": []}
    for i in range(3):
        time.sleep(15)  # Delay due to limitations of a free OpenAI API account
        review = reviewer.evaluate(llm=my_chatLLM, educational_context=educational_context, audience=audience, topic=topic, question=question["text"])
        print("Reviewer ", str(i+1))
        print(review)
        if "NO" in review:
            reviewers["no"] += 1
            reviewers["explanations"].append(review)
        else:
            reviewers["yes"] += 1
    if reviewers["no"] >= reviewers["yes"]:
        review_result = "Create new introduction, context and question taking into account following comments to the previous question you prepared:\n"
        for i, explanation in enumerate(reviewers["explanations"]):
            review_result += str(i+1)+". '"+explanation+"'\n"
        review_result += "\nTry harder than the last time. Give only the revised text."
        loop_counter += 1
    else:
        print("Created good question!")
        break
    if loop_counter >= REVIEW_LOOPS_MAX:
        print("Couldn't create a good question. Following with what we have.")
        break

chainEducator = question_reviewer.create(llm=my_chatLLM, educational_context=educational_context, audience=audience, topic=topic, question=question["text"])
conversationStudent = student.create(llm=my_chatLLM, audience=audience, topic=topic)

print("\nStarting conversation with student")
print(10*"-", "\n", "Educator\n", question["text"])
time.sleep(15)  # Delay due to limitations of a free OpenAI API account
answerStudent = conversationStudent({"question": question["text"]})
print(10*"-", "\n", "Student\n", answerStudent["text"])
time.sleep(15)  # Delay due to limitations of a free OpenAI API account
answerEducator = chainEducator.run(answerStudent["text"])
print(10*"-", "\n", "Educator\n", answerEducator)
