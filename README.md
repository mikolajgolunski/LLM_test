# Concept of using LLMs as generator and grader of a writing question

A concept system that uses LLMs to generate free response questions that test writing skills. Answer to the generated question is than graded by LLM.

![Diagram of the system](C:\Users\mikol\PycharmProjects\LLM_test\diagram.jpg)

Each LLM role is coded in separate file.

API key should be given in a main.ini file

## Transcripts

Transcripts folder includes a few transcripts from the testing sessions.

`student_test_1` and `student_test_2` use a flawed student LLM with prompt saying that "You should answer poorly providing only minimal viable answer." leading to worse grading. `student_test_3` uses an ideal student LLM, which, unsurprisingly, creates a great response that is graded as an A.

`baseball_vs_golf` shows an interaction between Questions' generator and Reviewers. Generator gets an instruction to prepare question about golf but reviewers think it should be about baseball. In theory Generator should rewrite the question to be about baseball, as Reviewers want. But it still needs some work. What's more Reviewers do not always detect that the golf question is wrong as it should be about baseball!