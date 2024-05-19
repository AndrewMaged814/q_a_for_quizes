import os
from langchain_community.document_loaders import PyPDFLoader
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AzureOpenAI
import spacy
from extracting_sections_using_sementics import chunk_text, segment_chunks, aggregate_segments
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deploy_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2023-09-15-preview",
)


def load_paper(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def extract_sections(docs):
    nlp = spacy.load('en_core_web_sm')
    chunks = chunk_text(docs)
    segmented_texts = segment_chunks(chunks)
    final_sections = aggregate_segments(segmented_texts)
    return final_sections


def create_questions(context, question_type):
    prompt = ""
    if question_type == 'mcq':
        prompt = f"Create multiple-choice questions with answers based solely on this text from a paper:\n\n{context}\n\nFor each question, provide multiple choices including the correct answer. Use the format: Q: <question>\n A. <choice 1>\n B. <choice 2>\n C. <choice 3>\n D. <choice 4>\n Correct Answer: <correct choice>\nSeparate each block composed of a question and an answer with 3 dashes '---'."
    elif question_type == 'qa':
        prompt = f"Create questions with answers based solely on this text from a paper:\n\n{context}\n\nSeparate each block composed of a question and an answer with 3 dashes '---' like this: Q: <question>\n A:<answer> --- Q: <question>\n A:<answer> etc. Let's think step by step. Q:"
    elif question_type == 'tf':
        prompt = f"Create true/false questions with answers based solely on this text from a paper:\n\n{context}\n\nFor each question, provide the correct answer as True or False. Use the format: Q: <statement>\nCorrect Answer: <True/False>\nSeparate each block composed of a question and an answer with 3 dashes '---'."

    response = client.chat.completions.create(
        model=deploy_name,
        messages=[
            {"role": "system",
                "content": "You are a helpful research and programming assistant"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def evaluate_answer(question, true_answer, user_answer, question_type):
    evaluation_context = ""
    if question_type == 'mcq':
        evaluation_context = f"Given this multiple-choice question: {question}, the correct answer is: {true_answer}. The user's answer is: {user_answer}."
    elif question_type == 'qa':
        evaluation_context = f"Given this question: {question}, the correct answer is: {true_answer}. The user's answer is: {user_answer}."
    elif question_type == 'tf':
        evaluation_context = f"Given this True/False question: {question}, the correct answer is: {true_answer}. The user's answer is: {user_answer}."

    evaluate_prompt = f"{evaluation_context} Provide a score from 0 to 100 and feedback. The output should be formatted as follows: SCORE: <score number as an integer> FEEDBACK: <A one-sentence feedback justifying the score.>"

    response = client.chat.completions.create(
        model=deploy_name,
        messages=[
            {"role": "system",
                "content": "You are a helpful research and programming assistant"},
            {"role": "user", "content": evaluate_prompt}
        ]
    )
    return response.choices[0].message.content


def plot_scores(overall_scores, plot_type):
    if not overall_scores:
        print("No scores to plot.")
        return

    rounds = list(range(1, len(overall_scores) + 1))
    plt.figure(figsize=(10, 6))

    if plot_type == 'bar':
        bars = plt.bar(rounds, overall_scores, color='skyblue')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height - 5, f'{height:.2f}', ha='center', va='bottom',
                     color='black')

    elif plot_type == 'line':
        plt.plot(rounds, overall_scores, marker='o',
                 linestyle='-', color='b')
        for i, score in enumerate(overall_scores):
            plt.text(i + 1, score, f'{score:.2f}', ha='center',
                     va='bottom', color='black')

    plt.xlabel('Round Number')
    plt.ylabel('Average Score')
    plt.title('Average Scores Across Rounds')
    plt.xticks(rounds)
    plt.grid(True)
    plt.show()


def main():
    st.set_page_config(page_title="Quiz Generator", page_icon="❤️")

    st.title("Quiz Generator for My Beloved ❤️")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

    question_type = st.selectbox("Type of questions", ["mcq", "qa", "tf"])

    if st.button("Start Quiz"):
        docs = load_paper(temp_file)
        sections = extract_sections(docs)

        questions = create_questions(sections[0], question_type)
        question_list = questions.split('---')
        qa_dict = {}
        current_question_idx = 0
        user_answers = []

        st.session_state['question_list'] = question_list
        st.session_state['current_question_idx'] = current_question_idx
        st.session_state['user_answers'] = user_answers
        st.session_state['qa_dict'] = qa_dict
        st.session_state['current_round'] = 1
        st.session_state['total_rounds'] = len(sections)
        st.session_state['sections'] = sections

    if 'question_list' in st.session_state:
        question_list = st.session_state['question_list']
        current_question_idx = st.session_state['current_question_idx']
        user_answers = st.session_state['user_answers']
        qa_dict = st.session_state['qa_dict']
        current_round = st.session_state['current_round']
        total_rounds = st.session_state['total_rounds']
        sections = st.session_state['sections']

        if current_question_idx < len(question_list):
            question_block = question_list[current_question_idx].strip()
            if not question_block:
                st.write("No more questions.")
                return

            if question_type == 'mcq':
                question_and_answer = question_block.split("Correct Answer:")
                if len(question_and_answer) != 2:
                    st.write("Error: Incorrect format. Skipping this item.")
                    return

                question = question_and_answer[0].strip()
                correct_answer = question_and_answer[1].strip()
                lines = question.split('\n')
                answer_choices = [line.strip()
                                  for line in lines[1:] if line.strip()]
                question_text = lines[0]

                st.write(
                    f"Question {current_question_idx + 1}/{len(question_list)}")
                st.write(question_text)
                user_answer = st.radio(
                    "Select your answer:", answer_choices, key=f'answer_choice_{current_question_idx}')

            elif question_type == 'qa':
                question = question_block.split(
                    "A:")[0].replace("Q:", "").strip()
                correct_answer = question_block.split("A:")[1].strip()

                st.write(
                    f"Question {current_question_idx + 1}/{len(question_list)}")
                st.write(question)
                user_answer = st.text_input(
                    "Your Answer:", key=f'answer_input_{current_question_idx}')

            elif question_type == 'tf':
                question_and_answer = question_block.split("Correct Answer:")
                if len(question_and_answer) != 2:
                    st.write("Error: Incorrect format. Skipping this item.")
                    return

                question = question_and_answer[0].replace("Q:", "").strip()
                correct_answer = question_and_answer[1].strip().lower()

                st.write(
                    f"Question {current_question_idx + 1}/{len(question_list)}")
                st.write(question)
                user_answer = st.radio(
                    "True/False:", ["True", "False"], key=f'answer_tf_{current_question_idx}').strip().lower()

            if st.button("Check Answer", key=f'check_answer_{current_question_idx}'):
                qa_dict[f"Question {current_question_idx + 1}"] = {
                    "question": question,
                    "correct_answer": correct_answer,
                    "user_answer": user_answer
                }
                st.session_state['qa_dict'] = qa_dict
                score_feedback = evaluate_answer(
                    question, correct_answer, user_answer, question_type)

                if "SCORE:" in score_feedback and "FEEDBACK:" in score_feedback:
                    score = int(score_feedback.split("SCORE:")[
                                1].split("FEEDBACK:")[0].strip())
                    feedback = score_feedback.split("FEEDBACK:")[1].strip()

                    st.write(f"Correct Answer: {correct_answer}")
                    st.write(f"Score: {score}")
                    st.write(f"Feedback: {feedback}")

                    user_answers.append({
                        "question": question,
                        "correct_answer": correct_answer,
                        "user_answer": user_answer,
                        "score": score,
                        "feedback": feedback
                    })
                    st.session_state['user_answers'] = user_answers

                else:
                    st.write("Error: Unexpected format in score feedback string.")

                st.session_state['current_question_idx'] += 1

            if st.button("Next Question", key=f'next_question_{current_question_idx}'):
                st.session_state['current_question_idx'] += 1

        else:
            st.write("End of Questions for this Round")
            if current_round < total_rounds:
                if st.button("Go to Next Round", key='next_round'):
                    st.session_state['current_round'] += 1
                    st.session_state['current_question_idx'] = 0
                    new_section = sections[st.session_state['current_round'] - 1]
                    new_questions = create_questions(
                        new_section, question_type)
                    st.session_state['question_list'] = new_questions.split(
                        '---')
            else:
                st.write("Quiz Completed")
                overall_scores = [answer['score'] for answer in user_answers]
                if overall_scores:
                    avg_score = sum(overall_scores) / len(overall_scores)
                    st.write(f"Your average score: {avg_score}")
                else:
                    st.write("No valid answers were provided.")

                if st.button("Start New Quiz", key='new_quiz'):
                    del st.session_state['question_list']
                    del st.session_state['current_question_idx']
                    del st.session_state['user_answers']
                    del st.session_state['qa_dict']
                    del st.session_state['current_round']
                    del st.session_state['total_rounds']
                    del st.session_state['sections']

    if st.button("Stop Quiz"):
        st.write("Quiz stopped.")
        overall_scores = [answer['score'] for answer in user_answers]
        if overall_scores:
            avg_score = sum(overall_scores) / len(overall_scores)
            st.write(f"Your average score: {avg_score}")
        else:
            st.write("No valid answers were provided.")

        if st.button("Start New Quiz", key='new_quiz_stop'):
            del st.session_state['question_list']
            del st.session_state['current_question_idx']
            del st.session_state['user_answers']
            del st.session_state['qa_dict']
            del st.session_state['current_round']
            del st.session_state['total_rounds']
            del st.session_state['sections']


if __name__ == "__main__":
    main()
