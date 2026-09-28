from dotenv import load_dotenv
import os
import dotenv

import mlflow
from openai import OpenAI
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines


# Configure environment
load_dotenv(override=True)
mlflow.set_experiment("Evaluation Quickstart")


client = OpenAI()

def my_agent(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions concisely.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


def qa_predict_fn(question: str) -> str:
    """Wrapper function for evaluation using ``my_agent``."""
    return my_agent(question)


# Define a simple Q&A dataset with questions and expected answers
eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "expectations": {"expected_response": "Wright Brothers"},
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"},
    },
]


@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if answer is concise (less than 5 words)"""
    return len(outputs.split()) <= 5

scorers = [
    Correctness(),
    Guidelines(name="is_english", guidelines="The answer must be in English"),
    is_concise,
]


if __name__ == "__main__":
    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=qa_predict_fn,
        scorers=scorers,
    )

