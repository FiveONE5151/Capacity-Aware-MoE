import random
import re
import string

import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc, results):
    """
    Custom process_results to handle the mismatch between:
    - doc_to_target: "The answer is (X)." for fewshot display
    - filtered result: "(X)" extracted by regex filter

    We compare the extracted answer directly with doc["answer"].
    """
    gold = doc["answer"]  # e.g. "(C)"
    result = results[0]  # filtered response from regex filter

    # Normalize both strings: lowercase and remove punctuation
    def normalize(s):
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        return s.strip()

    score = 1.0 if normalize(result) == normalize(gold) else 0.0
    return {"exact_match": score}
