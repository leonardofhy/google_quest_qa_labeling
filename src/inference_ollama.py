#!/usr/bin/env python3
"""
Ollama Qwen3 8B Inference Script for Google QUEST Q&A Labeling

Runs Qwen3 8B via Ollama on the Google QUEST test set and produces
a submission file matching sample_submission.csv.
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from ollama import chat

from labels import QUEST_LABELS
from prompts import build_prompt


def inference_single(
    qa_id: str,
    question: str,
    answer: str,
    context: str = "",
    model_name: str = "qwen3:8b"
) -> Dict[str, Any]:
    """Run inference on a single Q&A pair without thinking."""
    prompt = build_prompt(qa_id, question, answer, context)
    
    # Call the chat API without thinking for faster inference
    response = chat(
        model=model_name,
        messages=[{'role': 'user', 'content': prompt}],
        think=False,  # Disable thinking for faster inference
        stream=False,
    )
    
    try:
        # Extract JSON from response
        response_text = response.message.content
        # Find JSON in response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        json_str = response_text[start_idx:end_idx]
        result = json.loads(json_str)
        return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing response for qa_id {qa_id}: {e}", file=sys.stderr)
        return {
            "qa_id": qa_id,
            "scores": {label: 0.5 for label in QUEST_LABELS},
            "error": str(e)
        }


def inference_batch(data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Run inference on a batch of Q&A pairs."""
    results = []
    total = len(data)
    
    for idx, item in enumerate(data, 1):
        print(f"Processing {idx}/{total}: qa_id={item.get('qa_id', 'unknown')}")
        
        result = inference_single(
            qa_id=item.get('qa_id', ''),
            question=item.get('question_title', '') + ' ' + item.get('question_body', ''),
            answer=item.get('answer', ''),
            context=item.get('host', '')
        )
        results.append(result)
    
    return results


def process_csv(input_csv: str, output_csv: str = None, model_name: str = "qwen3:8b"):
    """Process test.csv and generate submission CSV."""
    input_path = Path(input_csv)
    output_path = Path(output_csv) if output_csv else Path("data/submission.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}\n")

    with input_path.open('r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))

    results: List[Dict[str, Any]] = []
    for item in tqdm(data, desc="Scoring Q&A pairs", unit="pair"):
        qa_id = item.get('qa_id', '')
        question = (item.get('question_title', '') + ' ' + item.get('question_body', '')).strip()
        answer = item.get('answer', '').strip()
        context = item.get('host', '').strip()

        try:
            result = inference_single(qa_id, question, answer, context, model_name)
            scores = result.get('scores', {label: 0.5 for label in QUEST_LABELS})
        except Exception as e:
            print(f"\nError on qa_id {qa_id}: {e}", file=sys.stderr)
            scores = {label: 0.5 for label in QUEST_LABELS}

        row = {'qa_id': qa_id, **{label: scores.get(label, 0.5) for label in QUEST_LABELS}}
        results.append(row)

    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['qa_id'] + QUEST_LABELS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSubmission saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Google QUEST Q&A Labeling with Ollama')
    parser.add_argument('--csv', type=str, default='data/test.csv', help='Input CSV (default: data/test.csv)')
    parser.add_argument('--output', type=str, default='data/submission.csv', help='Output CSV (default: data/submission.csv)')
    parser.add_argument('--model', type=str, default='qwen3:8b', help='Ollama model name (default: qwen3:8b)')

    args = parser.parse_args()
    process_csv(args.csv, args.output, args.model)
