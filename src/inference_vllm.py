#!/usr/bin/env python3
"""
vLLM Qwen3 8B Inference Script for Google QUEST Q&A Labeling

Runs Qwen3 8B via vLLM on the Google QUEST test set and produces
a submission file matching sample_submission.csv.

Supports two modes:
1. Standard mode: Sequential processing with vLLM
2. Batch mode: Parallel processing with Ray Data for large datasets
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams

from labels import QUEST_LABELS
from prompts import build_prompt


def create_llm_model(model_name: str = "Qwen/Qwen3-8B") -> LLM:
    """Initialize and return a vLLM model instance.
    
    Args:
        model_name: HuggingFace model identifier (default: Qwen/Qwen3-8B)
    
    Returns:
        Initialized LLM instance
    """
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    return llm


def inference_single(
    llm: LLM,
    sampling_params: SamplingParams,
    qa_id: str,
    question: str,
    answer: str,
    context: str = "",
) -> Dict[str, Any]:
    """Run inference on a single Q&A pair using vLLM.
    
    Args:
        llm: vLLM model instance
        sampling_params: Sampling parameters for generation
        qa_id: Unique identifier for the Q&A pair
        question: The question text
        answer: The answer text
        context: Optional context (e.g., host site)
    
    Returns:
        Dictionary with qa_id, scores, and justification
    """
    prompt = build_prompt(qa_id, question, answer, context)
    
    try:
        # Generate output
        outputs = llm.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text
        
        # Extract JSON from response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")
        
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


def inference_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    items: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Run batch inference on multiple Q&A pairs using vLLM.
    
    Args:
        llm: vLLM model instance
        sampling_params: Sampling parameters for generation
        items: List of dicts with qa_id, question, answer, context
    
    Returns:
        List of dictionaries with qa_id and scores
    """
    # Build all prompts
    prompts = []
    qa_ids = []
    for item in items:
        qa_id = item.get('qa_id', '')
        question = (item.get('question_title', '') + ' ' + item.get('question_body', '')).strip()
        answer = item.get('answer', '').strip()
        context = item.get('host', '').strip()
        
        prompt = build_prompt(qa_id, question, answer, context)
        prompts.append(prompt)
        qa_ids.append(qa_id)
    
    # Batch generation
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for qa_id, output in zip(qa_ids, outputs):
        try:
            response_text = output.outputs[0].text
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            scores = result.get('scores', {label: 0.5 for label in QUEST_LABELS})
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response for qa_id {qa_id}: {e}", file=sys.stderr)
            scores = {label: 0.5 for label in QUEST_LABELS}
        
        row = {'qa_id': qa_id, **{label: scores.get(label, 0.5) for label in QUEST_LABELS}}
        results.append(row)
    
    return results


def process_csv(
    input_csv: str,
    output_csv: str = None,
    model_name: str = "Qwen/Qwen3-8B",
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    batch_size: int = 1,
):
    """Process test.csv and generate submission CSV using vLLM.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        model_name: HuggingFace model identifier
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        batch_size: Number of items to process in each batch (default: 1 for sequential)
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv) if output_csv else Path("data/submission.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    print(f"Batch size: {batch_size}\n")

    # Initialize vLLM model
    print("Initializing vLLM model...")
    llm = create_llm_model(model_name)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    print(f"Sampling Params: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}\n")

    # Read input CSV
    with input_path.open('r', encoding='utf-8') as f:
        data = list(csv.DictReader(f))

    results: List[Dict[str, Any]] = []
    total_items = len(data)
    
    if batch_size == 1:
        # Sequential processing
        for item in tqdm(data, desc="Scoring Q&A pairs", total=total_items, unit="pair"):
            qa_id = item.get('qa_id', '')
            question = (item.get('question_title', '') + ' ' + item.get('question_body', '')).strip()
            answer = item.get('answer', '').strip()
            context = item.get('host', '').strip()

            try:
                result = inference_single(
                    llm, sampling_params, qa_id, question, answer, context
                )
                scores = result.get('scores', {label: 0.5 for label in QUEST_LABELS})
            except Exception as e:
                print(f"\nError on qa_id {qa_id}: {e}", file=sys.stderr)
                scores = {label: 0.5 for label in QUEST_LABELS}

            row = {'qa_id': qa_id, **{label: scores.get(label, 0.5) for label in QUEST_LABELS}}
            results.append(row)
    else:
        # Batch processing - show progress for individual items, not batches
        with tqdm(total=total_items, desc="Scoring Q&A pairs", unit="pair") as pbar:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                try:
                    batch_results = inference_batch(llm, sampling_params, batch)
                    results.extend(batch_results)
                except Exception as e:
                    print(f"\nError processing batch starting at index {i}: {e}", file=sys.stderr)
                    # Fallback to default scores for failed batch
                    for item in batch:
                        qa_id = item.get('qa_id', '')
                        row = {'qa_id': qa_id, **{label: 0.5 for label in QUEST_LABELS}}
                        results.append(row)
                
                # Update progress by the number of items processed in this batch
                pbar.update(len(batch))

    # Write output CSV
    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['qa_id'] + QUEST_LABELS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSubmission saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Google QUEST Q&A Labeling with vLLM'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='data/test.csv',
        help='Input CSV (default: data/test.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/submission.csv',
        help='Output CSV (default: data/submission.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen3-8B',
        help='HuggingFace model identifier (default: Qwen/Qwen3-8B)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature (default: 0.3)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Nucleus sampling p (default: 0.95)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate (default: 256)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing (default: 1 for sequential, higher values for batch mode)'
    )

    args = parser.parse_args()
    process_csv(
        args.csv,
        args.output,
        model_name=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
