"""Prompt templates for Google QUEST Q&A Labeling task."""


def build_prompt(qa_id: str, question: str, answer: str, context: str = "") -> str:
    """Build the prompt for Google QUEST Q&A Labeling task.
    
    Args:
        qa_id: Unique identifier for the Q&A pair
        question: The question text (title + body)
        answer: The answer text
        context: Optional context (e.g., host site like "stackoverflow.com")
    
    Returns:
        Formatted prompt string for the model
    """
    
    prompt = f"""You are an expert rater for the Google QUEST Q&A Labeling task. Your job is to evaluate a Question-Answer pair and assign PRECISE CONTINUOUS scores between 0.0 and 1.0 for 30 specific attributes.

### CRITICAL: Use Fine-Grained Continuous Scores
- DO NOT limit yourself to just 0.0, 0.5, or 1.0
- Use decimal precision (e.g., 0.23, 0.67, 0.84, 0.15)
- Think of scores as percentages: 0.23 = 23%, 0.67 = 67%
- Each attribute should have a nuanced score reflecting subtle differences

### Scoring Examples
- **0.0-0.2**: Very weak/absent (e.g., 0.05, 0.12, 0.18)
- **0.2-0.4**: Somewhat present (e.g., 0.23, 0.31, 0.39)
- **0.4-0.6**: Moderate (e.g., 0.42, 0.55, 0.58)
- **0.6-0.8**: Strong (e.g., 0.63, 0.72, 0.79)
- **0.8-1.0**: Very strong/dominant (e.g., 0.81, 0.91, 0.97)

### Task
Carefully analyze the following Q&A pair and output a JSON object with FINE-GRAINED scores for these 30 labels.

**Question Attributes:**
1. Intent: `question_asker_intent_understanding` (clarity of intent), `question_fact_seeking`, `question_opinion_seeking`, `question_multi_intent`.
2. Quality: `question_well_written`, `question_body_critical` (body's importance), `question_conversational`, `question_expect_short_answer`.
3. Type: `question_type_choice`, `question_type_compare`, `question_type_consequence`, `question_type_definition`, `question_type_entity`, `question_type_instructions`, `question_type_procedure`, `question_type_reason_explanation`, `question_type_spelling`.
4. Impact: `question_has_commonly_accepted_answer`, `question_interestingness_others`, `question_interestingness_self`, `question_not_really_a_question`.

**Answer Attributes:**
1. Quality: `answer_helpful`, `answer_level_of_information`, `answer_plausible`, `answer_relevance`, `answer_satisfaction`, `answer_well_written`.
2. Type: `answer_type_instructions`, `answer_type_procedure`, `answer_type_reason_explanation`.

### Input Data
**Context/Host:** {context if context else "General Forum"}
**Question:** {question}
**Answer:** {answer}

### Output Format
Return ONLY valid JSON with decimal scores (2-3 decimal places). Do not include markdown formatting.
{{
  "qa_id": "{qa_id}",
  "scores": {{
    "question_asker_intent_understanding": <float with 2-3 decimals, e.g., 0.73>,
    "question_body_critical": <float>,
    "question_conversational": <float>,
    "question_expect_short_answer": <float>,
    "question_fact_seeking": <float>,
    "question_has_commonly_accepted_answer": <float>,
    "question_interestingness_others": <float>,
    "question_interestingness_self": <float>,
    "question_multi_intent": <float>,
    "question_not_really_a_question": <float>,
    "question_opinion_seeking": <float>,
    "question_type_choice": <float>,
    "question_type_compare": <float>,
    "question_type_consequence": <float>,
    "question_type_definition": <float>,
    "question_type_entity": <float>,
    "question_type_instructions": <float>,
    "question_type_procedure": <float>,
    "question_type_reason_explanation": <float>,
    "question_type_spelling": <float>,
    "question_well_written": <float>,
    "answer_helpful": <float>,
    "answer_level_of_information": <float>,
    "answer_plausible": <float>,
    "answer_relevance": <float>,
    "answer_satisfaction": <float>,
    "answer_type_instructions": <float>,
    "answer_type_procedure": <float>,
    "answer_type_reason_explanation": <float>,
    "answer_well_written": <float>
  }},
  "justification": "Brief analysis of key scoring decisions."
}}
"""
    return prompt
