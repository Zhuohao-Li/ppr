import torch
import random
import requests
import time
from typing import Dict, List, Optional
import os
import asyncio
from openai import AsyncOpenAI

# FastAPI server configuration
_QWEN_API_URL = os.getenv("QWEN_API_URL", "http://localhost:30000/v1")
_QWEN_MODEL_NAME = "rm"
_TIMEOUT_S       = 300
_RETRIES         = 10
_CONCURRENCY     = 800

async def _single_chat_request(
    client: AsyncOpenAI,
    messages: List[Dict],
    retries: int = _RETRIES
) -> Optional[str]:

    for attempt in range(1, retries + 1):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model       = _QWEN_MODEL_NAME,
                    messages    = messages,
                    max_tokens  = 4096,
                    temperature = 0.0
                ),
                timeout=_TIMEOUT_S
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(1)
    return None


async def compute_process_reward_in_batch(
    rollout_results: List[Dict]
) -> List[Dict]:
    """
    Args
    ----
    rollout_results : [{'prompt_str': str, 'response_str': str}, ...]

    Returns
    -------
    List[Dict]  same length as rollout_results; elements like {'score': float, 'max_score': float}
    """

    batch_messages = [
        build_process_messages(r["prompt_str"], r["response_str"])
        for r in rollout_results
    ]

    client = AsyncOpenAI(base_url=_QWEN_API_URL, api_key="EMPTY")
    sem    = asyncio.Semaphore(_CONCURRENCY)

    async def _wrapped_call(msgs):
        async with sem:
            return await _single_chat_request(client, msgs)

    raw_replies = await asyncio.gather(
        *[_wrapped_call(m) for m in batch_messages],
        return_exceptions=True
    )

    score_results: List[Dict] = []
    for idx, reply in enumerate(raw_replies):
        if isinstance(reply, Exception) or reply is None:
            score_results.append(None)
            continue

        res = extract_process_score(reply)
        if res is None:
            score_results.append(None)
        else:
            score_results.append(res)

    if score_results:
        valid_results = [(i, r) for i, r in enumerate(score_results) if r is not None]
        if valid_results:
            sample_size = min(3, len(valid_results))
            sample_items = random.sample(valid_results, sample_size)
            for i, result in sample_items:
                print(
                    f"[Sample {i}] {result['score']}/"
                    f"{result['max_score']}\n"
                    f"P: {rollout_results[i]['prompt_str']}\n"
                    f"A: {rollout_results[i]['response_str']}"
                    f"Process reward model output: {raw_replies[i]}"
                )
    return score_results

def build_process_messages(query_str, solution_str):
    """
    Build messages for process model in OpenAI format.
    
    Args:
        query_str: The query/prompt string
        solution_str: The response string
        
    Returns:
        List of messages in OpenAI format
    """

    sys_prompt = """ 
    You are a very strict and skilled evaluator.
    Given **Query**, and **Response** pair, you should only evaluate the **Response**.
    This **Response** is one turn of multi-turn response, so it is okay that the response do not have final answer.
    For evaluation, generate reasonable principles and score each principle individually. 
    The most important principles you can refer are: 1. Whether it extracts correct information from <information> based on query. 2. Whether it provides correct search query for <search>  3. Whether it correctly decides conduct or not conduct <search> 
    Every string in **Response** must be wrapped in <think></think>,<search></search>, <information></information> or <answer></answer>. If not, the output SCORE must be 0. 

    [Output Format Requirements]
    Respond in exactly two lines:
        1. Analysis: Explain the reasoning and individual scores for each principle.
        2. Scores: <final_score>SCORE,MAX_SCORE</final_score>. e.g. <final_score>4,6</final_score>
    """.strip()

    user_fmt = f"""
    [Conversation Context]
    **Query**: {query_str}
    **Response**: {solution_str}
    """.strip()
    
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_fmt},
    ]

def extract_process_score(judge_str):
    """Extract score and max_score from process model output using <final_score></final_score> tags."""
    import re
    
    final_score_pattern = r'<final_score>(.*?)</final_score>'
    final_score_match = re.finditer(final_score_pattern, judge_str, re.DOTALL)
    final_score_match = list(final_score_match)
    
    if len(final_score_match) == 0:
        print(f"No final_score found in judge string")
        return None
    
    final_score_str = final_score_match[-1].group(1).strip()
    
    try:
        score, max_score = final_score_str.split(',')
        score = int(score.strip())
        max_score = int(max_score.strip())
    except Exception as e:
        print(f"Error in extracting score: {e}")
        return None

    return {
        'max_score': max_score,
        'score': score
    }


def filter_process_score(score_result):
    """
    Process process score results and validate them before applying rewards.
    
    Args:
        score_result: Result from extract_process_score
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    if score_result is None:
        print('No process result found')
        return False
    
    # Handle non-dictionary results (error cases)
    if not isinstance(score_result, dict):
        print(f"Process judge returned error or incomplete result: {score_result}")
        print("NO process reward added in this case")
        return False

    max_score_extracted = score_result.get('max_score')
    score = score_result.get('score')
    
    # Check if max_score_extracted is valid
    if max_score_extracted is None or max_score_extracted <= 0:
        print(f"Process Judge Score (no max score available)")
        return False

    if score is None or score < 0:
        print(f"Process Judge Score (score < 0)")
        return False

    # Check if score exceeds max score
    if score > max_score_extracted:
        print(f"Process Judge Score (score > max score extracted)")
        return False

    # success
    return True

def write_tag_rewards(
    reward_tensor: torch.Tensor,      # shape = [B, seq_len]
    batch_idx: int,                   
    turn_pos: int,
    score: int,
    max_score_extracted: int,     # max score extracted from process model
):

    reward_tensor[batch_idx, turn_pos] = score/max_score_extracted  # normalize PRM to [0, 1]
