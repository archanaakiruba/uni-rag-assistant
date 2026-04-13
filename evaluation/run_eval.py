"""
Evaluation runner — hits POST /ask for each test case, checks source overlap.

Usage:
  python evaluation/run_eval.py [--url http://localhost:8000]

Target: at least N-1/N test cases pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

EVAL_CASES_PATH = os.path.join(os.path.dirname(__file__), "eval_cases.json")


def load_cases() -> list[dict]:
    with open(EVAL_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def ask(client: httpx.Client, base_url: str, user_id: str, question: str) -> dict:
    resp = client.post(
        f"{base_url}/ask",
        json={"user_id": user_id, "question": question},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def source_overlap(returned: list[str], expected: list[str]) -> tuple[int, int]:
    """Return (matched_count, expected_count)."""
    matched = sum(1 for s in expected if s in returned)
    return matched, len(expected)


def evaluate_single_turn(
    client: httpx.Client,
    base_url: str,
    case: dict,
) -> dict:
    tc_id = case["id"]
    query = case["query"]
    expected = case["expected_sources"]

    result = ask(client, base_url, user_id=f"eval_{tc_id}", question=query)
    returned_sources = result.get("sources", [])
    answer = result.get("answer", "")

    # Special case: out-of-scope question — pass if no sources were retrieved
    if case.get("expect_out_of_scope"):
        passed = len(returned_sources) == 0
        return {
            "id": tc_id,
            "query": query,
            "expected": ["(no sources — out of scope)"],
            "returned": returned_sources,
            "matched": 1 if passed else 0,
            "total": 1,
            "passed": passed,
            "answer_preview": answer[:120],
        }

    # Special case: insufficient info — pass if the answer directs user to contact admissions
    if case.get("expect_no_answer"):
        contact_keywords = ["contact", "reach out", "admissions office", "directly"]
        passed = any(kw in answer.lower() for kw in contact_keywords)
        return {
            "id": tc_id,
            "query": query,
            "expected": ["(answer must recommend contacting admissions)"],
            "returned": [answer[:60] + "..."],
            "matched": 1 if passed else 0,
            "total": 1,
            "passed": passed,
            "answer_preview": answer[:120],
        }

    matched, total = source_overlap(returned_sources, expected)
    passed = matched >= max(1, total // 2)  # pass if >=50% expected sources returned

    return {
        "id": tc_id,
        "query": query,
        "expected": expected,
        "returned": returned_sources,
        "matched": matched,
        "total": total,
        "passed": passed,
        "answer_preview": answer[:120],
    }


def evaluate_multi_turn(
    client: httpx.Client,
    base_url: str,
    case: dict,
) -> dict:
    tc_id = case["id"]
    turns = case["turns"]
    user_id = f"{case.get('user_id') or f'eval_{tc_id}_mt'}_{int(time.time())}"

    last_result = None
    for turn_def in turns:
        # Support both "query" (legacy) and "question" (TC09 style)
        question = turn_def.get("query") or turn_def.get("question", "")
        last_result = ask(client, base_url, user_id=user_id, question=question)
        time.sleep(0.3)

    # Evaluate only the last turn
    final_turn = turns[-1]
    answer = (last_result or {}).get("answer", "")

    if final_turn.get("expects_clarification"):
        passed = "?" in answer
        return {
            "id": tc_id,
            "query": final_turn.get("query") or final_turn.get("question", ""),
            "expected": ["(clarifying question)"],
            "returned": ["?" if passed else "(no question mark)"],
            "matched": 1 if passed else 0,
            "total": 1,
            "passed": passed,
            "answer_preview": answer[:120],
        }

    expected = final_turn.get("expected_sources", [])
    returned_sources = last_result.get("sources", []) if last_result else []
    matched, total = source_overlap(returned_sources, expected)
    passed = matched >= max(1, total // 2)

    return {
        "id": tc_id,
        "query": f"[multi-turn] final: {final_turn.get('query') or final_turn.get('question', '')}",
        "expected": expected,
        "returned": returned_sources,
        "matched": matched,
        "total": total,
        "passed": passed,
        "answer_preview": answer[:120],
    }


def run(base_url: str) -> None:
    cases = load_cases()
    results: list[dict] = []

    with httpx.Client() as client:
        # Check server is up
        try:
            client.get(f"{base_url}/health", timeout=5).raise_for_status()
        except Exception as e:
            print(f"ERROR: Could not reach {base_url}/health — {e}")
            print("Make sure the server is running: uvicorn app:app --reload")
            sys.exit(1)

        for case in cases:
            print(f"  Running {case['id']}...", end=" ", flush=True)
            try:
                if "turns" in case:
                    r = evaluate_multi_turn(client, base_url, case)
                else:
                    r = evaluate_single_turn(client, base_url, case)
                status = "PASS" if r["passed"] else "FAIL"
                print(f"{status}  ({r['matched']}/{r['total']} sources)")
            except Exception as e:
                r = {"id": case["id"], "passed": False, "error": str(e)}
                print(f"ERROR — {e}")
            results.append(r)
            time.sleep(0.5)

    # Summary
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")
    print(f"{'='*60}\n")

    for r in results:
        if not r.get("passed"):
            print(f"FAIL [{r['id']}]")
            print(f"  Expected : {r.get('expected', [])}")
            print(f"  Returned : {r.get('returned', [])}")
            if "answer_preview" in r:
                print(f"  Answer   : {r['answer_preview']}...")
            print()

    if passed < total - 1:
        print(f"WARNING: target is {total-1}/{total}, got {passed}/{total}.")
        sys.exit(1)
    else:
        print(f"Target met: {passed}/{total}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the running API")
    args = parser.parse_args()
    run(args.url)
