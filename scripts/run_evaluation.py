#!/usr/bin/env python3
"""
MARS Ablation Study -- Automated LLM-as-Judge Evaluation

Performs blind pairwise evaluation of MARS baseline vs three ablation conditions
using an OpenAI-compatible API. Randomizes system labels to prevent position bias,
evaluates across 5 dimensions, and produces summary tables.

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/run_evaluation.py [--model gpt-4o] [--queries Query1,Query2] [--output-dir evaluation_results]

Environment variables:
    OPENAI_API_KEY  -- Required. Your OpenAI API key.
    OPENAI_BASE_URL -- Optional. Override the API base URL (for Azure, proxies, etc).
"""

import argparse
import json
import os
import random
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYSTEM_LABELS = {
    "evaluation": "MARS (Full Pipeline)",
    "ablation_3agent": "3-Agent Sequential (No RAG/KG)",
    "ablation_1agent_rag": "1-Agent + RAG/KG",
    "ablation_1agent_no_rag": "1-Agent (No RAG/KG)",
}

CONDITION_KEYS = ["evaluation", "ablation_3agent", "ablation_1agent_rag", "ablation_1agent_no_rag"]

# Injected into parsed judge JSON then popped before unblinding (not part of model output schema).
JUDGE_API_META_KEY = "_judge_api_metadata"


def load_rubric(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = PROJECT_ROOT / "config" / "evaluation_rubric.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_query_dirs() -> Dict[str, Path]:
    """Find query directories under results/ that have all 4 condition files.

    Supports both the new layout (results/QueryN/) and the legacy layout
    (pipeline_logs_QueryN/ at repo root).
    """
    results = {}

    # New layout: results/QueryN/
    results_dir = PROJECT_ROOT / "results"
    if results_dir.is_dir():
        for d in sorted(results_dir.iterdir()):
            if not d.is_dir() or d.name == "evaluation":
                continue
            query_name = d.name
            has_mars = (d / "mars.json").exists() or any(
                f.name.startswith("evaluation_") and f.suffix == ".json" for f in d.iterdir()
            )
            has_3agent = (d / "ablation_3agent.json").exists() or any(
                f.name.startswith("ablation_3agent_") and f.suffix == ".json" for f in d.iterdir()
            )
            has_1rag = (d / "ablation_1agent_rag.json").exists() or any(
                f.name.startswith("ablation_1agent_rag_") and f.suffix == ".json" for f in d.iterdir()
            )
            has_1norag = (d / "ablation_1agent_no_rag.json").exists() or any(
                f.name.startswith("ablation_1agent_no_rag_") and f.suffix == ".json" for f in d.iterdir()
            )
            if has_mars and has_3agent and has_1rag and has_1norag:
                results[query_name] = d

    # Legacy layout: pipeline_logs_QueryN/ at repo root
    if not results:
        for d in sorted(PROJECT_ROOT.iterdir()):
            if not d.is_dir() or not d.name.startswith("pipeline_logs_"):
                continue
            query_name = d.name.replace("pipeline_logs_", "")
            has_eval = any(f.name.startswith("evaluation_") and f.suffix == ".json" for f in d.iterdir())
            has_3agent = any(f.name.startswith("ablation_3agent_") and f.suffix == ".json" for f in d.iterdir())
            has_1rag = any(f.name.startswith("ablation_1agent_rag_") and f.suffix == ".json" for f in d.iterdir())
            has_1norag = any(f.name.startswith("ablation_1agent_no_rag_") and f.suffix == ".json" for f in d.iterdir())
            if has_eval and has_3agent and has_1rag and has_1norag:
                results[query_name] = d

    return results


def _load_json_file(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_condition_file(query_dir: Path, exact_name: str, prefix: str) -> Path:
    """Find a condition file by exact name first, then by prefix glob."""
    exact = query_dir / exact_name
    if exact.exists():
        return exact
    candidates = sorted(
        [f for f in query_dir.iterdir() if f.name.startswith(prefix) and f.suffix == ".json"],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {exact_name} or {prefix}*.json in {query_dir}")
    return candidates[0]


def load_condition_file(query_dir: Path, prefix: str) -> Dict[str, Any]:
    """Load the first JSON file matching the prefix in the query directory."""
    candidates = sorted(
        [f for f in query_dir.iterdir() if f.name.startswith(prefix) and f.suffix == ".json"],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.json in {query_dir}")
    return _load_json_file(candidates[0])


def load_all_conditions(query_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {
        "evaluation": _load_json_file(
            _find_condition_file(query_dir, "mars.json", "evaluation_")
        ),
        "ablation_3agent": _load_json_file(
            _find_condition_file(query_dir, "ablation_3agent.json", "ablation_3agent_")
        ),
        "ablation_1agent_rag": _load_json_file(
            _find_condition_file(query_dir, "ablation_1agent_rag.json", "ablation_1agent_rag_")
        ),
        "ablation_1agent_no_rag": _load_json_file(
            _find_condition_file(query_dir, "ablation_1agent_no_rag.json", "ablation_1agent_no_rag_")
        ),
    }


def strip_raw_responses(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove raw_responses to keep the judge prompt focused on structured output."""
    cleaned = {k: v for k, v in data.items() if k != "raw_responses"}
    if "metadata" in cleaned:
        cleaned["metadata"] = {
            k: v for k, v in cleaned["metadata"].items()
            if k not in ("ablation_condition", "pipeline_run_id")
        }
    return cleaned


def build_judge_prompt(
    query_sentence: str,
    systems: Dict[str, Dict[str, Any]],
    blind_mapping: Dict[str, str],
    rubric: Dict[str, Any],
) -> Tuple[str, str]:
    """Build the system and user prompts for the judge LLM.

    Returns (system_prompt, user_prompt).
    """
    dimensions = rubric["dimensions"]

    system_prompt = (
        "You are an expert materials scientist acting as a blind evaluator. "
        "You will receive outputs from four anonymized systems (labeled A, B, C, D) "
        "that each attempted to solve the same material substitution task. "
        "You must evaluate each system across multiple dimensions using the provided rubric.\n\n"
        "IMPORTANT:\n"
        "- Evaluate each system independently on its merits.\n"
        "- Do NOT try to guess which system is which.\n"
        "- Be critical and specific in your reasoning.\n"
        "- Use your materials science expertise to assess technical correctness.\n"
        "- A fabricated/hallucinated material should receive low candidate and hallucination scores "
        "even if the fabricated properties sound plausible.\n\n"
        "You MUST respond with a valid JSON object and nothing else."
    )

    rubric_text = ""
    for dim_key, dim in dimensions.items():
        rubric_text += f"\n### {dim['name']} (weight: {dim['weight']})\n{dim['rubric']}\n"

    system_blocks = ""
    for label in ["A", "B", "C", "D"]:
        condition_key = blind_mapping[label]
        data = strip_raw_responses(systems[condition_key])
        system_blocks += f"\n{'='*60}\nSYSTEM {label}\n{'='*60}\n"
        system_blocks += json.dumps(data, indent=2, ensure_ascii=False, default=str)
        system_blocks += "\n"

    dim_keys_list = list(dimensions.keys())
    json_schema = "{\n"
    for label in ["A", "B", "C", "D"]:
        json_schema += f'  "{label}": {{\n'
        for dk in dim_keys_list:
            json_schema += f'    "{dk}": {{"score": <int 1-10>, "reasoning": "<1-3 sentences>"}},\n'
        json_schema += f'    "overall_comment": "<1-2 sentences>"\n'
        json_schema += "  },\n"
    json_schema += '  "ranking": ["<best label>", "<2nd>", "<3rd>", "<worst>"],\n'
    json_schema += '  "ranking_reasoning": "<1-3 sentences explaining the ranking>"\n'
    json_schema += "}"

    user_prompt = (
        f"## Material Substitution Query\n\n{query_sentence}\n\n"
        f"## Evaluation Rubric\n{rubric_text}\n\n"
        f"## System Outputs (Anonymized)\n{system_blocks}\n\n"
        f"## Required Output Format\n\nRespond with ONLY a JSON object in this exact structure:\n\n"
        f"```\n{json_schema}\n```\n\n"
        "Provide integer scores (1-10) for each dimension and brief reasoning for each. "
        "Then provide an overall ranking from best to worst."
    )

    return system_prompt, user_prompt


def _should_retry_completion_with_max_completion_tokens(exc: BaseException) -> bool:
    """True when the API rejects max_tokens in favor of max_completion_tokens (e.g. GPT-5.x)."""
    msg = str(exc).lower()
    if "max_completion_tokens" in msg and "max_tokens" in msg:
        return True
    if "unsupported parameter" in msg and "max_tokens" in msg:
        return True
    return False


def _chat_completion_judge(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
):
    """
    Prefer max_tokens (works for GPT-4 class models). Newer models may require
    max_completion_tokens only; retry once with that parameter.
    """
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        if _should_retry_completion_with_max_completion_tokens(e):
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        raise


def call_judge(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Call the judge LLM and parse the JSON response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(max_retries):
        try:
            response = _chat_completion_judge(
                client, model, messages, temperature, max_tokens
            )
            text = response.choices[0].message.content.strip()

            # Extract JSON from possible markdown fencing
            if "```json" in text:
                text = text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].rsplit("```", 1)[0].strip()

            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace > first_brace:
                text = text[first_brace:last_brace + 1]

            parsed: Dict[str, Any] = json.loads(text)
            if isinstance(parsed, dict):
                choice = response.choices[0]
                meta: Dict[str, Any] = {
                    "finish_reason": getattr(choice, "finish_reason", None),
                }
                msg = getattr(choice, "message", None)
                if msg is not None and getattr(msg, "refusal", None):
                    meta["refusal"] = msg.refusal
                u = getattr(response, "usage", None)
                if u is not None:
                    meta["usage"] = {
                        "prompt_tokens": getattr(u, "prompt_tokens", None),
                        "completion_tokens": getattr(u, "completion_tokens", None),
                        "total_tokens": getattr(u, "total_tokens", None),
                    }
                parsed[JUDGE_API_META_KEY] = meta
            return parsed

        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}/{max_retries}: JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"error": f"JSON parse failed after {max_retries} attempts", "raw": text}
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{max_retries}: API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"error": str(e)}


def evaluate_query(
    query_name: str,
    query_dir: Path,
    client: OpenAI,
    rubric: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """Run blind evaluation for a single query."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {query_name}")
    print(f"{'='*70}")

    systems = load_all_conditions(query_dir)
    query_sentence = systems["evaluation"]["query"]["sentence"]

    # Randomize label assignment for blind evaluation
    shuffled_keys = list(CONDITION_KEYS)
    random.shuffle(shuffled_keys)
    blind_mapping = {label: key for label, key in zip(["A", "B", "C", "D"], shuffled_keys)}
    reverse_mapping = {v: k for k, v in blind_mapping.items()}

    print(f"  Query: {query_sentence[:100]}...")
    print(f"  Blind mapping (hidden from judge): {json.dumps({v: SYSTEM_LABELS[k] for v, k in sorted(blind_mapping.items())})}")

    system_prompt, user_prompt = build_judge_prompt(query_sentence, systems, blind_mapping, rubric)

    print(f"  Prompt size: {len(user_prompt):,} chars")
    print(f"  Calling {model}...")

    start = time.time()
    judge_result = call_judge(
        client, system_prompt, user_prompt, model,
        temperature=rubric.get("temperature", 0),
        max_tokens=rubric.get("max_tokens", 8000),
    )
    elapsed = time.time() - start
    print(f"  Judge responded in {elapsed:.1f}s")

    if "error" in judge_result:
        print(f"  ERROR: {judge_result['error']}")
        return {"query_name": query_name, "error": judge_result}

    judge_completion_meta = judge_result.pop(JUDGE_API_META_KEY, None)
    if judge_completion_meta:
        fr = judge_completion_meta.get("finish_reason")
        print(f"  Judge finish_reason: {fr}")
        if fr == "length":
            print("  WARNING: finish_reason is 'length' — completion may be truncated at max_tokens.")
        usage = judge_completion_meta.get("usage") or {}
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if pt is not None or ct is not None:
            print(f"  Token usage (prompt / completion): {pt} / {ct}")

    # Unblind: map labels back to condition names
    dimensions = list(rubric["dimensions"].keys())
    unblinded_scores = {}
    for condition_key in CONDITION_KEYS:
        label = reverse_mapping[condition_key]
        label_data = judge_result.get(label, {})
        scores = {}
        for dim in dimensions:
            dim_data = label_data.get(dim, {})
            scores[dim] = {
                "score": dim_data.get("score", 0),
                "reasoning": dim_data.get("reasoning", ""),
            }
        weight_sum = sum(rubric["dimensions"][d]["weight"] for d in dimensions)
        weighted_total = sum(
            scores[d]["score"] * rubric["dimensions"][d]["weight"] for d in dimensions
        )
        scores["weighted_total"] = round(weighted_total / weight_sum, 2) if weight_sum else 0
        scores["overall_comment"] = label_data.get("overall_comment", "")
        unblinded_scores[condition_key] = scores

    # Unblind ranking
    raw_ranking = judge_result.get("ranking", [])
    unblinded_ranking = [blind_mapping.get(r, r) for r in raw_ranking]

    result = {
        "query_name": query_name,
        "query_sentence": query_sentence,
        "blind_mapping": blind_mapping,
        "scores": unblinded_scores,
        "ranking": unblinded_ranking,
        "ranking_reasoning": judge_result.get("ranking_reasoning", ""),
        "judge_model": model,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "judge_elapsed_seconds": round(elapsed, 1),
    }
    if judge_completion_meta is not None:
        result["judge_completion"] = judge_completion_meta

    # Print per-query summary
    print(f"\n  {'System':<35} ", end="")
    for dim in dimensions:
        short = rubric["dimensions"][dim]["name"][:12]
        print(f"{short:<14}", end="")
    print(f"{'Weighted':>10}")
    print(f"  {'-'*35} ", end="")
    for _ in dimensions:
        print(f"{'-'*14}", end="")
    print(f"{'-'*10}")

    for ck in CONDITION_KEYS:
        s = unblinded_scores[ck]
        label = SYSTEM_LABELS[ck]
        print(f"  {label:<35} ", end="")
        for dim in dimensions:
            print(f"{s[dim]['score']:<14}", end="")
        print(f"{s['weighted_total']:>10.2f}")

    print(f"\n  Ranking: {' > '.join(SYSTEM_LABELS.get(r, r) for r in unblinded_ranking)}")
    print(f"  Reasoning: {result['ranking_reasoning']}")

    return result


def print_aggregate_summary(all_results: List[Dict[str, Any]], rubric: Dict[str, Any]):
    """Print aggregate summary across all queries."""
    dimensions = list(rubric["dimensions"].keys())

    print(f"\n{'='*90}")
    print("AGGREGATE SUMMARY ACROSS ALL QUERIES")
    print(f"{'='*90}")

    # Collect scores per condition
    totals = {ck: {d: [] for d in dimensions} for ck in CONDITION_KEYS}
    rank_positions = {ck: [] for ck in CONDITION_KEYS}

    for result in all_results:
        if "error" in result:
            continue
        for ck in CONDITION_KEYS:
            for dim in dimensions:
                score = result["scores"][ck][dim]["score"]
                totals[ck][dim].append(score)
        ranking = result.get("ranking", [])
        for pos, ck in enumerate(ranking):
            if ck in rank_positions:
                rank_positions[ck].append(pos + 1)

    # Average scores table
    print(f"\n{'System':<35} ", end="")
    for dim in dimensions:
        short = rubric["dimensions"][dim]["name"][:12]
        print(f"{short:<14}", end="")
    print(f"{'Avg Rank':>10}")
    print(f"{'-'*35} ", end="")
    for _ in dimensions:
        print(f"{'-'*14}", end="")
    print(f"{'-'*10}")

    for ck in CONDITION_KEYS:
        label = SYSTEM_LABELS[ck]
        print(f"{label:<35} ", end="")
        for dim in dimensions:
            scores = totals[ck][dim]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"{avg:<14.1f}", end="")
        ranks = rank_positions[ck]
        avg_rank = sum(ranks) / len(ranks) if ranks else 0
        print(f"{avg_rank:>10.2f}")

    # Win/loss table
    print(f"\n{'System':<35} {'1st':>6} {'2nd':>6} {'3rd':>6} {'4th':>6}")
    print(f"{'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for ck in CONDITION_KEYS:
        label = SYSTEM_LABELS[ck]
        ranks = rank_positions[ck]
        counts = [ranks.count(i) for i in range(1, 5)]
        print(f"{label:<35} {counts[0]:>6} {counts[1]:>6} {counts[2]:>6} {counts[3]:>6}")


def main():
    parser = argparse.ArgumentParser(description="MARS Ablation LLM-as-Judge Evaluation")
    parser.add_argument("--model", default=None, help="Judge model (default: from rubric or gpt-4o)")
    parser.add_argument("--queries", default=None, help="Comma-separated query names to evaluate (default: all)")
    parser.add_argument("--output-dir", default="results/evaluation", help="Directory for results")
    parser.add_argument("--rubric", default=None, help="Path to evaluation rubric YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for blind label shuffling")
    parser.add_argument("--base-url", default=None, help="Override OpenAI base URL")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is required.")
        print("  export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    rubric = load_rubric(args.rubric)
    model = args.model or rubric.get("judge_model", "gpt-4o")

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    random.seed(args.seed)

    query_dirs = discover_query_dirs()
    if not query_dirs:
        print("ERROR: No query directories found with all 4 condition files.")
        sys.exit(1)

    if args.queries:
        selected = [q.strip() for q in args.queries.split(",")]
        query_dirs = {k: v for k, v in query_dirs.items() if k in selected}
        if not query_dirs:
            print(f"ERROR: None of the specified queries found: {args.queries}")
            print(f"  Available: {', '.join(discover_query_dirs().keys())}")
            sys.exit(1)

    print(f"MARS Ablation Study -- LLM-as-Judge Evaluation")
    print(f"Judge model: {model}")
    print(f"Queries to evaluate: {', '.join(query_dirs.keys())}")
    print(f"Random seed: {args.seed}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for query_name, query_dir in query_dirs.items():
        result = evaluate_query(query_name, query_dir, client, rubric, model)
        all_results.append(result)

        # Save per-query result
        result_path = output_dir / f"eval_{query_name}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved: {result_path}")

    # Print aggregate summary
    valid_results = [r for r in all_results if "error" not in r]
    if valid_results:
        print_aggregate_summary(valid_results, rubric)

    # Save aggregate
    aggregate_path = output_dir / "aggregate_results.json"
    aggregate = {
        "judge_model": model,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "num_queries": len(all_results),
        "num_successful": len(valid_results),
        "per_query": all_results,
    }

    # Compute aggregate scores
    if valid_results:
        dimensions = list(rubric["dimensions"].keys())
        agg_scores = {}
        for ck in CONDITION_KEYS:
            agg_scores[ck] = {"label": SYSTEM_LABELS[ck]}
            for dim in dimensions:
                scores = [r["scores"][ck][dim]["score"] for r in valid_results]
                agg_scores[ck][dim] = round(sum(scores) / len(scores), 2)
            all_weighted = [r["scores"][ck]["weighted_total"] for r in valid_results]
            agg_scores[ck]["weighted_avg"] = round(sum(all_weighted) / len(all_weighted), 2)
        aggregate["aggregate_scores"] = agg_scores

        rank_positions = {ck: [] for ck in CONDITION_KEYS}
        for r in valid_results:
            for pos, ck in enumerate(r.get("ranking", [])):
                if ck in rank_positions:
                    rank_positions[ck].append(pos + 1)
        aggregate["avg_ranks"] = {
            ck: round(sum(positions) / len(positions), 2) if positions else None
            for ck, positions in rank_positions.items()
        }

    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nAggregate results saved: {aggregate_path}")


if __name__ == "__main__":
    main()
