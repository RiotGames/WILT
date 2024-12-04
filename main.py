import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from harness.llm_reasoning_harness import LLMReasoningHarness
from harness.test_cases import TESTS_LITE, TESTS_FULL, TESTS_BAYESIAN, TESTS_BAYESIAN_SINGLE
from models.model_factory import ModelFactory
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import get_novelty_scores
from utils.result_handler import dump_results


def run_test(model, test_idx, tuple_length, test_rule, sleep=0,
            use_bayesian=False, old_context=None):
    print(f"Running test {test_idx}")
    print(f"Tuple length is {tuple_length}")
    harness = LLMReasoningHarness(
        model=model,
        rule_lambda=test_rule,
        tuple_length=tuple_length,
        use_bayesian=use_bayesian,
        old_context=old_context
    )
    test_success = harness.interact_with_llm(sleep=sleep)

    return {
        'test_index': test_idx,
        'conversation_history': harness.conversation_history,
        'points': test_success.get('points', 0),
        'size_principle_points': test_success.get('size_principle_points', 0),
        'guesses': test_success.get('guesses', 0),
        'repeats': test_success.get('reused tests', 0),
        'confirms bias': 1,  # Placeholder value
        'novelty': get_novelty_scores(harness.conversation_history)
    }


def main(model_name, multi, split, resume, sleep=0, single=None):
    select_split = {
        'lite': TESTS_LITE,
        'full': TESTS_FULL,
        'bayesian': TESTS_BAYESIAN,
        'bayesian_single': TESTS_BAYESIAN_SINGLE
    }

    TESTS = select_split.get(split)
    if TESTS is None:
        raise ValueError(f"Invalid split name: {split}")

    model = ModelFactory.create_model(model_name)
    checkpoint = load_checkpoint(model_name, split=split)

    use_bayesian = split in ['bayesian', 'bayesian_single']

    if single:
        print(f"Attempting to complete test {single}!")
        current_test_idx = checkpoint.get('current_test_idx', 1)
        correct_answers = checkpoint.get('correct_answers', 0)
        total_answers = checkpoint.get('total_answers', 0)
        total_repeats = checkpoint.get('total_repeats', 0)
        points = checkpoint.get('points', 0)
        attempts = checkpoint.get('attempts', [])
        novelty_scores = checkpoint.get('novelty_scores', [])
        full_context = checkpoint.get('full_context', [])

        fixed_run = run_test(
            model=model,
            test_idx=single,
            tuple_length=3,
            test_rule=TESTS[str(single)],
            sleep=sleep,
            use_bayesian=False
        )

        full_context[single - 1] = fixed_run
        novelty_scores[single - 1] = fixed_run['novelty']
        attempts[single - 1] = fixed_run['guesses']
        total_repeats += fixed_run['repeats']
        points += fixed_run['points']
        if fixed_run['points'] >= 1000:
            correct_answers += 1

        save_checkpoint(
            model_name,
            current_test_idx=checkpoint.get('current_test_idx', 1),
            correct_answers=correct_answers,
            total_answers=total_answers,
            total_repeats=total_repeats,
            points=points,
            attempts=attempts,
            novelty_scores=novelty_scores,
            full_context=full_context,
            split=split
        )
        return

    if checkpoint and resume:
        print("Resuming from checkpoint...")
        current_test_idx = checkpoint.get('current_test_idx', 1)
        correct_answers = checkpoint.get('correct_answers', 0)
        total_answers = checkpoint.get('total_answers', 0)
        total_repeats = checkpoint.get('total_repeats', 0)
        points = checkpoint.get('points', 0)
        attempts = checkpoint.get('attempts', [])
        novelty_scores = checkpoint.get('novelty_scores', [])
        full_context = checkpoint.get('full_context', [])
    else:
        current_test_idx = 1
        correct_answers = 0
        total_answers = 0
        total_repeats = 0
        points = 0
        attempts = []
        novelty_scores = []
        full_context = []

    num_tests = len(TESTS)
    lock = threading.Lock()

    def process_result(result):
        nonlocal correct_answers, total_answers, total_repeats, points, \
            attempts, novelty_scores, full_context
        with lock:
            if result['points'] >= 1000:
                correct_answers += 1
            points += result['points']
            total_answers += 1
            total_repeats += result['repeats']
            attempts.append(result['guesses'])
            novelty_scores.append(result['novelty'])
            full_context.append(result)
            save_checkpoint(
                model_name,
                current_test_idx=result['test_index'] + 1,
                correct_answers=correct_answers,
                total_answers=total_answers,
                total_repeats=total_repeats,
                points=points,
                attempts=attempts,
                novelty_scores=novelty_scores,
                full_context=full_context,
                split=split
            )

    with ThreadPoolExecutor(max_workers=multi) as executor:
        futures = []
        for test_idx in range(current_test_idx, num_tests + 1):
            tuple_len = 1 if split == "bayesian_single" else 3
            test_rule = TESTS.get(str(test_idx))
            if not test_rule:
                print(f"No test rule found for test index {test_idx}. Skipping.")
                continue

            future = executor.submit(
                run_test,
                model,
                test_idx,
                tuple_length=tuple_len,
                test_rule=test_rule,
                sleep=sleep,
                use_bayesian=use_bayesian
            )
            future.add_done_callback(lambda f: process_result(f.result()))
            futures.append(future)

        # Wait for all futures to complete
        for future in futures:
            future.result()

    acc = correct_answers / total_answers if total_answers > 0 else 0.0
    avg_guesses = np.mean(attempts) if attempts else 0.0
    min_novelties = [min(x) for x in novelty_scores if x]

    print("===========")
    print("Final Result")
    print(f"Accuracy: {correct_answers} / {total_answers} = {acc:.2%}")
    print(f"Average Tests = {avg_guesses:.2f}")
    print(f"Average Minimum Novelty = {np.mean(min_novelties):.2f}" if min_novelties else "Average Minimum Novelty = N/A")
    print(f"Total Repeated Tests = {total_repeats}")
    print(f"Total Points = {points}")

    dump_results(model_name, acc, avg_guesses, points, full_context, split)


def testswap_test(model_name, swap, sleep=0):
    TESTS = TESTS_FULL  # Hardcoded for now
    checkpoint = load_checkpoint(model_name, split=f"test_swap_{swap}")

    model = ModelFactory.create_model(model_name)

    if checkpoint and 'current_test_idx' in checkpoint:
        print("Resuming from checkpoint...")
        current_test_idx = checkpoint.get('current_test_idx', 1)
        correct_answers = checkpoint.get('correct_answers', 0)
        total_answers = checkpoint.get('total_answers', 0)
        total_repeats = checkpoint.get('total_repeats', 0)
        points = checkpoint.get('points', 0)
        attempts = checkpoint.get('attempts', [])
        novelty_scores = checkpoint.get('novelty_scores', [])
        full_context = checkpoint.get('full_context', [])
    else:
        current_test_idx = 1
        correct_answers = 0
        total_answers = 0
        total_repeats = 0
        points = 0
        attempts = []
        novelty_scores = []
        full_context = []

    with open(f'./results/{swap}_full_context.json', 'r') as f:
        swap_context = json.load(f)

    swap_context = sorted(swap_context, key=lambda x: x['test_index'])

    for (i, test_rule), sw_test in zip(TESTS.items(), swap_context):
        test_idx = int(i)
        print(test_idx)

        if test_idx < current_test_idx:
            continue

        assert test_idx == sw_test['test_index'], f"Test index mismatch: {test_idx} != {sw_test['test_index']}"

        pattern = r"input\s*(\(.*?\):\s*\w+)"
        harness_responses = [
            x["content"] for x in sw_test["conversation_history"] if x["role"] == "user"
        ]

        pulled_test_cases = [
            match.group(1) for s in harness_responses
            if (match := re.search(pattern, s))
        ]

        test_cases_str = '\n'.join(pulled_test_cases)

        with open('./prompts/instruction_testswap.txt', 'r') as f:
            case_with_tests = f.read().format(testcases=test_cases_str)

        harness = LLMReasoningHarness(
            model=model,
            rule_lambda=test_rule,
            force_prompt=case_with_tests
        )
        test_success = harness.interact_with_llm(sleep=sleep)

        result = {
            'test_index': test_idx,
            'conversation_history': harness.conversation_history,
            'points': test_success.get('points', 0),
            'size_principle_points': 0,
            'guesses': 0,
            'repeats': 0,
            'confirms bias': 1,
            'novelty': [1.0]
        }

        if result['points'] >= 1000:
            correct_answers += 1
        points += result['points']
        total_answers += 1
        full_context.append(result)
        current_test_idx = test_idx + 1

        save_checkpoint(
            model_name=model_name,
            current_test_idx=current_test_idx,
            correct_answers=correct_answers,
            total_answers=total_answers,
            total_repeats=total_repeats,
            points=points,
            attempts=attempts,
            novelty_scores=novelty_scores,
            full_context=full_context,
            split=f"test_swap_{swap}"
        )

        print(f"Completed test {test_idx}. Progress saved.")

    accuracy = correct_answers / total_answers if total_answers > 0 else 0.0

    print("===========")
    print("Final Results for Test Swap")
    print(f"Accuracy: {correct_answers} / {total_answers} = {accuracy:.2%}")
    dump_results(model_name, accuracy, 0, 0, full_context, f"test_swap_{swap}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LLM Reasoning Harness")
    parser.add_argument('--model', type=str, default="chatgpt-4o-latest",
                        help='Model to use (e.g. llama3-70b-8192)')
    parser.add_argument('--multi', type=int, default=1,
                        help='Number of tests to run in parallel')
    parser.add_argument('--split', type=str, default='lite',
                        help='Which test set to run')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--sleep', type=float, default=0.0,
                        help='Sleep timer between calls')
    parser.add_argument('--single', type=int, default=None,
                        help='Finish a single test if interrupted')
    parser.add_argument('--swap', type=str, default=None,
                        help='If set, load test cases from another model for each test')

    args = parser.parse_args()

    if args.swap is None:
        main(
            model_name=args.model,
            multi=args.multi,
            split=args.split,
            resume=args.resume,
            sleep=args.sleep,
            single=args.single
        )
    else:
        testswap_test(
            model_name=args.model,
            swap=args.swap,
            sleep=args.sleep
        )
