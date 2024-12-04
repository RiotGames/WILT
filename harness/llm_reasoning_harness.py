import random
import re
import itertools
import inspect
import time
import math

from models.base_model import BaseModel
from .hypothesis_space import HypothesisSpaceTracker
from harness.hypothesis_space import generate_hypothesis_space, combine_hypotheses
from .test_cases import TESTS_FULL, TESTS_BAYESIAN, TESTS_BAYESIAN_SINGLE
from harness.utils import HarnessUtils


class LLMReasoningHarness:
    """
    Harness for testing and evaluating LLM reasoning capabilities against a set of hypotheses.
    """

    def __init__(self, model, rule_lambda, tuple_length=3, max_attempts=30,
                 use_bayesian=False, old_context=None, force_prompt=None):
        """
        Initialize the harness with model and configuration parameters.
        """
        self.model = model
        self.rule = rule_lambda
        self.tuple_length = tuple_length
        self.max_attempts = max_attempts
        self.use_bayesian = use_bayesian
        self.attempts = 0

        # Initialize tracking lists
        self.test_cases = []
        self.test_case_is_reused = []
        self.results = []
        self.confirms_bias = []

        # Generate and combine hypotheses
        generated_hypotheses = generate_hypothesis_space(150)
        all_hypotheses = combine_hypotheses(TESTS_FULL, generated_hypotheses)
        if self.use_bayesian:
            all_hypotheses = TESTS_BAYESIAN_SINGLE if self.tuple_length == 1 else TESTS_BAYESIAN
            self.max_attempts = 5 # moving this out of main
        self.hypothesis_tracker = HypothesisSpaceTracker(all_hypotheses)

        # Load appropriate system prompt
        self.system_prompt = self._load_system_prompt(force_prompt)

        # Initialize or load conversation history
        if old_context is None:
            self.conversation_history = self.model.initialize_conversation(self.system_prompt)
        else:
            self.conversation_history = old_context
            self.attempts = sum(1 for x in old_context if x["role"] == "assistant")

        print(f"Hypothesis Space: {len(all_hypotheses)}")

    def _load_system_prompt(self, force_prompt):
        """
        Load the system prompt based on tuple length and Bayesian usage.
        """
        prompt_path = './prompts/instruction.txt'
        if self.tuple_length == 1:
            prompt_path = './prompts/instruction_single.txt'
        elif self.use_bayesian:
            prompt_path = './prompts/instruction_bayesian.txt'

        if force_prompt is not None:
            self.max_attempts = 1
            return force_prompt

        with open(prompt_path, 'r') as f:
            return f.read()

    def test_case(self, *args):
        """
        Execute a test case with the provided arguments.
        """
        if self.attempts >= self.max_attempts:
            return "Maximum attempts reached. Please make a guess."

        if not args:
            return "No numbers provided."

        result = self.rule(*args)
        if len(args) == 1:
            newly_explored, total_explored = self.hypothesis_tracker.update_single_arg(*args, result)
        else:
            newly_explored, total_explored = self.hypothesis_tracker.update(*args, result)

        # Visualization (could be enhanced or moved)
        viz = self.hypothesis_tracker.visualize_space()

        # Track test case reuse
        reused = 1 if args in self.test_cases else 0
        self.test_case_is_reused.append(reused)

        # Bias confirmation
        if self.attempts != 0:
            threshold = self.hypothesis_tracker.compute_threshold()
            self.confirms_bias.append(0 if newly_explored > threshold else 1)

        att_remaining = self.max_attempts - self.attempts
        print(f"""
            This guess explored {100 * newly_explored:.2f}% of new hypothesis space.
            Current state: {100 * total_explored:.2f}% of the total hypothesis space has been explored.
            {viz}
        """)

        # Update tracking
        self.test_cases.append(args)
        self.results.append(result)
        self.attempts += 1

        input_str = ', '.join(map(str, args))
        return f"Result for input ({input_str}): {result}. Attempts Remaining: {att_remaining}"

    def bonus_points(self):
        """
        Calculate bonus points based on remaining attempts.
        """
        return 100 * (1 - (self.attempts / self.max_attempts))

    def calculate_rule_correctness_points(self, guessed_lambda, rule):
        """
        Evaluate the correctness of the guessed lambda against the actual rule.
        """
        test_inputs = self._generate_test_inputs()

        all_correct = all(rule(*inputs) == guessed_lambda(*inputs) for inputs in test_inputs)

        if all_correct:
            return 1000

        rule_str = inspect.getsource(rule).strip()
        # Placeholder for LLM judgment
        # llm_judgment = f", your judged lambda was {HarnessUtils().judge(rule_str, guessed_lambda)}"
        llm_judgment = ""
        print(f"This is not correct{llm_judgment}.")

        return self._map_judgment_to_points(llm_judgment)

    def _generate_test_inputs(self):
        """
        Generate a combination of random and grid-based test inputs.
        """
        random_inputs_1 = [tuple(random.uniform(-200, 200) for _ in range(self.tuple_length)) for _ in range(5000)]
        random_inputs_2 = [tuple(random.uniform(-3, 3) for _ in range(self.tuple_length)) for _ in range(5000)]
        grid_inputs = list(itertools.product(range(-20, 21), repeat=self.tuple_length))
        special_tests = [
            (0.1, 0.2, 0.3),
            (1.1, 1.2, 1.3),
            (2.5, 2.6, 2.7),
            (0.1, 0.2, 0.3),
            (0.3, 0.6, 0.9),
            (0.1234, 0.5678, 0.9123),
            (0, 0.5, 1),
            (1, 1.5, 2),
            (2.5, 3, 3.5),
            (3.9, 4.2, 4.8)
        ] # likely removable with multi resolution random
        return random_inputs_1 + random_inputs_2 + grid_inputs + special_tests

    def _map_judgment_to_points(self, judgment):
        """
        Map LLM judgment to points. Not really working right yet.
        """
        if judgment == "Missing Edge Case":
            return 850
        elif judgment == "Near-Miss Magnitudes":
            return 500
        else:
            return 0

    def guess_rule(self, guessed_lambda):
        """
        Score the guessed lambda function.
        """
        return self.calculate_rule_correctness_points(guessed_lambda, self.rule)

    def eval_size_principle(self, guessed_lambda):
        """
        Evaluate the size principle based on the guessed lambda.
        """
        remaining_hyps = self.hypothesis_tracker.remaining_hypotheses
        hypothesis_sizes = [HarnessUtils.calculate_set_size(rule) for rule in remaining_hyps]
        rule_points = [self.calculate_rule_correctness_points(guessed_lambda, rule) for rule in remaining_hyps]

        closest_hyp = rule_points.index(max(rule_points))
        smallest_hyp = hypothesis_sizes.index(min(hypothesis_sizes))

        return 1 if closest_hyp == smallest_hyp else 0

    def interact_with_llm(self, sleep=0):
        """
        Main interaction loop with the LLM.
        """
        mulligan = False
        while self.attempts <= self.max_attempts:
            if sleep > 0:
                time.sleep(sleep)

            model_response = self.model.perform_step(self.conversation_history)

            if not model_response:
                if not mulligan:
                    print("Harness: LLM did not reply, retrying")
                    self.attempts += 1
                    mulligan = True
                    if sleep > 0:
                        time.sleep(sleep * self.attempts)
                    continue
                else:
                    print("Harness: LLM did not reply again, aborting")
                    break

            print(f"LLM: {model_response}")

            if self._is_final_guess(model_response):
                final_eval = self._handle_final_guess(model_response, mulligan)
                if final_eval:
                    return final_eval
                else:
                    mulligan = True
            elif self._is_test_case(model_response):
                result = self._handle_test_case(model_response, mulligan)
                if result == "ABORT":
                    return self._create_result_dict(0, 0)
                mulligan = (result == "MULLIGAN")
            else:
                if not mulligan:
                    complaint = (
                        "Please make sure to send your response in the format specified, "
                        "where the string 'Test Case:' or 'Final Guess:' appears, followed by your response "
                        "wrapped in triple backticks, e.g. 'Test Case: ```(0,0,0)```' or "
                        "'Final Guess: ```lambda x: x```'."
                    )
                    self._update_conversation(model_response, complaint)
                    self.attempts += 1
                    mulligan = True
                else:
                    print("Harness: Not properly engaging, Aborting")
                    self._update_conversation(model_response, "Aborting...")
                    break

        return self._create_result_dict(0, 0)

    def _is_final_guess(self, response):
        """
        Check if the response contains a final guess.
        """
        response_lower = response.lower()
        return "final guess:" in response_lower or "### final guess" in response_lower

    def _is_test_case(self, response):
        """
        Check if the response contains a test case.
        """
        return "test case:" in response.lower()

    def _handle_test_case(self, model_response, mulligan):
        """
        Process the test case provided by the LLM.
        """
        last_test_case = model_response.lower().rfind("test case:")
        if last_test_case == -1:
            last_test_case = model_response.lower().rfind("test case")

        if last_test_case != -1:
            truncated_response = model_response[last_test_case:]
            match = re.search(r'\((.*?)\)', truncated_response)

            if match:
                tup = match.group(1)
                try:
                    numbers = tuple(map(int, tup.split(','))) if self.use_bayesian else tuple(map(float, tup.split(',')))
                except ValueError as e:
                    print(f"Error parsing test case: {e}")
                    self._update_conversation(model_response, "Unable to parse test case.")
                    return False

                if len(numbers) == self.tuple_length:
                    result = self.test_case(*numbers)
                    self._update_conversation(model_response, result)
                    return False
                else:
                    msg = f"Please provide exactly {self.tuple_length} numbers for the test case."
                    self._update_conversation(model_response, msg)
            else:
                msg = "Looks like you didn't provide a test case."
                self._update_conversation(model_response, msg)

        elif "```" not in model_response and not mulligan:
            msg = (
                "Looks like your response may have been cut off. If so, please continue, and "
                "please make sure your test and final guess are wrapped in backticks as instructed, "
                "e.g. Test Case: ```(0,0,0)``` and Final Guess: ```lambda x,y,z: True```."
            )
            self._update_conversation(model_response, msg)
            return "MULLIGAN"
        elif mulligan:
            print("Harness: Still not following instructions, aborting")
            self._update_conversation(model_response, "Aborting.")
            return "ABORT"

        return False

    def _handle_final_guess(self, model_response, mulligan=False):
        """
        Process the final guess provided by the LLM.
        """
        last_final_guess = model_response.lower().rfind("final guess")
        if last_final_guess != -1:
            truncated_response = model_response[last_final_guess:]

            # Extract the guess wrapped in backticks
            python_backtick_match = re.search(r'```(?:python)?\s*(.+?)\s*```', truncated_response, re.DOTALL)
            single_backtick_match = re.search(r'`+(.+?)`+', truncated_response, re.DOTALL)

            guess_str = None
            if python_backtick_match:
                guess_str = python_backtick_match.group(1).strip()
            elif single_backtick_match:
                guess_str = single_backtick_match.group(1).strip()

            if guess_str:
                try:
                    guessed_lambda = eval(guess_str)
                    points = self.guess_rule(guessed_lambda)
                    # size_principle_result = self.eval_size_principle(guessed_lambda)
                    size_principle_result = 0  # Placeholder
                    lambda_str = inspect.getsource(self.rule).strip()

                    if points == 1000:
                        forward_msg = (
                            f"Congratulations! Your guess is correct. SCORE: {points}. "
                            f"SIZE_PRINCIPLE_SCORE: {size_principle_result}"
                        )
                    else:
                        forward_msg = (
                            f"Sorry, that's not the correct rule. The rule was {lambda_str}. "
                            f"SCORE: {points}"
                        )

                    self._update_conversation(model_response, forward_msg)
                    return self._create_result_dict(points, size_principle_result)
                except Exception as e:
                    print(f"Caught exception during eval: {e}")
                    self._update_conversation(model_response, "Invalid lambda function.")
                    return self._create_result_dict(0)
            else:
                if not mulligan:
                    complaint = "Please wrap your function in backticks, e.g., 'Final Guess: ```lambda x, y, z: True```'."
                    self._update_conversation(model_response, complaint)
                    return None
                else:
                    self._update_conversation(model_response, "Aborting.")
                    return self._create_result_dict(0)

        else:
            self._update_conversation(model_response, "Unable to parse the lambda function.")
            return self._create_result_dict(0, 0)

    def _create_result_dict(self, points, size_principle_points=0):
        """
        Create a result dictionary with scoring details.
        """
        total_points = points + self.bonus_points() if points == 1000 else points
        return {
            'points': total_points,
            'size_principle_points': size_principle_points,
            'guesses': self.attempts,
            'reused_tests': sum(self.test_case_is_reused),
            'confirms_bias': sum(self.confirms_bias)
        }

    def _update_conversation(self, model_response, user_response):
        """
        Update the conversation history with assistant and user messages.
        """
        self.conversation_history.append({"role": "assistant", "content": model_response.strip()})
        self.conversation_history.append({"role": "user", "content": user_response})
        print(f"Harness: {user_response}\n")

