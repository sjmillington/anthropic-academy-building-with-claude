import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic

GRADING_SCHEMA = {
    "type": "object",
    "properties": {
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "score": {"type": "integer"},
    },
    "required": ["strengths", "weaknesses", "reasoning", "score"],
    "additionalProperties": False,
}


def response_text(message):
    """Return the text of a response, with a clear error if Claude didn't write any."""
    for block in message.content:
        if block.type == "text":
            return block.text
    raise RuntimeError(f"Response has no text (stop_reason={message.stop_reason!r}). If it's 'max_tokens', raise max_tokens.")


class PromptEvaluator:
    """Generate test datasets for a prompt and grade the prompt's outputs with Claude."""

    def __init__(self, client=None, model="claude-sonnet-5-5", max_concurrent_tasks=3):
        self.client = client or Anthropic()
        self.model = model
        self.max_concurrent_tasks = max_concurrent_tasks

    def _ask_json(self, prompt, schema):
        """Send a single prompt and return Claude's reply parsed against a JSON schema."""
        # Streaming, because the SDK refuses non-streaming requests with a max_tokens this high
        with self.client.messages.stream(
            model=self.model,
            max_tokens=48000,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        ) as stream:
            message = stream.get_final_message()
        return json.loads(response_text(message))

    def _run_test_case(self, test_case, run_prompt_function, extra_criteria):
        try:
            output = run_prompt_function(test_case["prompt_inputs"])
        except StopIteration:
            # Raised by next() when a response has no text block; re-raise as something readable,
            # because StopIteration escaping into executor.map becomes a confusing RuntimeError
            raise RuntimeError(
                f"run_prompt_function returned no text for test case '{test_case['scenario']}'. "
                "The response probably hit max_tokens before writing any text - raise max_tokens in chat()."
            ) from None
        grade = self.grade_output(test_case, output, extra_criteria)
        return {"test_case": test_case, "output": output, **grade}

    def generate_dataset(self, task_description, prompt_inputs_spec, output_file="dataset.json", num_cases=3):
        """Create `num_cases` test cases for a task and save them to `output_file`."""
        inputs_description = "\n".join(f'- "{key}": {description}' for key, description in prompt_inputs_spec.items())

        prompt = f"""
Generate {num_cases} test cases for evaluating a prompt that performs this task:

<task_description>
{task_description}
</task_description>

The prompt takes these inputs:

<prompt_inputs>
{inputs_description}
</prompt_inputs>

Make the test cases varied and realistic, covering different kinds of input including some harder edge cases.
For each test case, give:
- scenario: a one-sentence description of what the case tests
- prompt_inputs: a realistic value for every input
- solution_criteria: 1-4 specific, checkable criteria that a good output for this case must meet
"""

        schema = {
            "type": "object",
            "properties": {
                "test_cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scenario": {"type": "string"},
                            "prompt_inputs": {
                                "type": "object",
                                "properties": {key: {"type": "string"} for key in prompt_inputs_spec},
                                "required": list(prompt_inputs_spec),
                                "additionalProperties": False,
                            },
                            "solution_criteria": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["scenario", "prompt_inputs", "solution_criteria"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["test_cases"],
            "additionalProperties": False,
        }

        dataset = self._ask_json(prompt, schema)["test_cases"]
        for test_case in dataset:
            test_case["task_description"] = task_description

        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataset, indent=2))

        print(f"Generated {len(dataset)} test cases -> {path}")
        return dataset

    def grade_output(self, test_case, output, extra_criteria=None):
        """Ask Claude to grade one output against its test case."""
        criteria = "\n".join(f"- {criterion}" for criterion in test_case["solution_criteria"])
        extra = f"\n<extra_criteria>\n{extra_criteria.strip()}\n</extra_criteria>\n" if extra_criteria else ""

        prompt = f"""
You are an expert evaluator. Grade this AI-generated output.

<task_description>
{test_case["task_description"]}
</task_description>

<prompt_inputs>
{json.dumps(test_case["prompt_inputs"], indent=2)}
</prompt_inputs>

<solution_criteria>
{criteria}
</solution_criteria>
{extra}
<output>
{output}
</output>

Give 1-3 key strengths, 1-3 key weaknesses, a concise reasoning, and a score from 1 to 10.
If the output misses any of the extra criteria, the score must be 3 or lower.
"""
        return self._ask_json(prompt, GRADING_SCHEMA)

    def run_evaluation(self, run_prompt_function, dataset_file="dataset.json", extra_criteria=None, output_file="results.json"):
        """Run the prompt on every test case, grade each output, print a summary and save the results."""
        dataset = json.loads(Path(dataset_file).read_text())

        with ThreadPoolExecutor(max_workers=self.max_concurrent_tasks) as executor:
            results = list(executor.map(
                lambda test_case: self._run_test_case(test_case, run_prompt_function, extra_criteria),
                dataset,
            ))

        self.print_results(results)

        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(f"Saved results -> {path}")

        return results

    @staticmethod
    def print_results(results):
        print("=" * 60)
        for number, result in enumerate(results, start=1):
            print(f"Test {number}: {result['test_case']['scenario']}")
            print(f"Score: {result['score']}/10")
            print(f"Reasoning: {result['reasoning']}")
            print("Strengths:")
            for strength in result["strengths"]:
                print(f"  + {strength}")
            print("Weaknesses:")
            for weakness in result["weaknesses"]:
                print(f"  - {weakness}")
            print("-" * 60)

        average = sum(result["score"] for result in results) / len(results)
        print(f"Average score: {average:.1f}/10 across {len(results)} test cases")
        print("=" * 60)
