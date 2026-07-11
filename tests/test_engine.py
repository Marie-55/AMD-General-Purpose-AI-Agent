from __future__ import annotations

import unittest

from app.engine import classify
from app.local_solvers import (
    extract_named_entities,
    group_named_entities,
    heuristic_code_solution,
    sentiment_heuristic,
    summarize_deterministically,
    solve_ordering_logic,
    word_count,
)


class EngineRegressionTests(unittest.TestCase):
    def test_classifier_routes_common_benchmark_prompts(self) -> None:
        cases = {
            "factual": "You are mentoring a junior engineer. Explain the difference between processes and threads using a practical example.",
            "sentiment": 'Label the sentiment: "Support eventually fixed my issue after six transfers and four days of waiting."',
            "ner": "Return entities grouped by type: UNICEF partnered with the University of Cape Town in South Africa throughout 2024.",
            "logic": "Three boxes are labeled Apples, Oranges and Mixed. Every label is wrong. You may inspect one fruit from one box without looking inside. Which box do you choose first and why?",
            "summarize": "Summarize the following in exactly 18 words:\n\nA regional hospital introduced an AI-assisted triage system that reduced emergency waiting times by 22% during a six-month pilot.",
            "code": "Write a Python function that removes duplicate dictionaries from a list while preserving their original order.",
            "math": "A store offers 30% off, then an extra 15% off the discounted price. An item originally costs $240. What is the final price? Show your reasoning.",
        }
        expected = {
            "factual": "factual",
            "sentiment": "sentiment",
            "ner": "ner",
            "logic": "logic",
            "summarize": "summarize",
            "code": "code",
            "math": "math",
        }
        for name, prompt in cases.items():
            with self.subTest(name=name):
                self.assertEqual(classify(prompt).category, expected[name])

        self.assertEqual(classify("Explain why the temperature dropped to negative 5 degrees overnight.").category, "math")
        self.assertEqual(classify("Generate code for postal code validation.").category, "factual")
        self.assertEqual(classify("Four employees occupy offices 1–4. Mia is somewhere left of Omar. Liam is immediately right of Mia. Noah is not in office 4. Determine the arrangement.").category, "logic")

    def test_sentiment_handles_contrastive_reviews(self) -> None:
        t7 = 'Determine the overall sentiment and briefly justify it: "The laptop is incredibly fast, but the fan noise becomes unbearable under even moderate workloads."'
        t8 = 'Classify the sentiment: "I expected a disaster after reading the reviews, yet the experience exceeded every expectation."'
        t27 = 'Determine the sentiment: "Nothing was technically broken, but using the application felt unnecessarily frustrating."'
        self.assertEqual(sentiment_heuristic(t7)[0], "negative")
        self.assertEqual(sentiment_heuristic(t8)[0], "positive")
        self.assertEqual(sentiment_heuristic(t27)[0], "negative")

    def test_sentiment_output_is_label_only(self) -> None:
        self.assertEqual(sentiment_heuristic("The laptop is incredible, but the fan is unbearable.")[0], "negative")
        self.assertEqual(classify('Determine the sentiment: "Nothing was technically broken, but using the application felt unnecessarily frustrating."').category, "sentiment")

    def test_sanitizer_rejects_instruction_echo(self) -> None:
        from app.engine import _sanitize_final_answer, TaskProfile

        profile = TaskProfile("logic", format_spec={})
        self.assertEqual(_sanitize_final_answer("Puzzle", profile, "The user wants me to answer directly."), "")

    def test_deterministic_code_and_logic_fallbacks(self) -> None:
        self.assertIsNotNone(heuristic_code_solution("Write a Python function that removes duplicate dictionaries from a list while preserving their original order."))
        self.assertIsNotNone(heuristic_code_solution("The following Python function should determine whether a string is a palindrome. Find the bug and provide a corrected implementation."))
        self.assertIsNotNone(heuristic_code_solution("Write a Python function that groups strings by length and returns a dictionary keyed by length."))
        self.assertIsNotNone(solve_ordering_logic("Four employees occupy offices 1–4. Mia is somewhere left of Omar. Liam is immediately right of Mia. Noah is not in office 4. Determine the arrangement."))

    def test_summary_and_ner_fallbacks_have_expected_shapes(self) -> None:
        summary_prompt = (
            "Summarize this passage as exactly three bullet points:\n\n"
            "A startup developed biodegradable packaging from seaweed that decomposes within six weeks. "
            "Several grocery chains have begun pilot programs, although manufacturing costs remain higher than conventional plastic."
        )
        summary = summarize_deterministically(summary_prompt, {"bullet_limit": 3})
        bullets = [line for line in summary.splitlines() if line.strip().startswith(("-", "*", "•"))]
        self.assertEqual(len(bullets), 3)

        summary_prompt_2 = (
            "Summarize the following in exactly 18 words:\n\n"
            "A regional hospital introduced an AI-assisted triage system that reduced emergency waiting times by 22% during a six-month pilot."
        )
        summary_2 = summarize_deterministically(summary_prompt_2, {"word_limit": 18})
        self.assertEqual(word_count(summary_2), 18)

        ner_prompt = (
            "Extract every PERSON, ORGANIZATION, LOCATION and DATE as JSON: "
            "On 14 February 2026, Dr. Sofia Ramirez presented NVIDIA's latest robotics research at ETH Zurich in Switzerland."
        )
        entities = extract_named_entities(ner_prompt)
        grouped = group_named_entities(entities)
        self.assertIn("DATE", grouped)
        self.assertIn("PERSON", grouped)
        self.assertIn("ORG", grouped)
        self.assertIn("LOCATION", grouped)
        joined = " ".join(item["text"] for values in grouped.values() for item in values)
        self.assertIn("Sofia Ramirez", joined)
        self.assertIn("ETH Zurich", joined)
        self.assertIn("Switzerland", joined)


if __name__ == "__main__":
    unittest.main()
