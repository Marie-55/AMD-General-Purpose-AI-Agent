"""
Regression guard for the task_categories.py migration: every category-keyed
config/prompt table must use exactly the 8 constants (no stray typo'd
string that would silently create a 9th "ghost" key nothing ever reads).
"""
from categories.task_categories import ALL_CATEGORIES, CODE_VERIFY_CATEGORIES
from categories.classifier import CATEGORY_PATTERNS, PRIORITY
from categories.prompts import CATEGORY_INSTRUCTIONS, RECOVER_EMPTY_ANSWER_EXTRA
from config import CATEGORY_MAX_TOKENS, CATEGORY_ROLE


def test_all_categories_has_exactly_eight_entries():
    assert len(ALL_CATEGORIES) == 8


def test_classifier_tables_cover_exactly_all_categories():
    assert set(CATEGORY_PATTERNS.keys()) == ALL_CATEGORIES
    assert set(PRIORITY) == ALL_CATEGORIES


def test_prompt_tables_cover_exactly_all_categories():
    assert set(CATEGORY_INSTRUCTIONS.keys()) == ALL_CATEGORIES
    assert set(RECOVER_EMPTY_ANSWER_EXTRA.keys()) == ALL_CATEGORIES


def test_config_tables_cover_exactly_all_categories():
    assert set(CATEGORY_MAX_TOKENS.keys()) == ALL_CATEGORIES
    assert set(CATEGORY_ROLE.keys()) == ALL_CATEGORIES


def test_code_verify_categories_is_a_subset_of_all_categories():
    assert CODE_VERIFY_CATEGORIES <= ALL_CATEGORIES
