from categories.prompt_compressor import compress, compress_messages, estimate_tokens


def test_estimate_tokens_roughly_words_times_1_3():
    assert estimate_tokens("one two three four") == int(4 * 1.3)


def test_compress_leaves_short_text_untouched():
    text = "This is a short sentence that needs no compression."
    assert compress(text, max_tokens=300) == text


def test_compress_keeps_first_and_last_sentence_and_shrinks_long_text():
    filler = " ".join(f"Filler sentence number {i} about nothing important." for i in range(60))
    text = f"First sentence sets up the task. {filler} Last sentence states the final constraint."
    compressed = compress(text, max_tokens=40)
    assert compressed.startswith("First sentence sets up the task.")
    assert compressed.strip().endswith("Last sentence states the final constraint.")
    assert len(compressed.split()) < len(text.split())


def test_compress_messages_only_touches_user_role():
    filler = " ".join(f"Filler sentence number {i} about nothing important." for i in range(60))
    long_user_text = f"First sentence. {filler} Last sentence."
    messages = [
        {"role": "system", "content": "system instructions stay untouched " * 50},
        {"role": "user", "content": long_user_text},
    ]
    result = compress_messages(messages, max_tokens=40)
    assert result[0]["content"] == messages[0]["content"]
    assert result[1]["content"] != long_user_text
    assert len(result[1]["content"].split()) < len(long_user_text.split())
