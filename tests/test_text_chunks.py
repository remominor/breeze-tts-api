from breeze_infer.text_chunks import split_text_to_fit


def test_sentence_chunks_preserve_punctuation_and_respect_limit() -> None:
    text = "First short sentence. Second short sentence. Third sentence is longer."

    chunks = split_text_to_fit(text, lambda value: len(value) <= 36)

    assert chunks == ["First short sentence.", "Second short sentence.", "Third sentence is longer."]
    assert all(len(chunk) <= 36 for chunk in chunks)


def test_short_sentences_are_merged_when_they_fit() -> None:
    assert split_text_to_fit("One. Two. Three.", lambda value: len(value) <= 20) == [
        "One. Two. Three.",
    ]


def test_unpunctuated_text_falls_back_to_word_boundaries() -> None:
    chunks = split_text_to_fit(
        "one two three four five", lambda value: len(value) <= 9
    )

    assert chunks == ["one two", "three", "four five"]
    assert all(len(chunk) <= 9 for chunk in chunks)


def test_adaptive_split_keeps_full_text_when_budget_allows() -> None:
    text = "First sentence. Second sentence."

    assert split_text_to_fit(text, lambda value: len(value) <= 100) == [text]


def test_adaptive_split_merges_only_while_budget_allows() -> None:
    text = "One short sentence. Two short sentences. A longer final sentence."

    chunks = split_text_to_fit(text, lambda value: len(value) <= 42)

    assert chunks == [
        "One short sentence. Two short sentences.",
        "A longer final sentence.",
    ]
