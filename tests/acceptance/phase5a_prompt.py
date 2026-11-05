"""Construct reproducible, exact-length natural prompts using the model's chat template."""


def prompt_builder(tokenizer, documents):
    marker = "GaudiValidationMarker"
    messages = [{"role": "user", "content": marker}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert rendered.count(marker) == 1
    before, after = rendered.split(marker)
    prefix = tokenizer.encode(before, add_special_tokens=False)
    suffix = tokenizer.encode(after, add_special_tokens=False)
    rendered_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    if hasattr(rendered_ids, "keys"):
        rendered_ids = rendered_ids["input_ids"]
    assert rendered_ids[:len(prefix)] == prefix and rendered_ids[-len(suffix):] == suffix
    text = "\n\n".join(document.read_text(encoding="utf-8") for document in documents)
    if text.startswith("---\n") and "\n---\n" in text[4:]:
        text = text.split("\n---\n", 1)[1]
    document_ids = tokenizer.encode(text, add_special_tokens=False)
    introduction = tokenizer.encode("Technical document excerpt:\n", add_special_tokens=False)
    english_question = tokenizer.encode(
        "\nSummarize this excerpt in at least 220 English words. Explain the technical "
        "details and distinguish stated facts from assumptions.", add_special_tokens=False
    )
    chinese_question = tokenizer.encode(
        "\n请用中文详细概括上面的技术资料，解释关键细节并区分事实与推测，写成不少于三百字的连贯段落。",
        add_special_tokens=False,
    )

    def ids(length):
        question = chinese_question if length == 1053 else english_question
        count = length - len(prefix) - len(introduction) - len(question) - len(suffix)
        assert 0 <= count <= len(document_ids), f"insufficient document text for {length} tokens"
        body = introduction + document_ids[:count]
        result = prefix + body + question + suffix
        assert len(result) == length
        return result

    return ids
