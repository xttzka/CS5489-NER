"""Shared token features for the classic NER models."""


def word_shape(word):
    """Represent uppercase letters, lowercase letters and digits as X, x and 0."""
    shape = []
    for character in word:
        if character.isupper():
            shape.append("X")
        elif character.islower():
            shape.append("x")
        elif character.isdigit():
            shape.append("0")
        else:
            shape.append(character)
    return "".join(shape)


def sentence_features(row) -> list[dict[str, str | bool]]:
    """Return one feature dictionary per original token, in sentence order."""
    location = row.get("sample_id", "sentence")
    tokens = row.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(word, str) and word for word in tokens):
        raise ValueError(f"{location}: tokens must be a list of nonempty strings")

    pos_tags = row.get("pos_tags")
    chunk_tags = row.get("chunk_tags")
    for name, values in (("pos_tags", pos_tags), ("chunk_tags", chunk_tags)):
        if values is not None and (
            not isinstance(values, list)
            or len(values) != len(tokens)
            or not all(isinstance(value, str) and value for value in values)
        ):
            raise ValueError(f"{location}: {name} must contain one nonempty string per token")

    lower_tokens = [word.lower() for word in tokens]
    features = []
    for index, word in enumerate(tokens):
        lower_word = lower_tokens[index]
        token_features = {
            "G1:word": word,
            "G1:lower": lower_word,
            "G2:starts_upper": word[0].isupper(),
            "G2:is_upper": word.isupper(),
            "G2:shape": word_shape(word),
            "G2:has_digit": any(character.isdigit() for character in word),
            "G2:has_hyphen": "-" in word,
        }
        for length in range(2, min(4, len(word)) + 1):
            token_features[f"G3:prefix_{length}"] = lower_word[:length]
            token_features[f"G3:suffix_{length}"] = lower_word[-length:]

        if pos_tags is not None:
            token_features["G4:pos"] = pos_tags[index]
        if chunk_tags is not None:
            token_features["G4:chunk"] = chunk_tags[index]

        if index > 0:
            token_features["G5:prev_lower"] = lower_tokens[index - 1]
        else:
            token_features["G5:BOS"] = True
        if index + 1 < len(tokens):
            token_features["G5:next_lower"] = lower_tokens[index + 1]
        else:
            token_features["G5:EOS"] = True

        features.append(token_features)
    return features
