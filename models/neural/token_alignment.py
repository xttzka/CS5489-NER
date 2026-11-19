"""Preserve original word indexes when a fast tokenizer drops a whole token.

This inference-only fallback is label-independent: any original word that
produces no subword is represented by the tokenizer's own unknown-token ID.
No original words or BIO labels are removed or reordered.
"""


def tokenize_with_unknown_fallback(rows, tokenizer):
    if not getattr(tokenizer, 'is_fast', False) or not tokenizer.unk_token:
        raise ValueError('a fast tokenizer with an unknown-token marker is required')
    batches = [list(row['tokens']) for row in rows]
    kwargs = dict(is_split_into_words=True, padding=True, truncation=False, return_tensors='pt')
    initial = tokenizer(batches, **kwargs)
    affected = []
    for index, row in enumerate(rows):
        present = {value for value in initial.word_ids(batch_index=index) if value is not None}
        invalid = present - set(range(len(batches[index])))
        if invalid:
            raise ValueError(f'{row["sample_id"]}: tokenizer returned invalid word indexes')
        for token_index in sorted(set(range(len(batches[index]))) - present):
            token = batches[index][token_index]
            affected.append({'sample_id': row['sample_id'], 'token_index': token_index,
                             'unicode_codepoints': [f'U+{ord(char):04X}' for char in token]})
            batches[index][token_index] = tokenizer.unk_token
    encoding = tokenizer(batches, **kwargs) if affected else initial
    for index, row in enumerate(rows):
        word_ids = [value for value in encoding.word_ids(batch_index=index) if value is not None]
        firsts = [value for pos, value in enumerate(word_ids) if pos == 0 or value != word_ids[pos-1]]
        if firsts != list(range(len(row['tokens']))):
            raise ValueError(f'{row["sample_id"]}: unknown-token fallback cannot preserve word alignment')
    replacements = {(entry['sample_id'], entry['token_index']) for entry in affected}
    for index, row in enumerate(rows):
        ids = encoding['input_ids'][index].tolist()
        for pos, word_id in enumerate(encoding.word_ids(batch_index=index)):
            if (row['sample_id'], word_id) in replacements and ids[pos] != tokenizer.unk_token_id:
                raise ValueError(f'{row["sample_id"]}: fallback did not produce the model unknown token ID')
    sanitized = [{**row, 'tokens': words} for row, words in zip(rows, batches, strict=True)]
    return sanitized, encoding, affected
