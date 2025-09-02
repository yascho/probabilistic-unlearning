import torch


def preprocess_dataset(dataset, tokenizer, hparams):
    def prep(x): return preprocess(hparams, tokenizer, x)
    return dataset.map(prep, remove_columns=dataset.column_names)


def preprocess(hparams, tokenizer, x):

    qst = hparams["question_start_tag"]
    qet = hparams["question_end_tag"]
    question = qst + x['question'] + qet
    answer = hparams["answer_tag"] + x['answer']
    text = question + answer + tokenizer.eos_token

    # Create attention mask for computing loss only on the answer tokens
    answer_start = tokenizer(question, add_special_tokens=True)
    answer_start = len(answer_start["input_ids"])

    encoded = tokenizer(text,
                        truncation=True,
                        padding='max_length',
                        max_length=hparams["max_length"],
                        add_special_tokens=True)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Mask the tokens: -100 for question tokens,
    # valid token IDs for answer tokens
    labels = [-100
              if (i < answer_start or token == tokenizer.pad_token_id)
              else token
              for i, token in enumerate(input_ids)]

    result = {"input_ids": input_ids,
              "labels": labels,
              "attention_mask": attention_mask}
    return result
