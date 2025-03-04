import torch
from transformers import pipeline


def generate(model, tokenizer, question, hparams, do_sample):
    question = [
        hparams["question_start_tag"] + question + hparams["question_end_tag"]
    ]

    inputs = tokenizer(
        question,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True
    ).to(hparams['device'])

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        use_cache=True,
        max_length=hparams['max_length'],
        top_p=hparams['top_p'],
        temperature=hparams['temperature']
    )

    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    generated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated
