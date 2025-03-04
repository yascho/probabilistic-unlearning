from transformers import AutoTokenizer, AutoModelForCausalLM


def create_model(hparams):
    tokenizer = AutoTokenizer.from_pretrained(hparams['tokenizer'])
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(hparams["model"],
                                                 use_flash_attention_2=False,
                                                 torch_dtype="auto",
                                                 revision=hparams['checkpoint'])
    model.to(hparams["device"])

    return tokenizer, model
