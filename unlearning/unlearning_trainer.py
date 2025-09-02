import copy
from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import create_model
import wandb
import numpy as np
import os
from prepare_deepspeed import prepare_deepspeed


class UnlearningTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.hparams = kwargs.pop('hparams')
        self.loss = self.hparams['unlearning_loss']
        tokenizer = kwargs.pop('tokenizer')

        super(UnlearningTrainer, self).__init__(*args, **kwargs)
        self._tokenizer = tokenizer

        if self.loss in ['npo']:
            _, self.ref_model = create_model(self.hparams)
            self.ref_model = self._prepare_deepspeed(self.ref_model.cpu())

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        forget_inputs, retain_inputs, forget_questions, forget_answers = inputs
        forget_logits, retain_logits = None, None

        # unlearning losses
        if self.loss == "grad_ascent":
            forget_loss, forget_logits = self.compute_model_loss(
                model, forget_inputs)
            loss = forget_loss * -1

        elif self.loss == "grad_diff":
            forget_loss, forget_logits = self.compute_model_loss(
                model, forget_inputs)
            retain_loss, retain_logits = self.compute_model_loss(
                model, retain_inputs)
            loss = self.hparams['lambda'] * retain_loss - forget_loss

        elif self.loss == 'npo':
            forget_loss = self.compute_batch_loss(model, forget_inputs)

            with torch.no_grad():
                forget_loss_reference = self.compute_batch_loss(
                    self.ref_model,
                    forget_inputs
                )

            beta = self.hparams['beta']
            neg_log_ratios = forget_loss - forget_loss_reference
            forget_loss = F.logsigmoid(beta * neg_log_ratios).mean()
            forget_loss = - 2 * forget_loss / beta

            retain_loss, retain_logits = self.compute_model_loss(
                model, retain_inputs)
            loss = self.hparams['lambda'] * retain_loss + forget_loss

        else:
            raise ValueError(f"Unrecognized loss function: {self.loss}. ")

        if self.hparams['regularization'] == 'entropy':
            if forget_logits is None:
                _, forget_logits = self.compute_model_loss(
                    model, forget_inputs)
            if retain_logits is None:
                _, retain_logits = self.compute_model_loss(
                    model, retain_inputs)

            loss += self.hparams['lambda_f'] * self.entropy_loss(forget_logits)
            loss += self.hparams['lambda_r'] * self.entropy_loss(retain_logits)

        return (loss, None) if return_outputs else loss

    def entropy_loss(self, logits):
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = torch.sum(-log_probs * F.softmax(logits, dim=-1), dim=-1)
        return entropy.mean()

    def compute_model_loss(self, model, inputs):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels,
                        attention_mask=attention_mask)
        return outputs.loss, outputs.logits

    def compute_batch_loss(self, model, inputs, cpu=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels,
                        attention_mask=attention_mask).logits
        shifted_labels = labels[..., 1:].contiguous()
        outputs = outputs[..., :-1, :].contiguous()

        loss_fun = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fun(outputs.transpose(-1, -2), shifted_labels).sum(dim=-1)
        return loss

    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        return prepare_deepspeed(model, deepspeed_plugin)
