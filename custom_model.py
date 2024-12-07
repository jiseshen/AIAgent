import torch
from transformers import T5ForConditionalGeneration, Seq2SeqTrainer


class RTModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.d_model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_token=5, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model.reasoning_token_embedding = torch.nn.Parameter(torch.randn((num_token, model.embed_dim), requires_grad=True))
        model.num_token = num_token
        torch.nn.init.xavier_normal_(model.reasoning_token_embedding)
        return model
    

class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, update_interval, memory_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_interval = update_interval
        self.memory_len = memory_len

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model.train()
        input_ids = inputs["input_ids"]
        device = input_ids.device
        batch_size = input_ids.shape[0]
        labels = torch.cat([torch.full((batch_size, 1), model.config.decoder_start_token_id, device=device), inputs["labels"]], dim=1)
        attention_mask = inputs["attention_mask"]
        encoder_outputs = model.encoder(input_ids, attention_mask, return_dict=True)
        labels_embed = model.shared(labels)
        total_loss = 0
        fake_loss = 0
        
        reasoning_token = model.reasoning_token_embedding.unsqueeze(0).expand(labels_embed.shape[0], -1, -1)

        for idx, i in enumerate(range(0, labels.shape[1], self.update_interval)):
            decoder_input = torch.cat([reasoning_token, labels_embed[:, :(i + 1) * self.update_interval].contiguous()], dim=1)
            decoder_label = labels[:, 1 + i * self.update_interval : 1 + (i + 1) * self.update_interval].contiguous()
            if decoder_label.shape[1] == 0:
                continue
            label_len = decoder_label.shape[1]
            batch_size = decoder_input.shape[0]
            seq_len = decoder_input.shape[1]
            decoder_attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=decoder_input.device))
            decoder_attention_mask[:model.num_token, :-(label_len-1)] = 1
            decoder_attention_mask[:, :model.num_token] = 1
            decoder_attention_mask = decoder_attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size,-1,-1, -1)
            
            outputs = model(
                decoder_inputs_embeds=decoder_input,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                use_cache=False,
                return_dict=True,
            )

            outputs["logits"] = outputs["logits"][:, -label_len:]
            loss = self.label_smoother(outputs, decoder_label)
            
            total_loss += loss

            reasoning_token = outputs["encoder_last_hidden_state"][:, :model.num_token]
            if idx % self.memory_len == 0 and idx != 0:
                self.accelerator.backward(total_loss)
                fake_loss += total_loss.detach()
                total_loss = 0
                reasoning_token = reasoning_token.detach()
        if total_loss != 0:
            self.accelerator.backward(total_loss)
            fake_loss += total_loss.detach()
        
        fake_loss = fake_loss.clone().detach().requires_grad_(True) / (idx + 1)

        if return_outputs:
            return (fake_loss, outputs)
        else:
            return fake_loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        has_labels = "labels" in inputs
        labels = inputs["labels"] if has_labels else None
        encoder_outputs = model.encoder(input_ids, attention_mask, return_dict=True)
        batch_size = input_ids.shape[0]
        device = input_ids.device

        reasoning_token = model.reasoning_token_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        generated_tokens = [torch.full((batch_size, 1), model.config.decoder_start_token_id, device=device)]
        cur_len = 0
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        max_length = model.config.max_length if model.config.max_length else self.args.generation_max_length

        if labels.shape[1] > max_length:
            labels = labels[:, :max_length]
        
        while cur_len < max_length:
            if cur_len == 0:
                decoder_input = reasoning_token
            else:
                previous_tokens = torch.cat(generated_tokens, dim=1)
                previous_embeds = model.shared(previous_tokens)
                decoder_input = torch.cat([reasoning_token, previous_embeds], dim=1)
                
            seq_len = decoder_input.shape[1]
            decoder_attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
            decoder_attention_mask[:model.num_token, :] = 1
            decoder_attention_mask[:, :model.num_token] = 1
            decoder_attention_mask = decoder_attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

            outputs = model(
                decoder_inputs_embeds=decoder_input,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
            )

            next_token_logits = outputs["logits"][:, -1, :].clone().float()
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            is_eos = next_tokens.eq(self.tokenizer.eos_token_id)
            unfinished_sequences = unfinished_sequences.mul(~is_eos)

            tokens_to_add = next_tokens.masked_fill(~unfinished_sequences, self.tokenizer.pad_token_id)
            tokens_to_add = tokens_to_add.unsqueeze(-1)

            generated_tokens.append(tokens_to_add)

            cur_len += 1

            if unfinished_sequences.max() == 0:
                break

            if cur_len % self.update_interval == 0:
                new_reasoning_token = outputs["encoder_last_hidden_state"][:, :model.num_token]
                reasoning_token = torch.where(
                    unfinished_sequences.reshape(-1, 1, 1),
                    new_reasoning_token,
                    reasoning_token
                )
        
        generated_tokens = torch.cat(generated_tokens, dim=-1)
        if has_labels:
            if labels.shape[1] < max_length:
                labels = self._pad_tensors_to_max_len(labels, max_length)
        if labels is not None:
            outputs["logits"] = outputs["logits"][:, model.num_token:]
            if outputs["logits"].shape[1] < max_length:
                temp = self.tokenizer.pad_token_id * torch.ones((outputs["logits"].shape[0], max_length, outputs["logits"].shape[-1]), dtype=outputs["logits"].dtype, device=outputs["logits"].device)
                temp[:, :outputs["logits"].shape[1]] = outputs["logits"]
                outputs["logits"] = temp
            loss = self.label_smoother(outputs, labels)
        else:
            loss = None
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, generated_tokens, labels)
        