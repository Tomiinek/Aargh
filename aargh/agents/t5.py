import copy
import torch
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from pytorch_lightning.core.decorators import auto_move_data
from aargh.agents.hf_base import HFBaseAgent


class DoubleDecoderT5Base(HFBaseAgent):

    def get_target_label_keys(self):
        return ['response', 'belief']

    def forward(self, batch):
        outputs = self.model(
            input_ids=batch['conversation'],
            attention_mask=batch['conversation_mask'],
            decoder_input_ids=batch['query'],
            decoder2_input_ids=batch['response'],
            decoder_attention_mask=batch['query_mask'],
            decoder2_attention_mask=batch['response_mask']
        )

        outputs_label_pairs = {}
        if self.model.active_decoders == "both":
            outputs_label_pairs['query'] = (outputs[0], batch['query_labels'])                   
            outputs_label_pairs['response'] = (outputs[1], batch['response_labels'])
        elif self.model.active_decoders == "second":
            outputs_label_pairs['response'] = (outputs, batch['response_labels'])
        elif self.model.active_decoders == "first":  
            outputs_label_pairs['query'] = (outputs, batch['query_labels'])

        return outputs_label_pairs

    def set_tokenizer_reference(self, tokenizer):
        super().set_tokenizer_reference(tokenizer)
        if tokenizer is not None:
            self.sys_label_id = self.tokenizer.get_token_id(self.hparams.get('system_prefix', None))
            self.usr_label_id = self.tokenizer.get_token_id(self.hparams.get('user_prefix', None))

    def on_train_start(self):
        train_mode = self.hparams.get('train_mode', None)
       
        if train_mode == "response":
            self.model.activate_second_decoder()
            for param in self.model.decoder.parameters():
                param.requires_grad = False
        elif train_mode == "query":
            self.model.activate_first_decoder()
            for param in self.second_decoder.parameters():
                param.requires_grad = False
        
        if train_mode is not None:
            # freeze encoder because training of it would break the other decoder performance
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):   
        
        outputs = self(batch)

        loss = torch.tensor(0.0, device=self.device)
        
        for o, l in outputs.values():
            loss += self.process_step(o, l, self.train_metrics)

        self.log('train_total_loss', loss, sync_dist=True, prog_bar=True)
        self.log_dict({'train_' + k : m for k, m in self.train_metrics.items()}) 
            
        return loss

    def validation_step(self, batch, batch_idx):

        outputs = self(batch)

        loss = torch.tensor(0.0, device=self.device)
        for o, l in outputs.values():
            loss += self.process_step(o, l, self.val_metrics, validation=True, log_distribution=(batch_idx == 0))

        self.log('val_total_loss', loss, sync_dist=True, prog_bar=True)
        self.log_dict({'val_' + k : m for k, m in self.val_metrics.items()}, prog_bar=True)   

    @auto_move_data
    def respond(self, batch, *args, **kwargs):

        return_offset = kwargs.get('return_offset', False)
        greedy = kwargs.get('greedy', self.hparams.get('greedy', False))
        first_decoder = kwargs.get('first_decoder', True)
        decoder_key = kwargs.get('decoder_key', None)

        if first_decoder:
            self.model.activate_first_decoder()
        else:
            self.model.activate_second_decoder()

        decoder_input = batch[decoder_key]
        decoder_mask = batch[decoder_key + '_mask'] 
      
        # decoder attention mask cannot be used for masking the input of the decoder when generating
        # so we either need to provide the decoder_input_ids containing the same number of tokens for all
        # items in the batch, or we need to process them one by one

        generated = []
        for b in range(decoder_input.size(0)):
            di = torch.unsqueeze(decoder_input[b, :][decoder_mask[b]], 0)
         
            g = self.model.generate( 
                    torch.unsqueeze(batch['conversation'][b], 0),
                    attention_mask=torch.unsqueeze(batch['conversation_mask'][b], 0),
                    decoder_input_ids=di,
                    do_sample=(not greedy), 
                    max_length=128, 
                    min_length=1,
                    temperature=self.hparams.temperature if not greedy and hasattr(self.hparams, 'temperature') else 1.0,
                    top_k=self.hparams.top_k_sampling if not greedy and hasattr(self.hparams, 'top_k_sampling') else 50, 
                    top_p=self.hparams.nucleus_sampling if not greedy and hasattr(self.hparams, 'nucleus_sampling') else 1.0,
                    num_beams=self.hparams.num_beams if not greedy and hasattr(self.hparams, 'num_beams') else 1,
                    early_stopping=True,
                    eos_token_id=self.model.config.eos_token_id,
                    pad_token_id=self.hparams.padding_idx,
                    use_cache=False)

            if not return_offset:
                offset = di.size(1)
            else: 
                offset = 0
            
            generated.append(torch.squeeze(g[:, offset:], 0))

        self.model.activate_both_decoders()

        return torch.nn.utils.rnn.pad_sequence(generated, batch_first=True, padding_value=self.hparams.padding_idx)


class SmallDoubleDecoderT5Agent(DoubleDecoderT5Base):
    
    NAME = "goal_double_small_t5"

    def instantiate_model(self):
        return CausalDoubleDecoderT5.from_pretrained("t5-small")


class MediumDoubleDecoderT5Agent(DoubleDecoderT5Base):
    
    NAME = "goal_double_medium_t5"

    def instantiate_model(self):
        return CausalDoubleDecoderT5.from_pretrained("t5-base")


class CausalDoubleDecoderT5(T5ForConditionalGeneration):

    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
        r"second_decoder\.*",
        r"second_lm_head\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.second_decoder, self.second_lm_head = self.initialize_new_decoder(config)
        self.active_decoders = "both"

    def activate_both_decoders(self):
        self.active_decoders = "both"

    def activate_first_decoder(self):
        self.active_decoders = "first"

    def activate_second_decoder(self):
        self.active_decoders = "second"

    def initialize_new_decoder(self, config):
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        new_decoder = type(self.decoder)(decoder_config, self.shared)
        new_decoder.load_state_dict(self.decoder.state_dict())
        new_head = type(self.lm_head)(config.d_model, config.vocab_size, bias=False)
        new_head.load_state_dict(self.lm_head.state_dict())
        return new_decoder, new_head

    def resize_token_embeddings(self, *kargs, **kwargs):
        super().resize_token_embeddings(*kargs, **kwargs)
        self.second_decoder, self.second_lm_head = self.initialize_new_decoder(self.config)

    def parallelize(self, device_map=None):
        raise NotImplementedError()

    def deparallelize(self):
        raise NotImplementedError()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
        self.second_decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError()

    #def get_output_embeddings(self):
    #    raise NotImplementedError()

    def decoder_forward(self, decoder, lm_head, input_ids, labels, attention_mask, inputs_embeds, past_key_values, **other_inputs):

        if labels is not None and input_ids is None and inputs_embeds is None:
            input_ids = self._shift_right(labels)
        
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if input_ids is not None:
                input_ids = input_ids[:, -1:]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, -1:]

        outputs = decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **other_inputs
        )
        
        sequence_output = outputs[0]
        
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        
        logits = lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, outputs, loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        decoder2_input_ids=None,
        decoder_inputs_embeds=None,
        decoder2_inputs_embeds=None,
        decoder_attention_mask=None,
        decoder2_attention_mask=None,
        decoder_labels=None,
        decoder2_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.model_parallel:
            raise NotImplementedError()

        #
        # This stuff is copy pasted from the original T5 implementation in HF Transformers

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        #
        # This stuff is updadated and different

        other_inputs = {
            "past_key_values" : past_key_values,
            "encoder_hidden_states" : hidden_states,
            "encoder_attention_mask" : attention_mask,
            "head_mask" : head_mask,
            "use_cache" : use_cache,
            "output_attentions" : output_attentions,
            "output_hidden_states" : output_hidden_states,
            "return_dict" : return_dict
        }

        output1, output2 = None, None

        outputs1, outputs2 = None, None
        logits1, logits2 = None, None
        loss1, loss2 = None, None

        if self.active_decoders in ["second", "first"]:
            decoder = self.second_decoder if self.active_decoders == "second" else self.decoder
            lm_head = self.second_lm_head if self.active_decoders == "second" else self.lm_head

            if self.active_decoders == "second" and decoder2_input_ids is not None:
                # if we provided inputs for both decoders, but only the second is activated, use the inputs for the second (this happens during training)
                logits1, outputs1, loss1 = self.decoder_forward(decoder, lm_head, decoder2_input_ids, decoder2_labels, 
                                                                decoder2_attention_mask, decoder2_inputs_embeds, **other_inputs)
            else: 
                # otherwise, e.g. if we provided a single set of inputs and only the second is activated, use the only inputs (this happens during generation)
                logits1, outputs1, loss1 = self.decoder_forward(decoder, lm_head, decoder_input_ids, decoder_labels, 
                                                                decoder_attention_mask, decoder_inputs_embeds, **other_inputs)

        if self.active_decoders == "both":
            logits1, outputs1, loss1 = self.decoder_forward(self.decoder, self.lm_head, decoder_input_ids, decoder_labels, 
                                                            decoder_attention_mask, decoder_inputs_embeds, **other_inputs)
            logits2, outputs2, loss2 = self.decoder_forward(self.second_decoder, self.second_lm_head, decoder2_input_ids, decoder2_labels, 
                                                            decoder2_attention_mask, decoder2_inputs_embeds, **other_inputs)

        if not return_dict:
            
            output1 = ((logits1,) + outputs1[1:] + encoder_outputs) if logits1 is not None else None
            output2 = ((logits2,) + outputs2[1:] + encoder_outputs) if logits2 is not None else None

            output1 = ((loss1,) + output1) if loss1 is not None else output1
            output2 = ((loss2,) + output2) if loss2 is not None else output2

        else:
            if outputs1 is not None:
                output1 = Seq2SeqLMOutput(
                    loss=loss1,
                    logits=logits1,
                    past_key_values=outputs1.past_key_values,
                    decoder_hidden_states=outputs1.hidden_states,
                    decoder_attentions=outputs1.attentions,
                    cross_attentions=outputs1.cross_attentions,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                    encoder_hidden_states=encoder_outputs.hidden_states,
                    encoder_attentions=encoder_outputs.attentions,
                )

            if outputs2 is not None:
                output2 = Seq2SeqLMOutput(
                    loss=loss2,
                    logits=logits2,
                    past_key_values=outputs2.past_key_values,
                    decoder_hidden_states=outputs2.hidden_states,
                    decoder_attentions=outputs2.attentions,
                    cross_attentions=outputs2.cross_attentions,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                    encoder_hidden_states=encoder_outputs.hidden_states,
                    encoder_attentions=encoder_outputs.attentions,
                )

        if output1 is None:
            return output2
        elif output2 is None:
            return output1
        else: 
            return (output1, output2)