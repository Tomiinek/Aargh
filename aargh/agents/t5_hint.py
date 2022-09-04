import copy
import torch
import torch.nn.functional as F
import pickle
import os
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from torch.nn import Linear, Dropout, Embedding, Module, Parameter
from scipy import spatial
from torch.nn import CrossEntropyLoss
from transformers import AdamW, T5ForConditionalGeneration
from transformers import get_cosine_schedule_with_warmup as CosineWarmup
from transformers.models.t5.modeling_t5 import T5Stack 
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions
from pytorch_lightning.core.decorators import auto_move_data
from aargh.config import Params
from aargh.data.abstract import AutoTask, AutoTokenizer
from aargh.agents import AutoAgent
from aargh.agents import HFBaseAgent
from aargh.utils.data import batchify, tokens_from_transforms
from .action_base import ActionAgentBase, ActionBatchSampler


class HintDoubleDecoderT5Base(HFBaseAgent, ActionAgentBase):

    class ScalerLayer(Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self.w = Parameter(torch.tensor(float(scale)), requires_grad=True)

        def forward(self, x):     
            torch.clamp(self.w, 1e-6)     
            return self.w * x

    def __init__(self, params=None, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.embedded_hint_predictor = getattr(self.hparams, 'embedded_hint_predictor', False)
        
        if self.embedded_hint_predictor:
            
            # isntantiate retrieval model-specific modules
            self.hint_dropout = Dropout(self.hparams.output_dropout)
            self.hint_output = Linear(self.model.config.d_model, self.hparams.output_dim)
            self.api_embedding = Embedding(8, self.hparams.splitted_embedding_dim)
            self.similarity_scaler = self.ScalerLayer(self.hparams.similarity_scale)
            self.support_cache = None

            # activate manual optimization because we need multiple optimizers
            self.automatic_optimization = False

    def configure_optimizers(self):
        
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        group = {
            'shared' : ['model.encoder', 'model.shared'],
            'generation' : ['model.second_decoder', 'model.decoder', 'model.first_encoder'],
            'retrieval' : ['model.second_encoder']
        } 
        warmup = self.hparams.get('num_warmup_steps', None)

        if not self.embedded_hint_predictor:
            parameters = [
                {'params': [p for n, p in self.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': self.hparams.weight_decay},
                {'params': [p for n, p in self.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]
            optimizer = AdamW(parameters, lr=self.hparams.learning_rate)
            if warmup is None: return optimizer
            return [optimizer], [{'scheduler': CosineWarmup(optimizer, warmup, self.hparams.num_training_steps), 'interval': 'step'}]
                
        parameter_groups = {
            group_name : [
                {'params': [p for n, p in self.named_parameters() if p.requires_grad and any([m in n for m in modules]) and not any(nd in n for nd in no_decay)], 'weight_decay': self.hparams.weight_decay},
                {'params': [p for n, p in self.named_parameters() if p.requires_grad and any([m in n for m in modules]) and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ] for group_name, modules in group.items()
        }
        parameter_groups['retrieval'].extend([
            {'params': [p for n, p in self.similarity_scaler.named_parameters()], 'lr' : 0.01, 'weight_decay': 0.0 },
            {'params': [p for n, p in self.api_embedding.named_parameters()], 'weight_decay': 0.0 },
            {'params': [p for n, p in self.hint_output.named_parameters()], 'weight_decay': 0.0 }
        ])
        
        optimizers = [
            AdamW(parameter_groups['shared'], lr=self.hparams.learning_rate),
            AdamW(parameter_groups['generation'], lr=self.hparams.learning_rate),
            AdamW(parameter_groups['retrieval'], lr=self.hparams.learning_rate)
        ]

        schedulers = [
            {'scheduler': CosineWarmup(optimizers[0], warmup, self.hparams.num_training_steps), 'interval': 'step'},
            {'scheduler': CosineWarmup(optimizers[1], warmup, self.hparams.num_training_steps), 'interval': 'step'},
            {'scheduler': CosineWarmup(optimizers[2], warmup, self.hparams.num_training_steps), 'interval': 'step'}
        ]

        return optimizers, schedulers

    def get_target_label_keys(self):
        return ['response', 'belief']

    def pool(self, x, mask):
        mask = mask.unsqueeze(-1)
        x = x * mask
        x = reduce(x, 'b l d -> b d', 'sum')
        t = reduce(mask, 'b l d -> b d', 'sum')
        t = torch.clamp(t, min=1e-9)
        x = x / t
        return x

    def forward(self, batch, mode="both_decoders"):

        self.model.set_mode(mode)

        if self.training:

            def swtich_off(module):
                params = module.parameters()
                for param in params:
                    param.requires_grad = False

            for param in self.parameters():
                param.requires_grad = True

            if mode == "both_decoders": 
                if self.embedded_hint_predictor:
                    swtich_off(self.model.second_encoder)
            elif mode == "first_decoder": 
                if self.embedded_hint_predictor:
                    swtich_off(self.model.second_encoder)
                swtich_off(self.model.second_decoder)
            elif mode == "second_decoder": 
                if self.embedded_hint_predictor:
                    swtich_off(self.model.second_encoder)
                swtich_off(self.model.decoder)
            elif mode == "second_encoder": 
                swtich_off(self.model.first_encoder)
                swtich_off(self.model.decoder)
                swtich_off(self.model.second_decoder)
               # swtich_off(self.model.encoder)
               # swtich_off(self.model.shared)

        model_args = {
            'input_ids' : batch['conversation'],
            'attention_mask' : batch['conversation_mask'],
            'decoder_input_ids' : batch['query'],
            'decoder2_input_ids' : batch['response'],
            'decoder_attention_mask' : batch['query_mask'],
            'decoder2_attention_mask' : batch['response_mask']
        }

        if mode == "second_encoder":
            model_args['input_encoder_embeddings'] = self.api_embedding(batch['api_ids'])
            outputs = self.model(**model_args)
            ctxt_embedding = self.pool(outputs, batch['conversation_mask']) 
            ctxt_embedding = self.hint_output(self.hint_dropout(ctxt_embedding))
            ctxt_embedding = F.normalize(F.tanh(ctxt_embedding), dim=-1)
            return ctxt_embedding

        outputs = self.model(**model_args)

        outputs_label_pairs = {}
        if mode == "both_decoders":
            outputs_label_pairs['query'] = (outputs[0], batch['query_labels'])                   
            outputs_label_pairs['response'] = (outputs[1], batch['response_labels'])
        elif mode == "second_decoder":
            outputs_label_pairs['response'] = (outputs, batch['response_labels'])
        elif mode == "first_decoder":  
            outputs_label_pairs['query'] = (outputs, batch['query_labels'])

        return outputs_label_pairs

    def set_tokenizer_reference(self, tokenizer):
        super().set_tokenizer_reference(tokenizer)
        if tokenizer is not None:
            self.sys_label_id = self.tokenizer.get_token_id(self.hparams.get('system_prefix', None))
            self.usr_label_id = self.tokenizer.get_token_id(self.hparams.get('user_prefix', None))

    def get_batch_sampler(self):
        if self.embedded_hint_predictor:
            return ActionBatchSampler
        return None

    def on_train_start(self):

        def swtich_off(params):
            for param in params:
                param.requires_grad = False

        train_mode = self.hparams.get('train_mode', None)
       
        if train_mode == "response":
            self.model.activate_second_decoder()
            swtich_off(self.model.decoder.parameters())
        elif train_mode == "query":
            self.model.activate_first_decoder()
            swtich_off(self.model.second_decoder.parameters())
        
        if train_mode is not None:
            # freeze encoder because training of it would break the other decoder performance
            swtich_off(self.model.encoder.parameters())
            if self.embedded_hint_predictor:
                swtich_off(self.model.first_encoder.parameters())
                swtich_off(self.model.second_encoder.parameters())

        if self.embedded_hint_predictor:
            for opt in self.optimizers():
                opt.zero_grad()

    def training_step(self, batch, batch_idx):   

        if self.embedded_hint_predictor:

            sha_opt, dec_opt, ret_opt = self.optimizers()
            sha_sch, dec_sch, ret_sch = self.lr_schedulers()

            # only the generation task is accumulated: (0 0) (0 0) 1 (0 0) (0 0) 1 (0 0) (0 0) 1 ...
            if (self.trainer.global_step + 1) % (self.hparams.action_loss_weight * self.hparams.accumulation_steps + 1) == 0:
                batch = batch["custom"]

                outputs = self(batch, mode="second_encoder")
                outputs = rearrange(outputs, '(m n) d -> m n d', m=self.hparams.num_actions, n=self.hparams.num_examples_per_action)      
                loss = self.calculate_action_loss(outputs)
                
                self.log('train_action_loss', loss, sync_dist=True, prog_bar=True)
                self.log("train_sim_scale", self.similarity_scaler.w, prog_bar=True)

                self.manual_backward(loss)
                #sha_opt.step()
                ret_opt.step()       
                sha_opt.zero_grad()
                ret_opt.zero_grad()
                #sha_sch.step()
                ret_sch.step()

                return loss
                #batch = batch["default"]                  
            else:
                batch = batch["default"]

        else:
            batch = batch[0]
        
        #batch = batch["default"]
        outputs = self(batch, mode="both_decoders")
        loss = torch.tensor(0.0, device=self.device)
        for o, l in outputs.values():
            loss += self.process_step(o, l, self.train_metrics)

        self.log('train_total_loss', loss, sync_dist=True, prog_bar=True)
        self.log_dict({'train_' + k : m for k, m in self.train_metrics.items()}) 

        if self.embedded_hint_predictor:
            if self.hparams.accumulation_steps > 0:
                loss = loss / self.hparams.accumulation_steps
            self.manual_backward(loss)

            # only the generation task is accumulated: (0 1) (0 1) ? (0 1) (0 1) ? (0 1) (0 1) ? ...
            if ((self.trainer.global_step + 1) % (self.hparams.action_loss_weight * self.hparams.accumulation_steps + 1)) % self.hparams.accumulation_steps == 0:
            #if (self.trainer.global_step + 1) % self.hparams.accumulation_steps == 0:
                dec_opt.step()
                sha_opt.step()
                dec_opt.zero_grad()
                sha_opt.zero_grad()
                dec_sch.step()
                sha_sch.step()
            
        return loss

    def on_validation_epoch_start(self):

        if not self.embedded_hint_predictor or self.current_epoch % 2 != 0:
            return

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        ctxt_embeddings = []
        dataset_items = []

        def get_batch(size, dataset, message=None):
            batch = []
            transformed = []
            sampler = range(len(dataset))
            for idx in sampler:
                batch.append(dataset.items[idx])
                transformed.append(dataset[idx])
                if len(batch) == size:
                    yield batch, transformed
                    batch = []
                    transformed = []
            if len(batch) > 0:
                yield batch, transformed

        with torch.no_grad():
            for b, t in tqdm(get_batch(self.hparams.batch_size, self.trainer.datamodule.dataset.train)):
                prepared_batch = self.trainer.datamodule.dataset.collate(t)
                ctxt_encodings = self.encode_context(prepared_batch)
                ctxt_embeddings.append(ctxt_encodings)
                dataset_items.extend(b)

        output_ctxt_embeddings = {}
        for e in ctxt_embeddings:
            if e is None:
                continue
            for k in e:
                if k not in output_ctxt_embeddings:
                    output_ctxt_embeddings[k] = []
                output_ctxt_embeddings[k].append(e[k])
        for k in output_ctxt_embeddings:
            output_ctxt_embeddings[k] = torch.cat(output_ctxt_embeddings[k])

        self.support_cache = {
            'ctxt_encodings' : output_ctxt_embeddings,
            'items' : dataset_items
        }
        self.support_cache['embeddings_tree'] = spatial.KDTree(self.support_cache['ctxt_encodings']['embeddings'].cpu())

        with torch.no_grad():
            for m, f in [("train", self.trainer.datamodule.dataset.train), ("val", self.trainer.datamodule.dataset.val)]:
                for b, t in tqdm(get_batch(self.hparams.batch_size, f)):
                    prepared_batch = self.trainer.datamodule.dataset.collate(t)
                    _, items = self.respond(prepared_batch, return_responses=True, return_items=True, top_k=4, retrieval=True)            
                    for i, r in zip(b, items):
                        for response in r:
                            if response.conv_id == i.conv_id:
                                continue
                            i.hint = response.response
                            break
        
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def validation_step(self, batch, batch_idx, dataloader_idx=None):

        if not self.embedded_hint_predictor or dataloader_idx == 1:
            outputs = self(batch, mode="both_decoders")
            loss = torch.tensor(0.0, device=self.device)
            for o, l in outputs.values():
                loss += self.process_step(o, l, self.val_metrics, validation=True, log_distribution=(batch_idx == 0))
            self.log('val_total_loss', loss, sync_dist=True, prog_bar=True, add_dataloader_idx=False)
            self.log_dict({'val_' + k : m for k, m in self.val_metrics.items()}, prog_bar=True, add_dataloader_idx=False)   
        else:
            outputs = self(batch, mode="second_encoder")
            outputs = rearrange(outputs, '(m n) d -> m n d', m=self.hparams.num_actions, n=self.hparams.num_examples_per_action)         
            loss = self.calculate_action_loss(outputs)
            self.log('val_action_loss', loss, sync_dist=True, prog_bar=True, add_dataloader_idx=False)

    @auto_move_data
    def respond(self, batch, *args, **kwargs):

        # just to unify interface with retrieval models
        if kwargs.get('retrieval', False):
            return self.retrieval_respond(batch, *args, **kwargs)

        return_offset = kwargs.get('return_offset', False)
        greedy = kwargs.get('greedy', self.hparams.get('greedy', False))
        first_decoder = kwargs.get('first_decoder', True)
        decoder_key = kwargs.get('decoder_key', None)

        if first_decoder:
            self.model.set_mode('first_decoder')
        else:
            self.model.set_mode('second_decoder')

        decoder_input = batch[decoder_key]
        decoder_mask = batch[decoder_key + '_mask']
        
        # decoder attention mask cannot be used for masking the input of the decoder when generating
        # so we either need to provide the decoder_input_ids containing the same number of tokens for all
        # items in the batch, or we need to process them one by one

        #greedy = True

        generated = []
        for b in range(decoder_input.size(0)):
            di = torch.unsqueeze(decoder_input[b, :][decoder_mask[b]], 0)
            di_len = decoder_mask[b].sum()

            g = self.model.generate( 
                    torch.unsqueeze(batch['conversation'][b], 0),
                    attention_mask=torch.unsqueeze(batch['conversation_mask'][b], 0),
                    decoder_input_ids=di,
                    do_sample=(not greedy), 
                    max_length=256, 
                    min_length=int(di_len) if not first_decoder else None, # + 5 
                    temperature=self.hparams.temperature if not greedy and hasattr(self.hparams, 'temperature') else 1.0,
                    top_k=self.hparams.top_k_sampling if not greedy and hasattr(self.hparams, 'top_k_sampling') else 50, 
                    top_p=self.hparams.nucleus_sampling if not greedy and hasattr(self.hparams, 'nucleus_sampling') else 1.0,
                    num_beams=self.hparams.num_beams if not greedy and hasattr(self.hparams, 'num_beams') else 1,
                    #no_repeat_ngram_size=10,
                    early_stopping=True,
                    eos_token_id=self.model.config.eos_token_id,
                    pad_token_id=self.hparams.padding_idx,
                    use_cache=False)

            if not return_offset:
                offset = di.size(1)
            else: 
                offset = 0

            generated.append(torch.squeeze(g[:, offset:], 0))

        return torch.nn.utils.rnn.pad_sequence(generated, batch_first=True, padding_value=self.hparams.padding_idx)

    def try_load_support_cache(self, file_path):
        """ 
        Load encoded (training) data or better said the contexts and the correspoding dataset items. 
        The containig pickle file should contain a dictionary with the 'embeddings' and 'items' keys.
        """
        if not os.path.exists(file_path):
            return False

        import sys
        import aargh.data as data # because of pickle compatibility with older non-package checkpoints
        sys.modules["data"] = data

        with open(file_path, 'rb') as f:
            self.support_cache = pickle.load(f)

        if self.support_cache['ctxt_encodings'] is not None: 
            self.support_cache['embeddings_tree'] = spatial.KDTree(self.support_cache['ctxt_encodings']['embeddings'].cpu())

        return True

    def save_support_cache(self, embeddings, _, items, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'ctxt_encodings' : embeddings,
                'items' : items
            }, f)

    @auto_move_data
    def encode_context(self, batch):
        """ This helper method is here just to unify the interface with retrieval models. """
        if not self.embedded_hint_predictor:
            return None
        embeddings = self(batch, mode="second_encoder")
        return { 'embeddings' : embeddings }

    def encode_response(self, batch):
        """ This helper method is here just to unify the interface with retrieval models. """
        return None

    @auto_move_data
    def retrieval_respond(self, batch, return_responses=False, return_items=False, top_k=1, *args, **kwarg):    

        embeddings = self.encode_context(batch)['embeddings']
        score, label = self.support_cache['embeddings_tree'].query(embeddings.cpu(), k=list(range(1, top_k+1)))
        score = 1 / (score + 1e-6)
        items = self.support_cache['items']

        if not return_responses:
            return {
                'score' : score, 
                'label' : label,
                'item' : None if items is None else [items[idx] for idx in label[:, 0]]
            }

        responses = [[items[idx].response for idx in l[:top_k]] for l in label]
        if return_items:
            r_items = [[items[idx] for idx in l[:top_k]] for l in label]
            return responses, r_items

        return responses

    @auto_move_data
    def retrieve_hint(self, batch):

        def extract_retrieved_hints(b, use_heuristic=True):
            result = []
            for r in b:
                if not use_heuristic:
                    result.append(r[0])
                    continue
                best_r, best_n, best_p = '', -1, False
                for response in r:
                    num_slots = response.count('[')
                    venue_present = '[name]' in response or '[trainid]' in response
                    if venue_present or not best_p:
                        if num_slots >= best_n or (venue_present and not best_p):
                            best_n = num_slots
                            best_r = response
                        best_p = venue_present 
                result.append(best_r)
            return result

        if getattr(self.hparams, 'support_path', None) is None:
            return [None for _ in batch]
        
        if self.embedded_hint_predictor:

            if self.support_cache is None and not self.try_load_support_cache(self.hparams.support_path):
                raise ValueError(f"Pre-calculated retrieval embeddings are missing, given: {self.hparams.support_path}")

            batch_responses = self.retrieval_respond(batch, return_responses=True, top_k=3)
            
            return extract_retrieved_hints(batch_responses)

        if getattr(self, 'retrieval_model', None) is None:

            params = Params.from_checkpoint(self.hparams.retrieval_checkpoint)
            
            model_class = AutoAgent.from_config(params, return_instance=False)
            self.retrieval_model = model_class.load_from_checkpoint(self.hparams.retrieval_checkpoint)
            self.retrieval_model.to(device=self.device)
            self.retrieval_model.eval()
            self.retrieval_model.freeze()
            
            self.retrieval_task = AutoTask.from_config(params, return_instance=True)
            tokenizer = AutoTokenizer.from_config(params)
            tokenizer.add_tokens(self.retrieval_task.get_new_tokens() + tokens_from_transforms(self.retrieval_task.get_task_transforms(params)))
       
            self.retrieval_task.tokenizer = tokenizer
            self.retrieval_model.set_tokenizer_reference(tokenizer)
            self.retrieval_model.try_load_support_cache(self.hparams.support_path)

        t = self.retrieval_model.tokenizer
        context_token_ids, context_mask, _ = self.retrieval_task.inputs_concat([
            ('context',      batchify([x.replace('<|system|>', '[SYS]').replace('<|user|>', '[USR]') for x in batch['hint_query']['context']], t, padding=False, enable_wrapping=False)),
            ('prev_belief',  batchify(['[BS]' + x for x in batch['hint_query']['prev_belief']], t, padding=False, enable_wrapping=False)),
            ('api_result',   batchify(['[DB]' + x for x in batch['hint_query']['api_result']], t, padding=False, enable_wrapping=False)),    
        ])

        batch_responses = self.retrieval_model.respond({
            'conversation' : context_token_ids,
            'conversation_mask' : context_mask    
        }, return_responses=True, top_k=3) # top_k=1

        return extract_retrieved_hints(batch_responses)


class SmallHintDoubleDecoderT5Agent(HintDoubleDecoderT5Base):
    
    NAME = "hint_goal_double_small_t5"

    def instantiate_model(self):
        return model_factory(self.hparams).from_pretrained("t5-small")


class MediumHintDoubleDecoderT5Agent(HintDoubleDecoderT5Base):
    
    NAME = "hint_goal_double_medium_t5"

    def instantiate_model(self):
        return model_factory(self.hparams).from_pretrained("t5-base")


class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class T5CustomStack(T5Stack):
    def __init__(self, config, embed_tokens=None, use_pre_dropout=True, use_post_dropout=True):
        super().__init__(config, embed_tokens)
        self.use_pre_dropout = use_pre_dropout
        self.use_post_dropout = use_post_dropout

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        position_bias=None
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f":obj:`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        encoder_decoder_position_bias = None

        if self.use_pre_dropout:
            hidden_states = self.dropout(inputs_embeds)
        else:
            hidden_states = inputs_embeds

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)

        if self.use_post_dropout:
            hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # !!! CAUTION !!! the cross attentions are used to return the `position_bias`
        # This should not be a problem because cross_attentions are not used in the encoder

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    position_bias,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=position_bias,     
        )


def model_factory(hparams):

    splitted_encoder = getattr(hparams, 'embedded_hint_predictor', False)
    if splitted_encoder:
        splitted_depth = hparams.splitted_encoder_depth
        splitted_embedding_dim = hparams.splitted_embedding_dim

    class CausalDoubleDecoderT5(T5ForConditionalGeneration):

        _keys_to_ignore_on_load_missing = [
            r"encoder\.embed_tokens\.weight",
            r"decoder\.embed_tokens\.weight",
            r"lm_head\.weight",
            r"second_decoder\.*",
            r"second_lm_head\.weight",
            r"second_encoder_resize\.*"
        ]

        def __init__(self, config):

            # See the `resize_token_embeddings` for the rest of initialization (it is called from HFBase)
            
            super().__init__(config)
            self.second_decoder, self.second_lm_head = self.initialize_new_decoder(self.config)
            self.set_mode('both_decoders')
            if splitted_encoder:
                self.second_encoder_resize = Linear(splitted_embedding_dim + config.d_model, config.d_model)

        def set_mode(self, mode):
            available_modes = ['both_decoders', 'first_decoder', 'second_decoder', 'second_encoder']
            if mode not in available_modes:
                raise ValueError(f"The mode must be one of: {available_modes}!")
            self.mode = mode
            
        def initialize_new_decoder(self, config):
            decoder_config = copy.deepcopy(config)
            decoder_config.is_decoder = True
            new_decoder = type(self.decoder)(decoder_config, self.shared)
            new_decoder.load_state_dict(self.decoder.state_dict())
            new_head = type(self.lm_head)(config.d_model, config.vocab_size, bias=False)
            new_head.load_state_dict(self.lm_head.state_dict())
            return new_decoder, new_head

        def initialize_new_encoder(self, config):

            if not splitted_encoder:
                return None, None

            def duplicate_base_encoder(sd):
                encoder_config = copy.deepcopy(config)
                encoder_config.is_decoder = False
                encoder_config.use_cache = False
                new_encoder = T5CustomStack(encoder_config, self.shared)
                new_encoder.load_state_dict(sd)
                return new_encoder

            def duplicate_encoder(sd):
                new_encoder = duplicate_base_encoder(sd)
                while len(new_encoder.block) > splitted_depth:
                    new_encoder.block.__delitem__(0)
                new_encoder.embed_tokens = Identity()
                new_encoder.use_pre_dropout = False
                return new_encoder
            
            # build the new encoder `heads`, but use the original pretrained weights
            sd = self.encoder.state_dict()
            self.encoder = duplicate_base_encoder(sd)
            first_encoder = duplicate_encoder(sd)
            second_encoder = duplicate_encoder(sd)

            # remove the layers correspondig to `heads` in the original encoder
            self.encoder.final_layer_norm = Identity()
            self.encoder.use_post_dropout = False
            while len(self.encoder.block) > config.num_layers - splitted_depth:
                self.encoder.block.__delitem__(len(self.encoder.block) - 1)

            return self.encoder, first_encoder, second_encoder

        def resize_token_embeddings(self, *kargs, **kwargs):
            super().resize_token_embeddings(*kargs, **kwargs)
            self.second_decoder, self.second_lm_head = self.initialize_new_decoder(self.config)
            if splitted_encoder:
                self.encoder, self.first_encoder, self.second_encoder = self.initialize_new_encoder(self.config)

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

        def get_encoder(self):
            """ 
            Helper function for generation which first retrieves the model's generator and then calls it with arguments. 
            We have multiple encoders, but the retrieval one is never used in this way.
            """
            if not splitted_encoder:
                return self.encoder

            def encoder_func(input_ids, **kwargs):
                x = self.encoder(input_ids, **kwargs)
                kwargs["inputs_embeds"] = x.last_hidden_state
                kwargs["position_bias"] = x.cross_attentions
                x = self.first_encoder(**kwargs)
                return x

            return encoder_func

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
            decoder_head_mask=None, 
            cross_attn_head_mask=None, 
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            input_encoder_embeddings=None
        ):
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if self.model_parallel or decoder_head_mask is not None or cross_attn_head_mask is not None:
                raise NotImplementedError()

            encoder_outputs_cached = encoder_outputs is not None

            #
            # This stuff is copy pasted from the original T5 implementation in HF Transformers

            ### Possible modes:
            #
            #   both_decoders - the base & first encoder or original encoder -> both decoders
            #                   return: both decoders' outputs
            #                   expect: input_ids
            #
            #   first_decoder - the base & first encoder or original encoder -> the first decoder
            #                   return: first decoder outputs
            #                   expect: input_ids
            #
            #   second_decoder - the base & first encoder or original encoder -> the second decoder
            #                   return: second decoder outputs
            #                   expect: input_ids
            #
            #   second_encoder - the base & second encoder
            #                   return: second encoder outputs
            #                   expect: input_ids & embeddings of size (Batch, Dimension)
            #             
            ### Use cases:
            #   
            #   - train: both_decoders / second_encoder
            #   - validation: both_decoders / second_encoder
            #   - inference: first_decoder -> (second_encoder) -> second_decoder
            #       

            if not encoder_outputs_cached:
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            if not splitted_encoder and self.mode == "second_encoder":
                raise ValueError("Running in mode: `{self.mode}`, but the model does not have a splitted encoder!")
            
            if splitted_encoder:

                if self.mode == "second_encoder":

                    input_encoder_embeddings = repeat(input_encoder_embeddings, 'b d -> b l d', l=encoder_outputs.last_hidden_state.shape[1])
                    second_encoder_inputs = torch.cat([input_encoder_embeddings, encoder_outputs.last_hidden_state], dim=-1)
                    second_encoder_inputs = self.second_encoder_resize(second_encoder_inputs)

                    second_encoder_outputs = self.second_encoder(
                        inputs_embeds=second_encoder_inputs,
                        attention_mask=attention_mask,
                        head_mask=head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        position_bias=encoder_outputs.cross_attentions # caution: cross_attentions are misused and replaced with position bias!
                    )
                    return second_encoder_outputs.last_hidden_state

                if not encoder_outputs_cached:
                    encoder_outputs = self.first_encoder(
                        inputs_embeds=encoder_outputs.last_hidden_state,
                        attention_mask=attention_mask,
                        head_mask=head_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        position_bias=encoder_outputs.cross_attentions # caution: cross_attentions are misused and replaced with position bias!
                    )

            #
            # This stuff is updadated and different

            hidden_states = encoder_outputs[0]
            
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

            if self.mode in ["first_decoder", "second_decoder"]:
                decoder = self.second_decoder if self.mode == "second_decoder" else self.decoder
                lm_head = self.second_lm_head if self.mode == "second_decoder" else self.lm_head

                if self.mode == "second_decoder" and decoder2_input_ids is not None:
                    # if we provided inputs for both decoders, but only the second is activated, use the inputs for the second (this happens during training)
                    logits1, outputs1, loss1 = self.decoder_forward(decoder, lm_head, decoder2_input_ids, decoder2_labels, 
                                                                    decoder2_attention_mask, decoder2_inputs_embeds, **other_inputs)
                else: 
                    # otherwise, e.g. if we provided a single set of inputs and only the second is activated, use the only inputs (this happens during generation)
                    logits1, outputs1, loss1 = self.decoder_forward(decoder, lm_head, decoder_input_ids, decoder_labels, 
                                                                    decoder_attention_mask, decoder_inputs_embeds, **other_inputs)

            if self.mode == "both_decoders":
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
                        encoder_last_hidden_state=encoder_outputs.last_hidden_state, # This should be output of ecndoer `head`
                        encoder_hidden_states=encoder_outputs.hidden_states,         # This should be output of the base encoder
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

    # return from the factory wrapping function
    return CausalDoubleDecoderT5
