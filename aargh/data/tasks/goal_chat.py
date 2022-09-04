import torch
from dataclasses import dataclass
from torchvision import transforms
from aargh.utils.data import batchify, mask_until_last_nonempty
from aargh.data.tasks.base import BaseGoalTask, BaseLanguageModelTask
from aargh.data.transforms import ExtractLastUserUtteranceTransform, PrefixAttributeTransform


class LanguageModelGoalTask(BaseGoalTask, BaseLanguageModelTask):

    def _get_response_batch(self, batch, agent):
        batch['conversation_mask'] = mask_until_last_nonempty(batch['labels']['api_result'])
        return agent.respond(batch)

    def _get_query_batch(self, batch, agent):
        batch['conversation_mask'] = mask_until_last_nonempty(batch['labels']['context'])
        return agent.respond(batch, stop_token=self.tokenizer.get_token_id(self.params.database_prefix), greedy=True)

    def _decode_query_batch(self, batch):
        return self.tokenizer.decode_batch(batch, skip_special_tokens=True)
        
    def _decode_response_batch(self, batch):
        return self.tokenizer.decode_batch(batch, skip_special_tokens=True)
        
    def collate(self, batch):

        _, _, response, context, prev_belief, _, target_belief, api_call, api_result, *_ = zip(*batch)

        if self.differential_belief:         
            token_ids, mask, labels = self.inputs_concat([
                ('prev_belief', batchify(list(prev_belief), self.tokenizer, padding=False, enable_wrapping=False)), 
                ('context',     batchify(list(context), self.tokenizer, padding=False, enable_wrapping=False)), 
                ('belief',      batchify(list(target_belief), self.tokenizer, padding=False, enable_wrapping=False)), 
                ('api_call',    batchify(list(api_call), self.tokenizer, padding=False, enable_wrapping=False)),
                ('api_result',  batchify(list(api_result), self.tokenizer, padding=False, enable_wrapping=False)),
                ('response',    batchify(list(response), self.tokenizer, padding=False, enable_wrapping=False))
            ])

        else:
            token_ids, mask, labels = self.inputs_concat([
                ('context',    batchify(list(context), self.tokenizer, padding=False, enable_wrapping=False)), 
                ('belief',     batchify(list(target_belief), self.tokenizer, padding=False, enable_wrapping=False)), 
                ('api_call',   batchify(list(api_call), self.tokenizer, padding=False, enable_wrapping=False)),
                ('api_result', batchify(list(api_result), self.tokenizer, padding=False, enable_wrapping=False)),
                ('response',   batchify(list(response), self.tokenizer, padding=False, enable_wrapping=False))
            ])

        return {
            'conversation' :        token_ids,
            'conversation_mask' :   mask,
            'labels' :              labels
        }     


class DoubleLanguageModelGoalTask(BaseGoalTask, BaseLanguageModelTask):

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)

    def _get_response_batch(self, batch, agent):
        batch['response_mask'] = mask_until_last_nonempty(batch['response_labels']['api_result'])
        return agent.respond(batch, first_decoder=False, decoder_key='response')

    def _get_query_batch(self, batch, agent):
        # single SOS token is enough in this case
        batch['query_mask'] = torch.zeros_like(batch['query_labels']['belief'], dtype=torch.bool)
        batch['query_mask'][:, 0] = True
        return agent.respond(batch, first_decoder=True, greedy=True, decoder_key='query')

    def _decode_query_batch(self, batch):
        return self.tokenizer.decode_batch(batch, skip_special_tokens=True)
        
    def _decode_response_batch(self, batch):
        return self.tokenizer.decode_batch(batch, skip_special_tokens=True)
        
    def collate(self, batch):

        _, _, response, context, prev_belief, _, target_belief, api_call, api_result, *_ = zip(*batch)

        if not self.differential_belief:
            raise NotImplementedError()

        context_token_ids, context_mask, context_labels = self.inputs_concat([
            ('prev_belief', batchify(list(prev_belief), self.tokenizer, padding=False, enable_wrapping=False)), 
            ('context',     batchify(list(context), self.tokenizer, padding=False, enable_wrapping=False))
        ])

        query_token_ids, query_mask, query_labels = self.inputs_concat([
            ('belief',     batchify(list(target_belief), self.tokenizer, padding=False, enable_wrapping=False)), 
            ('api_call',   batchify(list(api_call), self.tokenizer, padding=False, enable_wrapping=False))
        ], first_prefix_shift=False)

        response_token_ids, response_mask, response_labels = self.inputs_concat([
            ('api_result',  batchify(list(api_result), self.tokenizer, padding=False, enable_wrapping=False)),
            ('response',    batchify(list(response), self.tokenizer, padding=False, enable_wrapping=False))
        ], first_prefix_shift=False)

        return_dict = {
            'conversation' :        context_token_ids,
            'conversation_mask' :   context_mask,    
            'conversation_labels' : context_labels,       
            'query' :               query_token_ids,
            'query_labels'  :       query_labels,
            'query_mask' :          query_mask,      
            'response' :            response_token_ids,
            'response_labels' :     response_labels,
            'response_mask' :       response_mask,      
        }

        return return_dict


class PolicyOptimizationLanguageModelGoalTask(BaseGoalTask, BaseLanguageModelTask):

    @dataclass
    class DatasetItem(BaseGoalTask.DatasetItem):
        last_user: str = None

    @classmethod
    def get_task_transforms(cls, params):
        return transforms.Compose([
            ExtractLastUserUtteranceTransform(),
            BaseLanguageModelTask.get_task_transforms(params),
            PrefixAttributeTransform('last_user', params.try_get('last_user_prefix', None))
        ])

    def _get_response_batch(self, batch, agent):
        batch['response_mask'] = mask_until_last_nonempty(batch['response_labels']['api_result'])
        return agent.respond(batch, first_decoder=False, decoder_key='response')

    def _get_query_batch(self, batch, agent):
        return batch['query_labels']

    def _decode_query_batch(self, batch):
        return self.tokenizer.decode_batch(batch, skip_special_tokens=True)
        
    def _decode_response_batch(self, batch):
        return self.tokenizer.decode_batch(batch, skip_special_tokens=True)
        
    def collate(self, batch):

        _, _, response, context, prev_belief, _, target_belief, api_call, api_result, last_user, *_ = zip(*batch)

        context_token_ids, context_mask, context_labels = self.inputs_concat([     
            ('context', batchify(list(context), self.tokenizer, padding=False, enable_wrapping=False)),
            ('prev_belief',  batchify(list(prev_belief), self.tokenizer, padding=False, enable_wrapping=False)), 
            ('api_result',  batchify(list(api_result), self.tokenizer, padding=False, enable_wrapping=False)),
        ])

        """
        no  ... 0
        0   ... 1
        1   ... 2
        2   ... 3
        3   ... 4
        5   ... 5
        10  ... 6
        >10 ... 7
        """

        api_ids = []
        for a in api_result:
            v = a.split(' ')[-1]
            if v != "[DB]":
                v = int(v)
                v += 1
                if v == 5 or v == 6:
                    v = 5
                elif v <= 11:
                    v = 6
                else: 
                    v = 7
                api_ids.append(v)
            else:
                api_ids.append(0)

        query_token_ids, query_mask, query_labels = self.inputs_concat([          
            ('api_result',  batchify(list(api_result), self.tokenizer, padding=False, enable_wrapping=False)),
        #    ('last_user',   batchify(list(last_user), self.tokenizer, padding=False, enable_wrapping=False))       
        ], first_prefix_shift=False)

        response_token_ids, response_mask, response_labels = self.inputs_concat([   
        #    ('api_result',  batchify(list(api_result), self.tokenizer, padding=False, enable_wrapping=False)),
            ('response',    batchify(list(response), self.tokenizer, padding=False, enable_wrapping=False))
        ], first_prefix_shift=False)

        return_dict = {
            'conversation' :        context_token_ids,
            'conversation_mask' :   context_mask,    
            'conversation_labels' : context_labels,    
            'query' :               query_token_ids,
            'query_labels'  :       query_labels,
            'query_mask' :          query_mask,           
            'response' :            response_token_ids,
            'response_labels' :     response_labels,
            'response_mask' :       response_mask,      
            'api_ids' :             torch.LongTensor(api_ids)
        }

        return return_dict
