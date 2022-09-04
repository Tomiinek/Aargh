import torch
import hashlib
import json
from torchvision import transforms
from collections import deque, OrderedDict
from aargh.data.tasks.multiwoz import MultiWOZTask, DoubleLanguageModelMultiWOZ, PolicyOptimizationLanguageModelMultiWOZ
from aargh.utils.data import batchify, mask_until_last_nonempty
from aargh.data.transforms import PrefixAttributeTransform
from .hint_base import HintBaseGoalTask


class HintMultiWOZTask(HintBaseGoalTask, MultiWOZTask):
    
    def get_cache_filename(self):
        hint_path = self.params.try_get("hint_path", None)
        to_hash = str(self.params.context_length) + (hint_path if hint_path is not None else "no_hint")
        params_hash = hashlib.md5(to_hash.encode()).hexdigest()         
        return self.NAME + "_" + params_hash + ("_test" if self.is_testing else "_devel")

    def _data_to_items(self, data, is_testing):

        # None for dummy hints (empty)
        # str if there is a valid file with hints
        hint_path = self.params.try_get("hint_path", None)

        if hint_path is None:
            self.hints = None
        else:
            with open(self.params.hint_path, "r") as f:
                self.hints = json.load(f)

        return super()._data_to_items(data, is_testing)
            
    def _conversation_to_items(self, conversation):

        def normalize_action(action):     
            normalized = {}
            for k, v in action.items():
                normalized[k.lower()] = [tuple(x) for x in v]      
            return normalized

        accepted_items, num_rejected_items = [], 0
        context = deque(maxlen=self.params.context_length)  

        belief, old_belief, old_results, prev_actions = {}, {}, {}, {}
        active_domain = None
        reject = False
       
        conv_id = conversation["dialogue_id"].lower().split('.')[0]
        hint_c = 0
        if type(self.hints) is dict:
            if conv_id not in self.hints:
                return accepted_items, num_rejected_items
            hints = self.hints[conv_id]
        else:
            hints = ""

        for turn in conversation["turns"]:
            if turn["speaker"].lower() == "system":
                
                turn_action = normalize_action(turn["dialog_act"])
                utterance = turn["utterance"]  
                belief_difference = self._get_belief_difference(belief, old_belief)     

                #
                # find active domain

                belief_active_domains = set(belief_difference.keys())
                action_active_domains = set()
                general_domain = False
                for action in turn_action:
                    domain, _ = action.split('-')
                    if domain in ['restaurant', 'hotel', 'taxi', 'train', 'hospital', 'police', 'attraction']:
                        action_active_domains.add(domain)
                    if domain == 'general':
                        general_domain = True

                possibly_active_domains = action_active_domains | belief_active_domains

                if len(possibly_active_domains) == 1:
                    active_domain = next(iter(possibly_active_domains))
                elif len(possibly_active_domains) > 1:
                    if active_domain in possibly_active_domains:
                        possibly_active_domains.remove(active_domain)
                        active_domain = next(iter(possibly_active_domains))
                    elif len(belief_active_domains) == 1:
                        active_domain = next(iter(belief_active_domains))
                    elif len(action_active_domains) == 1:
                        active_domain = next(iter(action_active_domains))
                    else:
                        active_domain = next(iter(possibly_active_domains))
                else:
                    if general_domain:
                        active_domain = None

                # 
                # delexicalize utterance and accept it only if it is possible to lexicalize it back

                if self.delexicalize and not reject:
                    delexicalized_utterance = self._delexicalize_utterance(utterance, turn["span_info"], self.is_testing)
                    if delexicalized_utterance is None:  
                        num_rejected_items += 1
                        continue
                    
                    if active_domain is not None:
                        results = self._intents_to_db_results(active_domain, belief, turn_action)
                    else:
                        results = {}
                    lexicalized = self._lexicalize(delexicalized_utterance, belief, results)
                    
                    if lexicalized is None and not self.is_testing:
                        lexicalized = self._lexicalize(delexicalized_utterance, belief, old_results)
                        if lexicalized is None:
                            reject = True
                    # else:
                    #     if lexicalized.lower() != turn["utterance"].lower():
                    #         get_logger(self.NAME).warning(f'{highlight(lexicalized)}')
                    #         get_logger(self.NAME).warning(f'{highlight(turn["utterance"])}')
                    old_results = results.copy()
        
                #
                # add new item if not rejecting

                if reject or (type(hints) is list and (hint_c >= len(hints) or len(hints[hint_c]) == 0)):
                    num_rejected_items += 1
                    reject = False
                    continue
                    
                target_belief = belief_difference if self.differential_belief else belief
                target_utterance = delexicalized_utterance if self.delexicalize else utterance

                target_belief = OrderedDict(sorted(target_belief.items()))

                # add the active domain as a key to the target belief
                if active_domain is not None:
                    if active_domain not in target_belief:
                        target_belief[active_domain] = {}
                    target_belief.move_to_end(active_domain, last=False)

                item = self.DatasetItem(None, conversation["dialogue_id"], target_utterance, list(context), old_belief, belief, target_belief, 
                                        active_domain, None, hint=hints[hint_c][0] if type(hints) is list else hints, actions=turn_action, prev_actions=prev_actions)
                hint_c += 1

                if self.delexicalize: # save also the original utterance as it can be useful during evaluation
                    item.response_raw = utterance

                accepted_items.append(item)  

            else:
                old_belief = belief.copy()
                belief = {}
                prev_actions = normalize_action(turn['dialog_act'])
                reject = False

                #
                # get the belief state

                for frame in turn["frames"]:                    
                    domain, slots = self._parse_slots(frame, turn["utterance"], context, old_belief)
                    if domain is None and slots is None and not self.is_testing:
                        reject = True
                    elif slots is None:
                        continue
                    else:           
                        if domain == "bus":
                            domain = "train"          
                        belief[domain] = slots
         
            context.append({'speaker': turn["speaker"].lower(), 'utterance': turn["utterance"]}) 

        return accepted_items, num_rejected_items


class HintDoubleLanguageModelMultiWOZ(HintMultiWOZTask, DoubleLanguageModelMultiWOZ):
    
    NAME = "hint_double_lm_multiwoz"

    @classmethod
    def get_task_transforms(cls, params):
        return transforms.Compose([
            PrefixAttributeTransform('hint', params.try_get('hint_prefix', None)),
            DoubleLanguageModelMultiWOZ.get_task_transforms(params)
        ])

    def _get_response_batch(self, batch, agent):
        batch['response_mask'] = mask_until_last_nonempty(batch['response_labels']['hint']) #['api_result'])
        return agent.respond(batch, first_decoder=False, decoder_key='response')

    def _get_hint_batch(self, batch, agent):
        return agent.retrieve_hint(batch)

    def collate(self, batch):

        _, _, response, context, prev_belief, _, target_belief, api_call, api_result, hint, *_ = zip(*batch)

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
            ('hint',        batchify(list(hint), self.tokenizer, padding=False, enable_wrapping=False)),
            ('response',    batchify(list(response), self.tokenizer, padding=False, enable_wrapping=False))
        ], first_prefix_shift=False)

        api_ids = []
        for a in api_result:
            v = a.split(' ')
            if len(v) > 1:
                v = int(v[-1])
                v += 1
                if v < 5:      pass
                elif v < 7:   v = 5
                elif v < 13:  v = 6
                else:         v = 7
                api_ids.append(v)
            else:
                api_ids.append(0)
        
        return_dict = {
            'conversation' :        context_token_ids,
            'conversation_mask' :   context_mask,    
            'conversation_labels' : context_labels,       
            'hint_query' : {
                'context'     : list(context),
                'prev_belief' : list(prev_belief),
                'api_result'  : list(api_result)
            },
            'query' :               query_token_ids,
            'query_labels'  :       query_labels,
            'query_mask' :          query_mask,      
            'response' :            response_token_ids,
            'response_labels' :     response_labels,
            'response_mask' :       response_mask,  
            'api_ids' :             torch.LongTensor(api_ids)    
        }

        return return_dict


class HintPolicyOptimizationLanguageModelMultiWOZ(HintMultiWOZTask, PolicyOptimizationLanguageModelMultiWOZ):
    
    NAME = "hint_policy_lm_multiwoz"

    @classmethod
    def get_task_transforms(cls, params):
        return transforms.Compose([
            PrefixAttributeTransform('hint', params.try_get('hint_prefix', None)),
            PolicyOptimizationLanguageModelMultiWOZ.get_task_transforms(params)
        ])