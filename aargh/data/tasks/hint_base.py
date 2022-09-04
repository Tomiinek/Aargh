import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from torchvision import transforms
import aargh.data as data
from aargh.data.tasks.base import BaseSupervisedTask


class HintBaseGoalTask(BaseSupervisedTask):
    
    @dataclass
    class DatasetItem(BaseSupervisedTask.DatasetItem):
        response: str = ''
        context: List[Dict[str, str]] = None
        prev_belief: Dict[str, Dict[str, str]] = field(default_factory=dict)   # dictionary of domain belief states from the previous dialog turn
        belief: Dict[str, Dict[str, str]] = field(default_factory=dict)        # dictionary of domain belief states
        target_belief: Dict[str, Dict[str, str]] = field(default_factory=dict) # same as belief if differential_belief is False
        api_call: Dict[str, Dict[str, str]] = None    # API calls, each has a dictionary of key-value arguments
        api_result: Dict[str, Dict[str, str]] = None  # API call results, each call has a dicionary of key-value results
        hint: str = ''                                # response hint, possibly from a retrieval model
        actions: Dict[str, List[Tuple[str]]] = None   # dictionary of dialog actions with list of slot-value pairs
        prev_actions: Dict[str, List[Tuple[str]]] = None # dictionary of dialog actions with list of slot-value pairs of the previous user turn

    def get_responses(self, agent, items, ground_truth_keys, **kwargs):

        # ground truth keys should be in {context, prev_belief, hint, belief, api_call, api_result}
        
        if 'api_result' in ground_truth_keys:
            ground_truth_keys.add('api_call')
            ground_truth_keys.add('belief')
         
        for i in items:
            i.reset_keys_except(ground_truth_keys)

        # save previous api_results because of the lexicalization
        if 'api_results' in kwargs:
            old_api_results = kwargs['api_results']
        else:
            old_api_results = [{} for i in items]

        if self.differential_belief and 'prev_belief' not in ground_truth_keys and 'state' in kwargs:
            for i, b in zip(items, kwargs['state']):
                i.prev_belief = b

        if 'api_call' in ground_truth_keys:
            gt_api_calls = [i.api_call for i in items]
        
        batch = self._prepare_inference_batch(items)
        query_batch = self._get_query_batch(batch, agent)
        predicted_beliefs, api_calls = self._decode_query_batch(query_batch)
            
        if 'api_call' in ground_truth_keys:
            api_calls = gt_api_calls

        if 'belief' in ground_truth_keys:
            merged_beliefs = [i.belief for i in items]
        else:
            merged_beliefs = [self._belief_postprocess(ob, pb) for ob, pb in zip([i.prev_belief for i in items], predicted_beliefs)]

        raw_items = [i for i in items]
        if 'api_result' not in ground_truth_keys:
            api_results = []
            for i, item in enumerate(items):
                item.target_belief = predicted_beliefs[i]
                item.belief = merged_beliefs[i]
                item.api_call = api_calls[i]
                item.api_result = None
                # a little bit ugly way to get API results and format belief string with respect to api calls
                self._lazy_prepare_item_hook(item)
                api_results.append(item.api_result)
                raw_items[i] = copy.copy(item)
                items[i] = self._prepare_item(item)
            batch = self.collate(items)
        else:
            api_results = [i.api_result for i in items]

        if 'hint' not in ground_truth_keys:
            items = raw_items
            hint_batch = self._get_hint_batch(batch, agent)
            for i, item in enumerate(items):
                item.hint = hint_batch[i]
                items[i] = self._prepare_item(item)
            batch = self.collate(items)

        response_batch = self._get_response_batch(batch, agent)
        responses = self._decode_response_batch(response_batch)

        # build output dictionary
        result = {'response' : responses, 'state' : merged_beliefs, 'api_call' : api_calls, 'api_results' : api_results, 'hint' : hint_batch}
        if self.differential_belief:
            result['state_update'] = predicted_beliefs
        
        # lexicalize predicted response
        if self.delexicalize:   
            lexicalized_responses = []
            for i, (r, b, a) in enumerate(zip(responses, merged_beliefs, api_results)):
                lexicalized = self._lexicalize(r, b, a)
                if lexicalized is None and old_api_results is not None:
                    lexicalized = self._lexicalize(r, b, old_api_results[i])
                lexicalized_responses.append(lexicalized if lexicalized is not None else r)
            
            result['response_raw'] = responses
            result['response'] = lexicalized_responses

        return result