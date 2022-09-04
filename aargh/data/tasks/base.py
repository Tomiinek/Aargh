import os
import copy
import torch
import pickle
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from dataclasses import dataclass, field, fields, astuple
from typing import List, Dict, Union, Tuple
from torch.nn.utils.rnn import pad_sequence
import aargh.data as data
from aargh.utils.logging import get_logger


class BaseTask(Dataset):
    
    NAME = None
    CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".cache"))
    
    @dataclass
    class DatasetItem:
        """ Class holding a single example used for training of the target model. """
        def __iter__(self):
            return iter(astuple(self)) 

        def reset_keys_except(self, keys):
            dummy = type(self)()
            for f in fields(self):
                if f.name in keys:
                    continue
                setattr(self, f.name, getattr(dummy, f.name)) 

    def __init__(self, params, is_testing, is_training=False, tokenizer=None, augment_transforms=None):
        """
        Build task for training, evaluating and testing.

        Arguments:
            params (dictionary): Dictionary with necessary hyperparameters.
            is_testing (bool): If False, train and val folds are build, otherwise only the test fold.
            tokenizer (TokenizerWrapper): Tokenizer to be used for data tokenization.
            augment_transforms (collable, optional): Transform used for data items augmentation (in the getter function).
        """
        super().__init__()     

        self.params = params
        self.is_testing = is_testing
        self.tokenizer = tokenizer
        self.task_transforms = self.get_task_transforms(params)
        self.augment_transforms = augment_transforms

        def set_transform_task(ts):
            if ts is None:
                return
            ts = ts.transforms if isinstance(ts, transforms.Compose) else ts
            for t in ts:
                if isinstance(t, transforms.Compose):
                    set_transform_task(t)
                else:
                    t.set_task(self)

        set_transform_task(self.task_transforms)
        set_transform_task(self.augment_transforms)

        if not is_testing and is_training: # i.e. we are validating
            self.set_training()
        else:
            self.unset_training()
        
    def collate(self, batch):
        """
        Collate function to merge a list of samples to mini-batch tensor.
         - Use self.tokenizer and tokenizer.encode_batch(list[str]) to prepare the batch.

        Arguments:
            batch (list[DatasetItem]): Mini-batch to be modified by this method.
        """
        raise NotImplementedError()

    def get_responses(self, agent, **inputs):
        """ 
        Prapare the given `context` for `agent` and use the agent for generating the next system response.
        Note: this must be done in the task, as it can prepare batches (use tokenizer etc.) for the agent and has special things such as the database.

        Arguments:
            contexts (List[List[str]]): A list of contexts for the particular conversations.
            agent (BaseTrainableAgent): The model trained that was trained on this task and that will be used for generaing the responses.
            inputs (List[...]): Additional system input arguments needed for successful response prediction, each is a list (contexts, states etc.)

        Returns:
            A list of dictionaries of predictions or any important ouputs (e.g. {'response' : ..., 'belief' : ..., 'my_custom_key' : ...}).
            The only key that must be present in each dictionary is `response`.
        """
        raise NotImplementedError()

    @classmethod
    def prepare_data(cls):
        """ Prepare the task data before the first usage, i.e. download data. """ 
        pass

    @classmethod
    def setup(cls):
        """ Setup and initialize the task, i.e. load data, environment, etc. """
        pass

    @classmethod
    def get_task_transforms(cls, params):
        """ Return a callable transform to be applied to items of the task (without user-defined augmentation transforms). """
        return None

    @staticmethod
    def _combine_transforms(task_transforms, augment_transforms):
        """ Combine the task transforms and user-defined augmentation transforms (their order may change for example). """
        return transforms.Compose([task_transforms, augment_transforms])

    def _lazy_prepare_item_hook(self, item):
        """ This hook can be used for lazy in-place item modification (such as database querying). """
        pass

    def _prepare_inference_batch(self, items):
        """
        Prepare a list of DatasetItem for inference, i.e. apply inference-time transforms and convert to a mini-batch.
         1) Prepare items, i.e. applies `_lazy_prepare_item_hook` and transforms.
         2) Performs the collate function which can run the tokenizer, prepare tensors, etc.

        Arguments:
            items (list[DatasetItem]): List of items to be used for mini-batch preparation.
        """
        items = [self._prepare_item(i) for i in items]
        return self.collate(items)

    def _prepare_item(self, item):
        self._lazy_prepare_item_hook(item)
        if self.transforms is not None:
            item = copy.deepcopy(item)
            item = self.transforms(item)
        return item

    @staticmethod
    def get_new_tokens():
        """ Returns list of task-specific tokens that might be used by the tokenizer. The list contains pairs of (token, is_special). """
        return []

    def set_training(self):
        if self.augment_transforms is None:
            self.transofrms = self.task_transforms
        else:
            self.transforms = self._combine_transforms(self.task_transforms, self.augment_transforms)
        
    def unset_training(self):
        self.transforms = self.task_transforms


class BaseSupervisedTask(BaseTask):

    VAL_SIZE = None
    DATASETS = []

    @dataclass
    class DatasetItem(BaseTask.DatasetItem):
        idx : int = None
        conv_id : Union[int, str] = None

    static_data = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.static_data is None:
            get_logger(self.NAME).warning(f"Using the task named: {self.NAME} without initialized data!")
            get_logger(self.NAME).warning(f"Consider calling {type(self).__name__}.prepare() beforehand.")
            return
    
        task_items_cache = os.path.join(self.CACHE_DIR, self.get_cache_filename())
        if os.path.exists(task_items_cache):
            get_logger(self.NAME).info(f"Dataset examples loaded from a cached file: {task_items_cache}")
            with open(task_items_cache, "rb") as f:
                train_items, val_items, test_items = pickle.load(f) 
        else:
            get_logger(self.NAME).info("Preparing dataset examples from raw data, this can take a while.")
            items = self._data_to_items(self.static_data, self.is_testing)
            
            if not os.path.exists(self.CACHE_DIR):
                os.makedirs(self.CACHE_DIR)
            with open(task_items_cache, "wb+" ) as f:
                get_logger(self.NAME).info(f"Dataset examples saved into a cached file: {task_items_cache}")
                pickle.dump(items, f)
            train_items, val_items, test_items = items

        if self.is_testing:
            self.items = test_items
            return
        
        self.items = train_items
        if val_items is None:
            val_size = int(self.VAL_SIZE if self.VAL_SIZE > 1 else len(train_items) * self.VAL_SIZE)
            train_indices, val_indices = random_split(range(len(self)), [int(len(self) - val_size),int(val_size)])
        else:
            tl = len(self.items)
            self.items.extend(val_items)
            train_indices, val_indices = range(tl), range(tl, len(self.items), 1) 

        val_dataset = copy.copy(self)
        val_dataset.unset_training()
        val_dataset.items = [self.items[i] for i in val_indices]
        self.val = val_dataset

        self.set_training()
        self.items = [self.items[i] for i in train_indices]
        self.train = self

    def get_cache_filename(self):
        return self.NAME + ("_test" if self.is_testing else "_devel")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        item.idx = index
        return self._prepare_item(item)

    def get_items(self):
        return self.items

    def _data_to_items(self, data, is_testing):
        """
        Construct task, i.e. convert a dictionary of raw dataset items into task items.
         - Should return three lists of items: train, val, test. Train cannot be None.
         - If val is None, VAL_SIZE parameter used to split train into two folds.
         - If test is None, testing is ommited at all further in the pipeline.
         - This is the suitable place to build database, delexicalize responses etc.

        Arguments:
            data (dictionary): A dictionary of raw loaded dataset items.
            is_testing (bool): True if preparing data for testing, otherwise False.
        """
        raise NotImplementedError()

    @classmethod
    def prepare_data(cls):
        for d, v in cls.DATASETS:
            data.abstract.AutoLoader.from_name(d, v).download()

    @classmethod
    def setup(cls):
        if cls.static_data is not None:
            return
        dataset_data = {}
        for d, v in cls.DATASETS:
            loader = data.abstract.AutoLoader.from_name(d, v)
            dataset_data[(d, v)] = loader.get_items()
        cls.static_data = dataset_data
        

class BaseLanguageModelTask(BaseSupervisedTask):

    @classmethod
    def get_task_transforms(cls, params):
        return transforms.Compose([
            data.transforms.LinearizeContextPairTransform(params.try_get('system_prefix', None), params.try_get('user_prefix', None)),
            data.transforms.LinearizeContextTransform(),
            data.transforms.PrefixAttributeTransform('response', params.try_get('response_prefix', None)),
            data.transforms.PrefixAttributeTransform('context', params.try_get('context_prefix', None)),
        ])

    @staticmethod
    def _combine_transforms(task_transforms, augment_transforms):
        return transforms.Compose([augment_transforms, task_transforms])

    def inputs_concat(self, inputs, pad_idx=-100, first_prefix_shift=True):
        
        token_ids, mask = [], []
        labels = { k[0] : [] for k in inputs } 
        add_bos_eos = self.params.try_get("sequence_tokens", False)

        # iterate through batch
        batch_size = len(inputs[0][1].token_ids)
        for i in range(batch_size):
            
            item_labels =  { k[0] : [] for k in inputs }
            item_token_ids = []
            item_mask_lens = [] 
            cl = 0
            
            for k, e in inputs:
                tokens = e.token_ids[i]
                item_token_ids.extend(tokens)
                item_mask_lens.append(len(tokens))
            
            if first_prefix_shift:
                for key in item_labels:
                    item_labels[key].append(pad_idx)
                l_addition = 0
                cl += 1
            else:
                l_addition = 1

            if add_bos_eos:
                for key in item_labels:
                    item_labels[key].append(pad_idx)
                cl += 1
                item_token_ids.insert(0, self.tokenizer.get_bos_token_id())
                item_token_ids.append(self.tokenizer.get_eos_token_id())
           
            for l, (k, _) in zip(item_mask_lens, inputs):
                l += l_addition
                for key in item_labels:
                    if cl + l > len(item_token_ids):
                        l -= 1
                    if key == k:
                        item_labels[key].extend(item_token_ids[cl:cl+l])
                    else:
                        item_labels[key].extend([pad_idx] * l) 
                cl += l
                l_addition = 0

            for key in item_labels:
                labels[key].append(torch.tensor(item_labels[key]))
            
            token_ids.append(torch.tensor(item_token_ids))
            mask.append(torch.ones_like(token_ids[-1]))  

        pad_id = self.tokenizer.get_pad_token_id()
        token_ids = pad_sequence(token_ids, padding_value=pad_id, batch_first=True)
        mask = pad_sequence(mask, padding_value=0, batch_first=True)
        for k, v in labels.items():
            labels[k] = pad_sequence(v, padding_value=pad_idx, batch_first=True)

        return token_ids, mask, labels


class BaseChatTask(BaseSupervisedTask):
    
    @dataclass
    class DatasetItem(BaseSupervisedTask.DatasetItem):
        response: str = ''
        context: List[Dict[str, str]] = None

    def _get_response_batch(self, batch, agent):
        return agent.respond(batch)
    
    def _decode_response_batch(self, batch):
        """ Decode the given batch of targets (predicitons) into a list of response strings. """
        raise NotImplementedError()

    def get_responses(self, agent, items, ground_truth_keys, **kwargs):
        for i in items:
            i.reset_keys_except(ground_truth_keys)
        batch = self._prepare_inference_batch(items)
        response_batch = self._get_response_batch(batch, agent)
        responses = self._decode_response_batch(response_batch)
        return {'response' : responses}


class BaseGoalTask(BaseSupervisedTask):
    
    @dataclass
    class DatasetItem(BaseSupervisedTask.DatasetItem):
        response: str = ''
        context: List[Dict[str, str]] = None
        prev_belief: Dict[str, Dict[str, str]] = field(default_factory=dict)   # dictionary of domain belief states from the previous dialog turn
        belief: Dict[str, Dict[str, str]] = field(default_factory=dict)        # dictionary of domain belief states
        target_belief: Dict[str, Dict[str, str]] = field(default_factory=dict) # same as belief if differential_belief is False
        api_call: Dict[str, Dict[str, str]] = None    # API calls, each has a dictionary of key-value arguments
        api_result: Dict[str, Dict[str, str]] = None  # API call results, each call has a dicionary of key-value results
        actions: Dict[str, List[Tuple[str]]] = None   # dictionary of dialog actions with list of slot-value pairs
        prev_actions: Dict[str, List[Tuple[str]]] = None # dictionary of dialog actions with list of slot-value pairs of the previous user turn

    def __init__(self, params, *args, **kwargs):
        self.delexicalize = params.try_get("delexicalize_response", False)
        self.differential_belief = params.try_get("differential_belief", False)
        self.remove_label = params.try_get("differential_remove_label", "none")
        super().__init__(params, *args, **kwargs)

    def api_call(self, api_name, **kwargs):
        """ Method implementing API or database calls. """
        raise NotImplementedError()

    def _lexicalize(self, utterance, belief, api_results):
        """ Lexicalize the given delexicalized response with respect to dialog state and results of api calls. """
        raise NotImplementedError()

    def _get_response_batch(self, batch, agent):
        """ Use the given batch and the agent to predict next responses. """
        raise NotImplementedError()

    def _get_query_batch(self, batch, agent):
        """ Use the given batch and the agent to predict next queries, i.e. the pairs of beliefs and api_calls. """
        raise NotImplementedError()

    def _decode_response_batch(self, batch):
        """ Decode the given batch of targets (predicitons) into a list of response strings. """
        raise NotImplementedError()

    def _decode_query_batch(self, batch):
        """ Decode the given batch of targets (predicitons) into a list of beliefs. """
        raise NotImplementedError()
 
    def _belief_postprocess(self, old_belief, new_belief):
        
        if not self.differential_belief:
            return new_belief

        combined = copy.deepcopy(old_belief)
        for domain in new_belief:
            if domain not in combined:
                combined[domain] = {}
            for slot in new_belief[domain]:
                if slot not in combined[domain] or (new_belief[domain][slot] != combined[domain][slot]):
                    combined[domain][slot] = new_belief[domain][slot]
                if new_belief[domain][slot] == [self.remove_label]:
                    combined[domain].pop(slot)
            if len(combined[domain]) == 0:
                combined.pop(domain)

        return combined

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
                items[i] = self._prepare_item(item)
            batch = self.collate(items)
        else:
            api_results = [i.api_result for i in items]


        response_batch = self._get_response_batch(batch, agent)
        responses = self._decode_response_batch(response_batch)

        # build output dictionary
        result = {'response' : responses, 'state' : merged_beliefs, 'api_call' : api_calls, 'api_results' : api_results}
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
