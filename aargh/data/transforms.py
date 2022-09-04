import random


class BaseTransform:
    NAME = None

    def __init__(self):
        self.task = None

    def __call__(self, item):
        raise NotImplementedError()
    
    def set_task(self, task):
        self.task = task

    def get_new_tokens(self):
        """ Return a list of pairs of (new_token, is_special). """
        return []


class ExtractLastUserUtteranceTransform(BaseTransform):
    """ Take the last user utterance and place it to the `last_user` data item. """

    NAME = "extract_user_utterance"

    def __call__(self, item):
        item.last_user = item.context[-1]['utterance'] if len(item.context) > 0 else ""
        return item
        

class ContextDropoutTransform(BaseTransform):
    """ Randomly drop some utterances from the dialog context. """

    NAME = "context_dropout"

    def __init__(self, dropout_ratio, preserve_latest):
        """
        Arguments:
            dropout_ratio (float): Probability of dropping each utterance of the dialog history.
            preserve_latest (bool): If set to True, the latest utterance is always preserved.
        """
        super().__init__()
        self.ratio = dropout_ratio
        self.preserve_latest = preserve_latest

    def __call__(self, item):

        #new_context = []
        for i, c in enumerate(item.context):
            if random.random() < self.ratio and (i != len(item.context) or not self.preserve_latest):
                item.context[i]['utterance'] = ''
            #new_context.append(c)

        # item.context = new_context
        return item


class StateDropoutTransform(BaseTransform):
    """ Randomly drop entries from the dialog state. """

    NAME = "state_dropout"

    def __init__(self, dropout_ratio):
        """
        Arguments:
            dropout_ratio (float): Probability of dropping each slot-value pair of the dialog state.
        """
        super().__init__()
        self.ratio = dropout_ratio

    def __call__(self, item):

        new_state = {}
    
        for domain, values in item.prev_belief.items():
            if domain not in new_state:
                new_state[domain] = {}
            for slot, value in values.items():
                if random.random() < self.ratio:
                    continue
                new_state[domain][slot] = value
            if len(new_state[domain]) == 0:
                new_state.pop(domain)

        item.prev_belief = new_state
        return item


class LinearizeContextPairTransform(BaseTransform):
    """ Convert (speker, turn) pairs of context vectors into strings. """

    NAME = "linearize_context_pair"

    def __init__(self, system_prefix, user_prefix):
        """
        Convert (speker, turn) pairs of context vectors into strings. If the context does not 
        contain infromation about speakers, i.e. `system` or `user`, the speaker labels are omitted at all. 
        Otherwise, `system_label` and `user_label` are used for the speaker `system` and `user`, respectively. 

        Arguments:
            system_prefix (str): System prefix prepended to system turns.
            user_prefix (str): User prefix prepended to user turns.
        """
        super().__init__()
        self.s = system_prefix
        self.u = user_prefix

    def __call__(self, item):
        for i, turn in enumerate(item.context):
            if isinstance(turn, str):
                continue
            speaker = turn["speaker"]
            utterance = turn["utterance"]
            if speaker == 'system':
                item.context[i] = self.s + utterance
            elif speaker == 'user':
                item.context[i] = self.u + utterance
            else:
                raise ValueError(f"Expected system/user labels for utterances, given: {speaker}!")
        return item

    def get_new_tokens(self):
        return [(self.s, True) , (self.u, True)]


class LinearizeContextTransform(BaseTransform):
    """ 
    Linearize context vector of shape List[str] into a single string. 
    Note: You can use `LinearizeContextPairTransform` if the context vector is made of (speaker, turn) pairs.
    """

    NAME = "linearize_context"

    def __init__(self, join_string=''):
        super().__init__()
        self.j = join_string

    def __call__(self, item):
        item.context = self.j.join(item.context)
        return item


class LinearizeAttributePairTransform(BaseTransform):
    """ Convert (key, value) pairs of the given attribute (e.g. balief state, api calls, ...) into strings."""

    NAME = "linearize_attribute_pairs"

    def __init__(self, attribute, join_string=': ', value_func=None):
        """
        Convert (key, value) pairs of a given attribute into a list of strings lexicographically sorted by the original keys.

        Arguments:
            attribute (str): Name of the attribute of a data item containing the key-value pairs to be converted to strings.
            join_string (str, default ': '): The delimiter to be added between keys and values.
            value_func (function, optional): Function applied to the second element of the pairs (value) in order to convert them to stirngs.
        """
        super().__init__()
        self.attribute = attribute
        self.join_string = join_string
        self.value_func = value_func

    def __call__(self, item):
        container = getattr(item, self.attribute)

        for key, value in container.items():
            new_values = []
            
            for k, v in sorted(value.items()):
                if self.value_func is None:
                    new_values.append(k + self.join_string + v)
                else:
                    new_values.append(k + self.join_string + self.value_func(v)) 

            container[key] = new_values   

        return item


class ApiResultsToTokenTransform(BaseTransform):
    
    NAME = "api_to_token"
    
    def __init__(self, attribute_name):
        super().__init__()
        self.attribute = attribute_name

    def __call__(self, item):

        items = getattr(item, self.attribute).items()

        booking = None
        quantity = None
        for domain, results in items:
            if quantity is None:
                quantity = len(results['results'])
            else:
                quantity = max(len(results['results']), quantity)
            if results['booking'] is not None:
                booking = results['booking']

        b = ''
        if booking is not None:
            if booking:
                b = '+success'
            else:
                b = '+fail'

        q = 'no'
        if quantity is not None:
            if quantity == 0:
                q = '0'
            elif quantity == 1:
                q = '1'
            elif quantity <= 3:
                q = 'few'
            else: 
                q = 'several'

        #print("--------------------------")
        #print({k : str(v['booking']) + str(len(v['results'])) for k, v in items})
        #print(f"<|{q+b}|>")

        setattr(item, self.attribute, f"<|{q+b}|>")

        return item

    def get_new_tokens(self):
        db_tokens = []
        for book in ['', '+success', '+fail']:
            for quantity in ['no', '0', '1', 'few', 'several']:
                db_tokens.append(quantity + book)
        return [(f'<|{x}|>', True) for x in db_tokens]


class LinearizeAttributeTransform(BaseTransform):
    """ 
    Linearize attribute (e.g. belief state, api calls, ...) of type Dict[str, str] or Dict[str, List[str]] into a single string. 
    Note: You can use `LinearizeAttributePairTransform` beforehand to linearize the contained (key, value) pairs.
    """

    NAME = "linearize_attribute"

    @staticmethod
    def wrap_parenthesis(x): return '' if x == '' else ' [' + x + ']'

    @staticmethod
    def to_list_str(key, value): 
        if not isinstance(value, list):
            value = [str(value)]
        return key, value

    def __init__(self, attribute_name, ignore_keys=False, key_join_string=', ', value_join_string=', ', 
                       preprocess_func=None, value_func=None, sort_by_key=True, sort_values=True):
        """ 
        Linearize attribute of type Dict[str, List[str]] or Dict[str, str] into a single string. If `preprocess_func` is defined, the attribute is first
        passed through this functions.

        Arguments:
            attribute_name (str): Name of the attribute of a data item to be converted into a single string (e.g. belief, api_calls, etc.)
            join_string (str, default ', '): The delimiter to be added between items.
            ignore_keys (bool, default `False`): If is true, keys are ignored and not used, otherwise results are wrapped into `Key { Linearized Values }`.
            preprocess_func (function): Function applied to the raw input key and attribute (accepts two argumnets and returns (key, value) pair).
            value_func (function): Function applied to values in order to wrap them into some parenthessis or something like that.
            sort_by_key (bool, default True): If is true, the order of linearized key-value pairs is determined by the key.
            sort_values (bool, default True): If is true and the values are lists, they are lexicographically sorted before the linearization.  
        """
        super().__init__()
        self.attribute = attribute_name
        self.key_join_string = key_join_string
        self.value_join_string = value_join_string
        self.ignore_keys = ignore_keys
        self.preprocess_func = preprocess_func if preprocess_func is not None else self.to_list_str
        self.value_func = value_func if value_func is not None else self.wrap_parenthesis
        self.sort_by_key = sort_by_key
        self.sort_values = sort_values

    def __call__(self, item):
        values = []

        items = getattr(item, self.attribute).items()

        if self.sort_by_key:
            items = sorted(items)

        for k, b in items:
            if self.preprocess_func is not None:
                k, b = self.preprocess_func(k, b)

            if self.ignore_keys:
                if len(b) == 0:
                    continue
                if self.sort_values:
                    b = sorted(b)
                if self.value_join_string is not None:
                    b = self.value_join_string.join(b)
                values.append(self.value_func(b))    
            else:
                if self.sort_values:
                    b = sorted(b)
                if self.value_join_string is not None:
                    b = self.value_join_string.join(b)
                values.append(k + self.value_func(b))

        setattr(item, self.attribute, self.key_join_string.join(values))

        return item


class PrefixAttributeTransform(BaseTransform):
    """ Add a prefix to string attribute of the item. """

    NAME = "prefix_attribute"

    def __init__(self, attribute_name, prefix):
        """
        Add `prefix` to a string stored in the attribute (defined by `attribute_name`) of dataset items. 

        Arguments:
            attribute_name (str): Name of the attribute of a data item to be changed.
            prefix (str): Prefix to be added to the attribute defined by `attribute_name`.
        """
        super().__init__()
        self.a = attribute_name
        self.p = prefix

    def __call__(self, item):
        if not self._is_used():
            return item
        attribute_str = getattr(item, self.a)
        if attribute_str is None:
            return item
        setattr(item, self.a, self.p + attribute_str)
        return item

    def get_new_tokens(self):
        if not self._is_used():
            return []
        return [(self.p, True)]

    def _is_used(self):
        return self.p is not None and self.p != ''


class HintToResponseTransform(BaseTransform):
    """ Randomly replace the respnse hint with ground-truth response. """

    NAME = "hint_to_response"

    def __init__(self, rate):
        """
        For the given example, the response hint is replaced with the corresponding response.

        Arguments:
            rate (float): Probability of replacement with the response.
        """
        super().__init__()
        self.rate = rate

    def __call__(self, item):
        if random.random() < self.rate:
            item.hint = item.response
        return item