import os
import re
from dataclasses import dataclass
from typing import List
from transformers import AutoTokenizer


class TokenizerWrapper:    
    """ This wrapper cannot subclass Tokenzier because it is a final class. """

    NAME = None
    DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".tokenizers"))

    @dataclass
    class Encoding:
        token_ids: List[int]
        attention_mask: List[int]

    def __init__(self):
        pass

    def __init__(self, sequence_tokens=False):
        super().__init__()
        self.add_sequence_tokens = sequence_tokens
        self.load()

    def load(self, *args, **kwargs):
        """ Load the underlying tokenizer from file and initialize it. """
        raise NotImplementedError()

    def encode_batch(self, batch):
        """ Encode a single batch of data. """
        raise NotImplementedError()

    def decode_batch(self, batch):
        """ Decode a single batch of data. """
        raise NotImplementedError()

    def add_tokens(self, tokens):
        special = [t[0] for t in tokens if t[1]]
        regular = [t[0] for t in tokens if not t[1]]
        self._add_special_tokens(special)
        self._add_regular_tokens(regular)

    def _add_regular_tokens(self, tokens):
        """ Add additional regular tokens that will not be splitted etc. """
        raise NotImplementedError() 

    def _add_special_tokens(self, tokens):
        """ Add additional special tokens that will not be splitted etc. """
        raise NotImplementedError() 

    def get_id_token(self, id):
        """ Convert the given id to its corresponding token if it exists (None otherwise). """
        raise NotImplementedError()

    def get_token_id(self, token):
        """ Convert the given token to its corresponding id. """
        raise NotImplementedError()

    def get_vocabulary_size(self):
        """ Get the size of the vocabulary used by this tokenizer. """
        raise NotImplementedError()

    def get_pad_token_id(self):
        """ Get ID of the pad token. """
        raise NotImplementedError()

    def get_eos_token_id(self):
        """ Get ID of the EOS token. """
        raise NotImplementedError()

    def get_bos_token_id(self):
        """ Get ID of the BOS token. """
        raise NotImplementedError()

    @classmethod
    def get_cache_file_path(cls, name): 
        return os.path.join(cls.DIR, f"{name}.json" )

    @classmethod
    def build(cls, files, name, *args, **kwargs):
        """ Train on files & save the underlying tokenizer. """
        raise NotImplementedError()


class HFTokenizerBase(TokenizerWrapper):

    def get_pretrained_name(self):
        """ Get name of the pretrained model used for instantiating the tokenizer. """
        raise NotImplementedError()

    def get_vocabulary_size(self):
        return len(self.tokenizer)

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def get_eos_token_id(self):
        return self.tokenizer.eos_token_id

    def get_bos_token_id(self):
        return self.tokenizer.bos_token_id

    def get_id_token(self, token_id):
        return self.tokenizer.convert_ids_to_tokens(token_id)

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def load(self):      
        self.tokenizer = AutoTokenizer.from_pretrained(self.SOURCE)
        #self.tokenizer.add_special_tokens({'pad_token': '<|padtoken|>'})
        
    def _add_regular_tokens(self, tokens):
        self.tokenizer.add_tokens(tokens)

    def _add_special_tokens(self, tokens):
        self.tokenizer.add_special_tokens({'additional_special_tokens' : tokens})
        
    def encode_batch(self, batch, padding=True, enable_wrapping=True):

        if enable_wrapping and self.add_sequence_tokens:
            for i, x in enumerate(batch):
                batch[i] = self.tokenizer.bos_token + x + self.tokenizer.eos_token

        encoding = self.tokenizer(batch, padding=padding)
        return self.Encoding(encoding.input_ids, encoding.attention_mask)

    def decode_batch(self, batch, skip_special_tokens=False):
        return self.tokenizer.batch_decode(batch, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)


class BertTokenizer(HFTokenizerBase):
    NAME = "bert"
    SOURCE = 'bert-base-uncased'

    def encode_batch(self, batch, padding=True, enable_wrapping=True):
        
        if enable_wrapping and self.add_sequence_tokens:            
            for i, x in enumerate(batch):                            
                batch[i] = self.tokenizer.cls_token + x + self.tokenizer.sep_token

        encoding = self.tokenizer(batch, padding=padding)
        encoding.input_ids = [x[1:-1] for x in encoding.input_ids] # because this tokenizer adds [CLS] + x + [SEP] by default 
        encoding.attention_mask = [x[1:-1] for x in encoding.attention_mask] # because this tokenizer adds [CLS] + x + [SEP] by default 

        return self.Encoding(encoding.input_ids, encoding.attention_mask)

    def get_bos_token_id(self):
        return self.tokenizer.cls_token_id

    def get_eos_token_id(self):
        return self.tokenizer.sep_token_id


class T5Tokenizer(HFTokenizerBase):
    NAME = "t5"
    SOURCE = 't5-base'
    
    def encode_batch(self, batch, padding=True, enable_wrapping=True):

        if enable_wrapping and self.add_sequence_tokens:
            for i, x in enumerate(batch):
                batch[i] = self.tokenizer.bos_token + x + self.tokenizer.eos_token

        encoding = self.tokenizer(batch, padding=padding)
        encoding.input_ids = [x[:-1] for x in encoding.input_ids] # because this tokenizer addds </s> at the end by default 
        encoding.attention_mask = [x[:-1] for x in encoding.attention_mask]  

        return self.Encoding(encoding.input_ids, encoding.attention_mask)
    
    def decode_batch(self, batch, skip_special_tokens=False):
        decoded = super().decode_batch(batch, skip_special_tokens=skip_special_tokens)
        return [re.sub(r"([^ ])\[", r"\1 [", d) for d in decoded]

    def get_bos_token_id(self):
        return self.tokenizer.pad_token_id