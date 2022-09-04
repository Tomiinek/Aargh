import os
import pickle
import torch
import torch.nn.functional as F
from scipy import spatial
from torch.nn import Module, Linear, Dropout, Embedding
from einops import reduce
from transformers import BertModel
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup as CosineWarmup
from pytorch_lightning.core.decorators import auto_move_data
from aargh.agents.hf_base import HFBaseAgent
from .action_base import ActionAgentBase, ActionBatchSampler


class RetrievalBertAgentBase(HFBaseAgent):

    class ScalerLayer(Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self.w = torch.nn.Parameter(torch.tensor(float(scale)), requires_grad=True)
            #self.b = torch.nn.Parameter(torch.tensor(-scale / 2), requires_grad=True)

        def forward(self, x):     
            torch.clamp(self.w, 1e-6)     
            return self.w * x #+ self.b
        
    def __init__(self, params=None, *args, **kwargs):
        super().__init__(params=params, *args, **kwargs)
        self.similarity_scaler = self.ScalerLayer(self.hparams.similarity_scale)
        self.support_cache = None
    
    def get_target_label_keys(self):
        return []

    def instantiate_model(self):
        return BertModel.from_pretrained("bert-base-uncased") #("TODBERT/TOD-BERT-JNT-V1") #("bert-base-uncased") 

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        parameters = [
            {'params': [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith('similarity_scaler') and not any(nd in n for nd in no_decay)], 
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith('similarity_scaler') and any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0},
            {'params': [p for n, p in self.similarity_scaler.named_parameters()], 'lr' : 0.01, 'weight_decay': 0.0 }
        ]
        optimizer = AdamW(parameters, lr=self.hparams.learning_rate)
        scheduler = {     
            'scheduler': CosineWarmup(optimizer, self.hparams.num_warmup_steps, self.hparams.num_training_steps),                
            'interval': 'step'
        }
        return [optimizer], [scheduler]

    def pool(self, x, mask):
        mask = mask.unsqueeze(-1)
        if self.hparams.pooling == "cls_token":
            x = x.pooler_output
        elif self.hparams.pooling == "max":
            x = x.last_hidden_state
            x[~mask] = -1e9  # Set padding tokens to large negative value
            x = reduce(x, 'b l d -> b d', 'max')
        elif self.hparams.pooling == "average":
            x = x.last_hidden_state * mask
            x = reduce(x, 'b l d -> b d', 'sum')
            t = reduce(mask, 'b l d -> b d', 'sum')
            t = torch.clamp(t, min=1e-9)
            x = x / t
        else:
            raise ValueError(f"Pooling strategy must be on of `cls_token`, `max`, `average`, given: {self.hparams.pooling}")
        return x

    def calculate_loss(self, batch, encoded):
        return None

    def training_step(self, batch, batch_idx, dataloader_idx=None):  
        if type(batch) is list:
            batch = batch[0]
        elif type(batch) is dict:
            batch = batch["custom"]

        encoded = self(batch)
        loss_dict = self.calculate_loss(batch, encoded)
        for n, l in loss_dict.items():
            self.log(f'train_{n}_loss', l, sync_dist=True, prog_bar=True)
        self.log("sim_scale", self.similarity_scaler.w, prog_bar=True)
        return loss_dict['total']

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is not None and dataloader_idx > 0:
            return

        encoded = self(batch)
        loss_dict = self.calculate_loss(batch, encoded)
        for n, l in loss_dict.items():
            self.log(f'val_{n}_loss', l, sync_dist=True, prog_bar=True)
        if encoded[-1] is not None:
            return torch.argsort(encoded[-1], dim=-1, descending=True), torch.arange(encoded[-1].size(0), device=self.device)
   
    def validation_epoch_end(self, outputs):
        predictions, labels = [], []
        # multiple dataloaders are being used, take the first one only
        if type(outputs[0]) is list:
            outputs = outputs[0]

        for p, l in outputs:
            if p.size(0) != self.hparams.batch_size:
                break
            predictions.append(p)
            labels.append(l)
        if len(predictions) == 0:
            return 
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)

        def _recall(p, l, k):
            p = p[:, :k]
            acc = 0
            for i, label in enumerate(l):
                if label in p[i]: acc += 1
            acc = 100 * acc / len(l)
            return acc

        self.log('val_r1', _recall(predictions, labels, 1), prog_bar=True)
        self.log('val_r4', _recall(predictions, labels, 4), prog_bar=True)
        self.log('val_r8', _recall(predictions, labels, 8), prog_bar=True)

    def forward(self, batch):
        ctxt_encoding = self.encode_context(batch)
        resp_encoding = self.encode_response(batch)
        context_embedding, scores = self.query(ctxt_encoding, resp_encoding, ctxt_encoding)
        return context_embedding, scores
    
    @auto_move_data
    def respond(self, batch, return_responses=False, return_items=False, top_k=1, *args, **kwarg):
        
        if self.support_cache is not None:
            ctxt_encodings = self.support_cache['ctxt_encodings']
            resp_encodings = self.support_cache['resp_encodings']
            items = self.support_cache['items']
        else:
            resp_encodings = self.encode_response(batch)
            ctxt_encodings = self.encode_context(batch)
            items = None
        
        ctxt_keys = self.encode_context(batch)
        
        results = None
        if self.support_cache is not None:
            results = self.inference_query(ctxt_keys, top_k)

        if results is None:
            results = self.query(ctxt_keys, resp_encodings, ctxt_encodings)
            score = results[-1]
            label = torch.argsort(score, dim=-1, descending=True)
        else:
            score, label = results

        if not return_responses:
            return {
                'score' : score, 
                'label' : label,
                'item' : None if items is None else [items[idx] for idx in label[:, 0]],
                #'response_embedding' : resp_embedding, 
                #'context_embedding' : ctxt_embedding
            }
        else:
            responses = []
            for i, l in enumerate(label):
                responses.append([items[idx].response for idx in l[:top_k]])
            if return_items:
                r_items = []
                for i, l in enumerate(label):
                    r_items.append([items[idx] for idx in l[:top_k]])
                return responses, r_items
            return responses

    def try_load_support_cache(self, file_path):
        """ 
        Load encoded (training) data or better said the contexts and the correspoding dataset items. 
        The containig pickle file should contain a dictionary with the 'encodings' and 'items' keys.
        """
        if not os.path.exists(file_path):
            return False

        import sys
        import aargh.data as data # because of pickle compatibility with older non-package checkpoints
        sys.modules["data"] = data

        with open(file_path, 'rb') as f:
            self.support_cache = pickle.load(f)

        if self.support_cache['ctxt_encodings'] is not None: 
            self.support_cache['ctxt_tree'] = spatial.KDTree(self.support_cache['ctxt_encodings']['embeddings'].cpu())
            if 'action_embeddings' in self.support_cache['ctxt_encodings']:
                self.support_cache['ctxt_action_tree'] = spatial.KDTree(self.support_cache['ctxt_encodings']['action_embeddings'].cpu())

        if self.support_cache['resp_encodings'] is not None: 
            self.support_cache['resp_tree'] = spatial.KDTree(self.support_cache['resp_encodings']['embeddings'].cpu())

        return True

    def save_support_cache(self, ctxt_encodings, resp_encodings, items, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                'ctxt_encodings' : ctxt_encodings,
                'resp_encodings' : resp_encodings,
                'items' :     items
            }, f)


class DualBertAgent(RetrievalBertAgentBase):

    NAME = "hint_dual_bert"

    def __init__(self, params=None, *args, **kwargs):
        super().__init__(params=params, *args, **kwargs) 
        self.dropout = Dropout(self.hparams.output_dropout)
        self.context_output = Linear(768, self.hparams.output_dim)
        self.response_output = Linear(768, self.hparams.output_dim)

    @auto_move_data
    def query(self, ctxt_key, resp_encoding, _):
        score = torch.matmul(ctxt_key['embeddings'], resp_encoding['embeddings'].T)
        return ctxt_key['embeddings'], score

    def inference_query(self, ctxt_key, k):
        neighbor_dist, neighbor_idxes = self.support_cache['resp_tree'].query(ctxt_key['embeddings'].cpu(), k=list(range(1, k+1)))
        neighbor_dist += 1e-6
        return 1 / neighbor_dist, neighbor_idxes

    @auto_move_data
    def encode_response(self, batch):
        resp_embeddings = self.model(batch['response'], attention_mask=batch['response_mask'])
        resp_embedding = self.pool(resp_embeddings, batch['response_mask'])
        resp_embedding = self.response_output(self.dropout(resp_embedding))
        resp_embedding = F.tanh(resp_embedding)
        resp_embedding = F.normalize(resp_embedding, dim=-1)
        return { 'embeddings' : resp_embedding }

    @auto_move_data
    def encode_context(self, batch):
        ctxt_embeddings = self.model(batch['conversation'], attention_mask=batch['conversation_mask'])
        ctxt_embedding = self.pool(ctxt_embeddings, batch['conversation_mask'])
        ctxt_embedding = self.context_output(self.dropout(ctxt_embedding))
        ctxt_embedding = F.tanh(ctxt_embedding)
        ctxt_embedding = F.normalize(ctxt_embedding, dim=-1)
        return { 'embeddings' : ctxt_embedding }

    def calculate_loss(self, batch, encoded):
        logits = self.similarity_scaler(encoded[1])
        labels = torch.arange(logits.size(0), device=self.device)
        loss = F.cross_entropy(logits, labels)
        return {'total' : loss}


class PolyBertAgent(RetrievalBertAgentBase):

    NAME = "hint_poly_bert"

    def __init__(self, params=None, *args, **kwargs):
        super().__init__(params=params, *args, **kwargs)
        self.dropout = Dropout(self.hparams.output_dropout)
        self.context_output = Linear(768, self.hparams.output_dim)
        self.response_output = Linear(768, self.hparams.output_dim)
        self.poly_code_embeddings = Embedding(self.hparams.poly_m, 768)

    def dot_attention(self, q, k, v, mask):
        attn_weights = torch.matmul(q, k.transpose(1, 2))     
        if mask is not None:
            attn_weights = attn_weights.transpose(1, 2)  
            attn_weights[~mask] = float('-inf')
            attn_weights = attn_weights.transpose(1, 2) 
        attn_weights = torch.nn.functional.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v)         
        return output

    @auto_move_data
    def query(self, ctxt_key, resp_encoding, _):
        
        candidates = resp_encoding['embeddings'].unsqueeze(1).permute(1, 0, 2) 
        candidates = candidates.expand(ctxt_key['embeddings'].shape[0], candidates.shape[1], candidates.shape[2])

        ctxt_embedding = self.dot_attention(
            candidates, 
            ctxt_key['embeddings'], 
            ctxt_key['embeddings'],
            None
        ) # [B, B, D]
    
        ctxt_embedding = self.context_output(self.dropout(ctxt_embedding))
        ctxt_embedding = F.tanh(ctxt_embedding)
        ctxt_embedding = F.normalize(ctxt_embedding, dim=-1)
        
        candidates = self.response_output(self.dropout(candidates))
        candidates = F.tanh(candidates)
        candidates = F.normalize(candidates, dim=-1)

        score = (ctxt_embedding * candidates).sum(-1)
        return ctxt_embedding, score
    
    def inference_query(self, ctxt_key, k):
        batch_size = ctxt_key['embeddings'].shape[0]
        candiate_splits = torch.split(self.support_cache['resp_encodings']['embeddings'], batch_size * 2, dim=0)
        all_scores = []
        for candidates in candiate_splits:
            diff = batch_size - candidates.shape[0]
            _, scores = self.query(ctxt_key, {'embeddings' : candidates}, None)
            all_scores.append(scores)
        all_scores = torch.cat(all_scores, dim=-1)
        all_scores, neighbor_idxes = torch.sort(all_scores, dim=-1, descending=True)
        neighbor_idxes = neighbor_idxes[:, :k]
        all_scores = all_scores[:, :k]
        return all_scores, neighbor_idxes
    
    @auto_move_data
    def encode_response(self, batch):
        resp_embeddings = self.model(batch['response'], attention_mask=batch['response_mask'])
        resp_embedding = self.pool(resp_embeddings, batch['response_mask'])
        return { 'embeddings' : resp_embedding }
    
    @auto_move_data
    def encode_context(self, batch):
        ctxt_embeddings = self.model(batch['conversation'], attention_mask=batch['conversation_mask'])[0]
        poly_code_ids = torch.arange(self.hparams.poly_m, dtype=torch.long, device=self.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch['conversation'].shape[0], self.hparams.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids)

        ctxt_embeddings = self.dot_attention(
            poly_codes, 
            ctxt_embeddings, 
            ctxt_embeddings, 
            batch['conversation_mask'].bool()
        )
        return { 'embeddings' : ctxt_embeddings }

    def calculate_loss(self, batch, encoded):        
        logits = self.similarity_scaler(encoded[1])
        labels = torch.arange(logits.size(0), device=self.device)
        loss = F.cross_entropy(logits, labels)
        return {'total' : loss}

    def try_load_support_cache(self, file_path):
        """ 
        Load encoded (training) data or better said the contexts and the correspoding dataset items. 
        The containig pickle file should contain a dictionary with the 'encodings' and 'items' keys.
        """

        import sys
        import aargh.data as data # because of pickle compatibility with older non-package checkpoints
        sys.modules["data"] = data

        if not os.path.exists(file_path):
            return False
        with open(file_path, 'rb') as f:
            self.support_cache = pickle.load(f)

        return True


class ActionBertAgent(ActionAgentBase, RetrievalBertAgentBase):
    
    NAME = "hint_action_bert"

    def __init__(self, params=None, *args, **kwargs):
        super().__init__(params=params, *args, **kwargs) 
        self.dropout = Dropout(self.hparams.output_dropout)
        self.context_output = Linear(768, self.hparams.output_dim)

    @auto_move_data
    def query(self, ctxt_key, _, ctxt_encoding):
        scores = torch.matmul(ctxt_key['embeddings'], ctxt_encoding['embeddings'].T)
        return ctxt_key['embeddings'], scores

    def inference_query(self, ctxt_key, k):
        neighbor_dist, neighbor_idxes = self.support_cache['ctxt_tree'].query(ctxt_key['embeddings'].cpu(), k=list(range(1, k+1)))
        neighbor_dist += 1e-6
        return 1 / neighbor_dist, neighbor_idxes

    def encode_response(self, batch):
        return None

    @auto_move_data
    def encode_context(self, batch):
        ctxt_embeddings = self.model(batch['conversation'], attention_mask=batch['conversation_mask'])
        ctxt_embeddings = self.pool(ctxt_embeddings, batch['conversation_mask'])           
        ctxt_embedding = self.context_output(self.dropout(ctxt_embeddings))
        ctxt_embedding = F.tanh(ctxt_embedding)
        ctxt_embedding = F.normalize(ctxt_embedding, dim=-1)
        return {
            'embeddings' : ctxt_embedding,
        }

    def forward(self, batch):
        ctxt_encoding = self.encode_context(batch)
        ctxt_embedding, scores = self.query(ctxt_encoding, None, ctxt_encoding)
        return ctxt_embedding, scores
    

class DualActionBertAgent(DualBertAgent, RetrievalBertAgentBase):
    
    NAME = "hint_action_dual_bert"

    def get_batch_sampler(self):
        return ActionBatchSampler

    def calculate_loss(self, batch, encoded):
        logits = self.similarity_scaler(encoded[1])
        n = self.hparams.num_examples_per_action
        labels = torch.block_diag(*[torch.ones(n, n, device=self.device) for _ in range(self.hparams.num_actions)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return {'total' : loss}
    