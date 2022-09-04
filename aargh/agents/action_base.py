import torch
import random
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch.utils.data import Sampler, SubsetRandomSampler


class ActionAgentBase:

    def get_batch_sampler(self):
        return ActionBatchSampler

    def calculate_loss(self, batch, encoded):

        score_loss = super().calculate_loss(batch, encoded)
        ctxt_encoded = rearrange(encoded[0], '(m n) d -> m n d', m=self.hparams.num_actions, n=self.hparams.num_examples_per_action)
        action_loss = self.calculate_action_loss(ctxt_encoded)

        if score_loss is None:
            return {'total' : action_loss}
        else:
            score_loss = score_loss['total']

        action_loss *= self.hparams.action_loss_weight / (self.hparams.action_loss_weight + self.hparams.score_loss_weight)
        score_loss *= self.hparams.score_loss_weight / (self.hparams.action_loss_weight + self.hparams.score_loss_weight)

        return {
            'total'  : score_loss + action_loss,
            'score'  : score_loss,
            'action' : action_loss
        }

    def calculate_action_loss(self, ctxt_encoded):
       
        centroids = ctxt_encoded.mean(dim=1)
        similarity_matrix = self.get_cossim(ctxt_encoded, centroids)
        similarity_matrix = rearrange(similarity_matrix, 'm n d -> (m n) d')
        similarity_matrix = self.similarity_scaler(similarity_matrix)

        labels = torch.arange(self.hparams.num_actions, device=self.device)
        labels = torch.repeat_interleave(labels, repeats=self.hparams.num_examples_per_action, dim=0)
        return F.cross_entropy(similarity_matrix, labels)

    def get_cossim(self, embeddings, centroids):

        utterance_centroids = self.get_utterance_centroids(embeddings)
        utterance_centroids_flat = rearrange(utterance_centroids, 'm n d -> (m n) d')
        embeddings_flat = rearrange(embeddings, 'm n d -> (m n) d')
        
        cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

        centroids_expand = centroids.repeat((embeddings.shape[0] * embeddings.shape[1], 1))
        embeddings_expand = repeat(embeddings_flat, 'm d -> m c d', c=embeddings.shape[0])
        embeddings_expand = rearrange(embeddings_expand, 'm n d -> (m n) d')

        cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
        cos_diff = rearrange(cos_diff, '(m n d) -> m n d', m=self.hparams.num_actions, n=self.hparams.num_examples_per_action, d=self.hparams.num_actions)

        same_idx = list(range(embeddings.size(0)))
        cos_diff[same_idx, :, same_idx] = rearrange(cos_same, '(m n) -> m n', m=self.hparams.num_actions, n=self.hparams.num_examples_per_action)
        cos_diff = cos_diff + 1e-6

        return cos_diff

    def get_utterance_centroids(self, embeddings):
        sum_centroids = reduce(embeddings, 'm n d -> m 1 d', 'sum')
        num_utterances = embeddings.shape[1] - 1
        centroids = (sum_centroids - embeddings) / num_utterances
        return centroids


class SubsetSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class ActionBatchSampler(Sampler):

    def __init__(self, dataset, params, shuffle=True):
        super().__init__(dataset.items)
        self.dataset = dataset
        self.params = params
        self.shuffle = shuffle

        assert params.num_examples_per_action * params.num_actions == params.batch_size, (
            f"The number of distinc actions in a batch (`num_actions`: {params.num_actions}) " 
            f"times the number of examples per each of them (`num_examples_per_action`: {params.num_examples_per_action}) "
            f"must be equal to `batch_size`: {params.batch_size}")

        def normalize(s):
            s = s.split('-')[1]
            if s == "offerbooked":
                return "book"
            elif s == "offerbook":
                return "book"
            return s

        self.unique_actions = {}
        for i in range(len(dataset)):
            item_actions = set(normalize(a) + '-' + n for a, sa in dataset.items[i].actions.items() for n, v in sa)
            for action in item_actions:
                if action not in self.unique_actions:
                    self.unique_actions[action] = []
                self.unique_actions[action].append(i)

        self.unique_actions = { a : i for a, i in self.unique_actions.items() if len(i) >= self.params.min_num_examples_per_action}

        self.iterators = None
        if shuffle:
            self._samplers = { k : SubsetRandomSampler(self.unique_actions[k]) for k in self.unique_actions }
        else:
            self._samplers = { k : SubsetSampler(self.unique_actions[k]) for k in self.unique_actions }

    def __iter__(self):
        
        if self.iterators is None:
            self.iterators = { k : iter(v) for k, v in self._samplers.items() }
        
        for _ in range(min(len(s) for s in self._samplers.values()) // self.params.num_examples_per_action):
            actions = list(self.unique_actions)
            if self.shuffle:
                random.shuffle(actions)
            for i in range(0, len(actions) - self.params.num_actions + 1, self.params.num_actions):  
                batch = []
                for j in range(i, i + self.params.num_actions):
                    a = actions[j]
                    for _ in range(self.params.num_examples_per_action):
                        s = next(self.iterators[a], None)
                        if s is None:
                            self.iterators[a] = iter(self._samplers[a])
                            s = next(self.iterators[a])
                        batch.append(s)
                yield batch 
      
    def __len__(self):
        return (len(self.unique_actions) // self.params.num_actions) * (min(len(s) for s in self._samplers.values()) // self.params.num_examples_per_action)
