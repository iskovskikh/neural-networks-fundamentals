import heapq

import torch
import torch.nn.functional as F


# ===========================
#.laborantum/texts/course/2. Fully Connected NNs/4. Word2Vec/word2vec.py

import torch
import torch.nn.functional as F


class BinaryIndexTree:
    def __init__(self, vocab_size):
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= 0:
            raise ValueError('vocab_size must be positive')
        self.depth = max(1, (self.vocab_size - 1).bit_length())
        self.max_path_length = self.depth
        self.num_internal_nodes = 2 ** self.depth - 1

    def targets_for_index(self, word_index):
        word_index = int(word_index)
        if word_index < 0 or word_index >= self.vocab_size:
            raise ValueError(f'word index {word_index} is outside vocabulary')
        return format(word_index, f'0{self.depth}b')

    @staticmethod
    def node_id_from_prefix(prefix_bits):
        if isinstance(prefix_bits, str):
            prefix = prefix_bits
        else:
            prefix = ''.join(str(int(bit)) for bit in prefix_bits)
        if not prefix:
            return 0
        depth = len(prefix)
        return (2 ** depth - 1) + int(prefix, 2)

    def path_and_targets(self, word_index):
        targets_string = self.targets_for_index(word_index)
        path = [
            self.node_id_from_prefix(targets_string[:step])
            for step in range(self.depth)
        ]
        targets = [int(bit) for bit in targets_string]
        return path, targets

    def __call__(self, context_word):
        device = context_word.device
        context_word = context_word.detach().cpu().view(-1).tolist()
        fallback_batch_size = len(context_word)
        # Fallback code for this task: keeps the notebook runnable before the method is solved.

        fallback = {
            'path': torch.zeros(
                fallback_batch_size,
                self.max_path_length,
                dtype=torch.long,
                device=device,
            ),
            'targets': torch.zeros(
                fallback_batch_size,
                self.max_path_length,
                dtype=torch.float32,
                device=device,
            ),
            'mask': torch.ones(
                fallback_batch_size,
                self.max_path_length,
                dtype=torch.float32,
                device=device,
            ),
        }

        ## YOUR CODE HERE
        

        paths = []
        targets = []
        
        for word_index in context_word:
            path, target = self.path_and_targets(word_index)
            paths.append(path)
            targets.append(target)

        path = torch.tensor(
            paths,
            dtype=torch.long,
            device=device,
        )

        target = torch.tensor(
            targets,
            dtype=torch.float32,
            device=device,
        )

        mask = torch.ones(
            len(context_word),
            self.max_path_length,
            dtype=torch.float32,
            device=device,
        )

        fallback = dict(
            path = path,
            targets = target,
            mask = mask,
        )

        return fallback
    


class HierarchicalSoftmax(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        # Fallback code for this task: creates a valid but deliberately weak decoder.
        # self.embedding_dim = int(embedding_dim)
        # self.targets = BinaryIndexTree(vocab_size)
        # self.decoder = torch.nn.Embedding(
        #     self.targets.num_internal_nodes,
        #     self.embedding_dim,
        # )
        # torch.nn.init.zeros_(self.decoder.weight)

        ## YOUR CODE HERE

        # Inside `__init__(self, embedding_dim, vocab_size)`, create:
        # - `self.embedding_dim`: integer embedding dimension;
        self.embedding_dim = int(embedding_dim)

        # - `self.targets`: a `BinaryIndexTree(vocab_size)` object;
        self.targets = BinaryIndexTree(vocab_size=vocab_size)

        # - `self.decoder`: a `torch.nn.Embedding(self.targets.num_internal_nodes, self.embedding_dim)` layer;
        self.decoder = torch.nn.Embedding(
            self.targets.num_internal_nodes,
            self.embedding_dim,
        )
        
        # - initialize `self.decoder.weight` with `torch.nn.init.normal_(..., mean=0.0, std=0.02)`.
        torch.nn.init.normal_(
            self.decoder.weight,
            mean=0.0,
            std=0.02,
        )

    @property
    def num_internal_nodes(self):
        return self.targets.num_internal_nodes

    @property
    def max_path_length(self):
        return self.targets.max_path_length

    def forward(self, embedding, target_word):
        # target_tensors = self.targets(target_word)
        # node_vectors = self.decoder(target_tensors['path'])
        # fallback_logits = torch.einsum('bd,bld->bl', embedding, node_vectors) * 0.0
        # fallback_probabilities = torch.sigmoid(fallback_logits)
        # fallback_target_probabilities = torch.where(
        #     target_tensors['targets'].bool(),
        #     fallback_probabilities,
        #     1.0 - fallback_probabilities,
        # )
        # fallback_total_probability = fallback_target_probabilities.prod(dim=1)
        # fallback_per_node_loss = F.binary_cross_entropy_with_logits(
        #     fallback_logits,
        #     target_tensors['targets'],
        #     reduction='none',
        # )
        # fallback_per_word_loss = (fallback_per_node_loss * target_tensors['mask']).sum(dim=1)
        # # Fallback code for this task: returns valid tensors from neutral probabilities.
        # fallback = {
        #     **target_tensors,
        #     'logits': fallback_logits,
        #     'probabilities': fallback_probabilities,
        #     'target_probabilities': fallback_target_probabilities,
        #     'total_probability': fallback_total_probability,
        #     'per_node_loss': fallback_per_node_loss,
        #     'per_word_loss': fallback_per_word_loss,
        #     'loss': fallback_per_word_loss.mean(),
        # }

        ## YOUR CODE HERE

        target_tensors = self.targets(target_word)

        path = target_tensors["path"]
        targets = target_tensors["targets"]
        mask = target_tensors["mask"]

        # The method receives:
        # - `embedding`: center-word vectors with shape `(batch_size, embedding_dim)`;
        # - `target_word`: target context-word indices with shape `(batch_size,)` or any flattenable shape.

        # Implement `forward` so it:
        # - calls `self.targets(target_word)` to get `path`, `targets`, and `mask`;
        # - uses `self.decoder(path)` to get one node vector per binary decision;
        
        node_vectors = self.decoder(path)

        # - computes `logits` as dot products between each center-word embedding and each node vector,
        #  with shape `(batch_size, max_path_length)`;
        
        logits = torch.matmul(embedding.unsqueeze(1), node_vectors.transpose(1, 2)).squeeze(1)


        # - computes `probabilities = torch.sigmoid(logits)`;
        probabilities = torch.sigmoid(logits)

        # - computes `target_probabilities`: 
        # use `probabilities` 
        # where the binary target is `1`,
        # and `1 - probabilities`
        # where the target is `0`;

        # target_probabilities = probabilities if targets.bool() else 1.0 - probabilities
        
        target_probabilities = torch.where(
            targets.bool(),
            probabilities,
            1.0 - probabilities,
        )
        
        # - computes `total_probability` as the product of target probabilities along the path;
        total_probability = target_probabilities.prod(dim=1)
        
        # - computes binary cross entropy with logits against `targets`, with `reduction="none"`;

        per_node_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        # - multiplies per-node losses by `mask`,
        #  sums over the path into `per_word_loss`,
        #  and averages into scalar `loss`;
        per_word_loss = (per_node_loss * mask).sum(dim=1)

        # - returns a dictionary containing the target tensors plus 
        #  `logits`,
        #  `probabilities`,
        #  `target_probabilities`,
        #  `total_probability`,
        #  `per_node_loss`,
        #  `per_word_loss`,
        #  and `loss`.

        fallback = {
            **target_tensors,
            'logits': logits,
            'probabilities': probabilities,
            'target_probabilities': target_probabilities,
            'total_probability': total_probability,
            'per_node_loss': per_node_loss,
            'per_word_loss': per_word_loss,
            'loss': per_word_loss.mean(),
        }


        return fallback
    
    
class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Fallback code for this task: creates valid modules with weak zero embeddings.
        # self.vocab_size = int(vocab_size)
        # self.embedding_dim = int(embedding_dim)
        # self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        # torch.nn.init.zeros_(self.encoder.weight)
        # self.hierarchical_softmax = HierarchicalSoftmax(
        #     self.embedding_dim,
        #     self.vocab_size,
        # )
        # self.decoder = self.hierarchical_softmax.decoder
        # self.num_internal_nodes = self.hierarchical_softmax.num_internal_nodes

        ## YOUR CODE HERE

        # - `self.vocab_size`: integer vocabulary size;
        self.vocab_size = int(vocab_size)

        # - `self.embedding_dim`: integer embedding dimension;
        self.embedding_dim = int(embedding_dim)

        # - `self.encoder`: a `torch.nn.Embedding(self.vocab_size, self.embedding_dim)` layer for center-word vectors;
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # - `self.hierarchical_softmax`: a `HierarchicalSoftmax(self.embedding_dim, self.vocab_size)` layer;
        self.hierarchical_softmax = HierarchicalSoftmax(
            self.embedding_dim,
            self.vocab_size,
        )

        # - `self.decoder`: an alias to `self.hierarchical_softmax.decoder`;
        self.decoder = self.hierarchical_softmax.decoder

        # - `self.num_internal_nodes`: an alias to `self.hierarchical_softmax.num_internal_nodes`;
        self.num_internal_nodes = self.hierarchical_softmax.num_internal_nodes

        # - initialize `self.encoder.weight` with `torch.nn.init.normal_(..., mean=0.0, std=0.02)`.

        torch.nn.init.normal_(
            self.encoder.weight,
            mean=0.0,
            std=0.02,
        )


    def forward(self, batch):
        # center_word = batch['data']['center_word']
        # embedding = self.encoder(center_word)
        # batch['signals'] = {
        #     'embedding': embedding,
        # }
        # batch['postprocessed'] = {}
        # if 'context_word' in batch['data']:
        #     target_tensors = self.hierarchical_softmax.targets(batch['data']['context_word'])
        #     fallback_logits = torch.zeros_like(target_tensors['targets'])
        #     fallback_probabilities = torch.sigmoid(fallback_logits)
        #     fallback_target_probabilities = torch.where(
        #         target_tensors['targets'].bool(),
        #         fallback_probabilities,
        #         1.0 - fallback_probabilities,
        #     )
        #     fallback_per_node_loss = F.binary_cross_entropy_with_logits(
        #         fallback_logits,
        #         target_tensors['targets'],
        #         reduction='none',
        #     )
        #     fallback_per_word_loss = (fallback_per_node_loss * target_tensors['mask']).sum(dim=1)
        #     batch['data'].update(target_tensors)
        #     batch['signals']['logits'] = fallback_logits
        #     batch['signals']['probabilities'] = fallback_probabilities
        #     batch['signals']['target_probabilities'] = fallback_target_probabilities
        #     batch['signals']['total_probability'] = fallback_target_probabilities.prod(dim=1)
        #     batch['signals']['loss'] = fallback_per_word_loss.mean()
        #     batch['postprocessed']['targets'] = (fallback_probabilities >= 0.5).long()
        # # Fallback code for this task: fills the batch with neutral predictions.
        # fallback = batch

        ## YOUR CODE HERE

        # Implement `forward(self, batch)` so it mutates and returns the same batch dictionary:
        # - read `center_word = batch['data']['center_word']`;
        center_word = batch['data']['center_word']
        # - compute `embedding = self.encoder(center_word)`;
        embedding = self.encoder(center_word)
        # - create `batch['signals'] = {'embedding': embedding}`;
        batch['signals'] = {'embedding': embedding}
        # - create `batch['postprocessed'] = {}`;
        batch['postprocessed'] = {}
        # - if `context_word` is present in `batch['data']`, 
        if 'context_word' in batch['data']:
        # call `self.hierarchical_softmax(embedding, batch['data']['context_word'])`;
            hs_output= self.hierarchical_softmax(embedding, batch['data']['context_word'])

        # - copy `path`, `targets`, and `mask` from the hierarchical-softmax output into `batch['data']`;
            batch["data"]["path"] = hs_output["path"]
            batch["data"]["targets"] = hs_output["targets"]
            batch["data"]["mask"] = hs_output["mask"]

        # - copy `logits`, `probabilities`, `target_probabilities`, `total_probability`, and `loss` into `batch['signals']`;
            batch["signals"]["logits"] = hs_output["logits"]
            batch["signals"]["probabilities"] = hs_output["probabilities"]
            batch["signals"]["target_probabilities"] = hs_output["target_probabilities"]
            batch["signals"]["total_probability"] = hs_output["total_probability"]
            batch["signals"]["loss"] = hs_output["loss"]
        
        # - store predicted binary decisions in `batch['postprocessed']['targets']` by thresholding probabilities at `0.5`;
            batch["postprocessed"]["targets"] = (
                hs_output["probabilities"] >= 0.5
            ).long()

        # - return `batch`.
        return batch

# ===========================================================
    
class HierarchicalSoftmaxTargets:
    def __init__(self, word_counts, word_to_index):
        active_counts = {
            word_to_index[word]: int(count)
            for word, count in word_counts.items()
            if word in word_to_index
        }
        if len(active_counts) <= 1:
            self.paths = {0: [0]}
            self.codes = {0: [1]}
            self.num_internal_nodes = 1
            self.max_path_length = 1
            return

        next_internal_id = 0
        serial = 0
        heap = []
        for word_index, count in active_counts.items():
            heapq.heappush(heap, (count, serial, {'word': word_index}))
            serial += 1

        while len(heap) > 1:
            left_count, _, left = heapq.heappop(heap)
            right_count, _, right = heapq.heappop(heap)
            node = {
                'node': next_internal_id,
                'left': left,
                'right': right,
            }
            next_internal_id += 1
            heapq.heappush(heap, (left_count + right_count, serial, node))
            serial += 1

        paths = {}
        codes = {}

        def walk(node, path, code):
            if 'word' in node:
                paths[node['word']] = path.copy()
                codes[node['word']] = code.copy()
                return
            walk(node['left'], path + [node['node']], code + [0])
            walk(node['right'], path + [node['node']], code + [1])

        walk(heap[0][2], [], [])
        self.paths = paths
        self.codes = codes
        self.num_internal_nodes = next_internal_id
        self.max_path_length = max(len(path) for path in self.paths.values())

    def __call__(self, context_word):
        device = context_word.device
        context_word = context_word.detach().cpu().view(-1).tolist()
        paths = []
        codes = []
        masks = []
        for word_index in context_word:
            path = self.paths[int(word_index)]
            code = self.codes[int(word_index)]
            padding = self.max_path_length - len(path)
            paths.append(path + [0] * padding)
            codes.append(code + [0] * padding)
            masks.append([1.0] * len(path) + [0.0] * padding)
        return {
            'path': torch.tensor(paths, dtype=torch.long, device=device),
            'code': torch.tensor(codes, dtype=torch.float32, device=device),
            'mask': torch.tensor(masks, dtype=torch.float32, device=device),
        }


class HierarchicalSoftmaxLoss(torch.nn.Module):
    def __init__(self, model, targets):
        super().__init__()
        self.model = model
        self.targets = targets

    def forward(self, batch):
        target_tensors = self.targets(batch['data']['context_word'])
        batch['data'].update(target_tensors)
        embedding = batch['signals']['embedding']
        node_vectors = self.model.decoder(batch['data']['path'])
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        batch['signals']['logits'] = logits
        batch['signals']['probabilities'] = torch.sigmoid(logits)
        batch['postprocessed']['code'] = (batch['signals']['probabilities'] >= 0.5).long()
        per_node_loss = F.binary_cross_entropy_with_logits(
            logits,
            batch['data']['code'],
            reduction='none',
        )
        masked_loss = per_node_loss * batch['data']['mask']
        return masked_loss.sum() / batch['data']['mask'].sum().clamp_min(1.0)


class Word2VecHierarchicalSoftmax(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_internal_nodes):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.num_internal_nodes = int(num_internal_nodes)
        self.encoder = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        self.decoder = torch.nn.Embedding(self.num_internal_nodes, self.embedding_dim)

        ## YOUR CODE HERE

    def __forward_kernel(self, center_word, path):
        embedding = self.encoder(center_word)
        node_vectors = self.decoder(path)
        logits = torch.einsum('bd,bld->bl', embedding, node_vectors)
        return embedding, logits

    def forward(self, batch):
        ## YOUR CODE HERE
        if 'signals' not in batch:
            batch['signals'] = {
                'embedding': self.encoder(batch['data']['center_word']),
            }
            batch['postprocessed'] = {}
        return batch
