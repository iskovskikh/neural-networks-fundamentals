from collections import Counter
import re

import torch


class SkipGramDataset:
    def __init__(self, corpus, window_size=2, min_count=1):
        self.window_size = int(window_size)
        self.min_count = int(min_count)
        self.tokens = []
        self.vocab = ['word']
        self.word_to_index = {'word': 0}
        self.index_to_word = {0: 'word'}
        self.word_counts = {'word': 1}
        self.pairs = [(0, 0)]

        ## YOUR CODE HERE


        # Inside `__init__`, create:
        # - `self.tokens`: normalized alphabetic tokens from the corpus.
        #  Token normalization has two steps:
        #  first lowercase the token, then cut common suffixes such as
        #  `-ing`, `-ed`, `-es`, and `-s`
        #  so related forms like `play`, `played`, and `playing` 
        # land closer together (but note that the remaining word should contain at least 2 letters);
        self.tokens = self._tokenize(corpus)
        
        # - `self.vocab`: words with frequency at least `min_count`,
        #  sorted by decreasing frequency and then alphabetically;

        counter = Counter(self.tokens)

        items = [word for word, count in counter.items() if count >= self.min_count]

        self.vocab =  sorted(
            items,
            key=lambda item: (-counter[item], item)
        )
        
        # - `self.word_to_index` and `self.index_to_word`;
        self.word_to_index = {
            word: idx for idx, word in enumerate(self.vocab)
        }

        self.index_to_word = {
            idx: word for word, idx in self.word_to_index.items()
        }
        
        # - `self.word_counts`: counts for words that remain in the vocabulary;
        self.word_counts = {
            word: counter[word] for word in self.vocab
        }
        
        # - `self.pairs`: `(center_word, context_word)` index pairs for every word within
        #  `window_size` positions.
        indexed_tokens = [
            self.word_to_index[word]
            for word in self.tokens
            if word in self.word_to_index
        ]

        self.pairs = self._make_pairs(
            indexed_tokens=indexed_tokens,
            window_size=self.window_size,
        )


    def _tokenize(self, corpus):
        text = ' '.join(corpus) if isinstance(corpus, (list, tuple)) else str(corpus)
        return [
            self._normalize_token(token)
            for token in re.findall(r'[a-z]+', text, flags=re.IGNORECASE)
        ]

    def _normalize_token(self, token):
        token = token.lower()
        if len(token) > 5 and token.endswith('ing'):
            return token[:-3]
        if len(token) > 4 and token.endswith('ed'):
            return token[:-2]
        if len(token) > 4 and token.endswith('es') and not token.endswith('ses'):
            return token[:-2]
        if len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
            return token[:-1]
        return token

    def _make_pairs(self, indexed_tokens, window_size):
        pairs = []
        for center_position, center_word in enumerate(indexed_tokens):
            left = max(0, center_position - window_size)
            right = min(len(indexed_tokens), center_position + window_size + 1)
            for context_position in range(left, right):
                if context_position != center_position:
                    pairs.append((center_word, indexed_tokens[context_position]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        # Return a dictionary with only:
        # - `center_word`: integer index of the input word in the vocabulary;
        # - `context_word`: integer index of the target nearby word in the vocabulary.

        center_word, context_word = self.pairs[index]
        return {
            'center_word': torch.tensor(center_word, dtype=torch.long),
            'context_word': torch.tensor(context_word, dtype=torch.long),
        }
