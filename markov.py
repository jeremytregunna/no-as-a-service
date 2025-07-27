import random
from collections import defaultdict
from typing import List, Dict


class MarkovChain:
    def __init__(self, order: int = 2):
        self.order = order
        self.chain: Dict[tuple, List[str]] = defaultdict(list)
        self.start_words: List[tuple] = []

    def train(self, texts: List[str]) -> None:
        for text in texts:
            words = text.strip().split()
            if len(words) < self.order + 1:  # Need at least order+1 words
                continue

            start_gram = tuple(words[:self.order])
            self.start_words.append(start_gram)

            # Build the chain
            for i in range(len(words) - self.order):
                current_gram = tuple(words[i:i + self.order])
                next_word = words[i + self.order]
                self.chain[current_gram].append(next_word)

    def generate(self, max_length: int = 50) -> str:
        if not self.start_words:
            return "No."

        current_gram = random.choice(self.start_words)
        result = list(current_gram)

        for _ in range(max_length - self.order):
            if current_gram not in self.chain:
                break

            next_word = random.choice(self.chain[current_gram])
            result.append(next_word)

            # Update current_gram to slide the window
            current_gram = tuple(result[-self.order:])

        return " ".join(result)

    def is_trained(self) -> bool:
        return len(self.chain) > 0
