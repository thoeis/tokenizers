from collections import Counter

from tqdm import tqdm
import datasets

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


# Build a tokenizer
unigram_tokenizer = Tokenizer(models.Unigram())
unigram_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
unigram_tokenizer.normalizer = normalizers.Lowercase()

# Initialize a dataset
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1")

counter = Counter()
for item in tqdm(dataset["train"].take(10000)):
    text = unigram_tokenizer.normalizer.normalize_str(item["text"])
    tokens = unigram_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    for w, _ in tokens:
        counter[w] += 1
    pass

# And finally train
unigram_tokenizer.train_from_counter(counter)
