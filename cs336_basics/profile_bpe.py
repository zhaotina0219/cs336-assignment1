from cs336_basics.tokenizer import train_bpe


train_bpe(
    input_path="data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size=1000,
    special_tokens=["<|endoftext|>"],
)