import time
import pickle
import resource
from pathlib import Path
from cs336_basics.tokenizer import train_bpe
input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
start_time = time.perf_counter()
print("开始训练 TinyStories 10K tokenizer...", flush=True)
vocab, merges = train_bpe(
    input_path=str(input_path),
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)

end_time = time.perf_counter()

longest_token = max(vocab.values(), key=len)
peak_memory_bytes = resource.getrusage(
    resource.RUSAGE_SELF
).ru_maxrss

print("训练时间：", end_time - start_time, "秒")
print("词表大小：", len(vocab))
print("合并次数：", len(merges))
print("最长 token:", longest_token)
print("最长 token 字节数：", len(longest_token))
print(
    "最长 token 文本：",
    longest_token.decode("utf-8", errors="replace"),
)
print(
    "峰值内存：",
    peak_memory_bytes / (1024 ** 2),
    "MB",
)
output_dir = Path("data/tokenizers")
output_dir.mkdir(parents=True, exist_ok=True)

vocab_path = output_dir / "tinystories_vocab_10000.pkl"
merges_path = output_dir / "tinystories_merges_10000.pkl"

with open(vocab_path, "wb") as file:
    pickle.dump(vocab, file)

with open(merges_path, "wb") as file:
    pickle.dump(merges, file)

print("词表保存到：", vocab_path)
print("合并规则保存到：", merges_path)

