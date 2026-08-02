import regex
import pickle
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
from collections import Counter
from collections.abc import Iterable, Iterator
def train_bpe(
        input_path:str,
        vocab_size:int,
        special_tokens:list[str]
):
    with open(input_path, "r", encoding="utf-8") as file:     # open的第三个位置的参数表示缓冲设置
        text = file.read()                           # 注意要明确写出参数名
    vocab = {} 
    merges = [] 
    for i in range(256):
        vocab[i] = bytes([i])  # 把列表[i]中的整数i转换成字节,字典变量[键]=值
    for i in special_tokens:
        spe = i.encode("utf-8")
        vocab[len(vocab)]=spe
    text_chunks = [text]     # 把字符串转换成列表的形式（整体变成一个列表）--字符串外加方括号
    for sep in special_tokens:
        text_ite = []
        for chunk in text_chunks:
            new_chunks = chunk.split(sep)  # 列表只能使用整数位置访问
            text_ite.extend(new_chunks)  # extend可以将可迭代的对象逐个元素插入到原对象中
        text_chunks = text_ite
    # 此时text_chunks是将特殊token分离出来的字符串列表
    pretokens_count = Counter()
    for i in range(len(text_chunks)):
        matches = regex.finditer(PAT,text_chunks[i])
        for match in matches:
            pretokens_count[match.group()] += 1
    word_counts = {}
    for pretoken,count in pretokens_count.items():
        pretoken_bytes = pretoken.encode("utf-8")
        symbols = []
        for i in pretoken_bytes:
            symbols.append(bytes([i]))
        symbols_tuple = tuple(symbols) 
        word_counts[symbols_tuple] = count 
    # 此时word_counts是预分词和对应的出现频率
    while(len(vocab) < vocab_size):
        pair_counts = {} # 这里统计相邻对需要累加，因此还是要使用counter，不能直接赋值
        for res,count in word_counts.items():  #字典的遍历不能用整数索引，必须是键和值
            for i in range(len(res)-1):
                samp = (res[i],res[i+1])
                pair_counts[samp] = pair_counts.get(samp,0)+count    
        if not pair_counts:
            break
        best_pair = max(
          pair_counts,
          key=lambda pair: (pair_counts[pair], pair),
)
        merges.append(best_pair)
        merged_tokens = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = merged_tokens
        new_word_counts = {}
        for symbols,count in word_counts.items():
            i = 0
            res = None
            while(i < len(symbols)):
                if(i + 1 < len(symbols) and (symbols[i],symbols[i+1]) == best_pair):
                    if(res is None):
                        res = list(symbols[:i])
                        res.append(merged_tokens)  # 这里的字节拼接用变量来优化
                        i += 2
                    else:
                        res.append(merged_tokens)
                        i += 2
                else:
                    if(res is not None):
                        res.append(symbols[i])
                    i += 1
            if res is None:
                res_tuple = symbols
            else:
                res_tuple = tuple(res)
            new_word_counts[res_tuple] = new_word_counts.get(res_tuple,0) + count
        word_counts = new_word_counts  
    return vocab,merges
class Tokenizer:
    @classmethod
    def from_files(
        cls,
        vocab_filepath:str,
        merges_filepath:str,
        special_tokens:list[str] | None = None
    ):
        with open(vocab_filepath,"rb") as file:  # 路径变量不能加引号
            vocab = pickle.load(file)
        with open(merges_filepath,"rb") as file:
            merges = pickle.load(file)
        return cls(vocab,merges,special_tokens)



    def __init__(self,vocab,merges,special_tokens = None):
        self.vocab = dict(vocab) # 为了避免后续添加特殊token修改原字典，此处进行字典拷贝
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = list(special_tokens)
        self.byte_to_id = {}
        for token_id,token_word in self.vocab.items():
            self.byte_to_id[token_word] = token_id
        for spe in self.special_tokens:
            spec_bytes = spe.encode("utf-8") # 使用in/not in可以快速判断字典是否存在某个特定键
            if spec_bytes in self.byte_to_id:
                continue
            else:
                new_id = len(self.vocab)
                self.byte_to_id[spec_bytes] = new_id
                self.vocab[new_id] = spec_bytes
        self.merge_ranks = {}
        for rank,pair in enumerate(self.merges):
            self.merge_ranks[pair] = rank
    
    def decode(self,ids:list[int]) -> str:
        bytes_list = []
        for i in ids:
            if i in self.vocab:
                bytes_list.append(self.vocab[i])
            else:
                raise ValueError(f"Token ID {i} 不在词表中") # raise抛出异常，终止当前函数运行
        res = b"".join(bytes_list)
        text = res.decode("utf-8",errors="replace")
        return text

    def encode(self,text:str) -> list[int]:
        token_ids = []
        pre_tokens = []
        sorted_special_tokens = sorted(self.special_tokens, key = len, reverse = True)
        escaped_special_tokens = []
        for special_token in sorted_special_tokens:
            escaped_special_tokens.append(regex.escape(special_token))
        if escaped_special_tokens:
            special_tokens_pattern = "(" + "|".join(escaped_special_tokens) + ")"
            res = regex.split(special_tokens_pattern, text)
        else:
            res = [text]
        special_tokens_set = set(self.special_tokens)
        for part in res:
            if part in special_tokens_set:
                pre_tokens.append(part)
            else:
                matches = regex.finditer(PAT, part)
                for match in matches:
                    pre_tokens.append(match.group())  # pre_tokens是预分词结果 
        for pre_token in pre_tokens:
            if pre_token in special_tokens_set:
                special_bytes = pre_token.encode("utf-8")
                if special_bytes not in self.byte_to_id:
                    raise ValueError(f"词表中不存在特殊 token: {pre_token}")
                token_ids.append(self.byte_to_id[special_bytes])
                continue
            pre_token_bytes = pre_token.encode("utf-8")
            symbols = []
            
            for i in pre_token_bytes:
                symbols.append(bytes([i]))
            while(True):
                best_pair = None
                best_rank = None
                for i in range(len(symbols) - 1):
                    pair = (symbols[i],symbols[i+1])
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if best_rank is None or rank < best_rank:
                            best_pair = pair
                            best_rank = rank
                if best_pair is None:
                        break
                else:
                    new_symbols = []
                    i = 0
                    while i < len(symbols):
                        if (
                        i + 1 < len(symbols)
                        and (symbols[i], symbols[i + 1]) == best_pair
                        ):
                            merged_symbol = symbols[i] + symbols[i + 1]
                            new_symbols.append(merged_symbol)
                            i += 2
                        else:
                            new_symbols.append(symbols[i])
                            i += 1
                    symbols = new_symbols
            for symbol in symbols:
                if symbol not in self.byte_to_id:
                    raise ValueError(f"词表中不存在token:{symbol}")
                token_ids.append(self.byte_to_id[symbol]) # 查询字典使用方括号
        return token_ids

    def encode_iterable(
            self,
            iterable:Iterable[str],
    ) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id # yield可以将函数变成生成器，返回一个迭代器对象

       








    

        



    
             # 创建列表直接使用方括号，不需要在方括号前添加关键字list



    

    