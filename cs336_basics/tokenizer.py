import regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
from collections import Counter
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


        



    
li = ["<|endoftext|>"]              # 创建列表直接使用方括号，不需要在方括号前添加关键字list
vocab,merges = train_bpe("tests/fixtures/corpus.en",500,li)


    

    