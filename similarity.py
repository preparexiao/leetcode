import pandas as pd
import numpy as np

# https://blog.csdn.net/irving_zhang/article/details/69440789
# 相似度分析：https://www.2cto.com/net/201704/622163.html
# glove库：https://nlp.stanford.edu/projects/glove/

def read_data_sets(train_dir):
    #
    # s1代表数据集的句子1
    # s2代表数据集的句子2
    # score代表相似度
    # sample_num代表数据总共有多少行
    #
    SICK_DIR = "SICK_data/SICK.txt"
    df_sick = pd.read_csv(SICK_DIR, sep="\t", usecols=[1,2,4], names=['s1', 's2', 'score'],
                          dtype={'s1':object, 's2':object, 'score':object})
    df_sick = df_sick.drop([0])
    s1 = df_sick.s1.values
    s2 = df_sick.s2.values
    score = np.asarray(map(float, df_sick.score.values), dtype=np.float32)
    sample_num = len(score)

    # 引入embedding矩阵和字典
    global sr_word2id, word_embedding
    sr_word2id, word_embedding = build_glove_dic()

    # word2id, 多线程将word转成id
    p = Pool()
    s1 = np.asarray(p.map(seq2id, s1))
    s2 = np.asarray(p.map(seq2id, s2))
    p.close()
    p.join()

    # 填充句子
    s1, s2 = padding_sentence(s1, s2)
    new_index = np.random.permutation(sample_num)
    s1 = s1[new_index]
    s2 = s2[new_index]
    score = score[new_index]

    return s1 ,s2, score

def get_id(word):
    if word in sr_word2id:
        return sr_word2id[word]
    else:
        return sr_word2id['<unk>']

def seq2id(seq):
    seq = clean_str(seq)
    seq_split = seq.split(' ')
    seq_id = map(get_id, seq_split)
    return seq_id


def build_glove_dic():
    # 从文件中读取 pre-trained 的 glove 文件，对应每个词的词向量
    # 需要手动对glove文件处理，在第一行加上
    # 400000 50
    # 其中400000代表共有四十万个词，每个词50维，中间为一个空格或者tab键
    # 因为word2vec提取需要这样的格式，详细代码可以点进load函数查看
    glove_path = 'glove.6B.50d.txt'
    wv = word2vec.load(glove_path)
    vocab = wv.vocab
    sr_word2id = pd.Series(range(1,len(vocab) + 1), index=vocab)
    sr_word2id['<unk>'] = 0
    word_embedding = wv.vectors
    word_mean = np.mean(word_embedding, axis=0)
    word_embedding = np.vstack([word_mean, word_embedding])

    return sr_word2id, word_embedding

def padding_sentence(s1, s2):
    #
    # 得到句子s1,s2以后，很直观地想法就是先找出数据集中的最大句子长度，
    # 然后用<unk>对句子进行填充
    #
    s1_length_max = max([len(s) for s in s1])
    s2_length_max = max([len(s) for s in s2])
    sentence_length = max(s1_length_max, s2_length_max)
    sentence_num = s1.shape[0]
    s1_padding = np.zeros([sentence_num, sentence_length], dtype=int)
    s2_padding = np.zeros([sentence_num, sentence_length], dtype=int)

    for i, s in enumerate(s1):
        s1_padding[i][:len(s)] = s

    for i, s in enumerate(s2):
        s2_padding[i][:len(s)] = s

    print("9840个句子填充完毕")
    return s1_padding, s2_padding
