# coding:utf-8

from gensim import models
import os
import sys
import numpy as np
import pandas as pd
import MeCab
from tfidf import set_folder_morph

corpus = "./corpus/"
folder = sys.argv[1]
directory = os.getcwd()

# 参考記事： http://qiita.com/okappy/items/32a7ba7eddf8203c9fa1
class LabeledListSentence(object):
    def __init__(self, words_list, labels):
        self.words_list = words_list
        self.labels = labels
    
    def __iter__(self):
        for i, words in enumerate(self.words_list):
            yield models.doc2vec.LabeledSentence(words, ['%s' % self.labels[i]])


morph_list, docs = set_folder_morph(corpus + folder)
sentences = LabeledListSentence(morph_list, docs)

# doc2vec の学習条件設定
# alpha: 学習率 / min_count: X回未満しか出てこない単語は無視
# size: ベクトルの次元数 / iter: 反復回数 / workers: 並列実行数
model = models.Doc2Vec(alpha=0.025, min_count=5,
                       size=100, iter=20, workers=4)

# doc2vec の学習前準備(単語リスト構築)
model.build_vocab(sentences)

# Wikipedia から学習させた単語ベクトルを無理やり適用して利用することも出来ます
# model.intersect_word2vec_format('./data/wiki/wiki2vec.bin', binary=True)

print("training...")

# 学習実行
model.train(sentences ,total_examples=model.corpus_count, epochs=model.iter)

if not os.path.isdir("./model"):
    os.mkdir("./model")
if not os.path.isdir("./model/doc2vec"):
    os.mkdir("./model/doc2vec")

# セーブ
model.save('./model/doc2vec/%s.model' %(folder))

# 学習後はモデルをファイルからロード可能
# model = models.Doc2Vec.load('./data/doc2vec.model')

# 順番が変わってしまうことがあるのでリストは学習後に再呼び出し
w2v = model.docvecs.offset2doctag

print("Finish!")
