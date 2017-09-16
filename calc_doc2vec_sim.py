# coding:utf-8

# 先にcalc_word2vec.pyを実行しておく
# ➞calc_cos_sim.pyも実行する必要あり
# gensimが入っていない場合
# conda install gensimを実行しておく

import os
import sys
from gensim import models
import shutil
from tfidf import set_folder_morph
import pandas as pd

argv = sys.argv
argc = len(argv)
if (argc != 3):
#引数がちゃんとあるかチェック
#正しくなければメッセージを出力して終了
    print('Usage: python %s arg1 arg2' %argv[0])
    quit()

doc1 = argv[1]
doc2 = argv[2]
folder = doc1 + "_" + doc2
corpus = "corpus/"

# フォルダが存在しない場合エラーを返し終了する
if not os.path.isdir(corpus + doc1):
    print("Not exist " + doc1)
    quit()
if not os.path.isdir(corpus + doc2):
    print("Not exist " + doc2)
    quit()

# 必要変数を定義
directory = os.getcwd() + "/"
model_path = "model/doc2vec/"
doc1_list = os.listdir(corpus + doc1)
doc2_list = os.listdir(corpus + doc2)

# ファイル名を取得
doc1_list = [doc for doc in doc1_list if doc[-4:] == ".txt"]
doc2_list = [doc for doc in doc2_list if doc[-4:] == ".txt"]
doc1_list = [doc for doc in doc1_list if not doc[0] == "."]
doc2_list = [doc for doc in doc2_list if not doc[0] == "."]


# folderがあれば初期化
# folderが無ければ新しく作成
if os.path.isdir(corpus + folder):
    shutil.rmtree(corpus + folder)
    os.mkdir(corpus + folder)
else:
    os.mkdir(corpus + folder)

print("copying...")

# doc1のファイルとdoc2のファイルをfolderにコピー
for f1 in doc1_list:
    shutil.copy(corpus + doc1 + "/" + f1, corpus + folder + "/" + f1)
for f2 in doc2_list:
    shutil.copy(corpus + doc2 + "/" + f2, corpus + folder + "/" +  f2)

# 学習用のクラスを定義
class LabeledListSentence(object):
    def __init__(self, words_list, labels):
        self.words_list = words_list
        self.labels = labels
    
    def __iter__(self):
        for i, words in enumerate(self.words_list):
            yield models.doc2vec.LabeledSentence(words, ['%s' % self.labels[i]])

# ラベル付けを行う
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

# モデルのセーブ
model.save(model_path + '%s.model' %(folder))

# モデルのロード
model = models.Doc2Vec.load(model_path + folder + ".model")


os.chdir(corpus + folder)
filename = os.listdir()

print("calculating...")

# 類似度を計算し平均する
DOC_SIM = 0
for f1 in doc1_list:
    for f2 in doc2_list:
        DOC_SIM += model.docvecs.similarity(d1=f1, d2=f2)
DOC_SIM = DOC_SIM / (len(doc1_list) * len(doc2_list))

os.chdir(directory)


# 計算結果を出力するための準備
if not os.path.isdir("./doc2vec_sim"):
    os.mkdir("./doc2vec_sim")

DOC_SIM_DF = pd.DataFrame({doc1: DOC_SIM}, index=[doc2])

# 文書集合の類似度を出力
# 1に近いほど似ている,0に近いほど似ていない
DOC_SIM_DF.to_excel("doc2vec_sim/%s.xlsx" % folder, encoding="shift-jis")

print("Finish!")
print("Doc2Vecでの%sと%sの類似度: " %(doc1, doc2) + str(DOC_SIM))
