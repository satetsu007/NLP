# coding:utf-8

import os
import sys
import pandas as pd
import numpy as np
import shutil
from tfidf import DF_, IDF_, TFIDF_, TF_, set_folder_morph

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

argv = sys.argv
argc = len(argv)
if (argc != 3):
#引数がちゃんとあるかチェック
#正しくなければメッセージを出力して終了
    print('Usage: python %s arg1 arg2' %argv[0])
    quit()

doc1 = argv[1]
doc2 = argv[2]
corpus = "corpus/"
folder = doc1 + "_" + doc2

if not os.path.isdir(corpus + doc1):
    print("Not exist " + doc1)
    quit()
if not os.path.isdir(corpus + doc2):
    print("Not exist " + doc2)
    quit()

directory = os.getcwd() + "/"
doc1_list = os.listdir(corpus + doc1)
doc2_list = os.listdir(corpus + doc2)

doc1_list = [doc for doc in doc1_list if doc[-4:] == ".txt"]
doc2_list = [doc for doc in doc2_list if doc[-4:] == ".txt"]


# folderがあれば初期化
# folderが無ければ新しく作成
if os.path.isdir(corpus + folder):
    shutil.rmtree(corpus + folder)
    os.mkdir(corpus + folder)
else:
    os.mkdir(corpus + folder)

# doc1のファイルとdoc2のファイルをfolderにコピー
for f1 in doc1_list:
    shutil.copy(corpus + doc1 + "/" + f1, corpus + folder + "/" + f1)

for f2 in doc2_list:
    shutil.copy(corpus + doc2 + "/" + f2, corpus + folder + "/" +  f2)

# TFIDFの計算

os.chdir(directory + corpus)
morph_list, docs = set_folder_morph(folder)
TF = TF_(morph_list)
DF = DF_(morph_list)
IDF = IDF_(DF)
TFIDF = TFIDF_(TF, IDF)

# TFIDFをcsvで出力する前準備
TFIDF.columns = docs

os.chdir(directory)
# csvファイルを出力
# tfidfフォルダがなければ作成
if not os.path.isdir("tfidf"):
    os.mkdir("tfidf")
TFIDF.to_csv("tfidf/" + folder + ".csv", encoding="utf-8")

# cos類似度を計算
# 1に近いほど似ている,0に近いほど似ていない
TFIDF = pd.read_csv("tfidf/" + folder + ".csv", index_col="Unnamed: 0", encoding="utf-8")

# TFIDFの行,列名を取得
filename = TFIDF.columns
word = TFIDF.index

# 計算結果を格納する行列を作成
# len(filename) * len(filename) 行列
result = np.zeros([len(filename), len(filename)])

# cos類似度をすべてのファイルに対して計算
for i, f1 in enumerate(filename):
    for j, f2 in enumerate(filename):
        result[i, j] = cos_sim(np.array(TFIDF[f1]), np.array(TFIDF[f2]))

# DFの行,列名をfilenameに変更
result_df = pd.DataFrame(result)
result_df.index = filename
result_df.columns = filename

# cos類似度の計算結果を出力するフォルダを準備
if not os.path.isdir("cos_sim"):
    os.mkdir("cos_sim")

# cos類似度を出力
result_df.to_csv("cos_sim/" + folder + ".csv", encoding="utf-8")
COS_SIM = pd.read_csv("cos_sim/" + folder + ".csv", index_col="Unnamed: 0", encoding="utf-8")

# 文書集合の類似度を計算

DOC_SIM = 0
for f1 in doc1_list:
    for f2 in doc2_list:
        DOC_SIM += COS_SIM.ix[f1, f2]
 
DOC_SIM = DOC_SIM / (len(doc1_list) * len(doc2_list))

# doc_simフォルダが存在しなければ作成
if not os.path.isdir("doc_sim"):
    os.mkdir("doc_sim")

DOC_SIM = pd.DataFrame({doc1: DOC_SIM}, index=[doc2])

# 文書集合の類似度を出力
# 1に近いほど似ている,0に近いほど似ていない
DOC_SIM.to_csv("doc_sim/" + folder + ".csv", encoding="utf-8")

