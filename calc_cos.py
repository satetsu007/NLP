# coding:utf-8

import os
import pandas as pd
import numpy as np
import sys

# cos類似度を計算する関数
# np.arrayを引数として類似度を返す
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# corpusごとにまとめられたフォルダ
corpus = "/corpus"
# cos類似度を計算したいフォルダ
folder = sys.argv[1]

directory = os.getcwd()
# csvからTFIDFを読み込む
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

