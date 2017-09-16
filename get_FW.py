# coding:utf-8

import os
import pandas as pd
import sys

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

if not os.path.isdir("FW"):
    os.mkdir("FW")

if not os.path.isdir("FW/" + folder):
    os.mkdir("FW/" + folder)

for fn in filename:
    FW = TFIDF[fn].sort_values(ascending=False)[:100]
    FW.to_csv("FW/" + folder + "/" + fn.replace(".txt", "") + ".csv", encoding="utf-8")
