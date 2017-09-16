#coding:utf-8

from tfidf import DF_, IDF_, TFIDF_, TF_, set_folder_morph
import os
import sys

# corpusごとにまとめられたフォルダ
corpus = "/corpus/"
# tfidfを計算したいフォルダ
folder = sys.argv[1]


directory = os.getcwd()
os.chdir(directory + corpus)
morph_list, docs = set_folder_morph(folder)
TF = TF_(morph_list)
DF = DF_(morph_list)
IDF = IDF_(DF)
TFIDF = TFIDF_(TF, IDF)
# print(TF)

# TFIDFをcsvで出力する前準備
TFIDF.columns = docs
# word = TFIDF[0]
# TFIDF.index = word
# TFIDF = TFIDF.ix[:, 1:]

os.chdir(directory)
# csvファイルを出力
# tfidfフォルダがなければ作成
if not os.path.isdir("tfidf"):
    os.mkdir("tfidf")
TFIDF.to_csv("tfidf/" + folder + ".csv", encoding="utf-8")
