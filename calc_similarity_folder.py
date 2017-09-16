# coding:utf-8

# gensimが入っていない場合
# conda install gensimを実行しておく
# 使用する際は
# python calc_similarity.py doc1 doc2
# ここでdoc1とdoc2はtourismフォルダ内に配置した文章集合群
# doc1を新たに用意した観光地の文章群
# doc2を既存の有名観光地の文章群として想定している

import os
# import sys
from gensim import models
import shutil
from tfidf import set_folder_morph
import pandas as pd
import numpy as np

# 学習用のクラスを定義
class LabeledListSentence(object):
    def __init__(self, words_list, labels):
        self.words_list = words_list
        self.labels = labels
    
    def __iter__(self):
        for i, words in enumerate(self.words_list):
            yield models.doc2vec.LabeledSentence(words, ['%s' % self.labels[i]])

# argv = sys.argv
# argc = len(argv)
# if (argc != 3):
# #引数がちゃんとあるかチェック
# #正しくなければメッセージを出力して終了
#     print('Usage: python %s arg1 arg2' %argv[0])
#     quit()

doc1 = "spot"#argv[1]
doc2 = "tourist_spot"#argv[2]

def calc_similarity(folder, doc1, doc2):
    
    # フォルダが存在しない場合エラーを返し終了する
    if not os.path.isdir("%s/%s" % (folder, doc1)):
        print("Not exist " + doc1)
        quit()
    if not os.path.isdir("%s/%s" % (folder, doc2)):
        print("Not exist " + doc2)
        quit()
    
    # 必要変数を定義
    directory = os.getcwd() + "/"
    # model_path = "model/doc2vec/"
    
    # tourism内の入力されたフォルダを読み込む
    spot = os.listdir("%s/%s" % (folder, doc1))
    spot = [sp for sp in spot if not sp == ".DS_Store"]
    tourist_spot = os.listdir("%s/%s" % (folder, doc2))
    tourist_spot = [tsp for tsp in tourist_spot if not tsp == ".DS_Store"]
    
    # tmpフォルダを新たに作る(存在する場合は初期化)
    if os.path.isdir("%s/tmp" % folder):
        shutil.rmtree("%s/tmp" % folder)
        os.mkdir("%s/tmp" % folder)
    else:
        os.mkdir("%s/tmp" % folder)
    
    # 入力されたフォルダ内のフォルダ内のテキストファイルを読み込む
    spot_list = []
    tourist_spot_list = []
    
    for sp in spot:
        tmp = os.listdir("%s/%s/%s" % (folder, doc1, sp))
        tmp = [fn for fn in tmp if fn[-4:] == ".txt"]
        tmp = [fn for fn in tmp if not fn[0] =="."]
        spot_list.append(tmp)
    
    for tsp in tourist_spot:
        tmp = os.listdir("%s/%s/%s" % (folder, doc2, tsp))
        tmp = [fn for fn in tmp if fn[-4:] == ".txt"]
        tmp = [fn for fn in tmp if not fn[0] =="."]
        tourist_spot_list.append(tmp)
    
    print("copying...")
    
    # 入力されたフォルダ内のフォルダ内ののテキストファイルをコピーする
    # ファイルの中身を100字取り出しておく
    # file_list = []
    text_list = []
    for sp, sp_l in zip(spot, spot_list):
        end = len(sp_l)
        flag = 0
        for tsp, tsp_l in zip(tourist_spot, tourist_spot_list):
            os.mkdir("%s/tmp/%s_%s" % (folder, sp, tsp))
            for s in sp_l:
                shutil.copy("%s/%s/%s/%s" % (folder, doc1, sp, s), "%s/tmp/%s_%s/%s" % (folder, sp, tsp, s))
                if flag in range(0, end):
                    f = open("%s/%s/%s/%s" % (folder, doc1, sp , s), encoding="utf-8")
                    text = f.read()
                    text_list.append(text[:100])
                    f.close()
                    flag += 1
            for t in tsp_l:
                shutil.copy("%s/%s/%s/%s" % (folder, doc2, tsp, t), "%s/tmp/%s_%s/%s" % (folder, sp, tsp, t))
    
    print("training...")
    
    morph_list, docs = [], []
    sentences = []
    model = []
    
    # ラベル付け, doc2vecモデルの定義を行う
    # alpha: 学習率 / min_count: X回未満しか出てこない単語は無視
    # size: ベクトルの次元数 / iter: 反復回数 / workers: 並列実行数
    # dm: 1の場合dmpvを使用, それ以外はDBoWを使用する
    # window: Doc2Vecで前後何単語まで入力とするか
    for sp in spot:
        for tsp in tourist_spot:
            ml, dc = set_folder_morph("%s/tmp/%s_%s" % (folder, sp, tsp))
            morph_list.append(ml)
            docs.append(dc)
            sentences.append(LabeledListSentence(ml, dc))
            model.append(models.Doc2Vec(alpha=0.025, dm=1, window=15, min_count=0,
                                        size=100, iter=20, workers=4))
    
    
    # doc2vecの学習前準備, 学習の実行
    index = 0
    for i in range(len(spot)):
        for j in range(len(tourist_spot)):
            model[index].build_vocab(sentences[index])
            model[index].train(sentences[index], total_examples=model[index].corpus_count, epochs=model[index].iter)
            index += 1
    
    if not os.path.isdir("./model"):
        os.mkdir("./model")
    if not os.path.isdir("./model/similarity"):
        os.mkdir("./model/similarity")
    
    # doc2vecモデルのセーブとロード
    index = 0
    for sp in spot:
        for tsp in tourist_spot:
            model[index].save("./model/similarity/%s_%s.model" % (sp, tsp))
            model[index] = models.Doc2Vec.load("./model/similarity/%s_%s.model" % (sp, tsp))
            index += 1
    
    
    print("calculating...")
    
    # 計算結果を格納するためのゼロ行列を作成
    
    spot_len = 0
    file_name = []
    label = []
    
    index = 0
    for sp_l in spot_list:
        spot_len += len(sp_l)
        for fn in sp_l:
            file_name.append(fn)
            label.append(spot[index])
        index += 1
    
    DOC_SIM = np.zeros((spot_len, len(tourist_spot)))
    
    # 類似度の計算
    index = 0
    start = 0
    end = 0
    for i, sp in enumerate(spot):
        start = end
        end += len(spot_list[i])
        for j, tsp in enumerate(tourist_spot):
            os.chdir(directory + "%s/tmp/%s_%s" % (folder, sp, tsp))
            flag = start
            for f1 in spot_list[i]:
                for f2 in tourist_spot_list[j]:
                    DOC_SIM[flag, j] += model[index].docvecs.similarity(d1=f1, d2=f2) / len(tourist_spot_list[j])
                flag += 1
            index += 1 
    
    os.chdir(directory)
    
    if not os.path.isdir("./similarity"):
        os.mkdir("./similarity")
    
    # 計算結果を出力するための準備
    DOC_SIM_DF = pd.DataFrame(np.c_[file_name, text_list,  DOC_SIM])
    DOC_SIM_DF.index = label
    DOC_SIM_DF.columns = ["file_name", "text"] + tourist_spot
    
    # 文書集合の類似度を出力
    # 1に近いほど似ている,0に近いほど似ていない
    DOC_SIM_DF.to_excel("./similarity/%s_%s.xlsx" % (doc1, doc2), encoding="shift-jis")
    
    print("Done.")
    
    # print("計算結果は")
    # print(DOC_SIM_DF)

if __name__ == "__main__":
    doc1 = "spot"
    doc2 = "tourist_spot"
    folder = "tourism"
    calc_similarity(folder, doc1, doc2)
