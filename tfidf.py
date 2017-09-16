#coding:utf-8

import MeCab
import os
from collections import Counter
from copy import copy
import numpy as np
import pandas as pd


# 入力されたテキストを形態素解析してリストで返す
def morph_analysis(file, mode=True):
    """テキストファイルを入力すると
mode=Falseのときは副詞、名詞、形容詞、動詞を単語毎に分かれたリストを返す
mode=Trueのときは形態素解析した結果を返す
"""
    if mode:
        # 形態素解析した結果を格納するリストを作成   
        BOW = []
        # MeCabを使用して分かち書き処理を行うためにタグを指定
        t = MeCab.Tagger("-Owakati")
        # テキストファイルを読み込む
        f = open(file, encoding="utf-8")
        tmp = f.read()
        f.close()
        
        # テキストファイルの下処理
        tmp = tmp.replace("\u3000", " ").split("\n")
        tmp = [line for line in tmp if not line == ""]
        text = copy(tmp)
        text = [t.parse(line) for line in text]
        text = [line.split("\n") for line in text]
        text = [word.split() for line in text for word in line if not word==""]
        #print(text)
        
        for line in text:
            for word in line:
                    BOW.append(word)
        
        # 形態素解析された結果を返す   
        return BOW
    else:   
        # 形態素解析した結果を格納するリストを作成
        BOW = []
        
        # MeCabを使用して分かち書き処理を行うためにタグを指定
        t = MeCab.Tagger("-Ochasen")
        # mecab-ipadic-neologdを使用する場合は上行をコメントアウトし下行のコメントアウトを外す
        # t = MeCab.Tagger("-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd/")
        # ファイルの読み込み
        # 文字コードをutf-8で読み込む
        f = open(file, encoding='utf-8')
        tmp = f.read()
        f.close()
        
        tmp = tmp.replace("\u3000", " ").split("\n")
        tmp = [line for line in tmp if not line == ""]
        text = copy(tmp)
        # text = copy(tmp[4:])
        text = [t.parse(line) for line in text]
        text = [line.split("\n") for line in text]
        text = [[word.split("\t") for word in line] for line in text]
        
        for line in text:
            for word in line:
                if len(word) == 1:
                    break
                elif word[3][:2] == "動詞" or word[3][:3] == "形容詞" or word[3][:2] == "名詞" or word[3][:2] == "副詞":
                    BOW.append(word[0])
        
        # 形態素解析結果を返す
        return BOW

# ファイル名と形態素解析結果の取得
def set_folder_morph(folder):
    """フォルダ名を入力するとフォルダ内のテキストファイルを
形態素解析してファイル名と解析結果を返す
"""
    
    # フォルダ内のファイル名を取得
    # .txt形式のみ取得する
    filename = os.listdir(folder)
    docs = [doc for doc in filename if doc[-4:] == ".txt"]
    docs = [doc for doc in docs if not doc[0] =="."]
    # 形態素解析の結果を格納するリストを作成
    morph_list = []
    
    
    for doc in docs:
        # フォルダ内のファイルのパスを代入
        file_path = os.getcwd() + "/" + folder + "/" + doc
        # 形態素解析結果を追加
        morph_list.append(morph_analysis(file_path))
    
    # 形態素解析とファイル名を返す
    return morph_list, docs



# TFの計算
def TF_(morph_list):
    """形態素解析の結果を入力すると
DataFrame(pandas)でTFを計算して返す
TF = テキスト内のある単語の出現回数 / テキスト内の単語の総数
"""
    # テキスト内の単語の出現回数を格納するリストを作成
    tf = []

    # 出現回数を計算する
    for morph in morph_list:
        # Counter(list)
        # リスト内の単語の出現回数を計算して辞書型で返す
        # テキスト内の単語の出現回数を格納
        tf.append(Counter(morph))

    # 全テキストで出てきた単語をtmpに格納するために
    # 数え上げたものを足し合わせる
    for i in range(len(tf)):
        # 1つ目はコピーする
        if i == 0:
            tmp = copy(tf[i])
        # 2つ目以降は足していく
        else:
            tmp += tf[i]

    # tmp = 文書集合全体の単語頻度

    # tfのkeysとvaluesを格納するリストを作成
    tf_keys = []
    tf_values = []

    # tfのkeysとvaluesを追加していく
    for i in tf:
        tf_keys.append(list(i.keys()))
        tf_values.append(list(i.values()))

    # TFを格納するリストを作成
    TF = []

    # TFを計算していく
    for i in range(len(tf)):
        # tmp内の全ての単語の出現回数を0に初期化する
        tmp_keys = list(tmp.keys())
        for key in tmp_keys:
            tmp[key] = 0


        # keyとvalueを同時に回すためにzipを使う
        # tmpに一時的にTFの計算結果を代入していく
        for key, value in zip(tf_keys[i], tf_values[i]):
            if key in tmp:
                tmp[key] = value / len(morph_list[i])

        # 計算結果をTFに追加していく
        TF.append(copy(tmp))

    # TFをDataFrame化する
    # 後で計算しやすいよう転置しておく
    TF = pd.DataFrame.from_dict(TF).T

    # TFを返す
    return TF


# DFの計算
def DF_(morph_list):
    """形態素解析した結果を入力すると
DataFrame(pandas)でDFを計算して返す
DF = ある単語を含む文書数 / 全文書数
"""
    # 形態素解析結果を数え上げたものを格納するリストを作成
    df = []
    # 形態素解析結果を数え上げる
    for i in morph_list:
        df.append(Counter(i))
    # 数え上げたものをtmpにひとまとめにする
    for i in range(len(df)):
        if i == 0:
            tmp = copy(df[i])
        else:
            tmp += df[i]

    # tmp = 文書集合全体の単語頻度

    # dfのkeysとvaluesを格納するリストを作成
    df_keys = []
    df_values = []

    # dfのkeysとvaluesを追加していく
    for i in df:
        df_keys.append(list(i.keys()))
        df_values.append(list(i.values()))

    # 文書内に単語が出てくると1出てこない場合は0のフラグを立てる
    # フラグを格納するリストを作成
    df_tmp = []

    for i in range(len(df)):

        # まずtmpを全て0に
        tmp_keys = list(tmp.keys())
        for key in tmp_keys:
            tmp[key] = 0

        # もし文書内に単語がでてくるならフラグを立てる
        for key in df_keys[i]:
            if key in tmp:
                tmp[key] = 1

        # フラグを立てたものを格納する
        df_tmp.append(copy(tmp))

    # DFの値を全て0で初期化
    DF = copy(tmp)
    for key in tmp_keys:
        DF[key] = 0

    # ある単語が何文書で出現したかの計算結果をDFに代入していく
    for dt in df_tmp:
        DF += dt

    DF = pd.DataFrame.from_dict(DF,orient='index')

    # DFを計算する
    DF = DF / len(morph_list)

    # DFを返す
    return DF

# IDFの計算
def IDF_(DF):
    """DFを入力するとIDFを計算して返す
IDF = log(1 / DF)
"""
    # IDFを求める
    IDF = np.log(1 / DF)

    # IDFを返す
    return IDF


# TFIDFの計算
def TFIDF_(TF, IDF):
    """TFとIDFを入力するとTFIDFを計算して返す
TFIDF = TF * IDF
"""
    # TFIDFには計算結果が入る
    # TFと同じ大きさのゼロ行列を作成
    TFIDF = np.zeros((len(TF),len(TF.T)))
    # ゼロ行列をDataFrame化する
    TFIDF = pd.DataFrame(TFIDF, index=TF.index)

    # テキスト毎にTFIDFを計算
    for i in range(len(TFIDF.T)):
        TFIDF[i] = (np.array(TF[i])*np.array(IDF).T).T

    # TFIDFを返す
    return TFIDF
