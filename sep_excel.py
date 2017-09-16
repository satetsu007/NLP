# encoding:utf-8
# 実行するとmain関数内が実行されます

import pandas as pd
import os
import shutil

def sep_excel(file_name, code_file_name):
    """file_nameをcode_file_nameに基づき分割する
"""
    # 必要変数の定義
    code_excel = code_file_name + ".xlsx"
    # excel(file_name, code_file_name)ファイルを読み込む
    main_df = pd.read_excel(file_name, encoding="utf-8")
    code_df = pd.read_excel("excel/%s" % code_excel, encoding="utf-8" )
    
    # code(証券コード)のみ取り出す
    code = (code_df["code"].drop_duplicates())
    
    # code毎に読み込んだデータフレームを分割する
    company = []
    for c in code:
        company.append(main_df[main_df["code"] == c])
    
    # file_nameフォルダを初期化
    if not os.path.isdir("excel/%s" % code_file_name):
        os.mkdir("excel/%s" % code_file_name)
    else:
        shutil.rmtree("excel/%s" % code_file_name)
        os.mkdir("excel/%s" % code_file_name)
    
    # code毎に分割されたデータフレームをexcelファイルとして出力
    for cm, c in zip(company, code):
        cm.to_excel("excel/%s/%s.xlsx" % (code_file_name, str(c)), encoding="shift-jis")

def preset(folder, code_file_name):
    """分割されたexcelファイルからテキストファイルを作成する
"""    
    # 分割先のfolderを初期化
    if not os.path.isdir(folder):
        os.mkdir(folder)
    else:
        shutil.rmtree(folder)
        os.mkdir(folder)
    
    # code毎にexcelファイルを読み込む
    company_df = []
    code = os.listdir("excel/%s" % code_file_name)
    code = [c for c in code if c[-5:] == ".xlsx" ]
    code = [c for c in code if not c[0] == "."]
    for c in code:
        company_df.append(pd.read_excel("excel/%s/%s" % (code_file_name, c), encoding="utf-8"))
    
    # 読み込んだexcelファイルからテキストファイルを作成する
    for c, c_df in zip(code, company_df):
        file_name = []
        text = []
        file_name = list(c_df["file_name"])
        file_name = [str(fn) + ".txt" for fn in file_name]
        text = list(c_df["text"])
        os.mkdir("%s/%s" % (folder, c[0:-5]))
        
        for t, fn in zip(text, file_name):
            f = open("%s/%s/%s" % (folder, c[0:-5], fn), mode="w",  encoding="utf-8") 
            f.write(t)
            f.close()

if __name__ == "__main__":
    # 分割したいexcelファイル名を入力する
    # 標準ではcode.xlsx内に記述のあるmain.xlsx内のコードを分割する
    
    # 分割したexcelファイルを入力
    file_name = "main"
    file_path = "excel/%s.xlsx" % file_name
    # 分割したexcelファイルを保存するフォルダ名を入力する
    folder = "test"
    # 分割したexcelファイル内で必要なcodeのみ記述されたファイル名を入力
    code_file_name = "code"
    
    sep_excel(file_path, code_file_name)
    print("excelファイルを分割しました")
