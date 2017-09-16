# coding:utf-8

# 必要モジュールの読み込み
from calc_similarity_folder import calc_similarity
from sep_excel import sep_excel, preset
import os

if __name__ == "__main__":
    # 新しくテキストファイルが展開されるフォルダを指定する
    # 後で作成するので存在しないフォルダでも構わない
    folder = "doc2vec"
    # 類似度を計算するexcelファイル名を入力
    main = "main"
    # 類似度を比較したいexcelファイル名を入力
    target = "target"
    
    # 必要変数の定義
    main_folder = folder + "/%s" % main
    target_folder = folder + "/%s" % target
    
    main_excel = "excel/%s.xlsx" % main
    target_excel = "excel/%s.xlsx" % target
    main_code = "%s_code" % main
    target_code = "%s_code" % target
    
    # folderが存在しなければ作成
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    # mainのexcelファイルをmain_codeで指定したcode毎に分割する
    sep_excel(main_excel, main_code)
    print("%s.xlsxを分割しました" % main)
    # tarfetのexcelファイルをtarget_codeで指定したcode毎に分割する
    sep_excel(target_excel, target_code)
    print("%s.xlsxを分割しました" % target)
    
    # 分割されたexcelファイルからテキストファイルを作成
    preset(main_folder, main_code)
    print("%sのtxtを作成しました" % main)
    preset(target_folder, target_code)
    print("%sのtxtを作成しました" % target)
    
    # 作成されたテキストファイルに対してDoc2Vecで単語を分散表現し,コサイン類似度を計算する
    calc_similarity(folder, main, target)

