使い方
- データセットはdatasets/"dataset_name"/trainA,trainB,testA,testB/*.jpgと配置してください
    - GoogleColaboratoryを使用する場合は google.colab.drive.mount('/content/drive')を実行してdrive/My Drive/Colab Notebooks/datasets/"dataset_name"/trainA,trainB,testA,testB/*.jpgとなるように配置してください
- モデルと損失値はmodels/"dataset_name"/に配置されます

- 1.runningインスタンスを生成する
    - colab使用の有無(bool=False), 
    - 出力される正方形の一辺(int=256),
    - 白黒画像での学習(bool=False), 
    - 最終出力の次元数(int=3), 
    - identitylossの使用の有無(bool=True)


- 2.running.train関数で学習を開始する
    - 学習するデータセット名(fine-tuningの際は源のデータセット(D)名で)(str=None), 
    - 学習済みepoch数(fine-tuningで未学習の際は源のモデルのepoch数)(int=0), 
    - データセットの画像の縦横ピクセル(tuple(int, int)=256,256), 
    - fine-tuningか否か(bool=False), 
    - fine-tuningの際の学習済みepoch数(int=None), 
    - fine-tuningの際の学習するデータセット名(str=None), 
    - データオーグメンテーションの使用の有無(bool=False) 

- 3.running.make_directory関数で学習結果を見る
    - 生成する画像を絞るか。Falseなら全てのファイルが変換される。低リソース下ならTrueがおすすめ(bool=False),
    - 生成するデータセット名(interpolationなら源のデータセット名)(str=None),
    - データセットの画像の縦横ピクセル(tuple(int, int)=256,256),
    - trainデータで生成するか、testデータで生成するか。trainデータの場合Trueを代入(bool=False),
    - 学習済みepoch数(interpolationの際は源のモデルのepoch数)(int=200),
    - fake画像を生成するかidentity画像やcycle画像を生成するか。'fakes'か'identity'か'cycle'を代入(str='fakes'),
    - interpolationの際の派生先データセット名(str=None),
    - interpolationの際の派生先モデルのepoch数
    - interpolationの際の変換の方向(例、源モデルのB変換と派生モデルのA変換を相互に実装するなら'BtoA'。でも大体'AtoA'か'BtoB')(str=None)

- 4.running.show_train_curve関数で損失関数の推移を見る
    - 学習したモデルのデータセット名(str)
    - 学習済みのepoch数(int)
