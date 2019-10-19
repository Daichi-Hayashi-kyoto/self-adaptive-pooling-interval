# マイクロサービス環境下における動的ポーリング自動調整アルゴリズム

- Sorry, this README is written by japanese only.
- Now, I won't release English version README.

## 概要

- 動的ポーリング自動調整分野において，初めて実データ(CPU使用率)に対して実験した結果を載せている．
- [最新手法](https://www.doc.ic.ac.uk/~dtuncer/papers/tnsm18_gt.pdf)が実データで使用可能なのかを確かめ，課題を明確にした．
- この分野において，確率分布を絡めた新たな手法を提案し，最新手法と比較して44%〜88%ほどポーリング回数の減少を達成．
- ポーリング間隔の推移を図示することにより，考察をしやすくした(先行研究は図示されていなかった)
- この分野における従来からの性能指標であるRMSEの必要性に疑問を投げかける結果になった．

## 使用データ

- AWS上にdockerコンテナを用いてRUBiSをビルドして，負荷をかけるサーバを立てた．
- 上と同様に，dockerを用いてPromethusをビルドして，監視サーバを立ててGUI上でも変動が観れるようにした．
- **使用データの公表に関しては結果を公に公表したのちに，載せる予定である．**

## ファイルの簡単な説明

- 中間発表で必要なソースコードをjupyter notebook形式で載せる．

### evaluation_by_rmse.py

- 線形補間して，rmseを計算するやつ．

### ipynb

- change_finder_0902.ipynb

  - Change-Finderを適用したもの

- final_report_implement_0906.ipynb

  - 確率分布を用いたポーリング調整を行ったやつ．

- sequential_update_using_distribution_method.ipynb

  - 逐次更新型で確率分布のパラメータを推定していき，分布とのズレを見るやつ．cos類似度の考えを用いている．

- SST_online.ipynb

  - 特異スペクトル変換(オンライン型)
