# FastLabel社の課題提出用レポジトリ
よろしくお願いいたします。

## セットアップ手順
1. リポジトリをクローンします。

```bash
git clone https://github.com/kwon-o/fastlabel.git
cd fastlabel
```

2. 必要なPythonライブラリをインストールします。
```bash
pip install -r requirements.txt
```

## 使い方
画像ファイルからテーブル文字を抽出し、CSVファイルとして保存します。
```bash
python script.py -i 入力画像ファイル名.png -o 出力CSVファイル名.csv
```
* -i / --input : 入力画像ファイルのパス
* -o / --output : 出力するCSVファイルのパス

例:
```bash
python script.py -i sample_table.png -o result.csv
```


## 注意事項
* 画像内の表が明瞭に描画されている必要があります。
* 入力画像によってはOCRの認識精度に影響する場合があります。
