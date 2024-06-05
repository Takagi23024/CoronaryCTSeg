# CTデータから冠動脈を抽出

## DICOMデータの描画





## Deeplearningでスライス毎に正解マスクをつけて、そのマスクを繋げて抽出

## 古典的な手法の組み合わせ
1. 上行大動脈を見つける(シード点検出)
　・頭方向から数スライス表示して一点をマニュアルで選択
　・閾値やハフ変換などで自動で検出

2. 上行大動脈から下に繋がる心臓を抽出
　・k-meansクラスタリングやGaussian Mixture Models(GMM)などのクラスタリング手法
　・領域拡張法

3. ボリュームデータから表面メッシュを生成(.obj)に変換
　スケルトン化: 冠動脈のボクセルデータからキー点を抽出し、これらの点を繋いで各枝の中心線を形成。
　再構築: キー点に基づいて、冠動脈の枝ごとにスムーズなメッシュを作成。 
　統合: 各枝のメッシュを結合して、完全な冠動脈のメッシュを構築

4. 上行大動脈から三つに分岐している場所を見つけてそれぞれ分けて抽出


これから参考にしたいURL
https://meknowledge.jpn.org/2020/12/24/python-ct-dicom/

https://meknowledge.jpn.org/2020/12/27/python-ct-dicom-obj-surface/

https://docs.blender.org/manual/en/latest/modeling/modifiers/generate/volume_to_mesh.html

https://github.com/KeremTurgutlu/dicom-contour/blob/master/dicom_contour/utils.py