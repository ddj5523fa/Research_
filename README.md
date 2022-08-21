# Research

* 內容簡介:
參考報告，[連結](https://drive.google.com/drive/folders/1xqlfBzIDdzSvvsFrwEuS95Sc-En4modQ?usp=sharing)
---
## 程式路徑說明:
* /reserach/data:放入AWA2資料集。
* /reserach/model/:大部分為訓練好的model，[下載連結](https://drive.google.com/drive/folders/1FcO4RMNtuRAZBnYzhKUnhaV1mMLsXj6x?usp=sharing)
* /reserach//AWA2_33c/:有放之前實驗的數據儲存，[下載連結](https://drive.google.com/drive/folders/1IxlxMW-wTV8Vd_Fla6zPup1JxDjA36qq?usp=sharing)

## 套件安裝
使用 Pytorch -GPU，CUDA 11.2
==透過pytorch官網篩選條件下載:== [**連結**](https://pytorch.org/get-started/locally/)
```
conda create -y -n "yourname" python=3.7
conda activate "yourname"
cd "yourname"
pip install -r requirements.txt
```
若pip install -r requirements.txt時候，遇到版本問題，建議可以將python改成3.8版本。

## Requirement.txt說明:
```
fasttext==0.9.2     
gensim==4.1.2  
matplotlib==3.4.3
nmslib==2.1.1  
numpy==1.21.2
pandas==1.3.5
Pillow==9.2.0
protobuf==4.21.5
scikit_learn==1.1.2
tqdm==4.62.3   

```
## 程式:
* VER2_Data_trasform.ipynb:
作用是將資料集Train的部分，將每個class的label對應好各自的一個Word Vector。
* W2V_Encoder_Training.ipynb:
主要是訓練Encoder
* Archtecture.py
Encoder架構打包。
* Evaluation.ipynb
測試label推薦。
* classifier_Stacking:
製作了五個不同的encoder，產出五個Classifier，進行stacking merge。

## Training遇到問題:
```
Expected more than 1 value per channel when training, got input size torch.Size
```
這是因為在BN層裡，遇到batch=1的情况，通常就是最後一個batch資料數只有1引起的

解決:
```
batch_size>1, 且 drop_last=True
```
