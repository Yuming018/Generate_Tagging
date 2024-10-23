# Generate_Tagging

## 安裝套件 (requirements)
```
cd Generate_Tagging
pip install -r requirements.txt 
```

## 下載資料 
資料連結 : https://drive.google.com/file/d/1q6UdAc_lhS49Ivx5HAigVHiS4j-qSTbi/view?usp=drive_link
```
cd Generate_Tagging
mkdir data
cd data
gdown 1q6UdAc_lhS49Ivx5HAigVHiS4j-qSTbi
unzip data.zip
```

## 訓練
### 指令註釋
``` 
python main.py -e 訓練幾次 -b 批次數 -g 訓練甚麼資料 -m 訓練哪種模型 -tm 是否為測試階段
```

## File Structure
```
|--- data
|   |--- Ranking
|      |--- train.csv
|      |--- valid.csv
|      |--- test.csv
|   |--- train.csv
|   |--- valid.csv
|   |--- test.csv
|
|--- dataloader.py
|--- inference.py
|--- main.py
|--- model.py
|--- training.py
|--- helper.py
|--- eval_pred.py
|--- visual_data.py
|--- LLM_generate.py
|
|--- combined extracted information
|   |--- main.py
|   |--- knowledge_graph.py
|   |--- generation.py
|   |--- dataloader.py
|   |--- coreference_resolution.py
|
|--- requirements.txt
|--- README.md
|--- .gitignore
```

## 結果csv
結果 : https://drive.google.com/file/d/1NYrX0DKeUcWI4xrmnFRwvQ739cF90vms/view?usp=drive_link
```
檔案名稱 : 
w_ans_correct_ratio_x : 同時生成問題與答案
wo_ans_correct_ratio_x : 單純生成問題
x : 為2, 3, 4 ，代表有多少個資訊輸入給模型生成問題

欄位 : 
Prediction : 生成的問題與答案
Correct ans ratio : 與 Golden ans 相比，相符的有多少個
Question_difficulty : Event(主詞與代名詞相離多少句子), Relation(兩起事件相離多少句子)
```
