# Generate_Tagging

## 安裝套件 (requirements)
```
cd Generate_Tagging
pip install -r requirements.txt 
```

## 下載資料 
```
cd Generate_Tagging
mkdir data
cd data
gdown 15fHorbAMDJWYPkMupn0lXTDco8Idr0b9
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
