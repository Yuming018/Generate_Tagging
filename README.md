# Generate_Tagging

## 安裝套件 (requirements)
```
cd Generate_Tagging
pip install -r requirements.txt 
```

## 下載資料 
```
cd Generate_Tagging
gdown 1Ff9qqYtkIIMsu6AiaLqToCnxq8-bl9yc
unzip data.zip
```


## 訓練
### 指令註釋
``` 
python main.py -e 訓練幾次 -b 批次數 -p 子圖片的大小 -tm 是否為測試階段
```

## File Structure
```
|--- data
|   |--- train.csv
|   |--- valid.csv
|   |--- test.csv
|
|--- dataloader.py
|--- inference.py
|--- main.py
|--- model.py
|--- training.py
|
|--- requirements.txt
|--- README.md
|--- .gitignore
```
