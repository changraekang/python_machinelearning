import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")


train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

#print(train_data[:2])
#print('훈련 샘플의 개수 :',len(train_data)) # 훈련용 리뷰 개수 출력
#print('테스트 샘플의 개수 :',len(test_data)) # 테스트용 리뷰 개수 출력





train_data['document'].nunique(), train_data['label'].nunique()

train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거

train_data = train_data.dropna(how = 'any')
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
train_data['document'] = train_data['document'].str.replace('^ +', "") # white space 데이터를 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)

#print(train_data[:5])

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()

okt.pos("안녕 하세요 ㅎㅎㅎ")

print(okt.pos("안녕 하세요 ㅎㅎㅎ"))
    
    


