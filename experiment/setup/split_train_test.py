import pandas as pd


path='/media/wilou/Elements/developmentNobra/ml-latest-small/ratings.csv'

df = pd.read_csv(path)
train=df.sample(frac=0.8,random_state=200) #random state is a seed value
test=df.drop(train.index)

train.to_csv("training")
test.to_csv("test")