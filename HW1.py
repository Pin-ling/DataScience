import re
import numpy
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy

wind_direction_mapping = {
    'N': 0,
    'NNE': 22.5,
    'NE': 45,
    'ENE': 67.5,
    'E': 90,
    'ESE': 112.5,
    'SE': 135,
    'SSE': 157.5,
    'S':180,
    'SSW': 202.5,
    'SW': 225,
    'WSW': 247.5,
    'W': 270,
    'WNW': 292.5,
    'NW': 315,
    'NNW': 337.5
}
today_rainy_mapping = {
    'Yes':1,
    'No':0
}

def PreprocessingData(raw_df):
    raw_df['Attribute8'] = raw_df['Attribute8'].map(wind_direction_mapping)
    raw_df['Attribute10'] = raw_df['Attribute10'].map(wind_direction_mapping)
    raw_df['Attribute11'] = raw_df['Attribute11'].map(wind_direction_mapping)
    raw_df['Attribute22'] = raw_df['Attribute22'].map(today_rainy_mapping)

    #轉為array
    ndarray = raw_df.values
    # 標準化(in book)
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_df = minmax_scale.fit_transform(ndarray)
    return scaled_df

#train
all_df = pd.read_csv("train.csv")
train_cols = ['Attribute2','Attribute3','Attribute4','Attribute5','Attribute6','Attribute7',
              'Attribute8','Attribute9','Attribute10','Attribute11','Attribute12','Attribute13','Attribute14',
              'Attribute15','Attribute16','Attribute17','Attribute18','Attribute19','Attribute20','Attribute21',
              'Attribute22','Attribute23']
train_df = all_df[train_cols]
train_df = train_df.dropna()

train_df['Attribute23'] = train_df['Attribute23'].map(today_rainy_mapping)
#down sampling
count_class_no, count_class_yes = train_df['Attribute23'].value_counts()
df_class_no = train_df[train_df['Attribute23'] == 0]
df_class_yes = train_df[train_df['Attribute23'] == 1]
df_class_no_under = df_class_no.sample(count_class_yes)
train_df = pd.concat([df_class_no_under, df_class_yes], axis=0)

train_df = PreprocessingData(train_df)

train_features = train_df[:,:21]
train_label = train_df[:,21]

#建立random forest
forest = RandomForestClassifier(n_estimators=200)
clf = forest.fit(train_features,train_label)

#result
submit_df = pd.read_csv("test.csv")
submit_cols = ['Attribute2','Attribute3','Attribute4','Attribute5','Attribute6','Attribute7',
              'Attribute8','Attribute9','Attribute10','Attribute11','Attribute12','Attribute13','Attribute14',
              'Attribute15','Attribute16','Attribute17','Attribute18','Attribute19','Attribute20','Attribute21',
              'Attribute22']
submit_df = submit_df[submit_cols]
submit_df = PreprocessingData(submit_df)

submit_predict = clf.predict(submit_df)

submit = pd.read_csv('ex_submit.csv')
submit['ans'] = submit_predict
submit['ans'] = submit['ans'].astype(int)
submit.to_csv('submit.csv',index = False)