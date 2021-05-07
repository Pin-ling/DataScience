import pandas as pd
from sklearn import preprocessing
#import data frame
all_df = pd.read_csv('data.csv')
all_df = all_df[['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10',
                'feature11','feature12','feature13','feature14','feature15','feature16','feature17','feature18','feature19']].values
#normalize
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
all_df = minmax_scale.fit_transform(all_df)

#k-means
from sklearn.cluster import KMeans
clf_kmeans = KMeans(n_clusters=5)
kmeans_cluster = clf_kmeans.fit_predict(all_df)

#heirarchical clustering
from sklearn import cluster
clf_hc = cluster.AgglomerativeClustering(n_clusters=4)
hc_cluster = clf_hc.fit_predict(all_df)

#DBSCAN
clf_dbscan = cluster.DBSCAN(eps=0.4)
db_cluster = clf_dbscan.fit_predict(all_df)

#spectural clustering
clf_sc = cluster.SpectralClustering(n_clusters=4,n_neighbors=20)
sc_cluster = clf_sc.fit_predict(all_df)

#test
test = pd.read_csv('test.csv')
test = test[['0','1']].values

kmeans_predict = list()
hc_predict = list()
db_predict = list()
sc_predict = list()
for i in range(400):
        if kmeans_cluster[(test[i][0])] == kmeans_cluster[(test[i][1])]:
                kmeans_predict.append(1)
        elif kmeans_cluster[(test[i][0])] != kmeans_cluster[(test[i][1])]:
                kmeans_predict.append(0)
        if hc_cluster[(test[i][0])] == hc_cluster[(test[i][1])]:
                hc_predict.append(1)
        elif hc_cluster[(test[i][0])] != hc_cluster[(test[i][1])]:
                hc_predict.append(0)
        if db_cluster[(test[i][0])] == db_cluster[(test[i][1])]:
                db_predict.append(1)
        elif db_cluster[(test[i][0])] != db_cluster[(test[i][1])]:
                db_predict.append(0)
        if sc_cluster[(test[i][0])] == sc_cluster[(test[i][1])]:
                sc_predict.append(1)
        elif sc_cluster[(test[i][0])] != sc_cluster[(test[i][1])]:
                sc_predict.append(0)

predict = list()
for i in range(400):
        vote = kmeans_predict[i]*0.2 + hc_predict[i]*0.3 + db_predict[i]*0.2 + sc_predict[i]*0.3
        if vote >= 0.7:
                predict.append(1)
        else:
                predict.append(0)

#submit
submit = pd.read_csv('submit.csv')
submit['ans'] = predict
#submit['ans'] = sc_predict
submit['ans'] = submit['ans'].astype(int)
submit.to_csv('submit.csv',index = False)