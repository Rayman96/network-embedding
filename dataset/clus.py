import numpy as np 
np.set_printoptions(suppress=True)
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt 
from sklearn.cluster import KMeans
import csv

def get_embedding(file):
	filename=open(file,'r')
	content=filename.readlines()
	filename.close()
	content=[x.strip() for x in content]
	content.pop(0)
	l=[]
	for i in range(len(content)):
		l.append(content[i].split())
	data = []
	for item in l:
		data.append([float(i) for i in item])  # Convert into float
	return(data)

data_file = 'D:\\embedding论文\\实验数据\\cora\\own4_cora.emb'
data = get_embedding(data_file)  # Evaluation file
data = np.array(sorted(data))  # sort the data in terms of its serial number
embedding_data = data[:,1:]

# filename = open('D:\\embedding论文\\实验数据\\facebook\\own_facebook.emb')
# b = np.loadtxt(filename,dtype = np.float32)
t = DBSCAN(eps =5 ).fit(embedding_data)
# t = KMeans(n_clusters=7, random_state=0).fit(embedding_data)
print(t.labels_)
la = t.labels_


# la = list(la)
# file = open('own6_label.csv','a',newline = '\n')
# csv_write = csv.writer(file)
# csv_write.writerow(la)
# file.close()

# print(la)
x= embedding_data[:,0]
y = embedding_data[:,1]

plt.scatter(x,y,c = t.labels_)
plt.show()