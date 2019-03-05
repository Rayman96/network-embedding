import networkx as nx
import matplotlib.pyplot as plt
from numpy import *
from sklearn.cluster import KMeans


# G = nx.generators.community.random_partition_graph((30,20,10),0.8,0.1,directed = False)

# G = nx.karate_club_graph()
# for node in G:
#     print(nx.neighbors(G,node)[0])
# with open('../dataset/karate.adjlist') as f:
#     context = f.read()
#     list_result = context.split('\n')
#     length = len(list_result)
#     for i in range(length):
#         list_result[i] = list_result[i].split(' ')
# f.close()
# for item in list_result:
#     for i in range(len(item)):
#         item[i] = int(item[i])
# print(list_result)

# nx.draw(G,with_labels=True)
# plt.show()
# print(G.adjacency_list())
# de = nx.degree(G)
# print(de)
# degree = []
# for i in range(len(de)):
#     degree.append(de[i])
# D = eye(len(degree))
# for i in range(len(degree)):
#     D[i][i] = degree[i]#degree matrix
# D = mat(D)#degree matrix
#
#
#
# def method2(D,G,k):
#     L = nx.laplacian_matrix(G)
#     p = D.I*L
#     a,b = linalg.eig(p)
#     Uk = mat(b[0:k+1])
#     Y = transpose(Uk)
#     return(a,Y)
#
# k = 2
# # Y = method1()
# a,Y = method2(D,G,k)
# print(Y)
# kmeans = KMeans(n_clusters=k, random_state= 0).fit(Y)#Y的行向量用来聚类
#
# nx.draw(G,with_labels=True ,node_color = kmeans.labels_)
# plt.show()
# # print('特征值',a)
# # print('特征向量',Y)
# # print('分类',kmeans.labels_)
# # a = mat([[1,-3,3,3],[-5,-1,-5,5],[-2,0,-4,0],[-2,0,-2,4]])
# # pni = mat([[-1,0,0,0],[0,0,1,0],[1,1,0,0],[0,1,1,1]])
# # b = mat([[0],[0],[-1],[-1]])
# # c = mat([[1,2,1,-2]])
# #
# #
# # print(pni.I)
filename = '../dataset/facebook_combined.txt'
G = nx.read_edgelist(filename)
# nx.write_adjlist(G, '../dataset/facebook.adjlist')
# cc = ['yellow', 'red', 'red', 'blue','red', 'blue', 'blue']
# nx.draw(G, with_labels=True)
nx.draw(G)
plt.show()
