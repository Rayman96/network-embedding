import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from numpy import *
set_printoptions(threshold=NaN)
from mpl_toolkits.mplot3d import Axes3D

def weight(adjdic,W):
    for i in range(len(adjdic)):
        for item in adjdic[i]:
            W[i][item-1] = 1
    return W

def draw(Y):
    x = []
    y = []
    z = []
    for i in range(Y.shape[0]):
        x.append(Y[i,0].real)
        y.append(Y[i,1].real)
        z.append(Y[i,2].real)
    return(x,y,z)

def noj(Y):
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = float(Y[i,j].real)
    return(Y)

def degree(p):
    m = len(p)
    D = [[0 for i in range(m)] for j in range(m)]
    for i in range(len(p)):
        for j in range(len(p[i])):
            if j != i:
                D[i][i] += p[i][j]
    return(D)

# def draw(Y):
#     x = []
#     y = []
#     z = []
#     for i in range(Y.shape[0]):
#         x.append(Y[i,0])
#         y.append(Y[i,1])
#         z.append((Y[i,2]))
#     return(x,y,z)

def main():
    filename = '../dataset/150points.adjlist'
    G = nx.read_adjlist(filename)
    # cc = ['yellow', 'red', 'red', 'red', 'blue', 'blue', 'blue']
    nx.draw(G,with_labels=True)
    plt.show()
    with open(filename) as f:
        context = f.read()
        list_result = context.split('\n')
        length = len(list_result)
        for i in range(length):
            list_result[i] = list_result[i].split(' ')
    f.close()
    for item in list_result:
        for i in range(len(item)):
            item[i] = int(item[i])
    # 读入数据
    dic = {}#邻接矩阵
    for i in range(len(list_result)):
        dic[list_result[i][0]] = list_result[i][1:]

    m = len(dic)
    W = [[0 for i in range(m)] for j in range(m)]
    W = mat(weight(dic,W))
    print('W',W)
    t = 100#路径长度
    A = 0.5*W+0.3*W**2+0.2*W**3
    # A= W
    print('weight matrix',A)
    D = degree(array(A))
    print('D',D)
    a, b = linalg.eig(mat(D)-mat(A))
    print(a, b)
    Uk = mat(b[0:3])
    Y = transpose(Uk)
    Y = noj(Y)
    print(Y)
    x,y,z = draw(Y)
    # node_label = ['red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'red', 'blue', 'red', 'red', 'red', 'red',
    #               'blue', 'blue', 'red', 'red', 'blue', 'red', 'blue', 'red', 'blue', 'blue', 'blue', 'blue', 'blue',
    #               'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
    # node_label = ['red','red','red','red','blue','blue','blue','blue']
    # node_label = ['red','red','red','red','yellow','yellow','yellow','yellow','blue','blue','blue','blue']
    # node_label = ['yellow','red','red','red','blue','blue','blue']
    plt.scatter(x,y)
    for i in range(len(x)):
        plt.text(x[i],y[i],i+1)
    plt.show()
    # x, y, z = draw(Y)
    # # plt.scatter(x,y,color = node_label)
    # ax = plt.figure().add_subplot(111, projection='3d')
    # ax.scatter(x, y, z,color = node_label)
    # plt.show()


if __name__ == "__main__":
    main()
