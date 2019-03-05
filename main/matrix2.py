import networkx as nx
import matplotlib.pyplot as plt
from numpy import *
from sklearn import preprocessing
from sklearn.manifold import TSNE
import random
import operator
# import community
# set_printoptions(threshold=NaN)
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


def modify_adjdic(adjdic):
    # print(len(adjdic))
    sorted_adj = sorted(adjdic.items(), key=operator.itemgetter(0))
    adjdic = dict(sorted_adj)
    for i in range(1, len(adjdic) + 1):
        for j in adjdic[i]:
            if i not in adjdic[j]:
                adjdic[j].append(i)

    return (adjdic)


def random_walk(adjdic, start=1, l=3):
    # l为random walk的长度
    path = [start]
    temp = start
    for i in range(l):
        next = random.sample(adjdic[temp], 1)
        temp = next[0]
        path.append(temp)
    return (path)


def trunked(randlist, adjdic, p, k=3):
    trun = []
    # temp = []
    for i in range(len(randlist) - k + 1):
        # trun.append(randlist[i:i+k])
        trun.append(randlist[i:i + k])
        # print(trun)
        # for i in range(len(trun)):
        #     for j in range(i+1,len(trun)):
        #         if trun[i] <= trun[j]:
        #             temp.append([trun[i],trun[j]])
        #         else:
        #             temp.append([trun[j],trun[i]])
        # for item in temp:
        #     if item[1] in adjdic[item[0]]:
        #         p[item[0] - 1][item[1] - 1] += 1
        #         p[item[1] - 1][item[0] - 1] += 1
        #         continue
        #     elif item[0] in adjdic[item[1]]:
        #         p[item[0] - 1][item[1] - 1] += 1
        #         p[item[1] - 1][item[0] - 1] += 1
    return (trun)


def findedge(adjdic, trunkedlist, p):
    temp = []
    for item in trunkedlist:
        for i in range(len(item)):
            for j in range(i + 1, len(item)):
                if item[i] <= item[j]:
                    temp.append([item[i], item[j]])
                else:
                    temp.append([item[j], item[i]])
    # print('temp',temp)
    for item in temp:
        if item[1] in adjdic[item[0]]:
            p[item[0] - 1][item[1] - 1] += 1
            p[item[1] - 1][item[0] - 1] += 1
            continue
        elif item[0] in adjdic[item[1]]:
            p[item[0] - 1][item[1] - 1] += 1
            p[item[1] - 1][item[0] - 1] += 1
    return p


def weight(adjdic, W):
    for i in range(len(adjdic)):
        for item in adjdic[i + 1]:
            W[i][item - 1] = 1
    return W


def degree(p):
    m = len(p)
    D = [[0 for i in range(m)] for j in range(m)]
    for i in range(len(p)):
        for j in range(len(p[i])):
            if j != i:
                D[i][i] += p[i][j]
    return (D)


def draw(Y):
    x = []
    y = []
    z = []
    for i in range(Y.shape[0]):
        x.append(Y[i, 0])
        y.append(Y[i, 1])
        # z.append((Y[i,2]))
    return (x, y, z)


def noj(Y):
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = float(Y[i, j].real)
    return (Y)


def GF(p, r, e, z):
    # p是权重矩阵，r是embedding维度，e是停止的准确度
    z_old = z
    # z_new = mat(zeros((len(p),r)))
    # print('z_new',z_new)
    t = 1  # 循环次数
    res = 10000  # 新旧误差
    u = 0.11  # 调参量
    while (res > e):
        print('times', t)
        # miu = 1/(1000*t**(0.5))
        miu = 0.0001
        t = t + 1
        z_temp = copy(z_old)
        # print('oriz_temp',z_temp)
        for i in range(len(p)):
            if res <= e:
                break
            # print('i',i)
            if i == 0:
                continue
            else:
                for j in range(i):
                    # print('j,i',j,i)
                    z_old[i] = mat(z_old[i]) + miu * ((p[i][j] - float(
                        mat(z_old[i]) * transpose(mat(z_old[j])))) * mat(
                            z_old[j]) + u * mat(z_old[i]))
        # print('z_temp',z_temp)
        res = (linalg.norm((z_old - z_temp), ord=None))**2  # 新旧矩阵差值的行列式的平方
        print('res', res)
        if res <= e:
            break
        # z_old[i] = z_new[i]
        # print('z_old',z_old)
    return (z_old)


def regulize_f(f, method):
    if method == 'minmax':
        minmaxscaler = preprocessing.MinMaxScaler()
        f_minmax = minmaxscaler.fit_transform(f)
        return (f_minmax)
    elif method == 'sigmoid':
        for i in range(len(f)):
            for p in range(len(f[i])):
                f[i][p] = 1.0 / (1 + exp(-float(f[i][p])))
        return (f)


def get_label(file):
    filename = open(file, 'r')
    content = filename.readlines()
    filename.close()
    content = [x.strip() for x in content]
    l = []
    for i in range(len(content)):
        l.append(content[i].split())
    label = []
    for item in l:
        label.append([int(i) for i in item])
    label = array(label)
    label = label[:, 1]
    return (label)


def savematrix(p):
    p = array(p)
    filename = '../dataset/cora2.matrix'
    with open(filename, 'w') as f:
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                f.write(str(p[i, j]) + ' ')
            f.write('\n')
    f.close()
    return ()


def main():
    filename = 'dataset/cora.adjlist'
    G = nx.read_adjlist(filename)
    node_label = get_label('dataset/cora_labels.txt')
    color_label = []
    for item in node_label:
        if item == 0 or item == '0':
            color_label.append('red')
        elif item == 1 or item == '1':
            color_label.append('green')
        elif item == 2 or item == '2':
            color_label.append('yellow')
        elif item == 3 or item == '3':
            color_label.append('blue')
        elif item == 4 or item == '4':
            color_label.append('brown')
        elif item == 5 or item == '5':
            color_label.append('black')
        elif item == 6 or item == '6':
            color_label.append('pink')
    # label = loadtxt('../dataset/300points.txt',dtype = int)
    # label = label[:,1]
    # nx.draw(G,node_color = label)
    # plt.show()
    with open(filename) as f:
        context = f.read()
        list_result = context.split('\n')
        length = len(list_result)
        for i in range(length):
            list_result[i] = list_result[i].split(' ')
    f.close()

    if list_result[0][0] == '0':
        for item in list_result:
            for i in range(len(item)):
                item[i] = int(item[i]) + 1
    else:
        for item in list_result:
            for i in range(len(item)):
                item[i] = int(item[i])
    # for item in list_result:
    #     for i in range(len(item)):
    #         if list_result[0][0] == 0:
    #             item[i] = int(item[i])+1
    #         else:
    #             item[i] = int(item[i])
    # 读入数据
    dic = {}
    for i in range(len(list_result)):
        dic[list_result[i][0]] = list_result[i][1:]
    # 产生dic
    print('dic1', dic)
    # if list_result[0][0] == '0':
    dic = modify_adjdic(dic)
    print('dic2', dic)
    m = len(dic)
    p = [[0 for i in range(m)] for j in range(m)]
    # 创建矩阵A
    # W = [[0 for i in range(m)] for j in range(m)]
    # W = mat(weight(dic,W))
    # print('W')

    # 构建自己的权重矩阵 W
    T = 1
    r = 16
    z_old = mat(ones((len(p), r)))
    # z_old=[[random.randint(-1, 1) for i in range(r)]for i in range(len(p))]
    # z_old = mat(z_old)
    print(z_old)
    # z_old = mat(z_old)
    # print('first z',z_old)
    print('walking')
    for t in range(T):
        s = 1
        # for s in range(1,len(dic)+1):
        path = random_walk(dic, start=s, l=len(p) * 10)
        # print('start = %d'%s)
        print('walking is over')
        # print('path',path)
        trunklist = trunked(path, dic, p, k=8)
        # print('trunked list',trunklist)
        p = findedge(dic, trunklist, p)
        # print("the %s node is over" %s)
    print('done walking!')
    # print('matirx p\n',mat(p))#!!!!!!!!!!!!!!!!!!!
    p = regulize_f(p, 'minmax')
    print('rugulized p', p)
    # D = degree(p)
    # print('D\n',mat(D))
    # savematrix(p)
    """
    Random Walk完成，矩阵W构造完成
    """

    # node_label = ['red','red','red','red','red','red','red','red','red','blue','red','red','red','red','blue','blue','red','red','blue','red','blue','red','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue','blue']
    # node_label = ['red','red','red','red','blue','blue','blue','blue']
    # node_label = ['red','blue','red','red','blue','blue','red','blue']
    # node_label = ['yellow','red','red','red','blue','blue','blue']
    # node_label = ['red','red','red','red','yellow','yellow','yellow','yellow','blue','blue','blue','blue']
    # node_label = ['red' for i in range(50)]
    # node_label2 = ['yellow' for i in range(100)]
    # node_label3 = ['blue' for i in range(150)]
    # node_label = node_label+node_label2+node_label3
    """
    方法一：our own method
    """
    # L = mat(D)-mat(p)
    # k = 2
    # a,b = linalg.eig(L)
    # print('a','b',a,b)
    # Uk = mat(b[0:r])
    # # # Uk = mat(b)
    # # # Uk_tsne = TSNE(n_components=2).fit_transform(Uk)
    # Y = Uk_tsne
    # Y = transpose(Uk)
    # Y = noj(Y)
    # Y = regulize_f(Y,'minmax')
    # print('Y',Y)
    # z_old = GF(array(W), r, 0.5, z_old)
    gap = [1000, 100, 10, 1, 0.1]
    for item in gap:
        z_now = z_old
        z_now = GF(p, r, item, z_old)
        # print('matrix z',z_old)
        print('get z')
        Uk_tsne = TSNE(n_components=2).fit_transform(z_now)
        z_now = Uk_tsne
        x, y, z = draw(z_now)
        # savetxt('../dataset/300point.txt',z_old)
        """
        画图展示
        # """
        plt.scatter(x, y, color=color_label)
        # for i in range(len(x)):
        #     plt.text(x[i],y[i],i+1)
        # # ax = plt.figure().add_subplot(111,projection = '3d')
        # # ax.scatter(x,y,z,color = node_label)
        # # plt.title('ours')
        # plt.show()
        plt.savefig('gap = %s.png' % item)
        plt.clf()

    # kmeans = KMeans(n_clusters=3, random_state=0).fit(z_old)  # Y的行向量用来聚类
    # print(kmeans.labels_)
    # for i in range(len(kmeans.labels_)):
    #     kmeans.labels_[i] = int(kmeans.labels_[i])
    # nx.draw(G, with_labels=True, node_color=kmeans.labels_+1)
    # plt.show()
    # gmm = GaussianMixture(n_components=3,covariance_type='tied',init_params='kmeans')
    # gmm.fit(z_old)
    # y_pred = gmm.predict(z_old)
    # print(y_pred)
    # print(len(x),len(y_pred))
    # # for i in range()
    # nx.draw(G,with_labels=True,node_color = y_pred)
    # plt.show()
    # eps = [0.1,0.5,1,3,4,5,8,10,20]
    # for p in eps:
    #     t = DBSCAN(eps = p).fit(z_old)
    #     mod = {}
    #     for i in range(len(t.labels_)):
    #         mod[str(i)] = t.labels_[i]
    #     print('mod',community.modularity(mod,G))
    # 写文件
    # filename = 'D:/embedding论文/实验数据/'+'cora/own4_cora.emb'
    # with open(filename, 'w') as f:
    #     f.write(str(z_old.shape[0])+' '+str(z_old.shape[1])+'\n')
    #     for i in range(z_old.shape[0]):
    #         f.write(str(i+1)+' '+str(z_old[i, 0])+' '+str(z_old[i, 1])+'\n')
    # f.close()


# ------------------------------------------------------ #

if __name__ == "__main__":
    main()
