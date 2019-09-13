import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import operator
import random

def read_file(file):
    filename = file
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
    # 读入数据
    dic = {}
    for i in range(len(list_result)):
        dic[list_result[i][0]] = list_result[i][1:]
    return dic

def modify_adjdic(adjdic):
    # print(len(adjdic))
    sorted_adj = sorted(adjdic.items(), key=operator.itemgetter(0))
    adjdic = dict(sorted_adj)
    for i in range(1, len(adjdic) + 1):
        for j in adjdic[i]:
            if i not in adjdic[j]:
                adjdic[j].append(i)
    return (adjdic)

def frontier_sample(B, m, c, graph):
    sample_list = []
    L = random.sample(range(1, len(graph)+1), m)
    # Initialize L = (v1, . . . , vm) with m randomly chosen vertices (uniformly)
    for time in range(B-m*c):
        Degree = {}
        degree_L = 0
        for node in L:
            Degree[node] = len(graph[node])
            degree_L += len(graph[node])
        probability = {}
        for node in L:
            probability[node] = Degree[node]/degree_L
        # 按照度的分布进行抽样
        probability = sorted(probability.items(), key=operator.itemgetter(1))  # 按概率从低到高排列
        print(probability)
        pro_u = random.random()
        print(pro_u)
        temp_probability = 0
        for key,value in probability:
            print(key)
            temp_probability += probability[key]
            print(temp_probability)
            if pro_u <= temp_probability:
                u = key
                break
        print(u)

    #
    #
    # print(Degree)
    # print(probability)

    # print(L)
    return sample_list

def main():
    graph = read_file('../dataset/karate.adjlist')
    graph = modify_adjdic(graph)
    # print(graph)
    sample_list = frontier_sample(15, 3, 1, graph)

    return 0


if __name__ == '__main__':
    main()
