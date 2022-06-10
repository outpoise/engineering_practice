# _*_ coding: utf-8 _*_

from dataToneo4jClass.DataToNeo4jClass import DataToNeo4j
import pymongo
import os
import pandas as pd

#pip install py2no ==5.0b1 注意版本不低于此版本


connect = pymongo.MongoClient(host='47.110.148.178', port=27017, username="root", password="8gd#729*1@5")
mongo_newsdb = connect['SinaDataBase']
collection = mongo_newsdb['Newsweibo_Test']
articles = list(collection.find())
print(len(articles))


def data_extraction():


    node1 = {}
    node2 = {}
    for article in articles:
        for triple in article['triple']:
            triple = triple.strip('(')
            triple = triple.strip(')')
            triple = triple.split(',')
            if triple[0] in node1:
                if triple[0] in article['triple_des']:

                   node1[triple[0]].add(article['triple_des'][triple[0]])

                else:
                    node1[triple[0]].add("null")

            else:
                node1[triple[0]] = set([])

                if triple[0] in article['triple_des']:

                   node1[triple[0]].add(article['triple_des'][triple[0]])

                else:
                   node1[triple[0]].add("null")

            if triple[2] in node2:
                if triple[2] in article['triple_des']:
                   node2[triple[2]].add(article['triple_des'][triple[2]])

                else:
                    node2[triple[2]].add("null")
            else:
                node2[triple[2]] = set([])

                if triple[2] in article['triple_des']:

                    node2[triple[2]].add(article['triple_des'][triple[2]])
                else:
                    node2[triple[2]].add("null")



    return node1,node2



def relation_extraction():
    #联系数据抽取


    links_dict={}
    node1_list=[]
    relation_list=[]
    node2_list=[]
    node1_des=[]
    node2_des=[]

    for article in articles:
        for triple in article['triple']:
            triple = triple.strip('(')
            triple = triple.strip(')')
            triple = triple.split(',')
            node1_list.append(triple[0])
            relation_list.append(triple[1])
            node2_list.append(triple[2])
            if triple[0] in article['triple_des']:
              node1_des.append(article['triple_des'][triple[0]])
            else:
              node1_des.append("null")
            if triple[2] in article['triple_des']:
              node2_des.append(article['triple_des'][triple[2]])
            else:
                node2_des.append("null")



    links_dict['node1'] = node1_list
    links_dict['relation'] =  relation_list
    links_dict['node2'] = node2_list
    links_dict['node1_des']=node1_des
    links_dict['node2_des']=node2_des
    df_data = pd.DataFrame(links_dict)
    # print(df_data)


    return df_data


create_data=DataToNeo4j()
create_data.create_node(data_extraction()[0], data_extraction()[1])
create_data.create_relation(relation_extraction())
