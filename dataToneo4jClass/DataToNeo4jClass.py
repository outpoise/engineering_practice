# _*_ coding: utf-8 _*_

from py2neo import Node, Graph, Relationship, NodeMatcher


class DataToNeo4j(object):
    """将excel中数据存入neo4j"""

    def __init__(self):
        """建立连接"""
        link = Graph("http://localhost:7474", username="neo4j", password="liwanrong")

        self.graph = link
        # self.graph = NodeMatcher(link)
        # 定义label
        self.buy = 'buy'
        self.sell = 'sell'
        self.graph.delete_all()
        self.matcher = NodeMatcher(link)




    def create_node(self, node1, node2):
        """建立节点"""



        for name in node1:
            # print(name)
            for des in node1[name]:
              node=Node(des,name=name)
            self.graph.create(node)
        for name in node2:
            for des in node2[name]:
                node = Node(des, name=name)
            self.graph.create(node)


    def create_relation(self, df_data):
        """建立联系"""


        m = 0
        for m in range(0, len(df_data)):
            try:


                rel = Relationship(
                    self.matcher.match(df_data['node1_des'][m], name=df_data['node1'][m]).first(),

                                          df_data['relation'][m], self.matcher.match(df_data['node2_des'][m], name=df_data['node2'][m]).first())
                self.graph.create(rel)
            except AttributeError as e:
                print(e, m)