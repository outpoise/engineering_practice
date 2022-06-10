from summarizer import Summarizer
from transformers import *
import pymongo
import re
import pymysql
def article_deal(s):

    contents = ""
    for line in s.split("\n"):

        # 逐行处理

        line = re.sub(' ', '', line)
        line = re.sub('(www\.(.*?)\.com)|(http://(.*?)\.com)|(www\.(.*?)\.cn)|(http://t.cn/(.*))', '',  line.lower())  # 去URL
        line = line.replace("#", "")
        line = line.replace("【", "")
        line = line.replace("】", "。")
        contents = contents + line

    return contents





def main():

    connect = pymongo.MongoClient(host='47.110.148.178', port=27017, username="root", password="8gd#729*1@5")
    mongo_newsdb = connect['SinaDataBase']
    collection = mongo_newsdb['Newsweibo_Test']
    articles = list(collection.find({}))

    custom_config = AutoConfig.from_pretrained('./bert-base-chinese/bert-base-chinese.json')
    custom_config.output_hidden_states=True
    custom_tokenizer = AutoTokenizer.from_pretrained('./bert-base-chinese/bert-base-chinese-vocab.txt')
    custom_model = AutoModel.from_pretrained('./bert-base-chinese/bert-base-chinese-pytorch_model.bin', config=custom_config)

    model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)

    for index,article in enumerate(articles):
        print("--------------------第%d篇文章------------------" % (index+1))

        text = article['text']
        text=article_deal(text)
        abstract=model(text, num_sentences=3)
        print("摘要：")

        for i in abstract:
          print(i)
        collection.update_one({'_id': article['_id']}, {'$set': {
            'new_text': text
        }}, upsert=False)
        collection.update_one({'_id': article['_id']}, {'$set': {
            'abstract': abstract
        }}, upsert=False)




        print("------------------------------")


if __name__ == '__main__':
    main()
