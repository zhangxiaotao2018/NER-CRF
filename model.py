# -*- coding: utf-8 -*-
"""
@version:NER-CRF
@time: 2019/01/17
@software: PyCharm
@Author：Zhang Xiaotao
@file: Model
"""

import sklearn_crfsuite
from sklearn.externals import joblib
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

from corpus import get_corpus
from utils import model_config, q_2_b, deal_with_entity


class NerModel(object):
    #初始化参数
    def __init__(self):
        self.corpus = get_corpus()
        self.corpus.initialize()
        self.model = None
    #初始化模型
    def initialize_model(self):
        algorithm = model_config.get("algorithm")
        c1 = float(model_config.get("c1"))
        c2 = float(model_config.get("c2"))
        max_iterations = int(model_config.get("max_iterations"))
        self.model = sklearn_crfsuite.CRF(algorithm=algorithm,
                                          c1=c1,
                                          c2=c2,
                                          max_iterations=max_iterations,
                                          all_possible_transitions=True)
        print("-> 完成模型初始化")
    #训练
    def train(self):
        self.initialize_model()
        x, y = self.corpus.generator()
        x_train, y_train = x[500:], y[500:]
        x_test, y_test = x[:500], y[:500]
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        print("-> 开始训练模型")
        self.model.fit(x_train, y_train)
        labels = list(self.model.classes_)
        labels.remove('0')
        y_predict = self.model.predict(x_test)
        print("-> 测试模型")
        metrics.flat_f1_score(y_test, y_predict, average='weighted', labels=labels)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
        print(metrics.flat_classification_report(y_test, y_predict, labels=sorted_labels, digits=3))
        print("-> 完成模型结果评测")
        self.save_model()

    #预测
    def predict(self, sentence, section_flag):
        self.load_model()
        x = q_2_b(sentence)
        word_lists = [['<BOS>'] + [c for c in x] + ['<EOS>']]
        word_grams = [self.corpus.segment_by_window(word_list) for word_list in word_lists]
        features = self.corpus.feature_extractor(word_grams)
        y_predict = self.model.predict(features)
        entity = ''
        entity_list = []
        tag = ""
        entity_tag = ["B_PER", "B_LOC", "B_ORG", "B_T", "I_LOC", "I_PER", "I_ORG", "I_T"]
        for index in range(len(y_predict[0])):
            if y_predict[0][index] == '0':
                if index ==0:
                    continue
                elif index > 0 and y_predict[0][index -1] != '0' and x[index] == x[index - 1]:
                    tag = y_predict[0][index - 1][2:]
                    entity += x[index]
            else:
                if index == 0:
                    entity += x[index]
                elif index > 0 and y_predict[0][index][-1] == y_predict[0][index - 1][-1]:
                    entity += x[index]
                    tag = y_predict[0][index][2:]
                elif index > 0 and y_predict[0][index][-1] != y_predict[0][index -1][-1]:
                    entity_list.append((entity, tag))
                    entity = ''
                    tag = ''
                    entity += x[index]
        if len(entity_list) > 0 and entity_list[0][1] == '':
            entity_list.pop(0)
        obj = deal_with_entity(entity_list, x, section_flag)
        return obj

    #保存模型
    def save_model(self, name='model'):
        model_path = model_config.get("model_path").format(name)
        joblib.dump(self.model, model_path)
        print("-> 完成模型存储")

     #加载模型
    def load_model(self, model_name ="model"):
        model_path = model_config.get("model_path").format(model_name)
        joblib.load(model_path)


def get_model():
    global __model
    __model = None
    if not __model:
        __model = NerModel()
    return __model