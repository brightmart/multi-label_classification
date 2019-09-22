# coding=utf-8
import json
import numpy as np
from zuo.predict_similarity import predict_single
# from zuo.predict_similarity import predict as predict_combine
#from predict_similarity_2 import predict as predict_2 # todo todo todo
# from main_buer import predict_standardized_output as predict_2
# from predict_similarity_s import predict as predict_baili
input_path_labor = "/input/labor/input.json"
output_path_labor = "/output/labor/output.json"
input_path_divorce = "/input/divorce/input.json"
output_path_divorce = "/output/divorce/output.json"
input_path_loan = "/input/loan/input.json"
output_path_loan = "/output/loan/output.json"

# todo todo todo todo ==============================要在本地测试main.py,打开这几行代码
input_path_labor = "zuo/data_all/labor/data_small_selected.json"
output_path_labor = "zuo/data_all/labor/output.json"
input_path_divorce = "zuo/data_all/divorce/data_small_selected.json"
output_path_divorce = "zuo/data_all/divorce/output.json"
input_path_loan = "zuo/data_all/loan/data_small_selected.json"
output_path_loan = "zuo/data_all/loan/output.json"
## todo todo todo todo ==============================要在本地测试main.py,打开这几行代码
# roeberta_zh_L-24_H-768_A-12.zip

def predict(input_path, output_path):
    category_en = ''
    if 'labor' in input_path:
        category_en = 'labor'
    if 'divorce' in input_path:
        category_en = 'divorce'
    if 'loan' in input_path:
        category_en = 'loan'

    inf = open(input_path, "r", encoding='utf-8')
    ouf = open(output_path, "w", encoding='utf-8')
    count = 0
    for line in inf:
        pre_doc = json.loads(line)
        new_pre_doc = []
        for sent in pre_doc:
            labels_prob_big=predict_single(str(sent['sentence']), category_en) # buer. labels_prob_buer:  {'LB1':0.01,'LB2':0.02,'LB3':0.0,...}
            print("###labels_prob_big:",labels_prob_big)
            label_list=get_label_list_single(labels_prob_big)
            print("###label_list:",label_list)
            sent['labels'] = label_list
            new_pre_doc.append(sent)
            #count = count + 1
        json.dump(new_pre_doc, ouf, ensure_ascii=False)
        ouf.write('\n')

    inf.close()
    ouf.close()

def check_whether_has_any_candidate(labels_with_prob_dict):
    """
    检测是否有有效候选项
    :param labels_with_prob_dict:
    :return: True if has candidate; False is not has candidate
    """
    candidate_list=[]
    for lable_tag_en, p_temp in labels_with_prob_dict.items():
        if float(p_temp)>0.01:
            candidate_list.append(lable_tag_en)

    return candidate_list
    #if len(candidate_list)>0:
    #    return candidate_list
    #else:
    #    return False

def combine_prob(labels_prob_1,labels_prob_2,weight_1=0.50):
    """
    整合两个概率，概率取加权平均
    :param labels_prob_1:
    :param labels_prob_2:
    :return: 加权平均后的概率
    """
    result_dict={}
    for tag, p_1 in labels_prob_1.items():
        # print("tag:==",tag,"===;p_1:",p_1) # tag: LB1 ;p_1: 0.0
        p_2=labels_prob_2[tag]
        p_avg=float(p_1)*weight_1+float(p_2) *(1.0-weight_1)
        result_dict[tag]=p_avg
    return result_dict

def get_label_list(labels_prob_1,labels_prob_2,threshold=0.5):
    """

    :param labels_prob_1: {'LB1': 0.0016595844645053148, 'LB2': 0.11449998617172241, 'LB3': 0.003680239664390683,
    :param labels_prob_2: {'LB1': 1.810735193430446e-05, 'LB2': 0.0016248620037610333, 'LB3': 1.8363494746154174e-05
    :return:
    """
    label_list=[]
    for tag_en, possibility_1 in labels_prob_1.items():
        possibility_2=labels_prob_2[tag_en]
        possibility=np.average([possibility_1,possibility_2])
        if possibility>threshold:
            label_list.append(tag_en)
    return label_list

def get_label_list_single(labels_prob_1,threshold=0.5):
    """

    :param labels_prob_1: {'LB1': 0.0016595844645053148, 'LB2': 0.11449998617172241, 'LB3': 0.003680239664390683,
    :param labels_prob_2: {'LB1': 1.810735193430446e-05, 'LB2': 0.0016248620037610333, 'LB3': 1.8363494746154174e-05
    :return:
    """
    label_list=[]
    for tag_en, possibility_1 in labels_prob_1.items():
        #possibility_2=labels_prob_2[tag_en]
        #possibility=np.average([possibility_1,possibility_2])
        if possibility_1>threshold:
            label_list.append(tag_en)
    return label_list

# labor领域预测
predict(input_path_labor, output_path_labor)

# loan领域预测
predict(input_path_loan, output_path_loan)

# divorce领域预测
predict(input_path_divorce, output_path_divorce)
