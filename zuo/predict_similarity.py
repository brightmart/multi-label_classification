#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-07-07 10:16
"""

import json
import time
import numpy as np
import random
from zuo.util import category2tags_dict, category2tag2id_dict, category2selectedtags, category2selectedtags_dict, \
    category_cn_dict, load_factor_with_additional_info,sentence_match_single
from zuo.run_classifier_predict_online import predict_online as predict_online_1

data_path = 'zuo/data_all/'
factorzh_additionalinfo_dict, _ = load_factor_with_additional_info(data_path + 'factor_desc_represent_add_re.csv')#'factor_desc_represent.csv')


def predict_single(sentence, category_en, candidate_list=None, threshold=0.5):
    """
    通过预测得到标签的列表: 会综合使用多个标签或句子来做预测
    :param sentence: 句子
    :param category_en: 类别信息，如 'labor'
    : return: label_result_list: 预测除的标签. e.g. label_result_list=['LN1','LN13']
    : return: tag_en_possibility_dict，这个模型融合时使用  e.g. tag_en_possibility_dict['LN1']=0.6
    """
    tags_dict, tag2id_dict, selectedtags, selectedtags_dict = category2tags_dict[category_en], category2tag2id_dict[
        category_en], category2selectedtags[category_en], category2selectedtags_dict[category_en]
    # print("###tags_dict:",tags_dict,";tag2id_dict:",tag2id_dict,";selectedtags:",selectedtags,";selectedtags_dict:",selectedtags_dict)
    selectedtag2index_dict = {v: k for k, v in selectedtags_dict.items()}

    label_result_list = []
    tag_en_possibility_dict = {}  # e.g. tag_en_possibility_dict['LN1']=0.6

    if candidate_list is not None and len(candidate_list) > 0:  # 有传入的候选项不为空，那么更新selectedtags即更新模型预测的标签范围
        index_list = [tag2id_dict[tag_en_] for tag_en_ in candidate_list]  # [0,1,3]
        selectedtags = [selectedtags_dict[indexx] for indexx in index_list]

    # print("#####bxul.selectedtags:",selectedtags)
    for candidate_tag_cn in selectedtags:
        if isinstance(sentence, float): return label_result_list
        # if len(sentence) > 90: sentence = sentence[0:45] + '。' + sentence[-45:] # OLD
        if len(sentence) > 250: sentence = sentence[0:125] + '。' + sentence[-125:]  # OLD

        key = category_cn_dict[category_en] + "_" + candidate_tag_cn
        possibility_list = []
        list_allow = get_allow_list_tag(factorzh_additionalinfo_dict[key])
        for k, candi_tag_cn in factorzh_additionalinfo_dict[key].items():  # k:'label_zh',candi_tag_cn: '拒绝履行偿还'
            if k not in list_allow: continue  # # if k not in ['label_zh','desc','sentence_repres1']:continue
            type_information = category_cn_dict[category_en] + '的' + candi_tag_cn
            ###############################################################
            # 添加一个判断，减少计算量即多数时候，只用一个模型计算就可以了，少数情况用两个模型
            _, possibility = predict_online_6(sentence, type_information)
            weight = 1  # 0.3333 #0.175 if k!='label_zh' else 0.3
            possibility_pos=possibility[1]
            possibility_list.append((possibility_pos, weight, k))
        p_list = [e[0] * e[1] for e in possibility_list]
        possibility_pos_final = np.average(p_list)

        index = selectedtag2index_dict[candidate_tag_cn]
        tag_en = tags_dict[index]  # e.g. tag_en='LN1'
        tag_en_possibility_dict[tag_en] = possibility_pos_final

        if possibility_pos_final > threshold:  # 如果超过阀值，加入到预测除的标签列表
            label_result_list.append(tag_en)
        ##################这里使用正则####这里使用正则####这里使用正则######################################################################################
        reugular_expression = factorzh_additionalinfo_dict[category_cn_dict[category_en] + "_" + candidate_tag_cn][
            'reugular_expression']
        if reugular_expression != '' and str(reugular_expression) != 'nan':
            flag = sentence_match_single(sentence, reugular_expression)
            if flag == True:
                tag_en_possibility_dict[tag_en] = 1.0
        ##################这里使用正则###这里使用正则###这里使用正则########################################################################################
    return tag_en_possibility_dict  # label_result_list,



def get_allow_list_tag(k_candi_tag_dict):  # k,candi_tag_cn
    """
    获取允许的候选项
    :param k_candi_tag_dict: '借款纠纷_拒绝履行偿还': {'label_zh': '拒绝履行偿还', 'desc': '未按时偿还借款|拒不偿还借款', 'sentence_repres1': '如果XX未按指定的期间履行给付金钱义务，应当依照xx规定，加倍支付迟延履行期间的债务利息', 'sentence_repres2': '被告本人没有向信用社贷款，没有签过合同和借款凭证，也没有实际领取贷款本金，因此不同意还款。', 'sentence_repres3': '合同约定的借款期限届满之日，XXX未能履行还款义务'}
    :return:
    """
    allow_tag_list = []
    allow_tag_list.append('label_zh')
    # tag_list = random.sample(['desc', 'sentence_repres1', 'sentence_repres2', 'sentence_repres3'], 2)
    #allow_tag_list.extend(tag_list)
    # todo
    return allow_tag_list


if __name__ == "__main__":
    text = "二、威海市文登区畜牧兽医技术服务中心向宋忠文支付2016年度带薪年休假工资1561.68元，于本判决生效后十日内付清；"
    domain = "labor"

    labels_prob = predict(text, domain)
    print(labels_prob)
