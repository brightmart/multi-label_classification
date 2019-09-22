#!/usr/bin/python
# -*- coding: utf-8 -*-
import codecs
import pandas as pd
import re
category_cn_dict = {'divorce': '婚姻家庭', 'labor': '劳动争议', 'loan': '借款纠纷'}

category_list = ['divorce', 'labor', 'loan']
data_path = 'zuo/data_all/'  # 'zuo/data_all/'

negative_word_list = ['不', '未', '没', '没有', '无', '非', '并未', '不再', '不能', '无法', '不足以', '不至于', '不存在',
                      '不能证明', '不认可','尚未', '不行', '不到', '不满', '未满', '未到', '没到', '没满', '没法'] # 否定词列表
negative_word = '(' + '|'.join(negative_word_list) + ')' # 否定词正则

def load_factor_with_additional_info(source_file):
    """
    加载要素-描述-代表性描述的文件
    :param source_file:
    :return: 返回一个dict: dict[factor_zh]={desc,sentence_repres1,sentence_repres2,sentence_repres3}
    """
    # 1.加载文件
    data = pd.read_csv(source_file)
    # 2.遍历并放入dict中
    factorzh_additionalinfo_dict = {}  # 中文标签对应的额外信息：描述、好的句子1，2，3
    factorzh_neg_sentenceorlabel_dict = {}  # 中文标签对应的候选负样本信息
    categorycn2labelcn2other = {}
    # 获得每个类别下所有的中文标签、描述、好的句子的集合
    category_cn_list = [v for k, v in category_cn_dict.items()]
    categorycn2totallist_dict = {}
    for indexx, row_ in data.iterrows():
        category_ = row_['纠纷类型']
        # label_en_ = row_['标签']
        label_zh_ = row_['中文标签']
        desc_ = row_['一句话标准描述']
        sentence_repres1_ = row_['好的句子1']
        sentence_repres2_ = row_['好的句子2']
        sentence_repres3_ = row_['好的句子3']
        sublist = categorycn2totallist_dict.get(category_, [])
        sublist.extend([label_zh_, desc_, sentence_repres1_, sentence_repres2_, sentence_repres3_])
        categorycn2totallist_dict[category_] = sublist

    # print("按类别统计的所有的标签、描述、有用的句子的集合:",len(categorycn2totallist_dict),";total_list:",categorycn2totallist_dict)
    for index, row in data.iterrows():
        category = row['纠纷类型']
        # label_en=row['标签']
        label_zh = row['中文标签']
        desc = row['一句话标准描述']
        sentence_repres1 = row['好的句子1']
        sentence_repres2 = row['好的句子2']
        sentence_repres3 = row['好的句子3']
        # reugular_expression=row.get('re','') # this field is not use at all.
        key = category + "_" + label_zh
        factorzh_additionalinfo_dict[key] = {"label_zh": label_zh, "desc": desc, "sentence_repres1": sentence_repres1,
                                             "sentence_repres2": sentence_repres2, "sentence_repres3": sentence_repres3,'reugular_expression':''}
        other_desc_list = [x for x in categorycn2totallist_dict[category] if
                           x not in [label_zh, desc, sentence_repres1, sentence_repres2, sentence_repres3]]
        factorzh_neg_sentenceorlabel_dict[key] = list(set(other_desc_list))
    return factorzh_additionalinfo_dict, factorzh_neg_sentenceorlabel_dict


def read_source_flies(data_path, category):
    """
    读取原始数据
    :param data_path:
    :param category:
    :return:
    """
    data_path_group = data_path
    file_path_divorce = data_path + category
    # tags
    divorce_tags_file = file_path_divorce + '/tags.txt'
    divorce_tags_object = open(divorce_tags_file, 'r')
    divorce_tags = divorce_tags_object.readlines();
    divorce_tags = [x.strip() for x in divorce_tags]
    divorce_tags_dict = {j: xx for j, xx in enumerate(divorce_tags)}
    divorce_tag2id_dict = {yy: xx for xx, yy in divorce_tags_dict.items()}
    # selectedtags
    divorce_selectedtags_file = file_path_divorce + '/selectedtags.txt'
    divorce_selectedtags_object = codecs.open(divorce_selectedtags_file, 'r',
                                              'utf-8')  # open(divorce_selectedtags_file, 'r')
    divorce_selectedtags = divorce_selectedtags_object.readlines()
    divorce_selectedtags = [xx.strip() for xx in divorce_selectedtags]
    divorce_selectedtags_dict = {jj: xxx for jj, xxx in enumerate(divorce_selectedtags)}  # {1:'婚后有子女',2:'限制行为能力子女抚养',...}
    # raw data
    if 'big' not in data_path:
        divorce_data_file = file_path_divorce + '/data_small_selected.json'
    else:
        divorce_data_file = file_path_divorce + '/train_selected.json'
    divorce_data_object = codecs.open(divorce_data_file, 'r', 'utf-8')  # open(divorce_data_file, 'r');
    divorce_lines = divorce_data_object.readlines()
    return divorce_tags_dict, divorce_tag2id_dict, divorce_selectedtags, divorce_selectedtags_dict, divorce_lines


category2tags_dict = {}
category2tag2id_dict = {}
category2selectedtags = {}
category2selectedtags_dict = {}
# factorzh_additionalinfo_dict={}

factorzh_additionalinfo_dict, factorzh_neg_sentenceorlabel_dict = load_factor_with_additional_info(
    data_path + 'factor_desc_represent.csv') # 'factor_desc_represent.csv'
#print("####factorzh_additionalinfo_dict:",factorzh_additionalinfo_dict)
# '借款纠纷_拒绝履行偿还': {'label_zh': '拒绝履行偿还', 'desc': '未按时偿还借款|拒不偿还借款', 'sentence_repres1': '如果XX未按指定的期间履行给付金钱义务，应当依照xx规定，加倍支付迟延履行期间的债务利息', 'sentence_repres2': '被告本人没有向信用社贷款，没有签过合同和借款凭证，也没有实际领取贷款本金，因此不同意还款。', 'sentence_repres3': '合同约定的借款期限届满之日，XXX未能履行还款义务'}
# print("-------------------------------------------------------------------------------------------------")
count_k = 0
# for k,v in factorzh_neg_sentenceorlabel_dict.items():
#    print(count_k,"k:",k,";v:",v)
#    count_k=count_k+1

for i, category in enumerate(category_list):
    tags_dict, tag2id_dict, selectedtags, selectedtags_dict, _ = read_source_flies(data_path, category)
    category2tags_dict[category] = tags_dict
    category2tag2id_dict[category] = tag2id_dict
    category2selectedtags[category] = selectedtags
    category2selectedtags_dict[category] = selectedtags_dict

#   reugular_expression = factorzh_additionalinfo_dict[domain_zn + "_" + tag_cn_candidate]['reugular_expression']
def sentence_match_single(sentence,keyword):
    flag_positive = len(re.findall(keyword, sentence)) > 0

    flag_neg_1 = len(re.findall(negative_word+keyword, sentence)) > 0
    flag_neg_2 = len(re.findall(keyword+negative_word, sentence)) > 0
    if flag_positive:
        if not flag_neg_1 and not flag_neg_2:
            return True
    return False

#result=sentence_match_single('原、被告于1980年结婚，婚后生有女儿范某香，大儿子范某荣，二儿子范某华均已成家另居。','divorce')
#print("result:",result)
# print("category2tags_dict:",category2tags_dict)
# print("category2tag2id_dict:",category2tag2id_dict)
# print("category2selectedtags_dict:",category2selectedtags_dict)
