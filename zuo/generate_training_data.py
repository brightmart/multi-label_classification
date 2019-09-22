# -*- coding: utf-8 -*-
import json
import random
from util import category_cn_dict, read_source_flies, factorzh_additionalinfo_dict, factorzh_neg_sentenceorlabel_dict


def generate_train_data(data_path,examples_source_path):
    """
    输入3个数据文件和
    """
    # 分别获取三个文件下的数据(divorce,labor,loan)
    category_list = ['divorce', 'labor', 'loan']
    count = 0
    count_empty_list = 0
    total_pos = 0  # 正样本的总数量
    total_neg = 0  # 负样本的总数量
    total_list = []
    avg_length = 0  # 平均长度
    ignore_count = 0

    category2tag_en_example_dict=load_pos_examples_files(examples_source_path)
    for i, category in enumerate(category_list):
        # 读取每个目录下的数据
        divorce_tags_dict, divorce_tag2id_dict, divorce_selectedtags, divorce_selectedtags_dict, divorce_lines = read_source_flies(data_path, category)
        divorce_id2tag_dict={v:k for k,v in divorce_tag2id_dict.items()}
        divorce_selectedtags2id_dict={v:k for k,v in divorce_selectedtags_dict.items()}
        for k, line in enumerate(divorce_lines):
            list_element = json.loads(line.strip())
            for m, element in enumerate(list_element):
                # 单个样本级别（句子和标签列表）
                sentence = element['sentence']
                sentence = sentence.replace("\t\n", "。").replace("\t", " ")
                if 'check-yuhan@gridsum.com' in sentence: ignore_count = ignore_count + 1;continue
                if len(str(sentence)) > 122: sentence = sentence[0:60] + "。" + sentence[-62:]  # 处理超长的文本 # if len(str(sentence)) > 90: sentence = sentence[0:43] + "。" + sentence[-45:]  # 处理超长的文本

                labels = element['labels']

                if len(labels) < 1:count_empty_list = count_empty_list + 1 # 统计空标签的行数
                labels_word_list = [divorce_selectedtags_dict[divorce_tag2id_dict[x]] for x in labels] # 得到中文标签名的列表
                #print('###labels_word_list.',k, m, "sentence:", sentence, ";labels:", labels, ";labels_word_list:", labels_word_list)
                count = count + 1

                # 传入一个标签列表和文本，正样本(从标准问法来）
                total_list, avg_length, total_pos=get_pos_example_data(labels_word_list, category,divorce_selectedtags2id_dict, divorce_id2tag_dict,
                                     category2tag_en_example_dict,sentence, count, total_pos, avg_length, total_list)

                # 传入一个标签列表和文本，负样本(经典：其他的中文标签)
                total_list, avg_length, total_neg=get_neg_example_data(divorce_selectedtags_dict, labels_word_list, category, sentence, count,divorce_selectedtags2id_dict,
                                     divorce_id2tag_dict, category2tag_en_example_dict,total_list, total_neg, avg_length)
                if k%5000==0:
                    print("Generate training data.length of total_list:",len(total_list),";total_pos:",total_pos,";total_neg:",total_neg)

    # 打印一些信息
    num_example = len(total_list)
    print("总行数:", count, ";空标签的行数:", count_empty_list, ";非空标签的行数：",count - count_empty_list)  # 总行数：22500 ；空标签的行数：13800； 非空行数9000。
    print("正样本总数量：", total_pos, ";负样本总数量：", total_neg, ";总样本数量：", total_pos + total_neg)
    print(category + "_tags_dict:",divorce_tags_dict)  # loan_tags_dict={0: 'LN1', 1: 'LN2', 2: 'LN3', 3: 'LN4', 4: 'LN5', 5: 'LN6', 6: 'LN7', 7: 'LN8', 8: 'LN9', 9: 'LN10', 10: 'LN11', 11: 'LN12', 12: 'LN13', 13: 'LN14', 14: 'LN15', 15: 'LN16', 16: 'LN17', 17: 'LN18', 18: 'LN19', 19: 'LN20'}
    print(category + "_selectedtags_dict:",divorce_selectedtags_dict)  # _selectedtags_dict={0: '债权人转让债权', 1: '借款金额x万元', 2: '有借贷证明', 3: '贷款人系金融机构', 4: '返还借款', 5: '公司|单位|其他组织借款', 6: '连带保证', 7: '催告还款', 8: '支付利息', 9: '订立保证合同', 10: '有书面还款承诺', 11: '担保合同无效|撤销|解除', 12: '拒绝履行偿还', 13: '免除保证人保证责任', 14: '保证人不承担保证责任', 15: '质押人系公司', 16: '贷款人未按照约定的日期|数额提供借款', 17: '多人借款', 18: '债务人转让债务', 19: '约定利率不明'}
    avg_length = float(avg_length) / float(num_example)
    print("平均长度：", avg_length, ";ignore_count:", ignore_count)  # 74. 120会够用

    # 写文件 train.tsv; dev.tsv
    write_data_to_file_system(total_list, data_path + 'train_data/')

def get_pos_example_data(labels_word_list,category,divorce_selectedtags2id_dict,divorce_id2tag_dict,category2tag_en_example_dict,sentence,count,total_pos,avg_length,total_list):
    """
    得到正样本的训练数据
    :return:
    """
    sub_result_list=[]
    for label_word in labels_word_list:  # e.g. label_word: '婚后有子女'; 其他可用的：1.一句话标准描述；2.好的句子1；3.好的句子2；4.好的句子3
        labelzh_and_additional_info_list_big = []
        # 中文标签对应的样本
        labelzh_and_additional_info_dict = factorzh_additionalinfo_dict[category_cn_dict[category] + "_" + label_word]
        labelzh_and_additional_info_list = [v for k, v in labelzh_and_additional_info_dict.items() if (not isinstance(v, float) and '*' not in v)] # 去掉有问题的元素
        labelzh_and_additional_info_sub_list= random.sample(labelzh_and_additional_info_list,2)
        labelzh_and_additional_info_list_big.extend(labelzh_and_additional_info_sub_list)
        #####################################################################################################
        # 添加标签对应的例子的随机选的5个例子
        # 从中文标签到index
        temp_index = divorce_selectedtags2id_dict[label_word]
        # 从index中英文标签
        temp_tag_en = divorce_id2tag_dict[temp_index]
        labels_word_list_examples = category2tag_en_example_dict[category].get(temp_tag_en, []) # 中文标签对应的例子的列表
        if len(labels_word_list_examples) > 0:
            temp_num_examples = len(labels_word_list_examples)
            temp_top5_example_list = random.sample(labels_word_list_examples, min(temp_num_examples, 3)) # 最多选3个
            labelzh_and_additional_info_list_big.extend(temp_top5_example_list)
        ######################################################################################################
        for label_candidate in labelzh_and_additional_info_list_big:
            if label_candidate == sentence: continue  # 如果要做句子对任务的双方是相同的，那么直接忽略
            if len(str(label_candidate)) > 122: label_candidate = label_candidate[0:60] + "。" + label_candidate[-62:]  # 处理超长的文本 # if len(str(sentence)) > 90: sentence = sentence[0:43] + "。" + sentence[-45:]  # 处理超长的文本
            strings = '1' + '\t' + category_cn_dict[category] + "的" + label_candidate + "\t" + sentence + "\n"  # label_word
            if isinstance(label_candidate, float): continue  # 跳过太短的，或有问题的例子
            sub_result_list.append(strings)
            total_pos = total_pos + 1
            avg_length = avg_length + len(strings)

    total_list.extend(sub_result_list)
    return total_list,avg_length,total_pos
    ##打印##############################################
    #if len(labels_word_list)>0:
    #    print(";labels_word_list:",labels_word_list," ;sentence:",sentence)
    #    for lll,ee in enumerate(sub_result_list):
    #        print("lll:",lll," ;ee:",ee)
    ###################################################

def get_neg_example_data(divorce_selectedtags_dict,labels_word_list,category,sentence,count,divorce_selectedtags2id_dict,divorce_id2tag_dict,category2tag_en_example_dict,total_list,total_neg,avg_length):
    """
    获得负样本
    :param divorce_selectedtags_dict:
    :param labels_word_list:
    :param category:
    :param sentence:
    :param count:
    :param divorce_selectedtags2id_dict:
    :param divorce_id2tag_dict:
    :param category2tag_en_example_dict:
    :param total_list:
    :param total_neg:
    :param avg_length:
    :return:
    """
    divorce_selectedtags_list = [v for k, v in divorce_selectedtags_dict.items()]
    #print("###divorce_selectedtags_list:", divorce_selectedtags_list)
    sub_neg_result_list=[]
    for label_word_neg in divorce_selectedtags_list:  # 标准负样本
        random_number = random.random()
        if label_word_neg not in labels_word_list: # 不在标签中的，但在标签集合中的标签，皆为负样本
            if random_number > 0.8:  #  TODO TODO TODO 通过改变这个值，你可以决定生成负训练样本的数量。将这个数改大一点，如果你希望产生更少的数据
                # a.负样本1：其他的中文标签
                strings = '0' + '\t' + category_cn_dict[category] + "的" + label_word_neg + "\t" + sentence + "\n"
                if isinstance(label_word_neg, float): continue  # 跳过太短的，或有问题的例子
                sub_neg_result_list.append(strings)
                total_neg = total_neg + 1
                avg_length = avg_length + len(strings)

                # b.负样本2：额外的标签的描述、好的句子1，2，3;外加标签对应的例子
                ############################################################################################
                temp_neg_index = divorce_selectedtags2id_dict[label_word_neg]
                # 从index中英文标签
                temp_neg_tag_en = divorce_id2tag_dict[temp_neg_index]
                labels_word_list_neg_examples = category2tag_en_example_dict[category].get(temp_neg_tag_en, [])
                neg_examples_list_big = []
                # 从标签对应的例子中采样出一部分
                if len(labels_word_list_neg_examples) > 0:
                    temp_num_neg_examples = len(labels_word_list_neg_examples)
                    temp_top5_neg_example_list = random.sample(labels_word_list_neg_examples,min(temp_num_neg_examples, 5))
                    neg_examples_list_big.extend(temp_top5_neg_example_list)
                ############################################################################################
                neg_sublist = factorzh_neg_sentenceorlabel_dict[category_cn_dict[category] + "_" + label_word_neg]
                neg_sublist = [xx for xx in neg_sublist if not isinstance(xx, float)]  # 去掉空值
                neg_examples_list_big.extend(neg_sublist)
                neg_sample_final_list = random.sample(neg_examples_list_big, 2) # 从描述和案例中随机选出2个
                # 添加到列表中
                for neg_sample in neg_sample_final_list:
                    if isinstance(neg_sample, float): continue
                    if len(str(neg_sample)) > 122: neg_sample = neg_sample[0:60] + "。" + neg_sample[-62:]
                    strings_neg = '0' + '\t' + category_cn_dict[category] + "的" + neg_sample + "\t" + sentence + "\n"
                    if neg_sample != label_word_neg:  # sample出来的负样本不应该是负样本本身
                        sub_neg_result_list.append(strings_neg)
                        total_neg = total_neg + 1
                        avg_length = avg_length + len(strings)

    total_list.extend(sub_neg_result_list)
    return total_list,avg_length,total_neg
    # 打印一些例子
    #for kkk, neg_elment in enumerate(sub_neg_result_list):
    #    print("kkk:",kkk," ;neg_elment:",neg_elment)


def write_file(data_list, target_file, file_type):
    """
    写单个文件
    :param data_list:
    :param target_file:
    :return:
    """
    random.shuffle(data_list)
    target_object = open(target_file, 'w')
    count_pos = 0
    count_neg = 0
    for string in data_list:
        #print("##string:",string)
        label_string = string.split("\t")[0]
        # 统计正负比例
        if file_type == 'train':
            if label_string == '0': count_neg = count_neg + 1
            if label_string == '1': count_pos = count_pos + 1
            target_object.write(string)
        if file_type == 'dev':  # 对于验证集，去掉一部分负样本
            if label_string == '1':
                target_object.write(string)
                count_pos = count_pos + 1
            else:
                if random.random() > 0.7:# TODO CHANGE AT08-27.0.6:
                    target_object.write(string)
                    count_neg = count_neg + 1
    print(file_type, "count_pos:", count_pos, ";count_neg:", count_neg, ";pert of pos:",(float(count_pos) / float(count_pos + count_neg)))
    target_object.close()


def write_data_to_file_system(total_list, target_path):
    """
    写多个文件
    :param total_list:
    :param target_path:
    :return:
    """
    random.shuffle(total_list)
    num_example = len(total_list)
    # 写训练集
    num_train = int(num_example * 0.95)
    train_list = total_list[0:num_train]
    target_train_file = target_path + 'train.tsv'
    write_file(train_list, target_train_file, 'train')
    # 写验证集
    dev_list = total_list[num_train:]
    target_dev_file = target_path + 'dev.tsv'
    write_file(dev_list, target_dev_file, 'dev')

def load_pos_examples_files(examples_source_path):
    """
    从文件中读取标签对应的代表性的正样本
    :param source_path: 文件所在位置
    :return:
    """
    category2tag_en_example_dict={}
    category_list = ['divorce', 'labor', 'loan']
    for category_en in category_list:
        source_file=examples_source_path+category_en+'_pos_examples.json'
        source_object=open(source_file,'r')
        line=source_object.readline()
        temp_dict=json.loads(line)
        print("type of temp_dict:",type(temp_dict))
        category2tag_en_example_dict[category_en]=temp_dict
    return category2tag_en_example_dict

examples_source_path= 'zuo/data_all/pos_examples/'
# category2tag_en_example_dict=load_pos_examples_files(examples_source_path)
# divorce_tagen_examples=category2tag_en_example_dict['labor']
# examples=divorce_tagen_examples['LB18']
# print("##examples:",examples)

data_path = 'zuo/data_all/'  # './data/'
generate_train_data(data_path,examples_source_path)
