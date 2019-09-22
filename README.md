# Multi-label Classification
Transform multi-label classification as sentence pair task &

Together with generating more training data, use more information and external knowledge


Multi-label Classification 多标签分类
--------------------------------------------------------------------
Multi-label Classification is a classification problem where multiple labels may be assigned to each instance.

Zero, one or multiple labels can be associated with an instance(or example). It is more general than multi-class

classification where one and only one label assigned to an example.

You can think the problem as a multiple binary classification that map input(X) to  

a value(y) either 0 or 1 for each label in label space. According to wiki:

    Multi-label classification is the problem of finding a model that maps inputs x to binary vectors y (assigning a value of 0 or 1 for each element (label) in y).

Transform Multi-label Classification as Sentence Pair Task 
--------------------------------------------------------------------
### 任务转化：将多标签分类转换为句子对任务

It is normal to get a baseline by trying use a single classification model to predict the labels for each input string. 

And it will be time-effective during inference since only one time computation is happened. However, the performance is quite poor especially when 

you only have few instances or examples for each label. one of the reasons for poor performance is that it try to map input to target labels directly

but fail to use more information and training example is not enough. 

By cast to sentence pair task, it is easy to use more information including label information, instance information, key words from each label.

Together with generating more training data, use more information and external knowledge
--------------------------------------------------------------------
###  产生更多训练数据、结合更多信息和额外的知识

Let's talk about how to use additional information and generate more data. 

Sentence pair task is like this: <sentence_1, sentence_2, label>. for sentence_1, it come from original input string. 

Then it be: <input_sentence, sentence_2, label(0,1)>. so where is sentence_2 come from? it is another input sentence. it can be come from: 
    
          1) chinese label string, or
          
          2) an sentence which can represent a label. you can randomly pick a sentence from a specific label to represent label, 
             
             you can also manual give an description to a label to represent this label;
    
          3) you can learn the keywords from the label and put keywords together to repsenent the label. 

In a words, there are many way to generate sentence_2 to do sentence pair task. and as a result, we generate more than 

1 million training instances for sentence pair task by using only 30k labeled instance. 

直接的多标签分类去预测，由于设法直接从输入的文本去影射标签，没有使用额外的信息，训练数据也有限，效果比较差。

通过将其转化为句子对任务，我们可以比较容易的利用额外的信息，并且产生极大数量的训练样本。这些额外的信息包括但不限于：

特定标签对应的训练样本中的部分输入文本、中文的标签信息、标签对应的top关键词的组合。这些额外的信息可用来代表这个标签。

在#Task Description和#Generate Training Data，我将详细展开并举例。


Procedure 流程
--------------------------------------------------------------------
1) Transform multi-label classification to sentence pair task with random instance from label --->

2) Additional information: add label information, which is chinese; or keys words from label, or export knowledge  --->

3) Additional domain knowledge: large scale domain pre-training


Performance 效果对比
--------------------------------------------------------------------

| No. 序列| Model | 描述 | Online 线上 |
| :-------| :------- | :------- | :---------: | 
|0 | Multi-label Classification(TextCNN)|多标签分类 | 61 | 
|1 | Multi-label Classification(Bert) |多标签分类| 64.9 | 
|2| Sentece-pair Task |句子对任务，标签对应的随机样本与输入文本| 68.9 |  
|3 |#2 + Instance Information |加上中文标签与输入文本的数据| 69.5 | 
|4 |#3 + bert_www_ext_law |bert_wwwm_ext基础上的领域预训练| 70.7 | 
|5 |#4 + RoBERTa-zh-domain | 结合RoBERTa和wwm的大规模领域预训练| 72.1 |  
|6 |#5 + RoBERTa-zh-Large-domain |24层的RoBERTa的预训练| 73.0| 
|7 |#6 + Ensemble| 使用3个模型概率求平均 & 候选项召回并预测加速 | 75.5|  
 
Additional information for pre-train RoBERTa chinese models, check: <a href="https://github.com/brightmart/roberta_zh">Roberta_zh</a>

Task Description 任务介绍 
--------------------------------------------------------------------
### About Task 任务是什么？
The purpose of this task is to extract important fact from description of legal case,
 
and map description of case to case elements according to system designed by experts in the field.

for each sentence in the paragraph from judicial document, the model need to identify the key element(s). 

multiple or zero elements may exist in a sentence.

本任务的主要目的是为了将案件描述中重要事实描述自动抽取出来，并根据领域专家设计的案情要素体系进行分类。

案情要素抽取的结果可以用于案情摘要、可解释性的类案推送以及相关知识推荐等司法领域的实际业务需求中。

具体地，给定司法文书中的相关段落，系统需针对文书中每个句子进行判断，识别其中的关键案情要素。

本任务共涉及三个领域，包括婚姻家庭、劳动争议、借款合同等领域。

### Examples of Data 数据介绍

本任务所使用的数据集主要来自于“中国裁判文书网”公开的法律文书，每条训练数据由一份法律文书的案情描述片段构成，其中每个句子都被标记了对应的类别标签

（需要特别注意的是，每个句子对应的类别标签个数不定），例如：
     
     {"labels": ["DV1", "DV4", "DV2"],"sentence": "In our opinion, according to the agreement between the two parties at the time of divorce, the plaintiff has paid 22210.00 yuan for the child's upbringing on the basis of ten-year upbringing. We can confirm that the plaintiff pays 200.00 yuan for the child's upbringing on a monthly basis."}
     
     {"labels": ["DV1", "DV4", "DV2"],"sentence": "本院认为，依据双方离婚时的协议约定，原告已按尚抚养十年一性支付孩子抚养费22210.00元，可确认原告每月支付孩子抚养费为200.00元。"}，
     {"labels": [],"sentence": "父母离婚后子女的抚养问题，应从有利于子女身心健康以及保障子女合法权益出发，结合父母双方的抚养能力和抚养条件等具体情况妥善解决。"},
     {"labels": ["DV1", "DV8", "DV4", "DV2"],"sentence": "二、关于抚养费的承担问题，本院根据子女的实际需要、父母双方的负担能力和当地的实际生活水平确定，由被告赵某甲每月给付孩子抚养费600.00元；"},
     {"labels": ["DV14", "DV9", "DV12"], "sentence": "原告诉称，原被告原系夫妻关系，双方于2015年3月18日经河南省焦作市山阳区人民法院一审判决离婚，离婚后原告才发现被告在婚姻关系存续期间，与他人同居怀孕并生下一男孩，给原告造成极大伤害。"},
     {"labels": [], "sentence": "特诉至贵院，请求判决被告赔偿原告精神损害抚慰金3万元。"}, 
     {"labels": [], "sentence": "被告辩称，1、原告在焦作市山阳区人民法院离婚诉讼中，提交答辩状期间就同意被告的离婚请求，被告并未出现与他人同居怀孕情况，原告不具备损害赔偿权利主体资格。"}, {"labels": [], "sentence": "2、被告没有与他人持续稳定的同居生活，不具备损害赔偿责任主体资格。"}, {"labels": [], "sentence": "3、原被告婚姻关系存续期间，原告经常对被告实施家庭暴力。"}, {"labels": [], "sentence": "原告存在较大过错，无权提起本案损失赔偿请求。"}, {"labels": ["DV1"], "sentence": "经审理查明：原被告于××××年××月××日登记结婚，××××年××月××日生育女孩都某乙。"}, {"labels": ["DV9"], "sentence": "被告于2014年9月23日向焦作市山阳区人民法院起诉与原告离婚，该院于2015年3月18日判决准予原被告离婚后，原告不服上诉至焦作市中级人民法院，该院于2015年6月15日作出终审判决，驳回原告上诉，维持原判。"}
     
     for each english label like DV1, it associated with a chinese label, such as 婚后有子女, which means 'Having children after marriage'.
     
     对于每个英文标签，都有一个对应的中文标签名称。 如 DV1、DV2、DV4对应的中文标签分别为：婚后有子女、限制行为能力子女抚养、支付抚养费。DV8对应：按月给付抚养费

Generate Training Data 生成训练数据
--------------------------------------------------------------------
### 标签下的代表性样本的产生
我们首先选取每个标签下一定数量样本，如随机的选取5个样本，来代表这个标签；并且由于我们知道这个标签对应的中文名称，所以，对于任何一个标签，我们都构造了6个句子来代表这个标签，

记为集合{representation_set}. 需要注意的是，为了使得样本更有代表性，在随机选取样本过程中，我们优先选择哪些只有一个标签的样本作为我们的代表性的样本。

如对于DV1,我们6个样本包括：
    
    婚后有子女（来自于中文标签）
    
    ××××年××月××日生育女儿赵某乙。（来自于DV1标签下的样本）
    
    综合全案证据及庭审调查，本院确认以下法律事实：××××年××月××日，原告林某和被告赵某甲办理结婚登记手续，婚后于××××年××月××日生育婚生女儿赵某乙。（来自于DV1标签下的样本）

    四、抚养权变更后，被告赵某甲有探望女儿赵某乙的权利，原告林某应为被告赵某甲探望孩子提供必要便利。（来自于DV1标签下的样本）

原始样本： {"labels": ["DV1", "DV4", "DV2"],"sentence": "本院认为，依据双方离婚时的协议约定，原告已按尚抚养十年一性支付孩子抚养费22210.00元，可确认原告每月支付孩子抚养费为200.00元。"}

### 正样本的构造

我们需要构造句子对任务：<sentence_1, sentence_2, label(0,1)>

句子对任务中的第一部分的输入为，原始的文本，即本院认为，依据双方离婚时的协议约定...)，第二部分的输入是数据产生过程的关键所在。

对于DV1这个标签，我们得到六个句子即representation_set[DV1]，并且对于原始输入与这里面的每个样本的组合，我们给label赋值为1。如：

    <"本院认为，依据双方离婚时的协议约定，原告已按尚抚养十年一性支付孩子抚养费22210.00元，可确认原告每月支付孩子抚养费为200.00元。", "婚后有子女", 1>
    
    <"本院认为，依据双方离婚时的协议约定，原告已按尚抚养十年一性支付孩子抚养费22210.00元，可确认原告每月支付孩子抚养费为200.00元。", “××××年××月××日生育女儿赵某乙。", 1>
    
    <"本院认为，依据双方离婚时的协议约定，原告已按尚抚养十年一性支付孩子抚养费22210.00元，可确认原告每月支付孩子抚养费为200.00元。", “综合全案证据及庭审调查，本院确认以下法律事实：××××年××月××日，原告林某和被告赵某甲办理结婚登记手续，婚后于××××年××月××日生育婚生女儿赵某乙。", 1>
    
    ....---

通过这个形式，我们将正样本扩大了6倍；我们可以通过标签下采样更多的例子，以及使用标签下统计得到的关键词的组合，来进一步扩大正样本的集合。

### 负样本的构造

对于一个标签集合，如婚姻家庭下有20个标签，那么对于一个句子，只要没有被打上标签的，就是负样本。我们也构造了一个标签下的可能的所有样本的集合外加中文标签，并通过随机样本的方式来得到需要的样本。

如对于我们的原始样本的输入文本："本院认为，依据双方离婚时的协议约定，原告已按尚抚养十年一性支付孩子抚养费22210.00元，可确认原告每月支付孩子抚养费为200.00元。"，

它被打上了三个标签["DV1", "DV4", "DV2"]，那么其他标签["DV3","DV5",...,"DV20"]，都可以用来构造负样本。对应DV3,我们选择4+1即5个样本。3即从DV3下随机的找出三个样本，1即中文标签做为样本。


### 总样本量和正负样本分布 Training data and Its distribution

由于正样本量扩大了至少6倍，负样本扩大了20倍左右，最终我们从原始的1万个样本中产生了100多万的数据。当然为了样本分布受控，我们也对负样本有部分下采样。

另外为了得到与任务目标接近的验证集，我们在验证集上对负样本精选了更大程度的下采样。

分布大致如下：

    train: 
       count_pos: 261001 ;count_neg: 1011322 ;pert of pos: 0.20513737470752316
    dev: 
       count_pos: 13863 ;count_neg: 16121 ;pert of pos: 0.4623465848452508


check this script: 

    python3 -u zuo/generate_training_data


Relationship with Few-Shot Learning 与Few-Shot Learning有什么关系
--------------------------------------------------------------------
According to Quora: with the term “few-shot learning”, the “few” usually lies between zero and five, 

meaning that training a model with zero examples is known as zero-shot learning, one example is one-shot learning, and so on. 

All of these variants are trying to solve the same problem with differing levels of training material.

We are not doing few shot learning. however we try to get maximum performance for tasks with not so many examples.

And also try to use information and knowledge,whether come from task or outside the task, as much as possible, to boost the performance.

Download Data 下载数据
--------------------------------------------------------------------
点击这里<a href="https://storage.googleapis.com/roberta_zh/roberta_model/data_all.zip">下载</a>数据，并解压缩到zuo目录下。这样你就有了一个新的包含所有需要的数据的目录./zuo/data_all

Training 训练模型 
--------------------------------------------------------------------
Run Command to Train the model：

    export BERT_BASE_DIR=./RoBERTa_zh_Large
    export TEXT_DIR=./zuo/data_all/train_data
    
    nohup python3 run_classifier.py   --task_name=sentence_pair   --do_train=true   --do_eval=true   --data_dir=$TEXT_DIR  \
     --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config_big.json   \
     --init_checkpoint=$BERT_BASE_DIR/roberta_zh_large_model.ckpt --max_seq_length=256 --train_batch_size=128  \
     --learning_rate=1e-5   --num_train_epochs=3 --output_dir=zuo/model_files/roberta-zh-large_law  &
      
    如果你从现有的模型基础上训练，指定一下BERT_BASE_DIR的路径，并确保bert_config_file和init_checkpoint两个参数的值能对应到相应的文件上。
    这里假设你下载了roberta的模型并放在本项目的这个RoBERTa_zh_Large目录下。


Inference and its acceleration 预测加速
--------------------------------------------------------------------
训练完成后，运行命令来进行预测 Run Command to Train Model：

    python3 -u main.py

    需要注意的是，你需要确保相应的目录有训练好的模型，见zuo/run_classifier_predict_online.py，特别注意这两个参数要能对应上：init_checkpoint和bert_config_file

### 预测阶段句子对任务的构建  Construct Sentence Pair During Inference

虽然训练阶段使用了很多信息和知识来训练，但是预测阶段我们只采用<原始输入的句子,候选标签对应的中文标签>来构造句子对任务。我们认为样本下的标签虽然能

代表标签，但中文标签具有最好的代表性，预测效果也好一些。

### 预测阶段加速 Accelerate Inference Time

由于采用了sentence pair任务即句子对形式，对于一个输入，有20个标签，每个标签都需要预测，那么总共需要预测20次，这会导致预测时间过长。

所以，我们采用的是快速召回+精细预测的形式。

快速召回，采用多标签分类的形式，只需一次预测就可以将可能的候选项找到，如概率大于0.05的标签都是候选项。实践中，多数时候候选的标签为只有0个或1个。

对于每个候选的标签，都会使用句子对任务（原始句子，标签对应的中文描述）的模型去预测这个标签的概率；当某个标签的概率超过0.5的时候，即认为是目标标签。

对于响应速度要求不是特别严格的时候，我们也可以通过训练层数比较少的句子对模型来作为快速召回模块。

Unfinished Work 未包含的工作
--------------------------------------------------------------------
利用标签间的关系，构造更多数据和更难的任务。

由于标签之间具有共现关系或排斥关系等，通过标签关系来生成更多数据，也是未来可以研究的一个方向。


Reference
--------------------------------------------------------------------
1、<a href="https://arxiv.org/pdf/1810.04805.pdf">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>

2、<a href="https://github.com/brightmart/roberta_zh">RoBERTa中文预训练模型：Roberta_zh</a>