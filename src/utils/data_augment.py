import collections
import time

import pandas as pd
from math import ceil
import re
import random
import jieba


def anti_join(x, y, on):
    """
    :param x:要保留的部分
    :param y:要排除的部分
    :param on:如没有特殊需求,可以不要这个参数
    :return: 返回x中不包含y的部分
    """
    ans = pd.merge(left=x, right=y, how='left', indicator=True, on=on)
    ans = ans.loc[ans._merge == 'left_only', :].drop(columns='_merge')
    return ans


def read_data(train_path, dev_path):
    """
    :param train_path: 训练集或全量数据集路径
    :param dev_path: 验证集路径
    """
    if train_path.endswith('.txt'):
        train_data = pd.read_csv(train_path, sep='\t')
    elif train_path.endswith('.csv'):
        train_data = pd.read_csv(train_path)
    elif train_path.endswith('.xlsx'):
        train_data = pd.read_excel(train_path)
    else:
        raise RuntimeError('不支持的数据格式')
    if dev_path.endswith('.txt'):
        dev_data = pd.read_csv(dev_path, sep='\t')
    elif dev_path.endswith('.csv'):
        dev_data = pd.read_csv(dev_path)
    elif dev_path.endswith('.xlsx'):
        dev_data = pd.read_excel(dev_path)
    else:
        raise RuntimeError('不支持的数据格式')

    return train_data, dev_data


def get_need(train_data, dev_data, k):
    """
        :return: 返回训练集分布不足k倍的类别和对应数量
    """
    # 计算对应类目下的数量
    dev_vc = pd.DataFrame(
        {'category_id': dev_data.value_counts('category_id').index,
         'count': dev_data.value_counts('category_id').values})
    train_vc = pd.DataFrame({'category_id': train_data.value_counts('category_id').index,
                             'count': train_data.value_counts('category_id').values})
    train_dev_vc = dev_vc.merge(train_vc, 'left', on='category_id').fillna(0)
    train_dev_vc['need'] = train_dev_vc.apply(lambda x: x[1] * k - x[2], axis=1)
    # need 为不平衡类别和对应所需数量
    need = train_dev_vc[train_dev_vc['need'] > 0]
    dev_count_dic = dev_vc.set_index('category_id').to_dict()['count']

    return need.set_index('category_id'), dev_count_dic


def get_need_count(need, key):
    return ceil(need.loc[key]['need'])


def get_cate_name(dev, need):
    """
    add category_name as normalized_name to data
    """
    res = dev[dev['category_id'].isin(need.index)][['category_name', 'category_id']].drop_duplicates()
    res.columns = ['normalized_name', 'category_id']

    return res


def get_aug_words_dic(need_aug):
    """
        generate each category_id's words(Redundant and Specification words) list from normalized_name
    """
    p = re.compile('\([^\(]+\)')
    q = re.compile('\[[^\[]+\]')
    o = re.compile('【[^【]+】')
    comlist = [p, q, o]

    cut_dic = collections.defaultdict(list)
    extra_dic = collections.defaultdict(list)

    for i, d in need_aug.iterrows():
        cid = d['category_id']
        name = d['normalized_name']
        extra = []
        cut = name.split()
        for cpil in comlist:
            extra += re.findall(cpil, name)
        for c in cut:
            if len(c) < 10:
                cut_dic[cid].append(c)
        extra_dic[cid].extend(extra)

    return extra_dic, cut_dic


def data_aug(name, tp='d', rlis=None, glis=None):
    """
    :param name: 需要增强的sku_name
    :param tp: 增强的模式
    :param rlis:冗余词的list，带括号的词
    :param glis:规格词的list
    :return: 增强后的new name
    """
    p = re.compile('\([^\(]+\)')
    q = re.compile('\[[^\[]+\]')
    o = re.compile('【[^【]+】')
    comlist = [p, q, o]

    if tp == 'd':
        namelis = jieba.lcut(name)
        index = random.randint(0, len(namelis) - 1)
        while namelis[index] == ' ':
            index = random.randint(0, len(namelis) - 1)
        namelis.pop(index)
        return ''.join(namelis)
    elif tp == 'r' and rlis:
        cples = random.sample(comlist, k=3)
        for cple in cples:
            if re.search(cple, name):
                if random.randint(0, 1) == 1:
                    add_r_seq = random.choice(rlis)
                    pos = random.randint(0, 1)
                    if pos == 0:
                        return add_r_seq + name
                    else:
                        return name + add_r_seq
                return re.sub(cple, '', name)
        add_r_seq = random.choice(rlis)
        pos = random.randint(0, 1)
        if pos == 0:
            return add_r_seq + name
        else:
            return name + add_r_seq
    elif tp == 'i' and glis:
        add_r_seq = random.choice(glis)
        namelis = name.split()
        pos = random.randint(0, len(namelis) - 1)
        namelis.insert(pos, add_r_seq)
        return ' '.join(namelis)
    elif tp == 's':
        namelis = jieba.lcut(name)
        random.shuffle(namelis)
        return ''.join(namelis)
    elif tp == 'e':
        namelis = jieba.lcut(name)
        if len(namelis) < 2:
            return name
        i, j = random.sample(range(len(namelis)), k=2)
        namelis[i], namelis[j] = namelis[j], namelis[i]
        return ''.join(namelis)
    return name

def raw_sampel(train_data, d, k):
    """
    sample data with k
    """
    res = []
    for cate_id, items in train_data.groupby('category_id'):
        if cate_id in d:
            smp_cnt = ceil(min(d[cate_id] * k, items.shape[0]))
            if smp_cnt > 0:
                res.append(items.sample(smp_cnt))
    return res

def first_aug(need_aug, extra_dic, cut_dic, mode, l):
    """
    first augmentation
    return augmentation data in dic with key as category_id
    """
    modelis = mode.split(',')
    dic = collections.defaultdict(set)
    for i, d in need_aug.iterrows():
        name = d['normalized_name']
        cid = d['category_id']
        rlis = extra_dic[cid]
        glis = cut_dic[cid]
        if name in l:
            new_name1 = data_aug(name, 'i', rlis, glis)
            if new_name1 != name: dic[cid].add((new_name1, cid))
            new_name2 = data_aug(name, 'r', rlis, glis)
            if new_name2 != name: dic[cid].add((new_name2, cid))
        else:
            for mode in modelis:
                new_name = data_aug(name, mode, rlis, glis)
                if new_name != name: dic[cid].add((new_name, cid))
    return dic


def first_sample(dic, need):
    """
    sample from aug_dic, return klis saves need second_aug category_id
    """
    ret = []
    klis = []
    for key, value in dic.items():
        if get_need_count(need, key) > len(value):
            klis.append(key)
        else:
            ret.extend(random.sample(value, get_need_count(need, key)))
    return ret, klis


def second_aug(dic, klis, need, mode, extra_dic, cut_dic):
    """
    second augment
    add data until meet demand
    """
    modelis = mode.split(',')
    for key in klis:
        vlis = dic[key].copy()
        rlis = extra_dic[key]
        glis = cut_dic[key]
        need_k = get_need_count(need, key) - len(vlis)
        add = 0
        while add < need_k:
            for item in vlis:
                name = item[0]
                mode = random.choice(modelis)
                new_name = data_aug(name, mode, rlis, glis)
                if (new_name, key) not in dic[key]:
                    dic[key].add((new_name, key))
                    add += 1
                    if add == need_k: break
    return dic


def data_augmentation(train_path, dev_path, k=3.5, sampled=True, base_mode='r,i,e', extra_mode='r,s,i,e,d'):
    """
    是用数据增强补齐不平衡的分布，单独增强单个name可以使用data_aug方法
    :param train_path: 需要增强的训练集或全量数据集路径
    :param dev_path: 验证集路径
    :param k:冗余词的list，带括号的词
    :param sampled:规格词的list
    :param base_mode:规格词的list
    :param extra_mode:规格词的list
    :return: 增强后的new name
    """
    st = time.time()
    train_data, dev_data = read_data(train_path, dev_path)
    train_data = anti_join(train_data, dev_data, on='normalized_name')[['normalized_name', 'category_id_x']]
    train_data.columns = ['normalized_name', 'category_id']

    need, dev_count_dic = get_need(train_data, dev_data, k)
    cate_name_data = get_cate_name(dev_data, need)
    need_aug = train_data[train_data['category_id'].isin(need.index)][['normalized_name', 'category_id']]
    extra_dic, cut_dic = get_aug_words_dic(need_aug)
    need_aug = pd.concat([need_aug, cate_name_data])


    l = cate_name_data['normalized_name'].tolist()
    dic = first_aug(need_aug, extra_dic, cut_dic, base_mode, l)
    ret, klis = first_sample(dic, need)
    dic = second_aug(dic, klis, need, extra_mode, extra_dic, cut_dic)
    ret, klis = first_sample(dic, need)
    assert len(klis) == 0
    if not sampled:
        ret.extend(raw_sampel(train_data, dev_count_dic, k))

    train_agued = pd.DataFrame(ret, columns=['normalized_name', 'category_id']).drop_duplicates('normalized_name')
    train_agued = train_agued[~train_agued['normalized_name'].isin(train_data['normalized_name'])]
    print('time out:', time.time() - st)

    return train_agued
