# -*- coding: utf-8 -*-
import os
import config as conf
from datetime import datetime, timedelta
from collections import Counter
import codecs
import json
from tqdm import tqdm

def iterate_data_files(from_dtm, to_dtm):
    from_dtm, to_dtm = map(str, [from_dtm, to_dtm])
    read_root = os.path.join(conf.data_root, 'read')
    for fname in os.listdir(read_root):
        if len(fname) != len('2018100100_2018100103'):
            continue
        if from_dtm != 'None' and from_dtm > fname:
            continue
        if to_dtm != 'None' and fname > to_dtm:
            continue
        path = os.path.join(read_root, fname)
        yield path, fname


def ts2time(ts):
    """
    ts 를 datetime 으로 변경.
    """
    tmp = datetime.fromtimestamp(ts / 1000)
    date_time = tmp.strftime("%Y%m%d%H")
    return int(date_time)


def build_dataset(words, sentences, n_words, min_count, skip_window):
    """
    raw input 들을 필요한 다른 data 들로 바꾸어줌.
    """
    under_min_count = 0
    count = Counter(words).most_common(n_words - 1)
    dictionary = {}
    for word, word_count in count:
        if word_count >= min_count:
            dictionary[word] = len(dictionary)
        else:
            under_min_count += 1
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    eliminate_sen_idxes = []
    for i, sentence in tqdm(enumerate(sentences), total=len(sentences)):
        new_sentence = []
        doc_count = 0
        for doc in sentence:
            if dictionary.get(doc, 0) == 0:
                new_sentence.append('UNK')
            else:
                new_sentence.append(doc)     
                doc_count += 1
        
        if doc_count <= 1:
            eliminate_sen_idxes.append(i)
        
        sentences[i] = new_sentence
    
    for idx in reversed(eliminate_sen_idxes):
        del sentences[idx]

    print ("# of under min count {}: {}".format(min_count, under_min_count))
    return sentences, count, dictionary, reversed_dictionary


def get_click_dist(date_list, test_days_len, additional_days_len):
    """
    테스트 기간 중 어떤 날짜에 작성된 문서가 얼마만큼 사용되는지 대략의 분포를 유추
    
    date_list: 테스트 하려는 날짜들 ex) [20190222, 20190221, ..., 20190201]
    test_days_len: 테스트 날짜로부터 이후 날짜 수 ex) 20
    additional_days_len: 테스트 날짜로부터 이전 날짜 수 ex) 4
    """
    diff = [e for e in reversed(range(test_days_len+1))] + [-(e+1) for e in range(additional_days_len)]
    
    dist_map = {}
    for i in diff:
        dist_map[i] = 0.0

    print ("Generating doc clicks distributions...")

    for input_date in tqdm(date_list):    

        t_obj = datetime.strptime(input_date, "%Y%m%d")
        d_obj = t_obj + timedelta(days=1)
        n_obj = t_obj - timedelta(days=test_days_len)
        c_obj = n_obj - timedelta(days=additional_days_len)

        test_end_date = d_obj.strftime("%Y%m%d")        
        test_start_date = n_obj.strftime("%Y%m%d")
        candidate_date = c_obj.strftime("%Y%m%d")

        test_end_date = int(test_end_date) * 100
        test_start_date = int(test_start_date) * 100
        candidate_date = int(candidate_date) * 100

        tmp_new_docs = {}
        with codecs.open('./res/metadata.json','rU','utf-8') as f:
            for line in f:
                j_map = json.loads(line)

                j_map['time'] = ts2time(j_map['reg_ts'])
                if j_map['time'] < test_end_date and j_map['time'] > candidate_date:
                    tmp_new_docs[j_map['id']] = j_map

        doc_cnt = {}
        for path, _ in iterate_data_files(test_start_date, test_end_date):
            for line in open(path):
                l = line.strip().split()
                seen = l[1:]

                for doc in seen:
                    doc_cnt[doc] = doc_cnt.get(doc, 0) + 1                

        new_doc_hit_cnt = 0
        new_doc_hit_per_date = {}

        for k, v in doc_cnt.items():
            if k in tmp_new_docs:
                new_doc_hit_cnt += v
                time = tmp_new_docs[k]['time'] // 100
                new_doc_hit_per_date[time] = new_doc_hit_per_date.get(time, 0) + v

        for i, (k, v) in enumerate(sorted(new_doc_hit_per_date.items(), key=lambda k: -k[0])):
            dist_map[diff[i]] += 100*v/new_doc_hit_cnt
            
    return dist_map
