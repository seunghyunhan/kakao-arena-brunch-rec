# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import codecs
import json
from datetime import datetime, timedelta
from collections import OrderedDict

import tensorflow as tf
from word2vec import word2vec
from util import iterate_data_files, ts2time, build_dataset, get_click_dist

def main():
    print ("Start inference!")
    test_end_date = "20190314"

    """
    아래 파라미터는 실험에 의해 적합한 값을 고름
    test_days_len: 테스트 날짜로부터 이후 날짜 수 ex) 20
    additional_days_len: 테스트 날짜로부터 이전 날짜 수 ex) 4

    테스트 시작 날짜로 부터 <앞뒤> 기간에 쓰여진 문서를 candidate doc 으로 사용
    """
    test_days_len = 20
    additional_days_len = 4
    candidates_len = test_days_len + additional_days_len
    
    users_dict = {}
    with codecs.open('./res/users.json','rU','utf-8') as f:
        for line in f:
            j_map = json.loads(line)
            users_dict[j_map['id']] = j_map
    
    cand_docs = {}

    t_obj = datetime.strptime(test_end_date, "%Y%m%d")
    doc_deadline_date = (t_obj + timedelta(days=1)).strftime("%Y%m%d")
    candidate_date = (t_obj - timedelta(days=candidates_len)).strftime("%Y%m%d")

    doc_deadline_date = int(doc_deadline_date) * 100
    candidate_date = int(candidate_date) * 100

    with codecs.open('./res/metadata.json','rU','utf-8') as f:
        for line in f:
            j_map = json.loads(line)

            # ts 를 datetime 으로 변경
            j_map['time'] = ts2time(j_map['reg_ts'])

            # [test 기간 + test 이전 몇 일 기간] 동안의 doc 정보 저장
            if j_map['time'] < doc_deadline_date and j_map['time'] > candidate_date:
                cand_docs[j_map['id']] = j_map

    print ("# of candidate articles from {} to {} : {}".format(candidate_date//100, test_end_date, len(cand_docs)))
    
    # 20190221 부터 한 달간의 클릭 문서 분포를 파악
    d_obj = datetime.strptime("20190221", "%Y%m%d")
    date_list = []
    for i in range(30):
        date_list.append((d_obj - timedelta(days=i)).strftime("%Y%m%d"))

    dist_map = get_click_dist(date_list, test_days_len, additional_days_len)    
    
    s_obj = datetime.strptime("20190222", "%Y%m%d")

    dist_sorted_map = sorted(dist_map.items(), key=lambda k: -k[1])
    click_rank_per_date = [((s_obj + timedelta(days=e[0])).strftime("%Y%m%d"), rank) for rank, e in enumerate(dist_sorted_map)]
    click_rank_per_date = dict(click_rank_per_date)
    print (click_rank_per_date)
    
    
    # 후보 doc 들을 writer 로 묶어줌
    cand_doc_writer = {}
    for doc_id, doc_info in cand_docs.items():
        writer = doc_info['user_id']
        cand_doc_writer[writer] = cand_doc_writer.get(writer ,[]) + [doc_id]

    for k, v in cand_doc_writer.items():
        c_v = [(e, int(e.split("_")[1])) for e in v]
        cand_doc_writer[k] = [(e[0], int(cand_docs[e[0]]['time'])) for e in sorted(c_v, key=lambda v: v[1])]
        
         
    user_seen = {}
    user_latest_seen = {}
    user_last_seen = {}

    # w2v 에 쓰일 sequences
    seen_seq = []
    all_articles = []

    # test 의 (겹치는)기간 동안의 doc 사용량
    doc_cnt = {}

    from_dtm = 2018100100
    to_dtm = 2019030100
    for path, _ in tqdm(iterate_data_files(from_dtm, to_dtm), mininterval=1):
        for line in open(path):
            l = line.strip().split()
            user = l[0]
            seen = l[1:]

            if len(seen) > 1:
                seen_seq.append(seen)
            all_articles += seen

            user_seen[user] = user_seen.get(user, []) + seen

            date_range = path.split("./res/read/")[1]
            fr = int(date_range.split("_")[0])

            if fr >= 2019020100:
                user_latest_seen[user] = user_latest_seen.get(user, []) + seen

            if fr < 2019022200:
                user_last_seen[user] = user_last_seen.get(user, []) + [fr]                

            if fr >= 2019022200:
                for doc in seen:
                    doc_cnt[doc] = doc_cnt.get(doc, 0) + 1

    for u, dates in user_last_seen.items():
        user_last_seen[u] = max(dates)

    doc_cnt = OrderedDict(sorted(doc_cnt.items(), key=lambda k: -k[1])) 
    pop_list = [k for k, v in doc_cnt.items()][:300]
    del doc_cnt
    
    # word2vec 에 이용하는 데이터 만들기
    vocabulary_size = len(set(all_articles))
    _, _, article2idx_map, idx2article_map = \
        build_dataset(all_articles, seen_seq.copy(), vocabulary_size, min_count=5, skip_window=4)
    filtered_vocabulary_size = len(article2idx_map)
    
    del all_articles
    del seen_seq

    print ("# of vocabulary : all ({}) -> filtered ({})".format(vocabulary_size, filtered_vocabulary_size))

    batch_size = 128
    embedding_size = 128
    num_sampled = 10  

    config = {}
    config['batch_size'] = batch_size
    config['embedding_size'] = embedding_size  
    config['num_sampled'] = num_sampled
    config['filtered_vocaulary_size'] = filtered_vocabulary_size

    # word2vec ckpt 불러오기    
    sess = tf.Session() 
    net = word2vec(sess, config)
    net.build_model()
    net.initialize_variables()
    net.restore_from_checkpoint(ckpt_path="./ckpt/", step=500000, use_latest=True)

    user_most_seen = {}
    for u, seen in user_latest_seen.items():
        for doc in seen:
            if doc.startswith("@"):
                writer = doc.split("_")[0]
                seen_map = user_most_seen.get(u, {})
                seen_map[writer] = seen_map.get(writer, 0) + 1
                user_most_seen[u] = seen_map

        if u in user_most_seen:
            user_most_seen[u] = dict([e for e in sorted(user_most_seen[u].items(), key=lambda k: -k[1])])
            
            
    #tmp_dev = ['./tmp/dev.users.recommend', './tmp/dev.users']
    dev = ['./res/predict/dev.recommend.txt', './res/predict/dev.users']
    test = ['./res/predict/recommend.txt', './res/predict/test.users']

    path_list = [dev, test]
    for output_path, user_path in path_list:

        print ("Start recommendation!")
        print ("Read data from {}".format(user_path))
        print ("Write data to {}".format(output_path))    
    
        ## word2vec 에 의한 top_n 먼저 계산
        articles_len = 5
        positives = []
        with codecs.open(user_path, mode='r') as f:
            for idx, line in enumerate(f):
                u = line.rsplit()[0]
                
                pos = [article2idx_map[e] for e in reversed(user_seen.get(u, [])) if e in article2idx_map][:articles_len]
                remain_len = articles_len - len(pos)
                pos += [filtered_vocabulary_size for _ in range(remain_len)]
                positives.append(np.array(pos))

        _, _, top_n_bests = net.most_similar(positives,
                                    idx2article_map=idx2article_map,
                                    top_n=300)

        top_n_bests = np.array(top_n_bests)[:, :, 0]


        with codecs.open(output_path, mode='w') as w_f:
            with codecs.open(user_path, mode='r') as f:
                for line in tqdm(f):
                    u = line.rsplit()[0]

                    user_most_seen_map = user_most_seen.get(u, {})

                    def rerank_doc(doc_list):
                        """
                        rerank : 세 가지 방식으로 doc_list 로 들어온 문서들을 재정렬함
                        - 우선순위 1. 유저가 과거(user_latest_seen) 에 본 에디터의 글 횟 수 -> 많을수록 우선
                        - 우선순위 2. 해당 날짜에 만들어진 문서가 클릭될 확률 순위(click_rank_per_date) -> rank 작을 수록 우선
                        - 우선순위 3. 문서가 만들어진 최신 순

                        """
                        n_doc_list = []
                        for e in doc_list:
                            if e[1] > user_last_seen.get(u, 0) and str(e[1]//100) in click_rank_per_date:
                                writer = e[0].split("_")[0]
                                writer_hit_cnt = user_most_seen_map.get(writer, 0)
                                n_doc_list.append((e[0], e[1], click_rank_per_date[str(e[1]//100)], writer_hit_cnt))

                        reranked_doc_list = [e[0] for e in sorted(n_doc_list, key=lambda k: (-k[3], k[2], k[1]))]
                        return reranked_doc_list


                    ### 추천은 아래 1 + 2 + 3 순서로 함

                    # 1. 구독한 에디터들의 글 중들을 candidate 에서 뽑기
                    following_list = users_dict.get(u, {'following_list': []})['following_list']
                    following_doc = []
                    if following_list:
                        for e in following_list:
                             following_doc += cand_doc_writer.get(e, [])
                        following_doc = rerank_doc(following_doc)


                    # 2. 유저가 많이 본 에디터의 글들을 candidate 에서 뽑기
                    most_seen_new_doc = []
                    if user_most_seen_map:
                        for e, writer_cnt in user_most_seen_map.items():
                            # writer 가 3 번 이상 본 경우에만 활용
                            if writer_cnt >= 3:
                                most_seen_new_doc += cand_doc_writer.get(e, [])
                        most_seen_new_doc = rerank_doc(most_seen_new_doc)


                    # 3. word2vec 모델에서 가장 최근에 본 n 개 문서와 가장 유사한 문서들을 뽑기
                    positive_input = [article2idx_map[e] for e in reversed(user_seen.get(u, [])) if e in article2idx_map][:articles_len]
                    if positive_input:
                        sim_list = list(top_n_bests[idx])
                    else:
                        sim_list = pop_list


                    # 최종 추천 (1 + 2 + 3)
                    rec_docs = following_doc + most_seen_new_doc + sim_list
                    rec_docs = list(OrderedDict.fromkeys(rec_docs))

                    # 이미 유저가 과거에 본 문서는 제거
                    n_rec_docs = []
                    for d in rec_docs:
                        if d not in user_seen.get(u, []):
                            n_rec_docs.append(d)

                    if len(n_rec_docs) < 100:
                        n_rec_docs = pop_list

                    line = "{} {}\n".format(u, ' '.join(n_rec_docs[:100]))
                    w_f.write(line)
        print ("Finish!")

if __name__ == "__main__":
    main()
