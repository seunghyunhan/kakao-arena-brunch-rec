# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
from time import time
from collections import deque
import argparse

from util import iterate_data_files, build_dataset
from word2vec import word2vec

data_idx = 0

def generate_batch(batch_size, all_targets, all_labels):
    global data_idx

    if data_idx + batch_size > len(all_targets):
        data_idx = 0

    batch = all_targets[data_idx:data_idx + batch_size]
    labels = all_labels[data_idx:data_idx + batch_size]
    labels = np.reshape(labels, [batch_size, 1])

    data_idx += batch_size

    return batch, labels

def train(args):
    from_dtm = 2018100100
    to_dtm = 2019030100

    all_articles = []
    seen_seq = []

    for path, _ in tqdm(iterate_data_files(from_dtm, to_dtm), mininterval=1):
        for line in open(path):
            l = line.strip().split()
            user = l[0]
            seen = l[1:]

            if len(seen) > 1:
                seen_seq.append(seen)
            all_articles += seen

    vocabulary_size = len(set(all_articles))

    new_seen, count, article2idx_map, idx2article_map = \
        build_dataset(all_articles, seen_seq.copy(), vocabulary_size, min_count=args.min_count, skip_window=args.skip_window)

    filtered_vocabulary_size = len(article2idx_map)

    print('Most common words', count[:5])
    print ("# of sentences : all ({}) -> filtered ({})".format(len(seen_seq), len(new_seen)))
    print ("# of vocabulary : all ({}) -> filtered ({})".format(vocabulary_size, filtered_vocabulary_size))

    # Reduce momory
    del all_articles
    del seen_seq


    span = 2 * args.skip_window + 1  # [ skip_window target skip_window ]
    buffer = deque(maxlen=span)  # pylint: disable=redefined-builtin
    skip_dummy = ['UNK'] * args.skip_window

    all_targets = []
    all_labels = []

    for sen_idx, sentence in tqdm(enumerate(new_seen), total=len(new_seen)):
        sentence = skip_dummy + sentence + skip_dummy    
        buffer.extend(sentence[0:span-1])

        for doc in sentence[span-1:]:
            buffer.append(doc)
            if buffer[args.skip_window] != 'UNK':
                context_words = [w for w in range(span) if w != args.skip_window and buffer[w] != 'UNK']
                _num_sample = len(context_words) if len(context_words) < args.num_skips else args.num_skips
                words_to_use = random.sample(context_words, _num_sample)

                for j, context_word in enumerate(words_to_use):
                    all_targets.append(article2idx_map[buffer[args.skip_window]])
                    all_labels.append(article2idx_map[buffer[context_word]])  


    t1 = time()
    print ("Shuffling indexes...")
    idxes = [e for e in range(len(all_targets))]
    random.shuffle(idxes)
    all_targets = np.array(all_targets)[idxes]
    all_labels = np.array(all_labels)[idxes]
    del idxes
    t2 = time()
    print ("Shuffling finished [{:.1f} s]".format(t2-t1))

    config = {}
    config['batch_size'] = args.batch_size
    config['embedding_size'] = args.embedding_size  
    config['skip_window'] = args.skip_window 
    config['num_skips'] = args.num_skips
    config['num_sampled'] = args.num_sampled
    config['filtered_vocaulary_size'] = filtered_vocabulary_size

    sess = tf.Session()
    net = word2vec(sess, config)
    net.build_model()
    net.initialize_variables()

    
    decay_alpha = (args.alpha - args.min_alpha) / args.num_steps
    alpha = args.alpha
    
    check_step = 10000
    save_step = 100000
    average_loss = 0
    t1 = time()
    for step in range(args.num_steps):
        batch_inputs, batch_labels = generate_batch(args.batch_size, all_targets, all_labels)

        loss_val = net.train(batch_inputs, batch_labels, alpha=alpha)
        alpha -= decay_alpha
        average_loss += loss_val

        if step % check_step == 0 and step > 0:
            average_loss /= check_step

            t2 = time()
            print("Average loss at step {}: {:.5} [{:.1f} s]".format(step, average_loss, t2-t1))
            t1 = t2
            average_loss = 0

        if (step % save_step == 0 and step > 0) or step+1 == args.num_steps:
            print ("Store checkpoints at step {}...".format(step))
            net.store_checkpoint(step=step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--skip_window", type=int, default=4)
    parser.add_argument("--num_skips", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=500001)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--min_alpha", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_sampled", type=int, default=10)
    args = parser.parse_args()

    print (args)
    train(args)
            
if __name__=="__main__":
    main()
