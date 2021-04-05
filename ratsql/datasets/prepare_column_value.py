import json
import pandas as pd
import os
import sqlite3
import re
import collections
from tqdm import tqdm
import copy
from nltk.util import ngrams
import argparse

import nltk.corpus
import string
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
STOPWORDS |= set(a for a in string.punctuation)

def get_unique_column_values(conn, table, column, col_type):
    column = column.replace(' ', '_')
    try:
        if col_type == 'number':
            max_val = conn.execute('SELECT MAX(%s) from %s where TYPEOF(%s)==\'integer\' OR TYPEOF(%s)==\'real\' ;'%(column, table, column, column)).fetchall()
            min_val = conn.execute('SELECT MIN(%s) from %s ;'%(column, table)).fetchall()
            vals = [min_val[0][0], max_val[0][0]]
        elif col_type == 'time':
            max_val = conn.execute('SELECT MAX(%s) from %s ;'%(column, table)).fetchall()
            min_val = conn.execute('SELECT MIN(%s) from %s ;'%(column, table)).fetchall()
            vals = [min_val[0][0], max_val[0][0]]
        else:
            vals = conn.execute('SELECT DISTINCT(%s) from %s ;'%(column, table)).fetchall()
            vals = [val[0] for val in vals if val[0] is not None]
    except (sqlite3.Warning, sqlite3.Error, sqlite3.DatabaseError,
                sqlite3.IntegrityError, sqlite3.ProgrammingError,
                sqlite3.OperationalError, sqlite3.NotSupportedError) as e:
        print(e, column, table)
        if col_type == 'number':
            vals = [1, -1]
        else:
            vals = []
    return vals

def prepare_col_values(vals, tab='', col=''):
    nxt_mask = {}
    n_grams = set()
    cell_values = set()
    raw_cell_values = set()
    sub_words = collections.Counter()
    for val in tqdm(vals, desc='{} {}'.format(tab, col)):
        if val.startswith('http://') or val.startswith('https://'):
            val = 'http'
        tokens = tokenizer.tokenize(val)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        sub_words.update(token_ids)
        normalized_token = ''
        i = 0
        words = []
        while i < len(tokens):
            if tokens[i].startswith('##'):
                normalized_token += tokens[i][2:]
            else:
                normalized_token += tokens[i]
            if normalized_token and (i+1 == len(tokens) or not tokens[i+1].startswith('##')):
                words.append(normalized_token)
                normalized_token = ''
            i += 1
        n_grams |= set(words)
        cell_values.add(' '.join(words[:5]))
#         for n in range(2,4):
#             n_grams |= set([' '.join(z) for z in ngrams(words, n)])
        last_token = tokenizer.bos_token_id
        for token_id in token_ids:
            if last_token not in nxt_mask:
                nxt_mask[last_token] = set()
            nxt_mask[last_token].add(token_id)
            last_token = token_id
        if last_token not in nxt_mask:
            nxt_mask[last_token] = set()
        nxt_mask[last_token].add(tokenizer.eos_token_id)
    return nxt_mask, n_grams, cell_values, sub_words

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
tokenizer.add_special_tokens({'bos_token':'[unused0]', 'eos_token':'[unused1]', 'unk_token':'[unused2]'})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="data dir")
    parser.add_argument('--dbs', help="dir for databases")
    args = parser.parse_args()

    with open(os.path.join(args.data,'tables.json'),'r') as f:
        db_tables = json.load(f)
    db_values = {}
    global_nxt_mask = {
        tokenizer.bos_token_id: set([
        tokenizer.convert_tokens_to_ids('true'),
        tokenizer.convert_tokens_to_ids('false'),
        tokenizer.convert_tokens_to_ids('0'),
        tokenizer.convert_tokens_to_ids('1'),
        tokenizer.convert_tokens_to_ids('2'),
        tokenizer.convert_tokens_to_ids('3'),
        ]),
        tokenizer.convert_tokens_to_ids('true'): set([tokenizer.eos_token_id]),
        tokenizer.convert_tokens_to_ids('false'): set([tokenizer.eos_token_id]),
        tokenizer.convert_tokens_to_ids('0'): set([tokenizer.eos_token_id]),
        tokenizer.convert_tokens_to_ids('1'): set([tokenizer.eos_token_id]),
        tokenizer.convert_tokens_to_ids('2'): set([tokenizer.eos_token_id]),
        tokenizer.convert_tokens_to_ids('3'): set([tokenizer.eos_token_id]),
    }
    for db in db_tables:
        db_id = db['db_id']
        if db_id in ['academic', 'scholar', 'imdb', 'yelp']:
            continue
        print(db_id)
        db_col_values = [[[], [], [], [], []]]
        db_nxt_mask = copy.deepcopy(global_nxt_mask)
        db_col_value_maps = [[]]
        with sqlite3.connect(os.path.join(args.dbs,db_id,'%s.sqlite'%db_id)) as conn:
            conn.text_factory = lambda x:str(x, 'latin1')
            cursor = conn.cursor()
            for (tab_id, col), col_type in zip(db['column_names_original'][1:], db['column_types'][1:]):
                tab = db['table_names_original'][tab_id]
                vals = get_unique_column_values(conn, tab, col, col_type)
                if not vals:
                    db_col_values.append([[], [], [], [], []])
                    db_col_value_maps.append([])
                    continue
                if col_type in ['number', 'time']:
                    sub_words = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('{} {}'.format(vals[0], vals[1])))
                    db_col_values.append([vals, [], [], sub_words, [1]*len(sub_words)])
                    db_col_value_maps.append([])
                elif col_type == 'boolean':
                    sub_words = [tokenizer.convert_tokens_to_ids('true'), tokenizer.convert_tokens_to_ids('false')]
                    db_col_values.append([[0, 1], [], [], sub_words, [1]*len(sub_words)])
                    db_col_value_maps.append(sub_words)
                else:
                    if col_type == 'others':
                        if vals and (isinstance(vals[0], int) or isinstance(vals[0], float)):
                            sub_words = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('{} {}'.format(min(vals), max(vals))))
                            db_col_values.append([[min(vals), max(vals)], [], [], sub_words, [1]*len(sub_words)])
                            db_col_value_maps.append(sub_words)
                            continue
                    if vals and (isinstance(vals[0], int) or isinstance(vals[0], float)):
                        db_col_values.append([[], [], [], [], []])
                        db_col_value_maps.append([])
                        print('------error------', tab, col)
                        continue
                    nxt_mask, words, cell_values, sub_words_count = prepare_col_values(vals)
                    sub_words = list(sub_words_count.keys())
                    db_col_values.append([[], list(words), list(cell_values), sub_words, [sub_words_count[sub_word] for sub_word in sub_words]])
                    db_col_value_maps.append(sub_words)
                    for token_id, mask in nxt_mask.items():
                        if token_id not in db_nxt_mask:
                            db_nxt_mask[token_id] = mask
                        else:
                            db_nxt_mask[token_id] = db_nxt_mask[token_id]|mask
        assert len(db_col_values) == len(db['column_names_original'])
        db_nxt_mask[tokenizer.unk_token_id] = [k for k in db_nxt_mask if k!=tokenizer.bos_token_id]+[tokenizer.unk_token_id,tokenizer.eos_token_id]
        db_values[db_id] = [db_col_values, {k:list(v)+[tokenizer.unk_token_id] for k,v in db_nxt_mask.items()}, db_col_value_maps]
        with open(os.path.join(args.data, 'tables_column_values.json'),'w') as f:
            json.dump(db_values, f)

if __name__ == "__main__":
    main()
