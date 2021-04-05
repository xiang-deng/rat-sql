import re
import string

import nltk.corpus

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)


# schema linking, similar to IRNet
def compute_schema_linking(question, column, table):
    def partial_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str in STOPWORDS or x_str in PUNKS:
            return False
        if re.search(rf"\b{re.escape(x_str)}\b", y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str == y_str:
            return True
        else:
            return False

    q_col_match = dict()
    q_tab_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0:
            continue
        col_id2list[col_id] = col_item

    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item

    # 5-gram
    n = 5
    while n > 0:
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n]
            n_gram = " ".join(n_gram_list)
            if len(n_gram.strip()) == 0:
                continue
            # exact match case
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        q_col_match[f"{q_id},{col_id}"] = "CEM"
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        q_tab_match[f"{q_id},{tab_id}"] = "TEM"

            # partial match case
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{col_id}" not in q_col_match:
                            q_col_match[f"{q_id},{col_id}"] = "CPM"
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{tab_id}" not in q_tab_match:
                            q_tab_match[f"{q_id},{tab_id}"] = "TPM"
        n -= 1
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match}


def compute_cell_value_linking(tokens, schema):
    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_match(word, column, table, db_conn):
        cursor = db_conn.cursor()

        p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or " \
                f"{column} like '% {word} %' or {column} like '{word}'"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except Exception as e:
            return False

    num_date_match = {}
    cell_match = {}

    for q_id, word in enumerate(tokens):
        if len(word.strip()) == 0:
            continue
        if word in STOPWORDS or word in PUNKS:
            continue

        num_flag = isnumber(word)

        CELL_MATCH_FLAG = "CELLMATCH"

        for col_id, column in enumerate(schema.columns):
            if col_id == 0:
                assert column.orig_name == "*"
                continue
            if column.synonym_for is not None:
                orig_column_id = column.synonym_for.id
                orig_match_key = f"{q_id},{orig_column_id}"
                syn_match_key = f"{q_id},{col_id}"
                if orig_match_key in num_date_match:
                    num_date_match[syn_match_key] = column.type.upper()
                if orig_match_key in cell_match:
                    cell_match[syn_match_key] = CELL_MATCH_FLAG
                continue
            # word is number 
            if num_flag:
                if column.type in ["number", "time"]:  # TODO fine-grained date
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
            else:
                ret = db_word_match(word, column.orig_name, column.table.orig_name, schema.connection)
                if ret:
                    # print(word, ret)
                    cell_match[f"{q_id},{col_id}"] = CELL_MATCH_FLAG

    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
    return cv_link

def compute_cell_value_linking_v1(tokens, schema, bert_idx_map=None):
    """check for value linking with cached db column values"""

    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    num_date_match = {}
    cell_match = {}
    for q_id, word in enumerate(tokens):
        if len(word.strip()) == 0:
            continue
        if word in STOPWORDS or word in PUNKS:
            continue 

        num_flag = isnumber(word)
        
        CELL_MATCH_FLAG = "CELLMATCH"

        for col_id, column in enumerate(schema.columns):
            if col_id == 0: 
                assert column.orig_name == "*"
                continue
            
            if column.synonym_for is not None:
                orig_column_id = column.synonym_for.id
                orig_match_key = f"{q_id},{orig_column_id}"
                syn_match_key = f"{q_id},{col_id}"
                if orig_match_key in num_date_match:
                    num_date_match[syn_match_key] = column.type.upper()
                if orig_match_key in cell_match:
                    cell_match[syn_match_key] = CELL_MATCH_FLAG
                continue
            
            if column.type == "time" and num_flag: # TODO unify date format and enable comparision
                num_date_match[f"{q_id},{col_id}"] = column.type.upper()
            elif column.type == "number" and num_flag:
                num = float(word)
                min_col_val, max_col_val = column.value_range
                try:
                    if min_col_val is not None and max_col_val is not None and num >= min_col_val and num <= max_col_val:
                        num_date_match[f"{q_id},{col_id}"] = column.type.upper()
                except TypeError:
                    pass
            else: 
                if word in column.value_vocab: 
                    cell_match[f"{q_id},{col_id}"] = CELL_MATCH_FLAG 
    cv_link = {"num_date_match": num_date_match, "cell_match" : cell_match}
    if bert_idx_map is not None:
        new_cv_link = {}
        for m_type in cv_link:
            _match = {}
            for ij_str in cv_link[m_type]:
                q_id_str, col_tab_id_str = ij_str.split(",")
                q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                real_q_id = bert_idx_map[q_id]
                _match[f"{real_q_id},{col_tab_id}"] = cv_link[m_type][ij_str]
            new_cv_link[m_type] = _match
        cv_link = new_cv_link
    return cv_link

def compute_cell_value_linking_v2(tokens, schema, bert_idx_map=None):
    """check for value linking with cached db column values. Match ngrams, include fine link mark like match start"""

    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    num_date_match = {}
    cell_match = {}
    
    q_id = 0
    while q_id < len(tokens):
        tmp_match = [{},{}]
        n = 5
        while n > 0:
            if q_id + n <= len(tokens):
                word = ' '.join(tokens[q_id:q_id+n])
                if len(word.strip()) == 0:
                    n -= 1
                    continue

                num_flag = isnumber(word)
                
                CELL_MATCH_FLAG = "CELLMATCH" # exact match to cell
                CELL_MATCH_START_FLAG = "CELLMATCHSTART" # exact match to cell, mark start of match
                CELL_P_MATCH_FLAG = "CELLTOKENMATCH" # match token vocabulary of column

                for col_id, column in enumerate(schema.columns):
                    if col_id == 0: 
                        assert column.orig_name == "*"
                        continue
                    
                    if column.synonym_for is not None:
                        orig_column_id = column.synonym_for.id
                        orig_match_key = f"{q_id},{orig_column_id}"
                        syn_match_key = f"{q_id},{col_id}"
                        if orig_match_key in num_date_match:
                            num_date_match[syn_match_key] = num_date_match[orig_match_key]
                        if orig_match_key in cell_match:
                            cell_match[syn_match_key] = cell_match[orig_match_key]
                        continue
                    
                    if column.type == "time" and num_flag: # TODO unify date format and enable comparision
                        num_date_match[f"{q_id},{col_id}"] = column.type.upper()
                    elif column.type == "number" and num_flag:
                        num = float(word)
                        min_col_val, max_col_val = column.value_range
                        try:
                            if min_col_val is not None and max_col_val is not None and num >= min_col_val and num <= max_col_val:
                                num_date_match[f"{q_id},{col_id}"] = column.type.upper()
                        except TypeError:
                            pass
                    else:
                        if f"{q_id},{col_id}" in cell_match:
                            continue
                        if n>1:
                            if word in column.cell_values: 
                                cell_match[f"{q_id},{col_id}"] = CELL_MATCH_START_FLAG
                                if n>1:
                                    for m_q_id in range(q_id+1,q_id+n):
                                        cell_match[f"{m_q_id},{col_id}"] = CELL_MATCH_FLAG
                        else:
                            if word in STOPWORDS or word in PUNKS:
                                continue
                            if word in column.cell_values: 
                                tmp_match[0][f"{q_id},{col_id}"] = CELL_MATCH_START_FLAG
                            elif word in column.tokens: 
                                tmp_match[1][f"{q_id},{col_id}"] = CELL_P_MATCH_FLAG
            n -= 1
        if len(tmp_match[0])!=0:
            for q_col in tmp_match[0]:
                if q_col not in cell_match:
                    cell_match[q_col] = tmp_match[0][q_col]
        elif len(tmp_match[0])==0 and len(tmp_match[1])<3:
            for q_col in tmp_match[1]:
                if q_col not in cell_match:
                    cell_match[q_col] = tmp_match[1][q_col]
        q_id += 1
    cv_link = {"num_date_match": num_date_match, "cell_match" : cell_match}
    if bert_idx_map is not None:
        new_cv_link = {}
        for m_type in cv_link:
            _match = {}
            for ij_str in cv_link[m_type]:
                q_id_str, col_tab_id_str = ij_str.split(",")
                q_id, col_tab_id = int(q_id_str), int(col_tab_id_str)
                real_q_id = bert_idx_map[q_id]
                _match[f"{real_q_id},{col_tab_id}"] = cv_link[m_type][ij_str]
            new_cv_link[m_type] = _match
        cv_link = new_cv_link
    return cv_link