import json
import re
import sqlite3
from copy import copy
from pathlib import Path
from typing import List, Dict

import attr
import torch
import networkx as nx
from tqdm import tqdm

from ratsql.utils import registry
from third_party.spider import evaluation


@attr.s
class SpiderItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    foreign_key_for = attr.ib(default=None)
    synonym_for = attr.ib(default=None)
    tokens = attr.ib(default=[])
    cell_values = attr.ib(default=[])
    value_vocab_ids = attr.ib(default=[])
    value_vocab_weights = attr.ib(default=[])
    value_range = attr.ib(default=[1,-1])


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()
    hidden = attr.ib()
    connection = attr.ib(default=None)
    nxt_masks = attr.ib(default=None) # NOT USED IN FINAL VERSION. the transition matrix of values in DB, for example, if the column only contain "New York", then only "York" can be decoded after "New"
    col_value_maps = attr.ib(default=None) # contain value in each column


def postprocess_original_name(s: str):
    return re.sub(r'([A-Z]+)', r' \1', s).replace('_', ' ').lower().strip()


def load_tables(paths, with_value = False):
    schemas = {}
    eval_foreign_key_maps = {}

    for path in paths:
        schema_dicts = json.load(open(path))
        if with_value:
            try:
                cached_value_path = path.replace('.json', '_column_values.json')
                with open(cached_value_path, 'r') as f:
                    cached_values = json.load(f)
                print('loaded cached column value from %s'%cached_value_path)
            except FileNotFoundError:
                print('cached value file not found')
                cached_values = {}

        for schema_dict in schema_dicts:
            if with_value:
                cached_value = cached_values.get(schema_dict['db_id'], [[], {}, []])
            tables = tuple(
                Table(
                    id=i,
                    name=name.split(),
                    unsplit_name=name,
                    orig_name=orig_name,
                )
                for i, (name, orig_name) in enumerate(zip(
                    schema_dict['table_names'], schema_dict['table_names_original']))
            )
            if with_value and len(cached_value[0])!=0:
                columns = tuple(
                    Column(
                        id=i,
                        table=tables[table_id] if table_id >= 0 else None,
                        name=col_name.split(),
                        unsplit_name=col_name,
                        orig_name=orig_col_name,
                        type=col_type,
                        value_range=value_range, 
                        tokens=set(tokens),
                        cell_values=set(cell_values),
                        value_vocab_ids=value_vocab_ids,
                        value_vocab_weights=value_vocab_weights
                    )
                    for i, ((table_id, col_name), (_, orig_col_name), col_type, (value_range, tokens, cell_values, value_vocab_ids, value_vocab_weights)) in enumerate(zip(
                        schema_dict['column_names'],
                        schema_dict['column_names_original'],
                        schema_dict['column_types'],
                        cached_value[0]
                        ))
                )
            else:
                columns = tuple(
                    Column(
                        id=i,
                        table=tables[table_id] if table_id >= 0 else None,
                        name=col_name.split(),
                        unsplit_name=col_name,
                        orig_name=orig_col_name,
                        type=col_type,
                    )
                    for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                        schema_dict['column_names'],
                        schema_dict['column_names_original'],
                        schema_dict['column_types']))
                )

            # Link columns to tables
            for column in columns:
                if column.table:
                    column.table.columns.append(column)

            for column_id in schema_dict['primary_keys']:
                # Register primary keys
                column = columns[column_id]
                column.table.primary_keys.append(column)

            foreign_key_graph = nx.DiGraph()
            for source_column_id, dest_column_id in schema_dict['foreign_keys']:
                # Register foreign keys
                source_column = columns[source_column_id]
                dest_column = columns[dest_column_id]
                source_column.foreign_key_for = dest_column
                foreign_key_graph.add_edge(
                    source_column.table.id,
                    dest_column.table.id,
                    columns=(source_column_id, dest_column_id))
                foreign_key_graph.add_edge(
                    dest_column.table.id,
                    source_column.table.id,
                    columns=(dest_column_id, source_column_id))

            # HACK: Introduce column synonyms as "phantom" columns
            if 'column_synonyms_original' in schema_dict:
                synonym_columns = []
                col_id = len(columns)
                for orig_name, synonyms in schema_dict['column_synonyms_original'].items():
                    orig_column = next((c for c in columns if c.orig_name == orig_name), None)
                    if not orig_column: continue
                    for syn_name in synonyms:
                        syn_column = copy(orig_column)
                        syn_column.synonym_for = orig_column
                        syn_column.orig_name = syn_name
                        syn_column.unsplit_name = postprocess_original_name(syn_name)
                        syn_column.name = syn_column.unsplit_name.split()
                        syn_column.id = col_id
                        col_id += 1
                        syn_column.table.columns.append(syn_column)
                        if orig_column.foreign_key_for is not None:
                            foreign_key_graph.add_edge(orig_column.table.id, orig_column.foreign_key_for.table.id,
                                                       columns=(syn_column.id, orig_column.foreign_key_for.id))
                            foreign_key_graph.add_edge(orig_column.foreign_key_for.table.id, orig_column.table.id,
                                                       columns=(orig_column.foreign_key_for.id, syn_column.id))
                        synonym_columns.append(syn_column)
                columns = tuple(list(columns) + synonym_columns)

            db_id = schema_dict['db_id']
            hidden = schema_dict.get('hidden', False)
            assert db_id not in schemas
            if with_value:
                schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict, hidden, nxt_masks={int(k):v for k,v in cached_value[1].items()}, col_value_maps=cached_value[2])
            else:
                schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict, hidden)
            eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps


@registry.register('dataset', 'spider')
class SpiderDataset(torch.utils.data.Dataset):
    def __init__(self, paths, tables_paths, db_path, demo_path=None, limit=None, with_value=False):
        self.paths = paths
        self.db_path = db_path
        self.examples = []

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths, with_value=with_value)

        for path in paths:
            raw_data = json.load(open(path))
            for entry in raw_data:
                item = SpiderItem(
                    text=entry['question_toks'],
                    code=entry['sql'],
                    schema=self.schemas[entry['db_id']],
                    orig=entry,
                    orig_schema=self.schemas[entry['db_id']].orig)
                self.examples.append(item)
        
        if demo_path:
            self.demos: Dict[str, List] = json.load(open(demo_path))
            
        # Backup in-memory copies of all the DBs and create the live connections
        for db_id, schema in tqdm(self.schemas.items()):
            sqlite_path = Path(db_path) / db_id / f"{db_id}.sqlite"
            sqlite_path = str(sqlite_path)
            schema.connection = sqlite3.connect(sqlite_path)
            # source: sqlite3.Connection
            # with sqlite3.connect(sqlite_path) as source:
            #     dest = sqlite3.connect(':memory:')
            #     dest.row_factory = sqlite3.Row
            #     source.backup(dest)
            # schema.connection = dest
            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __del__(self):
        for _, schema in self.schemas.items():
            if schema.connection:
                schema.connection.close()
    
    class Metrics:
        def __init__(self, dataset):
            self.dataset = dataset
            self.foreign_key_maps = {
                db_id: evaluation.build_foreign_key_map(schema.orig)
                for db_id, schema in self.dataset.schemas.items()
            }
            self.evaluator = evaluation.Evaluator(
                self.dataset.db_path,
                self.foreign_key_maps,
                'all')
            self.results = []

        def add(self, item, inferred_code, orig_question=None):
            ret_dict = self.evaluator.evaluate_one(
                item.schema.db_id, item.orig['query'], inferred_code)
            if orig_question:
                ret_dict["orig_question"] = orig_question
            self.results.append(ret_dict)

        def add_beams(self, item, inferred_codes, orig_question=None):
            beam_dict = {}
            if orig_question:
                beam_dict["orig_question"] = orig_question
            for i, code in enumerate(inferred_codes):
                ret_dict = self.evaluator.evaluate_one(
                    item.schema.db_id, item.orig['query'], code)
                beam_dict[i] = ret_dict
                if ret_dict["exact"] is True:
                    break
            self.results.append(beam_dict)

        def finalize(self):
            self.evaluator.finalize()
            return {
                'per_item': self.results,
                'total_scores': self.evaluator.scores
            }


@registry.register('dataset', 'spider_idiom_ast')
class SpiderIdiomAstDataset(torch.utils.data.Dataset):

    def __init__(self, paths, tables_paths, db_path, limit=None):
        self.paths = paths
        self.db_path = db_path
        self.examples = []

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths)

        for path in paths:
            for line in open(path):
                entry = json.loads(line)
                item = SpiderItem(
                    text=entry['orig']['question_toks'],
                    code=entry['rewritten_ast'],
                    schema=self.schemas[entry['orig']['db_id']],
                    orig=entry['orig'],
                    orig_schema=self.schemas[entry['orig']['db_id']].orig)
                self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    Metrics = SpiderDataset.Metrics
