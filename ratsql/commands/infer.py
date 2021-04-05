import argparse
import ast
import itertools
import json
import os
import sys

import _jsonnet
import asdl
import astor
import torch
import tqdm

from ratsql import beam_search
from ratsql import datasets
from ratsql import models
from ratsql import optimizers
from ratsql.models.spider import spider_beam_search
from ratsql.utils import registry
from ratsql.utils import saver as saver_mod


class Inferer:
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            torch.set_num_threads(1)

        # 0. Construct preprocessors
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])
        self.model_preproc.load()

    def load_model(self, logdir, step):
        '''Load a model (identified by the config used for construction) and return it'''
        # 1. Construct model
        model = registry.construct('model', self.config['model'], preproc=self.model_preproc, device=self.device)
        model.to(self.device)
        model.eval()
        model.visualize_flag = False

        # 2. Restore its parameters
        saver = saver_mod.Saver({"model": model})
        last_step = saver.restore(logdir, step=step, map_location=self.device, item_keys=["model"])
        if not last_step:
            raise Exception(f"Attempting to infer on untrained model in {logdir}, step={step}")
        return model

    def infer(self, model, output_path, args):
        output = open(output_path, 'w')

        with torch.no_grad():
            if args.mode == 'infer':
                orig_data = registry.construct('dataset', self.config['data'][args.section])
                model.load_orig_data(orig_data)
                preproc_data = self.model_preproc.dataset(args.section)
                if args.limit:
                    sliced_orig_data = itertools.islice(orig_data, args.limit)
                    sliced_preproc_data = itertools.islice(preproc_data, args.limit)
                else:
                    sliced_orig_data = orig_data
                    sliced_preproc_data = preproc_data
                assert len(orig_data) == len(preproc_data)
                self._inner_infer(model, args.beam_size, args.output_history, sliced_orig_data, sliced_preproc_data,
                                  output, args.use_heuristic)
            elif args.mode == 'debug':
                data = self.model_preproc.dataset(args.section)
                if args.limit:
                    sliced_data = itertools.islice(data, args.limit)
                else:
                    sliced_data = data
                self._debug(model, sliced_data, output)
            elif args.mode == 'visualize_attention':
                model.visualize_flag = True
                model.decoder.visualize_flag = True
                data = registry.construct('dataset', self.config['data'][args.section])
                if args.limit:
                    sliced_data = itertools.islice(data, args.limit)
                else:
                    sliced_data = data
                self._visualize_attention(model, args.beam_size, args.output_history, sliced_data, args.res1, args.res2, args.res3, output)
            

    def _inner_infer(self, model, beam_size, output_history, sliced_orig_data, sliced_preproc_data, output,
                     use_heuristic=True):
        for i, (orig_item, preproc_item) in enumerate(
                tqdm.tqdm(zip(sliced_orig_data, sliced_preproc_data),
                          total=len(sliced_orig_data))):
            decoded = self._infer_one(model, orig_item, preproc_item, beam_size, output_history, use_heuristic)
            output.write(
                json.dumps({
                    'index': i,
                    'beams': decoded,
                }) + '\n')
            output.flush()

    def _infer_one(self, model, data_item, preproc_item, beam_size, output_history=False, use_heuristic=True):
        if use_heuristic:
            # TODO: from_cond should be true from non-bert model
            beams = spider_beam_search.beam_search_with_heuristics(
                model, data_item, preproc_item, beam_size=beam_size, max_steps=1000, from_cond=False)
        elif use_heuristic==1: #oracle sketch
            beams = spider_beam_search.beam_search_with_oracle_sketch(model, data_item, preproc_item, beam_size=beam_size, max_steps=1000)
        else:
            beams = beam_search.beam_search(
                model, data_item, preproc_item, beam_size=beam_size, max_steps=1000)
        decoded = []
        for beam in beams:
            model_output, inferred_code = beam.inference_state.finalize(gold=data_item.code,oracle=[])
            _, oracle_select_inferred_code = beam.inference_state.finalize(gold=data_item.code,oracle=['select'])
            _, oracle_select_from_inferred_code = beam.inference_state.finalize(gold=data_item.code,oracle=['select', 'from'])

            decoded.append({
                'orig_question': data_item.orig["question"],
                'orig_code': data_item.code,
                'preproc_item': preproc_item[0],
                'model_output': model_output,
                'inferred_code': inferred_code,
                'oracle_select_inferred_code': oracle_select_inferred_code,
                'oracle_select_from_inferred_code': oracle_select_from_inferred_code,
                'score': beam.score,
                **({
                       'choice_history': beam.choice_history,
                       'score_history': beam.score_history,
                       'attention_history': beam.attention_history,
                       'column_history': beam.column_history
                   } if output_history else {})})
        return decoded

    def _debug(self, model, sliced_data, output):
        for i, item in enumerate(tqdm.tqdm(sliced_data)):
            (_, history), = model.compute_loss([item], debug=True)
            output.write(
                json.dumps({
                    'index': i,
                    'history': history,
                }) + '\n')
            output.flush()

    def _visualize_attention(self, model, beam_size, output_history, sliced_data, res1file, res2file, res3file, output):
        res1 = json.load(open(res1file, 'r'))
        res1 = res1['per_item']
        res2 = json.load(open(res2file, 'r'))
        res2 = res2['per_item']
        res3 = json.load(open(res3file, 'r'))
        res3 = res3['per_item']
        interest_cnt = 0
        cnt = 0
        for i, item in enumerate(tqdm.tqdm(sliced_data)):
        
            if res1[i]['hardness'] != 'extra':
                continue
            cnt += 1
            if (res1[i]['exact'] == 0) and (res2[i]['exact'] == 0) and (res3[i]['exact'] == 0):
                continue
            interest_cnt += 1
            '''
            print('sample index: ')
            print(i)
            beams = beam_search.beam_search(
                model, item, beam_size=beam_size, max_steps=1000, visualize_flag=True)
            entry = item.orig
            print('ground truth SQL:')
            print(entry['query_toks'])
            print('prediction:')
            print(res2[i])
            decoded = []
            for beam in beams:
                model_output, inferred_code = beam.inference_state.finalize()

                decoded.append({
                    'model_output': model_output,
                    'inferred_code': inferred_code,
                    'score': beam.score,
                    **({
                        'choice_history': beam.choice_history,
                        'score_history': beam.score_history,
                    } if output_history else {})})

            output.write(
                json.dumps({
                    'index': i,
                    'beams': decoded,
                }) + '\n')
            output.flush()
            '''
        print(interest_cnt * 1.0 / cnt)


def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')

    parser.add_argument('--step', type=int)
    parser.add_argument('--section', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--beam-size', required=True, type=int)
    parser.add_argument('--output-history', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--mode', default='infer', choices=['infer', 'debug', 'visualize_attention'])
    parser.add_argument('--use_heuristic', action='store_true')
    parser.add_argument('--res1', default='outputs/glove-sup-att-1h-0/outputs.json')
    parser.add_argument('--res2', default='outputs/glove-sup-att-1h-1/outputs.json')
    parser.add_argument('--res3', default='outputs/glove-sup-att-1h-2/outputs.json')
    args = parser.parse_args()
    return args


def main(args, model=None):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    if args.model_name is not None:
        args.logdir = args.model_name
    elif 'model_name' in config:
        args.logdir = os.path.join(args.logdir, config['model_name'])

    output_path = args.output.replace('__LOGDIR__', args.logdir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f'Output file {output_path} already exists')
        sys.exit(1)

    inferer = Inferer(config)
    if model is None:
        model = inferer.load_model(args.logdir, args.step)
    inferer.infer(model, output_path, args)
    return model

if __name__ == '__main__':
    args = add_parser()
    main(args)
