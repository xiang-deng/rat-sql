#!/usr/bin/env python

import argparse
import json

import _jsonnet
import attr
from seq2struct.commands import preprocess, train, infer, eval

import pdb

@attr.s
class PreprocessConfig:
    config = attr.ib()
    config_args = attr.ib()


@attr.s
class TrainConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()


@attr.s
class InferConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    beam_size = attr.ib()
    output = attr.ib()
    step = attr.ib()
    model_name = attr.ib()
    use_heuristic = attr.ib(default=False)
    mode = attr.ib(default="infer")
    limit = attr.ib(default=None)
    output_history = attr.ib(default=False)


@attr.s
class EvalConfig:
    config = attr.ib()
    config_args = attr.ib()
    logdir = attr.ib()
    section = attr.ib()
    inferred = attr.ib()
    output = attr.ib()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help="preprocess/train/eval")
    parser.add_argument('exp_config_file', help="jsonnet file for experiments")
    parser.add_argument('--model_config_args', help="optional overrides for model config args")
    parser.add_argument('--logdir', help="optional override for logdir")
    parser.add_argument('--model_name', help="optional override for model_name")
    parser.add_argument('--skip_infer', action='store_true', default=False)
    parser.add_argument('--skip_eval', action='store_true', default=False)
    parser.add_argument('--eval_name', help="optional override for eval name")
    args = parser.parse_args()

    exp_config = json.loads(_jsonnet.evaluate_file(args.exp_config_file))
    model_config_file = exp_config["model_config"]
    if "model_config_args" in exp_config:
        model_config_args = exp_config["model_config_args"]
        if args.model_config_args is not None:
            model_config_args_json = _jsonnet.evaluate_snippet("", args.model_config_args)
            model_config_args.update(json.loads(model_config_args_json))
        model_config_args = json.dumps(model_config_args)
    elif args.model_config_args is not None:
        model_config_args = _jsonnet.evaluate_snippet("", args.model_config_args)
    else:
        model_config_args = None

    logdir = args.logdir or exp_config["logdir"]
    if args.eval_name:
        exp_config["eval_name"] = args.eval_name

    if args.mode == "preprocess":
        preprocess_config = PreprocessConfig(model_config_file, model_config_args)
        preprocess.main(preprocess_config)
    elif args.mode == "train":
        train_config = TrainConfig(model_config_file,
                                   model_config_args, logdir)
        train.main(train_config)
    elif args.mode == "eval":
        print(exp_config["eval_name"])
        for step in exp_config["eval_steps"]:
            print(step)
            model = None
            for section in exp_config["eval_sections"]:
                infer_output_path = "{}/{}-step{}/{}.infer".format(
                    exp_config["eval_output"],
                    exp_config["eval_name"],
                    step,
                    section)
                infer_config = InferConfig(
                    model_config_file,
                    model_config_args,
                    logdir,
                    section,
                    exp_config["eval_beam_size"],
                    infer_output_path,
                    step,
                    args.model_name,
                    use_heuristic=exp_config["eval_use_heuristic"],
                    output_history=True
                )
                if not args.skip_infer:
                    model = infer.main(infer_config, model)
                if not args.skip_eval:
                    eval_output_path = "{}/{}-step{}/{}.eval".format(
                        exp_config["eval_output"],
                        exp_config["eval_name"],
                        step,
                        section)
                    eval_config = EvalConfig(
                        model_config_file,
                        model_config_args,
                        logdir,
                        section,
                        infer_output_path,
                        eval_output_path
                    )
                    eval.main(eval_config)
                    print(section)
                    for infer_type in ['inferred_code', 'oracle_select_inferred_code']:
                        print(infer_type)
                        res_json = json.load(open(eval_output_path.replace('.eval','_{}.eval'.format(infer_type))))
                        # print('%.4f %.4f %.4f %.4f'%(res_json['total_scores']['all']['exact'], res_json['total_scores']['all']['exact (with val)'], res_json['total_scores']['all']['exec'], res_json['total_scores']['all']['exec (non empty)']))
                        print('%.4f %.4f %.4f,'%(res_json['total_scores']['all']['exact'], res_json['total_scores']['all']['exec'], res_json['total_scores']['all']['exec (non empty)']))
                        # print('exact: {}'.format(res_json['total_scores']['all']['exact']))
                        # print('exact with val: {}'.format(res_json['total_scores']['all']['exact (with val)']))
                        # print('exec:{}, empty ratio:{}, error ratio:{}'.format(res_json['total_scores']['all']['exec'], res_json['total_scores']['all']['is_empty'], res_json['total_scores']['all']['is_error']))


if __name__ == "__main__":
    main()