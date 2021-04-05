import json

import argparse

import re

import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inferred', help="inference output")
    parser.add_argument('--output', help="pred output")
    args = parser.parse_args()
    with open(args.inferred, 'r') as f_in, open(args.output,'w') as f_out:
        for line in f_in:
            result = json.loads(line.strip())
            orig_question = result['beams'][0]["orig_question"]
            orig_question_uncased = orig_question.lower()
            pred = result['beams'][0]['inferred_code']
            pred_new = pred
            values = re.findall(r'\".+?\"', pred)
            if values:
                for value in values:
                    value = value.strip("\"")
                    if value in orig_question_uncased:
                        loc = re.search(r'\b'+re.escape(value)+r'\b', orig_question_uncased)
                        if loc is not None:
                            pred_new = pred_new.replace(f'\"{value}\"', f'\"{orig_question[loc.start():loc.end()]}\"')
            pred_new = pred_new.replace('$$','\'')
            like_vals = re.findall(r'LIKE \'\w+?\'',pred_new) + re.findall(r'LIKE \"\w+?\"',pred_new)
            if like_vals:
                for val in like_vals:
                    pred_new = pred_new.replace(val, 'LIKE \'%{}%\''.format(val[6:-1]))
            # fix CAST column
            pred_new = pred_new.replace('cast', '`cast`')
            f_out.write(pred_new+'\n')
