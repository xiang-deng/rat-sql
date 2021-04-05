mkdir tmpdata
pip install timeout_decorator
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
mkdir -p tmpdata/nl2code-1115,output_from=true,fs=2,emb=bert,cvlink,value,dec_min_freq=10,spideronly,newvalue
cp -r preprocessed_data/* tmpdata/nl2code-1115,output_from=true,fs=2,emb=bert,cvlink,value,dec_min_freq=10,spideronly,newvalue/
chmod 777 -R tmpdata/nl2code-1115,output_from=true,fs=2,emb=bert,cvlink,value,dec_min_freq=10,spideronly,newvalue
cp data/dev.json tmpdata/
cp data/tables.json tmpdata/
python seq2struct/datasets/prepare_column_value.py --data tmpdata/ --dbs database/
python run.py preprocess codalab_exp/$1.jsonnet
mkdir logdirs

python run.py eval codalab_exp/$1.jsonnet --skip_eval --model_name checkpoint_0 --model_config_args "{dec_min_freq: 10, use_column_value: false, mask_column: -1, use_encoder_action: false, use_bert_subtask_loss: false, use_rat_subtask_loss: false, local_pretrained: '', cv_link: true}"
python extract_preds.py --inferred logdirs/$1-step40000/$1.infer --output checkpoint_0.$1.predicted_sql.txt
python run.py eval codalab_exp/$1.jsonnet --skip_eval --model_name checkpoint_1 --model_config_args "{dec_min_freq: 10, use_column_value: false, mask_column: -1, use_encoder_action: false, use_bert_subtask_loss: false, use_rat_subtask_loss: false, local_pretrained: '', cv_link: false}"
python extract_preds.py --inferred logdirs/$1-step40000/$1.infer --output checkpoint_1.$1.predicted_sql.txt

python evaluation.py --gold dev_gold.sql --pred checkpoint_0.$1.predicted_sql.txt --etype all --db database --table data/tables.json
python evaluation.py --gold dev_gold.sql --pred checkpoint_1.$1.predicted_sql.txt --etype all --db database --table data/tables.json