cd ../

csv=experiments/all_csvs/baseline/baseline_test.csv

eng_ref=data/test_data/devtest_english.txt
khm_ref=data/test_data/devtest_khmer.txt
lao_ref=data/test_data/devtest_lao.txt
tha_ref=data/test_data/devtest_thai.txt

baseline_ckpt=model/baseline_small_fixed/checkpoints/model-epoch=94-val_loss=2.6131.ckpt
baseline_config=experiments/all_configs/baseline/baseline.yaml
baseline_preds=model/baseline/preds

srun python train.py $baseline_config $baseline_ckpt $csv
sacrebleu $eng_ref -i $baseline_preds/[KHMER]-[ENGLISH].txt -m bleu chrf -b
sacrebleu $eng_ref -i $baseline_preds/[LAO]-[ENGLISH].txt -m bleu chrf -b
sacrebleu $eng_ref -i $baseline_preds/[THAI]-[ENGLISH].txt -m bleu chrf -b
sacrebleu $khm_ref -i $baseline_preds/[ENGLISH]-[KHMER].txt --tokenize flores200 -m bleu chrf -b
sacrebleu $lao_ref -i $baseline_preds/[ENGLISH]-[LAO].txt --tokenize flores200 -m bleu chrf -b
sacrebleu $tha_ref -i $baseline_preds/[ENGLISH]-[THAI].txt --tokenize flores200 -m bleu chrf -b