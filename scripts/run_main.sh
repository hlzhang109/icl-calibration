model_size='gpt2-xl'
model=gpt2-xl

CUDA_VISIBLE_DEVICES=2 nohup python run_classification_calibrate.py \
--model=$model \
--dataset="sst2" \
--num_seeds=10 \
--all_shots="0, 1, 4, 8" \
--subsample_test_set=300 \
--approx > output_sst2_imdb_$model_size.out 2>&1