echo "cot_extract_generation_results"

python get_cot_results.py \
--model=$model \
--dataset="worldtree_cot, open_book_qa_cot, commonsense_qa_cot, strategy_qa_cot, " \
--num_seeds=1 \
--all_shots="0, 1, 4, 8" \
--subsample_test_set=100  \
--approx > cot_100_get_all_results_1seed_$model_size.out



echo "cot_extract_generation_results commonsense_qa_cot, strategy_qa_cot (last two)"
python get_cot_results.py \
--model=$model \
--dataset="commonsense_qa_cot, strategy_qa_cot" \
--num_seeds=1 \
--all_shots="0, 1, 4, 8" \
--subsample_test_set=100  \
--approx > cot_100_get_commonsense_qa_cot_strategy_qa_cot_1seed_$model_size.out



echo "cot_extract_generation_results worldtree_cot, open_book_qa_cot (first two)"
python get_cot_results.py \
--model=$model \
--dataset="worldtree_cot, open_book_qa_cot" \
--num_seeds=1 \
--all_shots="0, 1, 4, 8" \
--subsample_test_set=100  \
--approx > cot_100_get_worldtree_cot_open_book_qa_cot_1seed_$model_size.out

