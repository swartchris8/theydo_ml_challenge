python run_glue.py \
  --model_name_or_path bert-base-cased \
  --dataset_name imdb  \
  --do_train \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --max_train_samples 5 \
  --max_eval_samples 5 \
  --max_predict_samples 5 \
  --output_dir test-sent-bert