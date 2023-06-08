python run_translation.py \
    --model_name_or_path Helsinki-NLP/opus-mt-de-en \
    --do_train \
    --do_eval \
    --source_lang de \
    --target_lang en \
    --dataset_name wmt16 \
    --dataset_config_name de-en \
    --output_dir finetuned_de-en_translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --max_train_samples 2000 \
    --max_eval_samples 500 \
    --max_predict_samples 500 \
    --predict_with_generate 