set PRE_SEQ_LEN=128
set LR=2e-2
set NUM_GPUS=1

python main.py ^
    --do_train ^
    --train_file knowledge/train.json ^
    --validation_file knowledge/dev.json ^
    --preprocessing_num_workers 4 ^
    --prompt_column content ^
    --response_column summary ^
    --overwrite_cache ^
    --model_name_or_path d:\\ChatGLM\\ChatGLM2-6B-main\\chatglm2-6b-int4 ^
    --output_dir output3\\knowledge-chatglm2-6b-int4-pt-%PRE_SEQ_LEN%-%LR% ^
    --overwrite_output_dir ^
    --max_source_length 64 ^
    --max_target_length 128 ^
    --per_device_train_batch_size 1 ^
    --per_device_eval_batch_size 1 ^
    --gradient_accumulation_steps 16 ^
    --predict_with_generate ^
    --max_steps 600 ^
    --logging_steps 10 ^
    --save_steps 100 ^
    --learning_rate %LR% ^
    --pre_seq_len %PRE_SEQ_LEN% ^