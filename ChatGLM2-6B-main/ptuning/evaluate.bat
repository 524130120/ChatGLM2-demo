set PRE_SEQ_LEN=128
set CHECKPOINT=adgen-chatglm2-6b-int4-pt-128-2e-2
set STEP=2000
set NUM_GPUS=1

python main.py ^
    --do_predict ^
    --validation_file AdvertiseGen/dev.json ^
    --test_file AdvertiseGen/dev.json ^
    --overwrite_cache ^
    --prompt_column content ^
    --response_column summary ^
    --model_name_or_path d:\\ChatGLM\\ChatGLM2-6B-main\\chatglm2-6b-int4 ^
    --ptuning_checkpoint .\\output2\\%CHECKPOINT%\\checkpoint-%STEP% ^
    --output_dir .\\output2\\%CHECKPOINT% ^
    --overwrite_output_dir ^
    --max_source_length 64 ^
    --max_target_length 64 ^
    --per_device_eval_batch_size 1 ^
    --predict_with_generate ^
    --pre_seq_len %PRE_SEQ_LEN% ^