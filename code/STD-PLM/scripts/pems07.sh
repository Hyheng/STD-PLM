
python main.py \
    --data_path '../../../data/traffic/PEMS07/PEMS07.npz' \
    --adj_filename ../../../data/traffic/PEMS07/PEMS07.csv \
    --dataset PEMS07FLOW \
    --desc PEMS07_pre\
    --sample_len 12 \
    --predict_len 12 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --epoch 500 \
    --val_epoch 1 \
    --test_epoch 5 \
    --batch_size 64\
    --lr 0.001 \
    --causal 0 \
    --model gpt2 \
    --patience 50 \
    --ln_grad \
    --lora \
    --t_dim 64 \
    --node_emb_dim 64 \
    --node_embedding \
    --llm_layers 3 \
    --time_token \
    --dropout 0.05 \
    --trunc_k 64 \
    --weight_decay 0 \
    --task prediction \
    --sandglassAttn \
    --sag_dim 128 \
    --sag_tokens 128 \
    --input_dim 1\
    --output_dim 1
