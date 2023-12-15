fairseq-train \
    path_to_data-bin/ \
    --arch TransformerModelBaseWithTM  --TM_num 5 \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam_inverse_sqrt --clip-norm 0.1 \
    --lr 0.0007  \
    --max-tokens 4000 \
    --save-dir path_to_data_checkpoints/ 
