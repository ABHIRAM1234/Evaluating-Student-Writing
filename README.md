# Evaluating-Student-Writing

## Introduction
We divide our solution into two stages to reduce post-processing
stage 1:
15 class bert token prediction:
   · huggingface download pretrain model
   · split train.csv to 5 flod
   · train 8 models
    deberta-v2-xxlarge
    deberta-v2-xlarge
    longformer-large-4096
    distilbart-mnli-12-9
    bart-large-finetuned-squadv1
    roberta-large
    distilbart-cnn-12-6
    distilbart-xsum-12-6
   · Model weighted average ensemble to get the file:
    data_6model_offline712_online704_ensemble.pkl

stage 2:
lgb sentence prediction
   · First recall as many candidate samples as possible by lowering the threshold. On the training set, we recall three million samples to achieve a mean of 95% of recalls.
   · After getting the recall samples, we select sample with high boundary threshold and choice 65% length with the highest probability of the current class as a new sample.
   · Finally, We made about 170 features for lightgbm training, and select samples as the final submission.

hardware
  GPU: A100 * 4
  CPU: 60core +
  memory: 256G

## Main files:
   stage1_train_eval.py: train the Bert in stage1
   stage1_pred.py: generate the cv result
   stage1_merge.py: merge the cv result for next stage
   stage2_recall.py: generate the word segments as data for stage 2
   stage2_lstm_pca_train.py: train the lstm & pca model as features of lightgbm
   stage2_lgb_train.py: train the lightgbm model
   init.sh: the scripts to run all the pipeline

