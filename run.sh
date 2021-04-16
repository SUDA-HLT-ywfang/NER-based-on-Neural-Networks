# to reproduce result of benchmarks

## chinese, weiboNER
python train.py \
--train ../Data/weibo/weiboNER_train_bio \
--dev ../Data/weibo/weiboNER_dev_bio \
--test ../Data/weibo/weiboNER_test_bio \
--word_emb_dim 300 \
--pretrain_emb ../Data/pretrain_emb/sgns.merge.char \
--optimizer SGD \
--learning_rate 0.015 \
--batch_size 1 \
--gpu 7 > log/weibo.lstmcrf.log 2>&1 &