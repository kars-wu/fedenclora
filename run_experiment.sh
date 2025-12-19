python -m experiments.run_privacy_attacks\
 --model-path /data/wuhao/model/Qwen2.5-3B-Instruct\
 --dataset sst2\
 --num-samples 1000\
 --num-clients 3\
 --num-rounds 3\
 --local-epochs 2\
 --batch-size 4\
 --device cuda:3