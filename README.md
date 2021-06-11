# AFCFM

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in FCFM.py or AFCFM.py).
* FCFM on ml-1m
```
python FCFM.py --dataset ml-1m --epoch 200 --pretrain -1 --batch_size 4096 --hidden_factor 64 --keep_prob 0.8 --lr 0.01 --net_channel [32,32,32,32,32,32] 
```

* FCFM (different dataset)
```
python FCFM.py --dataset lastfm --epoch 200 --pretrain -1 --batch_size 4096 --hidden_factor 64 --keep_prob 0.8 --lr 0.01 --net_channel [32,32,32,32,32,32] 
```

* FCFM on ml-1m
```
python FCFM.py --dataset ml-1m --epoch 200 --pretrain -1 --batch_size 4096 --hidden_factor 64 --keep_prob 0.8 --lr 0.01 --net_channel [32,32,32,32,32,32] 
```

* AFCFM on ml-1m
```
python AFCFM.py --dataset ml-1m --epoch 200 --pretrain 1 --batch_size 4096 --hidden_factor 64 --lamda 1 --keep_prob 0.8 --lr 0.005 --net_channel [32,32,32,32,32,32] --eps 0.5 --adv 1 --onpara 0
```

* AFCFM (different dataset)
```
python AFCFM.py --dataset lastfm --epoch 200 --pretrain 1 --batch_size 4096 --hidden_factor 64 --lamda 1 --keep_prob 0.8 --lr 0.005 --net_channel [32,32,32,32,32,32] --eps 0.5 --adv 1 --onpara 0
```



