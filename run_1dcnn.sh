DATA_DIR="/home/jhjoo/data/VAR/UCF101-Frames"

python3 main.py -data $DATA_DIR -b 24 -dim 256 -a 0.1 -iter 15 -lr 0.001 -optim sgd -model 1dcnn -eval_interval 5
