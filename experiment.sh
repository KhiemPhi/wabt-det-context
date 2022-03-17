#Experiment 1: Effects of Focal CB Loss Augmentation w/ Context
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -ta -sn "context_all_augments" -c 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -sn "context train augments" -c 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -ta -sn "context test augments" -c 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20  -sn "context no augments" -c 

#Experiment 2 : Effects of Softmax CB Loss w/ Context
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -ta -sn "context_all_augments" -l "softmax" -c 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -sn "context train augments" -l "softmax" -c 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -ta "context test augments" -l "softmax" -c 
python -u train.py --gpu 5 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -sn "context no augments" -l "softmax" -c

#Experiment 3 : Effects of Cross Entropy Loss w/ Context
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -ta -sn "context_all_augments" -l "cross-entropy" -c 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -sn "context train augments" -l "cross-entropy" -c 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -ta -sn "context test augments" -l "cross-entropy" -c 
python -u train.py --gpu 1 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -sn "context no augments" -l "cross-entropy" -c


# Experiment 4: Effects of Cross Entropy    
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -ta -sn "no_context_all_augments" -l "cross-entropy"
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -sn "no_context train augments" -l "cross-entropy" 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -ta -sn "no_context test augments" -l "cross-entropy" 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -sn "no_context no_augments" -l "cross-entropy" 

# Exerpiemtnts with CB Loss w/o Context
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -ta -sn "cb-focal-loss_no_context_all_augments"
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -tra -sn "cb-focal-loss_no_context train augments" 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -ta -sn "cb-focal-loss_no_context test augments" 
python -u train.py --gpu 3 --batch 70 --learning_rate 1e-4 --epochs 20 --pruning -sn "cb-focal-loss_no_context no_augments"


