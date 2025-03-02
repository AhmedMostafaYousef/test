#!/bin/bash
# declare -a AR_METHOD=("mudhog" "fedavg" "median" "gm" "krum" "mkrum" "foolsgold")
declare -a AR_METHOD=("mudhog" "fedavg" "gm" "krum" "mkrum" "foolsgold")

#### LIST CLIENTS for MNIST dataset
# Unrelibale: upto 4 nodes
# LIST_UNRELIABLE=("2" "2,18" "2,18,20" "2,18,20,37" "2,18,20,37" "2,18,20,37")

# Untargeted: Sign-flipping attackers up to 5 nodes
# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13" )
# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15" )
# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15,16,17,18" )
# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23" )

# Untargeted: Additive noise attackers, upto 6 node
# LIST_AN_ATTACKERS=("24" "24,25" "24,25,26" "24,25,26,27" "24,25,26,27,28" "24,25,26,27,28,30")

# Targeted: Label-flipping attackers: increasing from 3 nodes to 8 nodes.
# LIST_LF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "17,29,6,8,23,39,31,1")

# Targeted: Multi-labeli-flipping attackers: increasing from 3 nodes to 8 nodes.
# LIST_MLF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "17,29,6,8,23,39,31,1")

# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38" )
# LIST_LF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "39,40,41,42,43")
# LIST_MLF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "44,45,46,47,48,49,50,51,52,53")



# LIST_SF_ATTACKERS=(
#        "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
#        "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24"
#        "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
#         "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26"
#          "0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"
#             )
# DATASET="mnist"
# LR=1e-2 # SGD optimizer learning rate
# WD=1e-4  # weight-decay
# MOMENTUM=0.5 # SGD optimizer momentum
# GLOBAL_EPOCHS=40
# LOCAL_EPOCHS=4
# STD_UR=30
# NUM_CLIENTS=100
# MINIMUM_ITERATION_TO_START_THE_ATTACK=15
# LOADER_TYPE=dirichlet

# LIST_SF_ATTACKERS=("0" "0,10" "0,10,11" "0,10,11,12" "0,10,11,12,13" "0,10,11,12,13,14,15,16,17,18" )
# LIST_LF_ATTACKERS=("17,29,6" "17,29,6,8" "17,29,6,8,23" "17,29,6,8,23,39" "17,29,6,8,23,39,31" "19,20,21,22,24,25,26,27,28")

# Untargeted: Sign-flipping attackers up to 5 nodes
LIST_SF_ATTACKERS=("0,10" "0,10,11,18" "0,10,11,18,19,20" "0,10,11,18,19,20,24,28" "0,10,11,18,19,20,24,28,29,30" "0,10,11,18,19,20,24,28,29,30,31")
# Targeted: Multi-labeli-flipping attackers: increasing from 3 nodes to 8 nodes.
LIST_MLF_ATTACKERS=("32,33,34" "32,33,34,35" "32,33,34,35,36" "32,33,34,35,36,37" "32,33,34,35,36,37,38" "32,33,34,35,36,37,38,39")

DATASET="mnist"
LR=1e-2 # SGD optimizer learning rate
WD=1e-4  # weight-decay
MOMENTUM=0.5 # SGD optimizer momentum
GLOBAL_EPOCHS=40
LOCAL_EPOCHS=4
STD_UR=30
NUM_CLIENTS=40
MINIMUM_ITERATION_TO_START_THE_ATTACK=10
LOADER_TYPE=dirichlet

# We train the FL model with SGD optimizer (learning
# rate = 1e-2, momentum = 0.5 for MNIST and 0.9 for Fashion-MNIST, and
# weight-decay = 1e-4 for Fashion-MNIST) over 40 communication rounds, 4 local
# epochs;

# Series Exp1A (AN_LF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+3) label-flipping (LF) attackers; where j in range [2,6].
#EXP_NAME="sExp1Bb_40ep" 
#for ((i=0; i < 1; i++));
#do
#for ((j=4; j<6; j++));
#do
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))LF --epochs $GLOBAL_EPOCHS --num_clients 40 --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --lr $LR --weight_decay $WD --momentum $MOMENTUM --max_std_unreliable $STD_UR &
#done
#done

# sExp2A (AN_MLF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+3) multi-label-flipping (MLF) attackers; where j in range [2,6].
#EXP_NAME="sExp2Ba_40ep"
#for ((i=1; i < 2; i++));
#do
#for ((j=2; j<4; j++));
#do
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients 40  --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_multi_label_flipping ${LIST_MLF_ATTACKERS[$j]} --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --source_labels 1,2,3
#done
#done


# Series Exp3A (SF_AN_MLF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+3) label-flipping (MLF) attackers; where j in range [2,6].
#EXP_NAME="sExp3Bb_40ep"
#for ((i=0; i < 1; i++));
#do
#for ((j=2; j<3; j++));
#do
#CUDA_VISIBLE_DEVICES=6 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type dirichlet --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))LF --epochs $GLOBAL_EPOCHS  --num_clients 40 --inner_epochs $LOCAL_EPOCHS --list_unreliable ${LIST_UNRELIABLE[$j]} --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_uatk_add_noise ${LIST_AN_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR 
#done
#done

# Series Exp4A (SF_AN_MLF): x1 unreliable clients, and j additive-noise (AN) attackers,
# and (j+2) multi-label-flipping (MLF) attackers; where j in range [1,5].
# EXP_NAME="sExp4Ba_40ep10"

for ((i=3; i < 4; i++));
do
for ((j=5; j<6; j++));
do
STARTING_THRESHOLD=0
EXP_NAME="krum_${j}_"
# EXP_NAME="Exp_40C_40ep_10start_${j}_"
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --list_tatk_multi_label_flipping ${LIST_MLF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --attacks "backdoor" --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_backdoor ${LIST_BD_ATTACKERS[$j]} --list_tatk_label_flipping ${LIST_LF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 
# CUDA_VISIBLE_DEVICES=0 python main.py --save_model_weights --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 
CUDA_VISIBLE_DEVICES=0 python main.py --save_model_weights --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --list_tatk_multi_label_flipping ${LIST_MLF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset $DATASET --AR ${AR_METHOD[$i]} --loader_type $LOADER_TYPE --experiment_name ${EXP_NAME}_${DATASET}_${AR_METHOD[$i]}_40C_$((j+3))MLF --epochs $GLOBAL_EPOCHS --num_clients $NUM_CLIENTS --inner_epochs $LOCAL_EPOCHS --list_uatk_flip_sign ${LIST_SF_ATTACKERS[$j]} --source_labels 1,2,3 --lr $LR --weight_decay $WD --momentum $MOMENTUM --verbose  --max_std_unreliable $STD_UR  --minimumIterationToStartTheAttack $MINIMUM_ITERATION_TO_START_THE_ATTACK 

done
done
