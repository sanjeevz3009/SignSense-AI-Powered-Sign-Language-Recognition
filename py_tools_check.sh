#!/bin/sh

CMDS=(
    "black --check --diff ."
    "pylint collect_training_data.py gestures_to_detect.py opencv_mp_feed.py sign_language_recog.py train_nn.py utils.py"
    "flake8 ."
    "isort --check-only ."
    "bandit -r ."
)

for i in "${CMDS[@]}"
do 
    if $i
    then
        echo "$i command successfully executed"
    else
        echo "$i command failed to execute or display errors that needs to be corrected"
    fi
done
