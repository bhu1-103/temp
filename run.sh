#!/usr/bin/zsh

alias red='echo -e -n "\033[38;2;255;0;0m"'
alias green='echo -e -n "\033[38;2;0;255;0m"'
alias blue='echo -e -n "\033[38;2;0;0;255m"'

./create-folders.sh
cd step1;
#nvim AI_challenge_sce1.java;
echo -n "enter your preferred text editor: "
read EDITOR
$EDITOR input-java.csv; ./step1.sh; red && echo -n "Step 1 done" && echo -e "\033[0m" ;cd ../
cd step2; ./step2.sh; 
#./points.sh | wl-copy; pwd; ./v0.4/pls-work $(wl-paste)
#xdotool key super+f; red && 
red && echo -n "Step 2 done" && echo -e "\033[0m";cd ../
cd step3; bash step3.sh; red && echo -n "Step 3 done" && echo -e "\033[0m";cd ../
cd step4; ./step4.sh; red && echo -n "Step 4 done" && echo -e "\033[0m"
#cp -r sce1a_output/rssi rssi
#cp -r sce1a_output/throughput thruput
echo "done"
