#!/bin/bash

# Declare a string array with type
declare -a UserPrompt=("Can I have a cup of strawberry milk"
    "I want a cup of matcha milk"
    "May I have a some milk with taro"
    "Can I get a cup of strawberry boba milk"
    "I want to order a cup of strawberry matcha milk with boba")

declare -a InteruptIntruction=("I want to add boba into the drink"
    "Sorry, I want boba milk instead"
    "Can I replace the taro with strawberry?"
    "Sorry, I don't want to boba anymore"
    "Can I just get matcha boba milk instead?")

COUNTER=0
for index in {0..4}; do
 for interupt_step in {1..3}; do
  let COUNTER=COUNTER+1
  echo "python run.py --user-prompt "\"${UserPrompt[$index]}\"" --interupt-instruction "\"${InteruptIntruction[$index]}"\" --interupt-step $interupt_step --name $COUNTER"
  python run.py --user-prompt "\"${UserPrompt[$index]}\"" --interupt-instruction "\"${InteruptIntruction[$index]}"\" --interupt-step $interupt_step --name $COUNTER
  sleep 60
 done
done
