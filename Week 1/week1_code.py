# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:41:56 2022

@author: Sasuke
"""
#importing neccesary modules
import sys
import os
from collections import Counter

#defining circuit start and end tokens
CIRCUIT_START = ".circuit"
CIRCUIT_END = ".end"

ROOT_dir = os.getcwd() #gets your root directory

#tokenizer
def tokenizer(line): 
    '''
    Parameters
    ----------
    line : A string that has space seperated
    value.
    Returns
    -------
    A dictionary containing values of the 
    different elemnts given in the INPUT and there corresponding keys
    '''
    tokens = line.split()
    token_dict = dict() # empty dictionary
    # checks for the number of elements in the INPUT and stores them to a dictionary 
    if len(tokens) == 4:
        token_dict["name"] = tokens[0]
        token_dict["node0"] = tokens[1]
        token_dict["node1"] = tokens[2]
        token_dict["value"] = tokens[3]
    
    elif len(tokens) == 5:
        token_dict["name"] = tokens[0]
        token_dict["node0"] = tokens[1]
        token_dict["node1"] = tokens[2]
        token_dict["source_voltage"] = tokens[3]
        token_dict["value"] = tokens[4]

    elif len(tokens) == 6:
        token_dict["name"] = tokens[0]
        token_dict["node0"] = tokens[1]
        token_dict["node1"] = tokens[2]
        token_dict["source_node0"] = tokens[3]
        token_dict["source_node1"] = tokens[4]
        token_dict["value"] = tokens[5]
    else:
        return -1; #Number of eleme4nts in INPUT cant be greater than 6
    
    return token_dict # returns a dictionary containing tokens

assert len(sys.argv) == 2,"Please Input only the file name" #Ouput for wrong number of inputs
file_path = os.path.join(ROOT_dir, sys.argv[1]) # FIle Location
assert file_path.find("netlist") != -1, "Invalid file" # Incorrect Filename Entered

try:
    with open(file_path, "r") as f:
        text = f.read().splitlines()
    #print(text)

    occurence = Counter(text) # Counting word occurance

    assert occurence[CIRCUIT_START] ==1 and occurence[CIRCUIT_END] == 1, " INVALID FILE FORMAT, file contains incorrect number of start or end token"

    answer = []

    try:
        start_idx = text.index(CIRCUIT_START) + 1
        end_idx = text.index(CIRCUIT_END)
        assert start_idx<=end_idx #start_idx should be less or equal to end_idx
        for _ in range(start_idx, end_idx): #interating though every line of input text
            line = text[_].split("#")[0]
            answer.append(" ".join(reversed(tokenizer(line).values()))) # reversing tokens and joining them
        print(*reversed(answer), sep = '\n') #Printing reversed input

    except:
        print("Invalid file formating") #output for incorrect format

except FileNotFoundError as FNFE:
    print(FNFE) # output if no file is found

    
