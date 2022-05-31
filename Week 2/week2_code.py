# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 18:32:51 2022

@author: Sasuke
"""
#Phase is measured in degrees
#Importing necessary modules
import numpy as np
import sys
from pprint import pprint
import cmath
import re
import math

# Helper tokens
CIRCUIT_START = ".circuit"
CIRCUIT_END = ".end"
ac_flag = False
AC = ".ac"
np.set_printoptions(
    formatter={"float": lambda x: "{0:0.5f}".format(x)}
)  # Setting print options

# parses the string to find the value
def parse_val(value):
    """


    Parameters
    ----------
    value : String
    This function parses the input string
    Returns
    -------
    float

    """
    
    split = re.split("([A-Z]+)", value.upper())
    if len(split) == 1 or split[1] == "E": #if not special charectar return value as it is.
        return float(value)
    
    suf = split[1]
    # 
    if suf[0] == "K":  
        return float(split[0]) * 10**3
    #Femto
    elif suf[0] == "F":  
        return float(split[0]) * 10**-15
    # Mega
    elif suf[0:3] == "MEG":  
        return float(split[0]) * 10**6
    # Micro
    elif suf[0:5] == "MICRO" or suf[0] == "U":
          return float(split[0]) * 10**-6 
    # Giga
    elif suf[0] == "G":  
        return float(split[0]) * 10**9
    # Pico
    elif suf[0] == "P":  
        return float(split[0]) * 10**-12
    # Tera
    elif suf[0] == "T":  
        return float(split[0]) * 10**12
    # Milli
    elif suf[0] == "M":  
        return float(split[0]) * 10**-3
    # Nano
    elif suf[0] == "N":  
        return float(split[0]) * 10**-9
    


# Defining necessary classes
class Passive:
    def __init__(self, tokens, flag):
        if flag:
            self.name = tokens[0]
            self.n1 = re.sub("\D", "", tokens[1]) if tokens[1] != "GND" else tokens[1]
            self.n2 = re.sub("\D", "", tokens[2]) if tokens[2] != "GND" else tokens[2]
            self.value = str(parse_val(tokens[3]))
           
        else:
            self.name = tokens[0]
            self.n1 = re.sub("\D", "", tokens[1]) if tokens[1] != "GND" else tokens[1]
            self.n2 = re.sub("\D", "", tokens[2]) if tokens[2] != "GND" else tokens[2]
            self.value = str(parse_val(tokens[3]))


class IndependentSource:
    def __init__(self, tokens, flag):
        if flag:
            self.name = tokens[0]
            self.n1 = re.sub("\D", "", tokens[1]) if tokens[1] != "GND" else tokens[1]
            self.n2 = re.sub("\D", "", tokens[2]) if tokens[2] != "GND" else tokens[2]
            self.value = str(parse_val(tokens[3]))
            self.pp = str(parse_val(tokens[4]))
            self.phase = str(math.pi*parse_val(tokens[5])/180)
            
        else:
            self.name = tokens[0]
            self.n1 = re.sub("\D", "", tokens[1]) if tokens[1] != "GND" else tokens[1]
            self.n2 = re.sub("\D", "", tokens[2]) if tokens[2] != "GND" else tokens[2]
            self.value = str(parse_val(tokens[3]))


class DependentSource:
    def __init__(self, tokens, flag):
        if flag:
            self.current = True
            if len(tokens) == 6:
                self.name = tokens[0]
                self.n1 = (
                    re.sub("\D", "", tokens[1]) if tokens[1] != "GND" else tokens[1]
                )
                self.n2 = (
                    re.sub("\D", "", tokens[2]) if tokens[2] != "GND" else tokens[2]
                )
                self.n3 = (
                    re.sub("\D", "", tokens[3]) if tokens[3] != "GND" else tokens[3]
                )
                self.n4 = (
                    re.sub("\D", "", tokens[4]) if tokens[4] != "GND" else tokens[4]
                )
                self.value = str(parse_val(tokens[5]))
                self.current = False
            else:
                self.name = tokens[0]
                self.n1 = (
                    re.sub("\D", "", tokens[1]) if tokens[1] != "GND" else tokens[1]
                )
                self.n2 = (
                    re.sub("\D", "", tokens[2]) if tokens[2] != "GND" else tokens[2]
                )
                self.cVoltage = tokens[3]
                self.value = str(parse_val(tokens[4]))

        else:
            self.current = True
            if len(tokens) == 6:
                self.name = tokens[0]
                self.n1 = (
                    re.sub("\D", "", tokens[1]) if tokens[1] != "GND" else tokens[1]
                )
                self.n2 = (
                    re.sub("\D", "", tokens[2]) if tokens[2] != "GND" else tokens[2]
                )
                self.n3 = (
                    re.sub("\D", "", tokens[3]) if tokens[3] != "GND" else tokens[3]
                )
                self.n4 = (
                    re.sub("\D", "", tokens[4]) if tokens[4] != "GND" else tokens[4]
                )
                self.value = str(parse_val(tokens[5]))
                self.current = False
            else:
                self.name = tokens[0]
                self.n1 = (
                    re.sub("\D", "", tokens[1]) if tokens[1] != "GND" else tokens[1]
                )
                self.n2 = (
                    re.sub("\D", "", tokens[2]) if tokens[2] != "GND" else tokens[2]
                )
                self.cVoltage = tokens[3]
                self.value = str(parse_val(tokens[4]))


# Helper variables
nodes = set()
num_variables = 0
vol_mapping = dict()
voltages = []

try:
    with open(sys.argv[1], "r") as f:
        text = f.read().splitlines()
except FileNotFoundError as FNFE:
    print(FNFE)
    sys.exit(1)

try:
    start_idx = text.index(CIRCUIT_START) + 1
    end_idx = text.index(CIRCUIT_END)

    assert start_idx > 0 and start_idx < end_idx

    if end_idx + 1 != len(text):
        if AC == text[end_idx + 1].split()[0]:
            ac_flag = True  # AC enable
            W = 2 * math.pi * float(str(parse_val(text[end_idx + 1].split()[-1])))
    elements = []

    for line in text[start_idx:end_idx]:
        tokens = line.split("#")[0].split()
        if tokens[0][0] in ["R", "L", "C"]:
            element = Passive(tokens, ac_flag)
            [nodes.add(x) for x in [element.n1, element.n2]]

        elif tokens[0][0] in ["V", "I"]:
            element = IndependentSource(tokens, ac_flag)
            if element.name[0] == "V":
                if ac_flag:
                    vol_mapping[element.name] = [
                        element.n1,
                        element.n2,
                        element.value,
                        element.pp,
                        element.phase,
                    ]

                else:
                    vol_mapping[element.name] = [element.n1, element.n2, element.value]
                num_variables += 1
                voltages.append(element.name)
            [nodes.add(x) for x in [element.n1, element.n2]]

        else:
            element = DependentSource(tokens, ac_flag)
            if element.current:
                [nodes.add(x) for x in [element.n1, element.n2]]
                if element.name[0] == "H":
                    voltages.append(element.name)
                    num_variables += 1
            else:
                [nodes.add(x) for x in [element.n1, element.n2, element.n3, element.n4]]

                if element.name[0] == "E":
                    voltages.append(element.name)
                    num_variables += 1

        elements.append(element)
except (ValueError, TypeError):
    sys.exit("Error: Invalid value in circuit file")
except AssertionError:
    sys.exit("Error: Invalid circuit file")

num_variables += len(nodes)
if ac_flag:

    S = np.zeros((num_variables + 1, 1), dtype=complex)

    M = np.zeros((num_variables + 1, num_variables + 1), dtype=complex)

else:
    S = np.zeros((num_variables + 1, 1))
    M = np.zeros((num_variables + 1, num_variables + 1))


node_mapping = {"GND": 0} # Assuming GND as ground
GND_Flag = False
for node in nodes:
    if node not in node_mapping.keys():
        node_mapping[node] = int(node)
    if node == "GND":
        GND_Flag = True
        
assert GND_Flag, " Please label ground as GND"

S[0] += 0


def fill_M(element):
    """


    Parameters
    ----------
    element : class objects
        Fill M and S matrix with appropriate stamps

    Returns
    -------
    None.

    """
    global node_mapping
    global nodes, ac_flag
    global M, W
    global voltages
    global vol_mapping
    if element.name[0] == "R":
        G = 1 / float(element.value)
        n1 = node_mapping[element.n1]
        n2 = node_mapping[element.n2]
        M[n1][n1] += G
        M[n2][n2] += G
        M[n1][n2] -= G
        M[n2][n1] -= G

    elif element.name[0] == "L":
        G = complex(0, -1 / (W * float(element.value)))
        n1 = node_mapping[element.n1]
        n2 = node_mapping[element.n2]
        M[n1][n1] += G
        M[n2][n2] += G
        M[n1][n2] -= G
        M[n2][n1] -= G

    elif element.name[0] == "C":
        G = complex(0, W * float(element.value))
        n1 = node_mapping[element.n1]
        n2 = node_mapping[element.n2]
        M[n1][n1] += G
        M[n2][n2] += G
        M[n1][n2] -= G
        M[n2][n1] -= G

    elif element.name[0] == "V":
        if ac_flag:
            n1 = node_mapping[element.n1]
            n2 = node_mapping[element.n2]
            i12 = voltages.index(element.name) + len(nodes)
            phase = float(element.phase)
            A = float(element.pp) / 2
            M[n1][i12] += 1
            M[n2][i12] -= 1
            M[i12][n1] += 1
            M[i12][n2] -= 1
            z = complex(A * math.cos(phase), A * math.sin(phase))
            S[i12] += z

        else:
            n1 = node_mapping[element.n1]
            n2 = node_mapping[element.n2]
            i12 = voltages.index(element.name) + len(nodes)
            M[n1][i12] += 1
            M[n2][i12] -= 1
            M[i12][n1] += 1
            M[i12][n2] -= 1

            S[i12] += float(element.value)

    elif element.name[0] == "I":
        if ac_flag:
            n1 = node_mapping[element.n1]
            n2 = node_mapping[element.n2]
            phase = float(element.phase)
            A = float(element.pp) / 2
            z = complex(A * math.cos(phase), A * math.sin(phase)) 
            S[n1] -= z
            S[n2] += z
            
        else:
            n1 = node_mapping[element.n1]
            n2 = node_mapping[element.n2]
            S[n1] -= float(element.value)
            S[n2] += float(element.value)

    elif element.name[0] == "G":
        n1 = node_mapping[element.n1]
        n2 = node_mapping[element.n2]
        n3 = node_mapping[element.n3]
        n4 = node_mapping[element.n4]
        G = float(element.value)
        M[n1][n3] += G
        M[n1][n4] += -G
        M[n2][n4] += G
        M[n2][n3] += -G

    elif element.name[0] == "E":
        n1 = node_mapping[element.n1]
        n2 = node_mapping[element.n2]
        n3 = node_mapping[element.n3]
        n4 = node_mapping[element.n4]
        i12 = voltages.index(element.name) + len(nodes)

        M[n1][i12] += 1
        M[n2][i12] -= 1
        M[i12][n1] += 1
        M[i12][n2] -= 1
        M[i12][n3] -= float(element.value)
        M[i12][n4] += float(element.value)

    elif element.name[0] == "H":
        n1 = node_mapping[element.n1]
        n2 = node_mapping[element.n2]
        value = float(element.value)
        source = vol_mapping[element.cVoltage]
        n3 = node_mapping[source[0]]
        n4 = node_mapping[source[1]]
        source_value = float(source[2])
        i12 = voltages.index(element.name) + len(nodes)
        i34 = voltages.index(element.cVoltage) + len(nodes)
        M[n1][i12] += 1
        M[n2][i12] -= 1
        M[i12][n1] += 1
        M[i12][n2] -= 1
        M[i12][i34] -= value

    elif element.name[0] == "F":

        n1 = node_mapping[element.n1]
        n2 = node_mapping[element.n2]
        value = float(element.value)
        source = vol_mapping[element.cVoltage]
        n3 = node_mapping[source[0]]
        n4 = node_mapping[source[1]]
        source_value = float(source[2])
        i34 = voltages.index(element.cVoltage) + len(nodes)
        M[n1][i34] += value
        M[n2][i34] -= value

#Opening values
M[-1][0] += 1
M[0][-1] += 1
try:
    for element in elements:
        fill_M(element)
    x = np.matmul(np.linalg.inv(M), S)
except np.linalg.LinAlgError:
    sys.exit("Circuit is unsolvable")
except ValueError:
    print("Error in file format")
    sys.exit()
#Printing Values
x[0] = 0
if ac_flag:
    for idx, value in enumerate(x):
        if idx < len(nodes):
            tmp = cmath.polar(x[idx][0])
            print(
                "Volate at node {0} is: {1:0.4f}exp({2:0.4f}j)".format(
                    idx, tmp[0], tmp[1]
                )
            )
        elif idx < len(nodes) + len(voltages):
            tmp = cmath.polar(x[idx][0])
            print(
                f"Current through Voltage source {voltages[idx - len(nodes)]}",
                "is: {0:0.4f}exp({1:0.4f}j)".format(tmp[0], tmp[1]),
            )


else:
    for idx, value in enumerate(x):
        if idx < len(nodes):
            print("Volate at node {0} is: {1:0.4f}".format(idx, x[idx][0]))
        elif idx < len(nodes) + len(voltages):
            print(
                f"Current through Voltage source {voltages[idx - len(nodes)]}",
                "is: {0:0.4f}".format(x[idx][0]),
            )
