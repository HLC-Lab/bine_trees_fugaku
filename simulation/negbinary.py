#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: negbinary.py
# Generates all the negbinary numbers between 0 and 2^n - 1
# The negbinary number system is a base -2 number system
import math as m
import sys
import numpy as np

def neg2dec(arr):
    n = 0
    for i, num in enumerate(arr[::-1]):
        n+= ((-2)**i)*num
    return n

def dec2neg(num):
    if num == 0:
        digits = ['0']
    else:
        digits = []
        while num != 0:
            num, remainder = divmod(num, -2)
            if remainder < 0:
                num, remainder = num + 1, remainder + 2
            digits.append(str(remainder))
    return ''.join(digits[::-1])

def find_switch_points(s):
    switch_points = []
    last = 0
    for i in range(0, len(s)):
        bit = int(s[i])
        if bit:
            if last == 0:
                if i > 0:
                    switch_points += [i-1]
            last = 1
        else:
            if last == 1:
                switch_points += [i-1]
            last = 0

    if last == 1:
        switch_points += [i]
    return switch_points

p = int(sys.argv[1])
d = {}
nbits = m.ceil(m.log2(p))
for i in range(-p*2 - 1, p*2 + 1):
    n = dec2neg(i)
    if len(n) <= nbits:
        n = n.zfill(nbits)
        if i in d:
            d[i % p].append(n)
        else:
            d[i % p] = [n]
'''
for i in range(p):
    print(i, d[i])

print("")
print("")
'''
    
ranks = [0]*p
r = 0
for q in range(p):
    pos = 0
    if ((r % 2 == 0) and (q % 2 != 0)) or ((r % 2 != 0) and (q % 2 == 0)): # Even talking to odd or viceversa, even number of steps
        pos = (q-r) % p
    else:
        pos = (r-q) % p
    ranks[pos] = d[q]

for i in range(p):
    print(i, ranks[i])

print("")
print("")


# Flip the bits for even destinations (assuming even source),
# so that the processing is homogeneous

idxs = {}

for i in range(p):
    for j in ranks[i]:
        string = j[::-1]
        switch_points = find_switch_points(string)        
        if i in idxs:
            idxs[i] += [switch_points]
        else:
            idxs[i] = [switch_points]

for i in range(p):
    print(i, idxs[i])
