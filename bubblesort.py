### This file incoporates the "Bubble Sort" Algorithm in Python 3.x.

import random

random.seed("portfolio")

numbers_to_sort = [random.randint(0, 100000) for i in range(100)]

def bubble_sort(number_list):
    swapped = True
    while swapped:
        swapped = False
        for i in range(1,len(number_list)):
            if number_list[i-1] > number_list[i]:
                number_list [i-1], number_list[i] = number_list[i], number_list[i-1]
                swapped = True
    return number_list



print(numbers_to_sort)
bubble_sort(numbers_to_sort)
print(numbers_to_sort)
