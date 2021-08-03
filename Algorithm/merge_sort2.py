### This is an implementation of Merge Sort algorithm

import random

random.seed("portfolio")

numbers_to_sort = [random.randint(0, 100000) for i in range(100)]


def merge(left, right):
    final_merge_list = []
    left_pointer = 0
    right_pointer = 0

    while left_pointer < len(left) and right_pointer < len(right):
        if left[left_pointer] < right[right_pointer]:
            final_merge_list.append(left[left_pointer])
            left_pointer = left_pointer + 1
        else:
            final_merge_list.append(right[right_pointer])
            right_pointer = right_pointer + 1
    if left_pointer == len(left):
        final_merge_list.extend(right[right_pointer:])
    else:
        final_merge_list.extend(left[left_pointer:])
    return final_merge_list


def merge_sort(test_numbers):
    # Check for Base Case
    if len(test_numbers) <= 1:
        return test_numbers

    mid = len(test_numbers) // 2
    left, right = merge_sort(test_numbers[:mid]), merge_sort(test_numbers[mid:])

    return merge(left, right)


print(numbers_to_sort)
sort_numbers = merge_sort(numbers_to_sort)
print(sort_numbers)
