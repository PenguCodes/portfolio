### This is an implementation of Merge Sort algorithm


import random

random.seed("portfolio")

numbers_to_sort = [random.randint(0, 100000) for i in range(100)]


def merge_sort(numbers_list, left, right):

    if left >= right:
        return

    mid = (left + right) // 2

    merge_sort(numbers_list, left, mid)
    merge_sort(numbers_list, mid + 1, right)

    merge(numbers_list, left, right, mid)

def merge(numbers_list, left, right, mid):

        left_numbers_copy = numbers_list[left:mid + 1]
        right_numbers_copy = numbers_list[mid + 1:right + 1]

        l_counter, r_counter = 0, 0
        sorted_counter = left

        while l_counter < len(left_numbers_copy) and r_counter < len(right_numbers_copy):
            if left_numbers_copy[l_counter] <= right_numbers_copy[r_counter]:
                numbers_list[sorted_counter] = left_numbers_copy[l_counter]
                l_counter = l_counter + 1
            else:
                numbers_list[sorted_counter] = right_numbers_copy[r_counter]
                r_counter = r_counter + 1
            sorted_counter = sorted_counter + 1
        while l_counter < len(left_numbers_copy):
            numbers_list[sorted_counter] = left_numbers_copy[l_counter]
            l_counter = l_counter + 1
            sorted_counter = sorted_counter + 1
        while r_counter < len(right_numbers_copy):
                numbers_list[sorted_counter] = right_numbers_copy[r_counter]
                r_counter = r_counter + 1
                sorted_counter = sorted_counter + 1

# if numbers.count() == 0:
#     print("No Array Provided")

print(len(numbers_to_sort))

print(numbers_to_sort)
merge_sort(numbers_to_sort, 0, len(numbers_to_sort) - 1)

print(numbers_to_sort)

