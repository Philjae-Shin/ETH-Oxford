import itertools

def operation_on_permutation(permutation):
    # placeholder operation
    return sum(permutation)


def compute_all_permutations(no_of_ones: int, no_of_zeros: int):
    # build the list of 1s and 0s
    bit_list = [1] * no_of_ones + [0] * no_of_zeros

    # generate every unique permutation of the list
    unique_permutations = set(itertools.permutations(bit_list))

    # compute result for each permutation
    results = []
    for perm_tuple in unique_permutations:
        perm_str = ''.join(perm_tuple)   # Convert tuple to string, e.g. ('1','0','1') -> "101"
        result_value = operation_on_permutation(perm_str)
        results.append({
            "permutation": perm_str,
            "result": result_value
    })


    # return a list of dictionaries
    return results