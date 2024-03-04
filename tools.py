
def ranking_dict(input_dict):
    len_dict = {}
    output_list = []
    for key, value in input_dict.items():
        len_dict[key] = len(value[0]) + len(value[1])
    sorted_dict = sorted(len_dict.items(), key=lambda x: x[1], reverse=True)
    for key in sorted_dict:
        output_list.append(key[0])
    return output_list

if __name__ == '__main__':
    test_case = {'a':[[1,2,3],[4]], 'b':[[3],[4,]], 'c':[[1,2,3,7],[4,5,6]], 'd':[[1,3],[4,5,6]]}

    print(ranking_dict(test_case)[:3])
