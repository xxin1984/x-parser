import re
import os
from sklearn.metrics import roc_auc_score
import numpy as np


def my_norm_list(list_input):
    sum_value = 0.0
    for i in list_input:
        sum_value += float(i)
    if sum_value <= 0.0:
        return my_list
    else:
        result = []
        for i in list_input:
            result.append(float(i)/sum_value)
        return result


def my_avg_list(my_list):
    sum = 0.0
    length = len(my_list)+0.0
    for i in my_list:
        sum = i+sum
    if length > 0:
        return sum/length
    else:
        return 0.0


def my_random_generater(size):
    return np.random.randint(0, size)


def my_get_sorted_keys(table):
    def f(k):
        return table[k]
    return sorted(table.keys(), key=f)


def my_auc(y_true, y_scores):
    #y_true = np.array([0, 0, 1, 1])
    #y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    values= {}
    for label in y_true:
        values[label] = True
    if len(values)==1:
        return 1.0
    else:
        return roc_auc_score(y_true, y_scores)


def my_strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 32:
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring


def my_open_file(path):
    file = open(path, "r", encoding="utf-8")
    result = []
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        result.append(line)
    file.close()
    return result


def my_open_file_content(path):
    file = open(path, "r", encoding="utf-8")
    result = ""
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        result = result + line + " "
    file.close()
    return result


def my_write_file(path, content):
    file = open(path, "w", encoding="utf-8")
    file.write(content)
    file.close()


def my_write_file_append(path, content):
    file = open(path, "a", encoding="utf-8")
    file.write(content)
    file.close()


def my_read_ix(path):
    tag_2_ix = {}
    lines = my_open_file(path)
    for line in lines:
        list = line.split(" ")
        tag_2_ix[list[0]] = int(list[1])
    return tag_2_ix


def my_read_ix_reverse(path):
    tag_2_ix = {}
    lines = my_open_file(path)
    for line in lines:
        list = line.split(" ")
        tag_2_ix[int(list[1])] = list[0]
    return tag_2_ix


def my_str_index_of(sub:str, text:str):
    result = re.findall(sub,text)
    if len(result) == 0:
        return -1
    else:
        return text.index(sub)


def my_copy(source, target):
    fileSource = open(source, "rb")
    content = fileSource.read()
    fileSource.close()
    fileTarget = open(target, "wb")
    fileTarget.write(content)
    fileTarget.close()


def my_move(source, target):
    os.rename(source, target)


def my_list(root):
    return os.listdir(root)


def my_match(text, query_re):
    result = re.findall(query_re, text)
    if(len(result)>0):
        return re.findall(query_re, text)[0]
    else:
        return ""

def my_reverse_list(my_list):
    return list(reversed(my_list))


