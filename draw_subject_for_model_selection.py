import random

random.seed(3988632922)  #this number is randomly generated

subject_list = list(range(0, 123))
random.shuffle(subject_list)


def get_ms_subject_ids():
    return subject_list[:12]


def get_test_subject_ids():
    return subject_list[12:]
