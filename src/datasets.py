import pandas as pd
import numpy as np

def timespan_to_refs_sequence(start, end):
    start_year = int(str(start)[:-2])
    start_month = int(str(start)[-2:])

    end_year = str(end)[:-2]
    end_month = str(end)[-2:]
    refs_list = [start]
    
    while refs_list[-1] != end:
        prev_year = int(str(refs_list[-1])[:-2])
        prev_month = int(str(refs_list[-1])[-2:])

        next_ref_month = (prev_month + 1)%12 != 0 and (prev_month + 1)%12 or 12
        next_ref_year = next_ref_month < prev_month and prev_year + 1 or prev_year
        if(next_ref_month<10):
            next_ref_month = "0" + str(next_ref_month)
        next_ref = str(next_ref_year) + str(next_ref_month)
        refs_list.append(int(next_ref))
    return refs_list

def load_data(path, timespan= None):
    dataset = pd.read_csv(path, sep=",")
    return dataset.values