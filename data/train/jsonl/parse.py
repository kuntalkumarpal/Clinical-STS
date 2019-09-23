import json
import ast
def readJsonl(filename):
    json_file = open(filename)
    json_str = json_file.readlines()
    pairs = list()
    for s in json_str:
        oneDict = ast.literal_eval(s)
        if oneDict['gold_label'] == '-':
            continue
        pairs.append(oneDict)
    return pairs
