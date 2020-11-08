import pandas as pd
import pickle
import re
import javalang

with open('/homes/cs510/project-3/data/train.pickle', 'rb') as handle:
    train = pickle.load(handle)
with open('/homes/cs510/project-3/data/valid.pickle', 'rb') as handle:
    valid = pickle.load(handle)
with open('/homes/cs510/project-3/data/test.pickle', 'rb') as handle:
    test = pickle.load(handle)

# Get top 50000 rows of each
train = train.head(50000)
valid = valid.head(50000)
test = test.head(50000)

# Get num buggy 
train_buggy = len(train.loc[train['is_buggy'] == 1])
valid_buggy = len(valid.loc[valid['is_buggy'] == 1])
test_buggy = len(test.loc[test['is_buggy'] == 1])

# Print Num buggy rows
print("Number Train Buggy: " + str(train_buggy))
print("Number Valid Buggy: " + str(valid_buggy))
print("Number Test Buggy: " + str(test_buggy))

# Print number of rows where is_buggy = 1
print("Train Buggy Rate: " + str(train_buggy/50000))
print("Valid Buggy Rate: " + str(valid_buggy/50000))
print("Test Buggy Rate: " + str(test_buggy/50000))