import pandas as pd

# Modify the script for study-specific alterations (filter my prompt, response length, etc.)

# Loads all items in from the original datasets (files ending in _items_sctt)
dfTrain = pd.read_csv("C:\\Users\\samgr\\PycharmProjects\\ca-polyencoder-official\\data\\training_items_sctt.csv")
dfVal = pd.read_csv("C:\\Users\\samgr\\PycharmProjects\\ca-polyencoder-official\\data\\validation_items_sctt.csv")
dfTest = pd.read_csv("C:\\Users\\samgr\\PycharmProjects\\ca-polyencoder-official\\data\\test_items_sctt.csv")

# Keep only relevant columns
dfTrain = dfTrain[["text", "response", "label"]]
dfVal = dfVal[["text", "response", "label"]]
dfTest = dfTest[["text", "response", "label"]]

# Save filtered files
dfTrain.to_csv("C:\\Users\\samgr\\PycharmProjects\\ca-polyencoder-official\\data\\train.csv")
dfVal.to_csv("C:\\Users\\samgr\\PycharmProjects\\ca-polyencoder-official\\data\\val.csv")
dfTest.to_csv("C:\\Users\\samgr\\PycharmProjects\\ca-polyencoder-official\\data\\test.csv")