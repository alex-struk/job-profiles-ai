import pandas as pd

def loadCSVData():
    df = pd.read_csv("out.csv").copy()
    return df