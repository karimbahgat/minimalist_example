from predict import predict
from train import train
import pandas as pd

def assess(predictions, observations_fn):
    df = pd.read_csv(observations_fn)
    observations = df['disease_cases'] #What if we wanted last months disease cases - how to easily get lagged data
    print('predictions:', predictions)
    print('observations:', observations)
    print('MAE:', sum(abs(pred-obs) for pred,obs in zip(predictions, observations))/len(predictions))

train("input/trainData.csv", "output/model.bin")
predictions = predict("output/model.bin", "input/trainData.csv", "input/futureClimateData.csv", "output/predictions.csv")
assess(predictions, 'input/futureDiseaseData.csv')