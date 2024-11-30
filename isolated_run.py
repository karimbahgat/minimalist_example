from predict import predict
from train import train

train("input/trainData.csv", "output/model.bin")
predictions = predict("output/model.bin", "input/trainData.csv", "input/futureClimateData.csv", "output/predictions.csv")
print("Predictions: ", predictions)