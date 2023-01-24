# Adapted CityLearn 2021 Dataset

The Dataset provided for the CityLearn Challenge 2021 differs from the one provided for CityLearn 2022.
In order to profit from this larger amount of data, some changes need to be made.

Building_i.csv:
- Handle Daylight Savings
- combine cooling load, DHW load, and Equipment Electric Power into single metric
- 

carbon__intensity.csv:
- do nothing

pricing.csv:
- create file from simple pricing rules

weather.csv:
- rename columns

schema.json:
- remove unused appliances and storage devices
- resize things

model:
- somehow incorporate the fact that buildings are different now.

