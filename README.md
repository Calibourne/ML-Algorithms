# ML-Algorithms

## Abstract

This Repo contains ML models made from scratch.

At this time, 3 models are implemented:
- Desicion Tree
- Adaptive Boost
- Gradient Boost

## Data

The data consists of features of real estate in different areas of Bangalore. It 
was pre-processed for convenience. The original data can be found [here](https://www.kaggle.com/datasets/aryanfelix/bangalore-housing-prices). 
 
### Variables
- availability: is the property available immediately (1) or in the near future (0). 
- total_sqft: the area of the property in square feet (1 foot = 30.54 cm). 
- bedrooms: the number of bedrooms in the property. 
- bath: the number of bathrooms in the property. 
- balcony: the number of balconies in the property. 
- rank: the ranking of the neighborhood in terms of average price (1 is the highest). 
- area_type: is the property type a built up area (B) or plot area (P). 
- price in rupees: the price of the property. 
 
### Data Splits
- Train: rows 1-8040.  
- Validation: rows 8041-10050. 
- Test: rows 10051-12563. 
