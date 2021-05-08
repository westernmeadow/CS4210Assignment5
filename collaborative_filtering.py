#-------------------------------------------------------------------------
# AUTHOR: Wesley Kwan
# FILENAME: collaborative_filtering
# SPECIFICATION: Make user-based recommendations for travel locations
# FOR: CS 4200- Assignment #5
# TIME SPENT: 90 min
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   #--> add your Python code here
scores = []
for i in range(0,99):
   vec1 = [np.delete(np.array([df.iloc[i]]),[0,1,4])]
   vec2 = [np.delete(np.array([df.iloc[99]]),[0,1,4])]
   scores.append((i,cosine_similarity(vec1, vec2)[0][0]))

#find the top 10 similar users to the active user according to the similarity calculated before
#--> add your Python code here
scores.sort(key=lambda x:x[1], reverse=True)
predictors = scores[:10]

#Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
#--> add your Python code here
total = 0
for value in np.delete(np.array([df.iloc[99]]),[0,1,4]):
   total += value
user_avg = total/8

top_sum = 0
bot_sum = 0
for (x,y) in predictors:
   total = 0
   for value in np.delete(np.array([df.iloc[x]]),[0,1,4]):
       total += value
   avg = total/8
   top_sum += y*(float(df.iloc[x][1])-avg)
   bot_sum += y
predicted_rating = user_avg + top_sum/bot_sum
print("Predicted rating for galleries = %.2f" % predicted_rating)

top_sum = 0
bot_sum = 0
for (x,y) in predictors:
   total = 0
   for value in np.delete(np.array([df.iloc[x]]),[0,1,4]):
       total += value
   avg = total/8
   top_sum += y*(float(df.iloc[x][4])-avg)
   bot_sum += y
predicted_rating = user_avg + top_sum/bot_sum
print("Predicted rating for restaurants = %.2f" % predicted_rating)


