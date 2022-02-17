#####
# Author: Juan Camilo
######
# I always liked simulation(s) both in life as a way to solve problems and in games as a way to pass time.
# This python program is meant to be a simple discrete simulation of a restaurant trying to seat, serve and bill
# clients, with 2 graphs at the end to see what a some basic questions and answers might be coming from the simulation
#
# As a clarification, this project is a proof of concept, there could be many new features or simulations added, such as
# a simulated kitchen, simulated warehouse/inventory systems etc... but this project will try to get a functional
# simulation running and trying to gather insights from it
########

import random
import pandas as pd
import simpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 2277  # I like this number.
NUM_OF_TABLES = 10  # Number of tables in restaurant
WAITERS = NUM_OF_TABLES  # For simplicity we will say we have the same number of waiters as we do tables
AVG_WAIT_TIME = 30  # Minutes it takes to completely serve a table on average
AVG_TIME_BETWEEN_CUSTOMERS = 10  # Create a customer every 10 minutes on average
SIM_TIME = 600  # Simulation time in minutes, 600 being 10 hours. This could be any number of minutes
AVG_CUSTOMER_PATIENCE = 3  # Average amount of minutes a table is willing to wait
TABLES_READY_FOR_CUSTOMERS = max(3, NUM_OF_TABLES // 3)
# number of tables ready for use. This is to account for any sort of delay in tables such as cleaning etc....
# if we do NOT have enough tables ready then people will leave hungry and angry... hangry!
AVERAGE_FOOD_PRICE = 100  # average price per plate
np.random.seed(RANDOM_SEED)  # This helps reproducing the results
event_log = []  # Empty list to gather information about our events!


class Restaurant(object):

    def __init__(self, env, num_tables, wait_time, waiters, ws):
        self.env = env
        self.tables = simpy.Resource(env, num_tables)
        self.wait_time = wait_time
        self.waiters = simpy.Resource(env, waiters)
        self.ws = simpy.Resource(env, ws)

    def feed(self, table, name, ws):
        arrive = env.now
        completion = random.randint(20, 100)
        price = max(15, np.random.normal(AVERAGE_FOOD_PRICE, 50))
        print("%s ARRIVES AT THE TABLE %.2f." % (name, arrive))

        with ws.ws.request() as req:
            patience = max(1, np.random.normal(AVG_CUSTOMER_PATIENCE, 1))
            results = yield req | self.env.timeout(patience)
            wait = env.now - arrive
            if req in results:
                print("Waiters satisfied %d%% of %s's orders.Costumer paid %d" % (completion, table, price))
                event_log.append({
                    "Table": table,
                    "Found_Table": "YES",
                    "Completion": completion,
                    "Paid": price})
                yield self.env.timeout(max(1, np.random.normal(self.wait_time, 20)))
            else:
                print('%7.4f %s: LEFT THE CAMILO RESTAURANT SAD WITHOUT FOOD, DIEGO ATE IT after %6.3f' % (
                    env.now, name, wait))
                event_log.append({
                    "Table": table,
                    "Found_Table": "NO",
                    "Completion": 0,
                    "Paid": 0})


#########
# The above is a class that has the initialization of our restaurant and the feed function, because that's what we do
# in our restaurant, we (try to) feed  people!
# This takes self, a table (described below) and ws (available tables)
#
# We create some variables such as arrive, completion, price, patience, results...
# These help us create a logical check to see if the customer leaves because of no tables or if they can successfully
# take a table or they leave. It also adds the event to the log such that we can later add it to a data frame and
# analyze it to deliver any required insight#
#########


def table(env, name, restaurant):
    print('%s arrives at the restaurant at %.2f.' % (name, env.now))
    with restaurant.tables.request() as request:
        yield request
    with restaurant.waiters.request() as request:
        yield request

        print('%s enters the Restaurant at %.2f.' % (name, env.now))
        yield env.process(restaurant.feed(name, name, restaurant))

        print('%s leaves the Resturant at %.2f.' % (name, env.now))


##########
# This is a table for our restaurant, we use the with, yield and process methods to "send" tables to the "feed" function
# and then the feed function does it's magic! It also prints some messages so that we have some assurance its workings#
######


def setup(env, num_tables, wait_time, time_between_customers, waiters, ws):
    # Create the Resaurant
    restaurant = Restaurant(env, num_tables, wait_time, waiters, ws)

    # Create some starting customers
    for i in range(7):  # lucky 7!
        env.process(table(env, 'Customer %d' % i, restaurant))

        # Create more customers while the simulation is running
        while True:
            yield env.timeout(max(1, np.random.normal(time_between_customers, 5)))
            i += 1
            env.process(table(env, 'Customer %d' % i, restaurant))


##########
# This is our setup function, in here we call the restaurant class and process(create) the starting tables. We then
# keep creating customers as long as the simulation is running. We also label them with the "i" iterator so we can
# know how many customers we generated, etc...
#######

# Setup and start the simulation
print('Restaurant')

########
# Create an environment and start the setup process, and how many iterations we want to run. Iterations could be
# interpreted as business days. We leave the iterator at 10 so we save on computing time, but setting it to 365 could be
# a whole year!
#######

for i in range(1, 10):
    env = simpy.Environment()
    env.process(
        setup(env, NUM_OF_TABLES, AVG_WAIT_TIME, AVG_TIME_BETWEEN_CUSTOMERS, WAITERS, TABLES_READY_FOR_CUSTOMERS))
    env.run(until=SIM_TIME)


########
# Our simulations works~
#######

df = pd.DataFrame(event_log)
data = df.reset_index()
data["Quality"] = "Bad"
data.loc[data["Completion"] > 25, "Quality"] = "Poor"
data.loc[data["Completion"] > 50, "Quality"] = "OK"
data.loc[data["Completion"] > 75, "Quality"] = "Good"
data.loc[data["Completion"] > 90, "Quality"] = "Complete"
data.loc[data["Found_Table"] == "NO", "Quality"] = "Hangry :("

########
# Setting up our data frames and doing some classification on different columns.
########


count_of_eats = df["Found_Table"].value_counts().reset_index()
sns.barplot(data=count_of_eats, y="Found_Table", x="index")
plt.xlabel("Was Customer Properly Seated?")
plt.ylabel("Quantity")
plt.show()


########
# Creating a plot on the data about our customer availability, how many did we properly seat and how many left.
########

quality = data["Quality"].value_counts().reset_index()
sns.barplot(data=quality, x="index", y="Quality")
plt.xlabel("Service Quality")
plt.ylabel("Quantity")
plt.show()


########
# Creating a plot on the data about our customer satisfaction, how many orders we had correct and how many left hangry :(
########

print(round(df["Paid"].values.sum()),2)

########
# Just printing the sum of the amounts paid in the simulation, good to see if it covers fixed costs. This could be put
# in a loop (as any of the variables can) so we could find means, variances and standard deviations within the desired
# scenario
########
