import numpy as np
import pandas as pd
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import random

#RandomActivation: Activates each agent once per step, 
#in random order, with the order reshuffled every step.

class Consumer(Agent):
    def __init__(self, unique_id, model):#, tech_preference):
        super().__init__(unique_id, model)
        self.cons_tech_preference = random.choice(['brown', 'green'])

    def buy(self):

        # available_brown_products = self.model.total_brown_products
        # available_green_products = self.model.total_green_products

        # product_color = random.choice(['brown', 'green'])

        # if product_color == 'brown' and available_brown_products > 0:
        #         self.model.total_brown_products -= 1
        # elif product_color == 'green' and available_green_products > 0:
        #         self.model.total_green_products -= 1

        return random.choice(['brown', 'green'])#self.cons_tech_preference
    
    
    def cons_payoff(self, ):
        return 
    def __str__(self):
        return f"Consumer {self.unique_id}"
    
class Producer(Agent):
    def __init__(self, unique_id, model):#, tech_preference):
        super().__init__(unique_id, model)
        self.prod_tech_preference = random.choice(['brown', 'green'])

    def produce(self):
        return random.choice(['brown', 'green'])#self.prod_tech_preference
    
    def prod_payoff(self):
        return
    
    def __str__(self):
        return f"Producer {self.unique_id}"
    
class Jurisdiction(Model):

    def __init__(self, n_producers, n_consumers,tax):

        self.tax = tax
        
        self.schedule = RandomActivation(self)

        self.n_producers = n_producers
        self.n_consumers = n_consumers

        self.total_brown_products = 0
        self.total_green_products = 0

        self.total_brown_producers = 0
        self.total_green_producers = 0

        self.total_brown_consumers = 0
        self.total_green_consumers = 0

        self.perc_brown_prod = 0
        self.perc_green_prod = 0

        self.perc_brown_cons = 0
        self.perc_green_cons = 0

        # Create consumers
        for i in range(n_consumers):
            #tech_pref = random.choice
            consumer = Consumer(i, self)
            self.schedule.add(consumer)

        # Create producers
        for i in range(n_producers):
            producer = Producer(n_consumers + i, self)
            self.schedule.add(producer)
         
        # trackers
        # adoption rate
        # welfare in jurisdiction
        # externalities

        self.datacollector = DataCollector(model_reporters={"Total Brown Products": "total_brown_products",
                             "Total Green Products": "total_green_products",
                             "Total Brown Producers": "total_brown_producers",
                             "Total Green Producers": "total_green_producers",
                             "Total Brown Consumers": "total_brown_consumers",
                             "Total Green Consumers": "total_green_consumers",
                             "Percentage brown Producers": "perc_brown_prod",
                             "Percentage green Producers": "perc_green_prod",
                             "Percentage brown Consumers": "perc_brown_cons",
                             "Percentage green Consumers": "perc_green_cons"})
        

    def step(self):

        self.trading_cycle()
        self.datacollector.collect(self)


    def trading_cycle(self):
        # every time we call the step function, we set the total products in the market to 0
        self.total_brown_products = 0
        self.total_green_products = 0

        self.total_brown_producers = 0
        self.total_green_producers = 0

        self.total_brown_consumers = 0
        self.total_green_consumers = 0

        # Producers produce one product each
        for agent in self.schedule.agents:
            #print(agent)
            if isinstance(agent, Producer):
                product_color = agent.produce()
                if product_color == 'brown':
                    self.total_brown_products += 1
                    self.total_brown_producers += 1
                elif product_color == 'green':
                    self.total_green_products += 1
                    self.total_green_producers += 1
        
        self.perc_brown_prod = self.total_brown_producers / self.n_producers
        self.perc_green_prod = self.total_green_producers / self.n_producers

        # Consumers buy one product each
        for agent in self.schedule.agents:
            if isinstance(agent, Consumer):
                product_color = agent.buy()
                #print(product_color)
                if product_color == 'brown':
                    self.total_brown_consumers += 1
                if product_color == 'brown' and self.total_brown_products > 0:
                    self.total_brown_products -= 1
                if product_color == 'green':
                    self.total_green_consumers += 1
                if product_color == 'green'and self.total_green_products > 0:
                    self.total_green_products -= 1
                    #self.total_green_consumers += 1

        self.perc_brown_cons = self.total_brown_consumers / self.n_consumers
        self.perc_green_cons = self.total_green_consumers / self.n_consumers
           
        #super().__init__()


# RUN MODEL AND PRINT OUTPUTS
if __name__ == "__main__":
    model = Jurisdiction(n_consumers=10, n_producers=10, tax=1)
    for i in range(5):
        model.step()

    # Retrieve and plot data collected by the DataCollector
    model_data = model.datacollector.get_model_vars_dataframe()
    #print(model_data)
    # Separate data for brown and green products
    brown_data = model_data.filter(like='Brown')
    green_data = model_data.filter(like='Green')

    # Create separate plots for brown and green products
    plt.figure(figsize=(10, 5))

    # Plot brown products
    plt.subplot(1, 3, 1)
    for column in brown_data.columns:
        plt.plot(brown_data[column], label=column)
    plt.title('Brown Products')
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks(range(0, len(brown_data)), map(int, brown_data.index))

    # Plot green products
    plt.subplot(1, 3, 2)
    for column in green_data.columns:
        plt.plot(green_data[column], label=column)
    plt.title('Green Products')
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks(range(0, len(green_data)), map(int, green_data.index))

    # adoption rates
    plt.subplot(1, 3, 3)
    #plt.plot(model_data['Percentage brown Producers'], label='Percentage Brown Producers')
    plt.plot(model_data['Percentage green Producers'], label='Percentage Green Producers')
    #plt.plot(model_data['Percentage brown Consumers'], label='Percentage Brown Consumers')
    plt.plot(model_data['Percentage green Consumers'], label='Percentage Green Consumers')
    plt.title('Adoption of green tech')
    plt.xlabel('Steps')
    plt.ylabel('Percentage')
    plt.legend()
    plt.xticks(range(0, len(model_data)), map(int, model_data.index))

    plt.tight_layout()
    plt.show()