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

# CONSUMER CLASS
class Consumer(Agent):
    def __init__(self, unique_id, model):#, tech_preference):
        super().__init__(unique_id, model)
        self.cons_tech_preference = random.choice(['brown', 'green'])
        self.benefit = 10 #random.uniform(0, 1)  # can make it random for heterogeneous agents
        self.ext = 3 #random.uniform(0, 1) 
        self.price_green = 10 
        self.price_brown = 8
        self.payoff = 0

    def buy(self):

        # available_brown_products = self.model.total_brown_products
        # available_green_products = self.model.total_green_products

        # product_color = random.choice(['brown', 'green'])

        # if product_color == 'brown' and available_brown_products > 0:
        #         self.model.total_brown_products -= 1
        # elif product_color == 'green' and available_green_products > 0:
        #         self.model.total_green_products -= 1

        return random.choice(['brown', 'green']) # change to self.cons_tech_preference when switching is possible
    
    
    def cons_payoff(self, tech_preference):
        if tech_preference == 'green':
            price = self.price_green
            ext = self.ext
        else:
            price = self.price_brown
            ext = 0
        self.payoff = - price + self.benefit + ext
        return self.payoff
    
    def cons_switch(self, other_consumer):
        payoff_cons =  self.payoff #self.cons_payoff(self.cons_tech_preference)
        payoff_other_cons = other_consumer.payoff #other_consumer.cons_payoff(other_consumer.cons_tech_preference)
        #print(payoff_cons, payoff_other_cons)
        #print(self.cons_tech_preference)
                                                       
        if payoff_cons < payoff_other_cons:
            #print('switch')
            self.cons_tech_preference = other_consumer.cons_tech_preference
        #print(self.cons_tech_preference)
    
    def __str__(self):
        return f"Consumer {self.unique_id}"
    


# PRODUCER CLASS
class Producer(Agent):
    def __init__(self, unique_id, model):#, tech_preference):
        super().__init__(unique_id, model)
        self.prod_tech_preference = random.choice(['brown', 'green'])
        self.cost_brown = 5
        self.cost_green = 6 
        self.tax = 1  
        self.fixed_cost = 1 
        self.price_brown = 8  
        self.price_green = 10
        self.payoff = 0

    def produce(self):
        return random.choice(['brown', 'green']) # change to self.prod_tech_preference when swithcing is possible
    
    def prod_payoff(self, tech_preference):
        if tech_preference == 'green':
            price = self.price_green
            cost = self.cost_green
            tax = 0 
        else:
            price = self.price_brown
            cost = self.cost_brown
            tax = self.tax
        self.payoff = price - cost - tax - self.fixed_cost
        return self.payoff
    
    def prod_switch(self, other_producer):
        payoff_prod = self.payoff 
        payoff_other_prod = other_producer.payoff
        #payoff_diff = payoff_prod - payoff_other_prod
        # if self.prod_tech_preference == 'brown' and other_producer.prod_tech_preference == 'green':
        #     prob = (1 + np.exp(-1* (payoff_other_prod - payoff_prod))) ** - 1 

        # elif self.prod_tech_preference == 'green' and other_producer.prod_tech_preference == 'brown':
        #     prob = (1 + np.exp(-1* (payoff_prod - payoff_other_prod))) ** - 1

        # else:
        #    prob = 0

        #if prob > random.uniform:
        #    self.prod_tech_preference = other_producer.prod_tech_preference
        #print(payoff_prod, payoff_other_prod)
        #print(self.prod_tech_preference)                                               
        if payoff_prod < payoff_other_prod:
            #print('switch')
            self.prod_tech_preference = other_producer.prod_tech_preference
        #print(self.prod_tech_preference)
    
    def __str__(self):
        return f"Producer {self.unique_id}"
    


# JURISDICTION CLASS
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

        self.consumers = [agent for agent in self.schedule.agents if isinstance(agent, Consumer)]
        self.producers = [agent for agent in self.schedule.agents if isinstance(agent, Producer)]
        #print(self.consumers)
        #print(self.producers)
         
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

        # Random list of producers
        shuffled_producers = list(self.producers)
        random.shuffle(shuffled_producers)

        # Producers produce one product each
        for agent in shuffled_producers:
            product_color = agent.prod_tech_preference
            if product_color == 'brown':
                self.total_brown_products += 1
                self.total_brown_producers += 1
            elif product_color == 'green':
                self.total_green_products += 1
                self.total_green_producers += 1

            agent.payoff = agent.prod_payoff(agent.prod_tech_preference)# update payoff
                #print(agent, product_color, agent.payoff)
        #print('brown producers:',self.total_brown_producers, 'green producers:', self.total_green_producers)
        self.perc_brown_prod = self.total_brown_producers / self.n_producers
        self.perc_green_prod = self.total_green_producers / self.n_producers

        # Random list of producers
        shuffled_consumers = list(self.consumers)
        random.shuffle(shuffled_consumers)

        # Consumers buy one product each
        for agent in shuffled_consumers:
            product_color = agent.cons_tech_preference
            #print(product_color)
            if product_color == 'brown':
                self.total_brown_consumers += 1
                if self.total_brown_products > 0:
                    self.total_brown_products -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer is able to buy brown
                else:
                    agent.payoff = 0 # consumer is not able to buy
            if product_color == 'green':
                self.total_green_consumers += 1
                if self.total_green_products > 0:
                    self.total_green_products -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer is able to buy green
                else:
                    agent.payoff = 0 # consumer not able to buy

        self.perc_brown_cons = self.total_brown_consumers / self.n_consumers
        self.perc_green_cons = self.total_green_consumers / self.n_consumers

        # after consumers have bought, subtract from payoff of random producers that havent sold
        if self.total_brown_products > 0: # check if we have to perform the subtraction for brown
            brown_producers = [agent for agent in self.producers if agent.prod_tech_preference == 'brown']
            selected_producers_b = random.sample(brown_producers, self.total_brown_products)
            for prod in selected_producers_b:
                prod.payoff -= prod.price_brown

        if self.total_green_products > 0: # check if we have to perform the subtraction for green
            green_producers = [agent for agent in self.producers if agent.prod_tech_preference == 'green']
            selected_producers_g = random.sample(green_producers, self.total_green_products)
            for prod in selected_producers_g:
                prod.payoff -= prod.price_green
      

        # MAKE SURE TO NOT UPDATE IMMEDIATELY but after going through all agents
        # Compare payoff and possible switch
        # iterate over producers and consumers, if they switch -> update preference
        prod_payoff_diffs = {}
        for prod in self.producers:

            # if we work with 1000 consumers, can maybe skip this list to save time. can maybe neglect that 1/1000 agent picks himself
            other_producers = [pr for pr in self.producers if pr != prod] 
            other_prod = random.choice(other_producers)

            prod_payoff_diffs[(prod, other_prod)] = prod.payoff - other_prod.payoff # change to probability later

        #print(prod_payoff_diffs)

        for prod, diff in prod_payoff_diffs.items():
            if diff < 0: # change to compare probability with number between 0 and 1
                prod[0].prod_tech_preference = prod[1].prod_tech_preference
            # CAN DO THE SWITCHING HERE OR IN AGENT CLASS
            # print(payoff_prod, payoff_other_prod)
            # print(self.prod_tech_preference) 
            # if prod.prod_payoff(prod.prod_tech_preference) < other_prod.prod_payoff(other_prod.prod_tech_preference):
            #     print('switch')
            #     prod.prod_tech_preference = other_prod.prod_tech_preference

        #    prod.prod_switch(other_prod)

        for cons in self.consumers:
            other_consumers = [co for co in self.consumers if co != cons]
            other_cons = random.choice(other_consumers)

            cons.cons_switch(other_cons)
        
        #super().__init__()


# RUN MODEL AND PRINT OUTPUTS
if __name__ == "__main__":
    model = Jurisdiction(n_consumers=8, n_producers=8, tax=1)
    for i in range(2):
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