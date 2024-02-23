import numpy as np
import pandas as pd
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from numba import jit

#RandomActivation: Activates each agent once per step, 
#in random order, with the order reshuffled every step.

# CONSUMER CLASS
class Consumer(Agent):
    def __init__(self, unique_id, model):#, tech_preference):
        super().__init__(unique_id, model)
        self.cons_tech_preference = random.choices(['brown', 'green'], weights=[5, 5])[0]
        self.benefit = 0.5 #random.uniform(0, 1)  # can make it random for heterogeneous agents
        self.ext_brown = 0.1 #random.uniform(0, 1) 
        self.ext_green = 0.3
        self.price_green = 0.5 
        self.price_brown = 0.5
        self.payoff = 0
        self.jurisdiction = random.choice([1,2]) # every agent belongs to either jurisdiction

    def buy(self):
        return random.choice(['brown', 'green']) # change to self.cons_tech_preference when switching is possible
    
    
    def cons_payoff(self, tech_preference):
        if tech_preference == 'green':
            price = self.price_green
            ext = self.ext_green
        else:
            price = self.price_brown
            ext = self.ext_brown
        self.payoff = - price + self.benefit + ext
        return self.payoff
    
    
    def cons_switch(self, other_consumer):
        payoff_cons =  self.payoff 
        payoff_other_cons = other_consumer.payoff
        if self.cons_tech_preference == other_consumer.cons_tech_preference:
            return 0
        else:
            return (1 + np.exp(-1 * (payoff_other_cons - payoff_cons))) ** - 1
                                                       

    def __str__(self):
        return f"Consumer {self.unique_id}"
    


# PRODUCER CLASS
class Producer(Agent):
    def __init__(self, unique_id, model):#, tech_preference):
        super().__init__(unique_id, model)
        self.prod_tech_preference = random.choices(['brown', 'green'], weights=[5, 5])[0]
        self.cost_brown = 0.25
        self.cost_green = 0.45
        self.tax = 0
        self.fixed_cost = 0 
        self.price_brown = 0.5  
        self.price_green = 0.5
        self.payoff = 0
        self.jurisdiction = random.choice([1,2]) # every agent belongs to either jurisdiction

    def produce(self):
        return random.choice(['brown', 'green']) # change to self.prod_tech_preference when swithcing is possible
    
    
    def prod_payoff(self, tech_preference, jurisdiction):
        if tech_preference == 'green':
            price = self.price_green
            cost = self.cost_green
        else:
            price = self.price_brown
            cost = self.cost_brown
        if tech_preference == 'brown' and jurisdiction == 1:
            tax = self.tax
        else:
            tax = 0
        self.payoff = price - cost - tax - self.fixed_cost
        return self.payoff
    
    
    def prod_switch(self, other_producer):
        payoff_prod = self.payoff 
        payoff_other_prod = other_producer.payoff
        if self.prod_tech_preference == other_producer.prod_tech_preference:
            return 0
        else:
            return (1 + np.exp(-1 * (payoff_other_prod - payoff_prod))) ** - 1

    
    def __str__(self):
        return f"Producer {self.unique_id}"
    


# JURISDICTION CLASS
class Jurisdiction(Model):

    def __init__(self, n_producers, n_consumers,tax,alpha,beta,gamma):

        self.tax = tax
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
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

        self.green_externality_j1 = 0
        self.brown_externality_j1 = 0
        self.green_externality_j2 = 0
        self.brown_externality_j2 = 0

        self.welfare = 0

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

        self.consumers_j1 = [agent for agent in self.consumers if agent.jurisdiction == 1]
        self.consumers_j2 = [agent for agent in self.consumers if agent.jurisdiction == 2]

        self.producers_j1 = [agent for agent in self.producers if agent.jurisdiction == 1]
        self.producers_j2 = [agent for agent in self.producers if agent.jurisdiction == 2]

        self.n_consumers_j1 = len(self.consumers_j1)
        self.n_consumers_j2 = len(self.consumers_j2)

        self.n_producers_j1 = len(self.producers_j1)
        self.n_producers_j2 = len(self.producers_j2)

            
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
                             "Percentage green Consumers": "perc_green_cons",
                             "Percentage brown Producers J1": "perc_brown_prod_j1",
                             "Percentage green Producers J1": "perc_green_prod_j1",
                             "Percentage brown Producers J2": "perc_brown_prod_j2",
                             "Percentage green Producers J2": "perc_green_prod_j2",
                             "Percentage brown Consumers J1": "perc_brown_cons_j1",
                             "Percentage green Consumers J1": "perc_green_cons_j1",
                             "Percentage brown Consumers J2": "perc_brown_cons_j2",
                             "Percentage green Consumers J2": "perc_green_cons_j2",
                             "externality":"externality",
                             "welfare": "welfare"})
        
    
    def step(self):

        self.trading_cycle(self.alpha)
        self.datacollector.collect(self)

    
    def trading_cycle(self,alpha):

        # every time we call the step function, we set the total products in the market to 0
        # need to track this per jurisdiction later...
        self.total_brown_products = 0
        self.total_green_products = 0

        self.total_brown_producers = 0
        self.total_green_producers = 0

        self.total_brown_consumers = 0
        self.total_green_consumers = 0

        self.green_externality_j1 = 0
        self.brown_externality_j1 = 0
        self.green_externality_j2 = 0
        self.brown_externality_j2 = 0
        self.welfare = 0

        # calculate market distribution per jurisdiction
        self.total_brown_products_j1 = len([agent for agent in self.producers_j1 if agent.prod_tech_preference == 'brown'])
        self.total_green_products_j1 = len([agent for agent in self.producers_j1 if agent.prod_tech_preference == 'green'])

        self.total_brown_producers_j1 = len([agent for agent in self.producers_j1 if agent.prod_tech_preference == 'brown'])
        self.total_green_producers_j1 = len([agent for agent in self.producers_j1 if agent.prod_tech_preference == 'green'])

        self.total_brown_consumers_j1 = len([agent for agent in self.consumers_j1 if agent.cons_tech_preference == 'brown'])
        self.total_green_consumers_j1 = len([agent for agent in self.consumers_j1 if agent.cons_tech_preference == 'green'])

        self.total_brown_products_j2 = len([agent for agent in self.producers_j2 if agent.prod_tech_preference == 'brown'])
        self.total_green_products_j2 = len([agent for agent in self.producers_j2 if agent.prod_tech_preference == 'green'])

        self.total_brown_producers_j2 = len([agent for agent in self.producers_j2 if agent.prod_tech_preference == 'brown'])
        self.total_green_producers_j2 = len([agent for agent in self.producers_j2 if agent.prod_tech_preference == 'green'])

        self.total_brown_consumers_j2 = len([agent for agent in self.consumers_j2 if agent.cons_tech_preference == 'brown'])
        self.total_green_consumers_j2 = len([agent for agent in self.consumers_j2 if agent.cons_tech_preference == 'green'])


        for agent in self.producers:
            agent.payoff = agent.prod_payoff(agent.prod_tech_preference, agent.jurisdiction)


        self.perc_brown_prod_j1 = self.total_brown_producers_j1 / self.n_producers_j1
        self.perc_green_prod_j1 = self.total_green_producers_j1 / self.n_producers_j1

        self.perc_brown_prod_j2 = self.total_brown_producers_j2 / self.n_producers_j2
        self.perc_green_prod_j2 = self.total_green_producers_j2 / self.n_producers_j2

        self.perc_brown_cons_j1 = self.total_brown_consumers_j1 / self.n_consumers_j1
        self.perc_green_cons_j1 = self.total_green_consumers_j1 / self.n_consumers_j1

        self.perc_brown_cons_j2 = self.total_brown_consumers_j2 / self.n_consumers_j2
        self.perc_green_cons_j2 = self.total_green_consumers_j2 / self.n_consumers_j2



        # # Producers produce one product each
        # for agent in self.producers:
        #     product_color = agent.prod_tech_preference
        #     if product_color == 'brown':
        #         self.total_brown_products += 1
        #         self.total_brown_producers += 1
        #     elif product_color == 'green':
        #         self.total_green_products += 1
        #         self.total_green_producers += 1

        #     agent.payoff = agent.prod_payoff(agent.prod_tech_preference)# update payoff
        #         #print(agent, product_color, agent.payoff)
        # #print('brown producers:',self.total_brown_producers, 'green producers:', self.total_green_producers)
        # self.perc_brown_prod = self.total_brown_producers / self.n_producers
        # self.perc_green_prod = self.total_green_producers / self.n_producers

        # Consumers have to buy in random order, also random between jurisdictions
        shuffled_consumers = list(self.consumers)
        random.shuffle(shuffled_consumers)

        # Consumers buy one product each if possible
        for agent in shuffled_consumers:
            product_color = agent.cons_tech_preference
            juris = agent.jurisdiction
            if product_color == 'brown' and (self.total_brown_products_j1 != 0 or self.total_brown_products_j2 != 0):
                if juris == 1:
                    #print('brown 1')
                    prob_j1 = self.total_brown_products_j1 / (self.total_brown_products_j1 + alpha * self.total_brown_products_j2)
                    #prob_j2 =  alpha * self.total_brown_products_j2 / (self.total_brown_products_j1 + alpha * self.total_brown_products_j2)
                    if prob_j1 > random.random():
                        self.total_brown_products_j1 -= 1
                    else:
                        self.total_brown_products_j2 -= 1
                if juris == 2:
                    #print('brown 2')
                    prob_j2 = self.total_brown_products_j2 / (alpha* self.total_brown_products_j1 + self.total_brown_products_j2)
                    if prob_j2 > random.random():
                        self.total_brown_products_j2 -= 1
                    else:
                        self.total_brown_products_j1 -= 1

                agent.payoff = agent.cons_payoff(agent.cons_tech_preference)

            elif product_color == 'green' and (self.total_green_products_j1 != 0 or self.total_green_products_j2 != 0):
                if juris == 1:
                    #print('green 1')
                    prob_j1 = self.total_green_products_j1 / (self.total_green_products_j1 + alpha * self.total_green_products_j2)
                    #prob_j2 =  alpha * self.total_brown_products_j2 / (self.total_brown_products_j1 + alpha * self.total_brown_products_j2)
                    if prob_j1 > random.random():
                        self.total_green_products_j1 -= 1
                    else:
                        self.total_green_products_j2 -= 1
                if juris == 2:
                    #print('green 2')
                    prob_j2 = self.total_green_products_j2 / (alpha* self.total_green_products_j1 + self.total_green_products_j2)
                    if prob_j2 > random.random():
                        self.total_green_products_j2 -= 1
                    else:
                        self.total_green_products_j1 -= 1

                agent.payoff = agent.cons_payoff(agent.cons_tech_preference)

            else:
                agent.payoff = 0
                

            # this code is for first depleting local supply
            # if product_color == 'brown':
            #     if self.total_brown_products_j1 > 0 and juris == 1:
            #         self.total_brown_products_j1 -= 1
            #         agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer in J1 is able to buy brown
            #         self.brown_externality_j1 += agent.ext_brown
            #     elif self.total_brown_products_j2 > 0 and juris == 2:
            #         self.total_brown_products_j2 -= 1
            #         agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer in J2 is able to buy brown
            #         self.brown_externality_j2 += agent.ext_brown
            #     else:
            #         agent.payoff = 0 # consumer is not able to buy
            # if product_color == 'green':
            #     #self.total_green_consumers += 1
            #     if self.total_green_products_j1 > 0 and juris == 1:
            #         self.total_green_products_j1 -= 1
            #         agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer in J1 is able to buy green
            #         self.green_externality_j1 += agent.ext_green
            #     elif self.total_green_products_j2 > 0 and juris == 2:
            #         self.total_green_products_j2 -= 1
            #         agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer in J2 is able to buy green
            #         self.green_externality_j2 += agent.ext_green
            #     else:
            #         agent.payoff = 0 # consumer not able to buy
        
        # Let global consumers buy what is left in the other jurisdiction
        # if self.total_brown_products_j2 > 0:
        #     poss_cons_j1b = [agent for agent in self.consumers_j1 if agent.payoff == 0 and agent.cons_tech_preference == 'brown']
        #     amount_j1b = int(alpha * len(poss_cons_j1b))
        #     subset_j1b = random.sample(poss_cons_j1b, amount_j1b)
        #     if len(subset_j1b) != 0:
        #         for cons in subset_j1b:
        #             if self.total_brown_products_j2 == 0:
        #                 break  
        #             self.total_brown_products_j2 -= 1
        #             cons.payoff = cons.cons_payoff(cons.cons_tech_preference)

        # if self.total_brown_products_j1 > 0:
        #     poss_cons_j2b = [agent for agent in self.consumers_j2 if agent.payoff == 0 and agent.cons_tech_preference == 'brown']
        #     amount_j2b = int(alpha * len(poss_cons_j2b))
        #     subset_j2b = random.sample(poss_cons_j2b, amount_j2b)
        #     if len(subset_j2b) != 0:
        #         for cons in subset_j2b:
        #             if self.total_brown_products_j1 == 0:
        #                 break  
        #             self.total_brown_products_j1 -= 1
        #             cons.payoff = cons.cons_payoff(cons.cons_tech_preference)
            

        # if self.total_green_products_j2 > 0:
        #     poss_cons_j1g = [agent for agent in self.consumers_j1 if agent.payoff == 0 and agent.cons_tech_preference == 'green']
        #     amount_j1g = int(alpha * len(poss_cons_j1g))
        #     subset_j1g = random.sample(poss_cons_j1g, amount_j1g)
        #     if len(subset_j1g) != 0:
        #         for cons in subset_j1g:
        #             if self.total_green_products_j2 == 0:
        #                 break  
        #             self.total_green_products_j2 -= 1
        #             cons.payoff = cons.cons_payoff(cons.cons_tech_preference)

        # if self.total_green_products_j1 > 0:
        #     poss_cons_j2g = [agent for agent in self.consumers_j2 if agent.payoff == 0 and agent.cons_tech_preference == 'green']
        #     amount_j2g = int(alpha * len(poss_cons_j2g))
        #     subset_j2g = random.sample(poss_cons_j2g, amount_j2g)
        #     if len(subset_j2g) != 0:
        #         for cons in subset_j2g:
        #             if self.total_green_products_j1 == 0:
        #                 break  
        #             self.total_green_products_j1 -= 1
        #             cons.payoff = cons.cons_payoff(cons.cons_tech_preference)


        # after consumers have bought, subtract from payoff of random producers that havent sold
        if self.total_brown_products_j1 > 0: # check if we have to perform the subtraction for brown J1
            brown_producers = [agent for agent in self.producers_j1 if agent.prod_tech_preference == 'brown']
            selected_producers_b1 = random.sample(brown_producers, self.total_brown_products_j1)
            for prod in selected_producers_b1:
                prod.payoff -= (prod.price_brown - prod.tax) # they dont pay tax if they dont sell the product?

        if self.total_brown_products_j2 > 0: # check if we have to perform the subtraction for brown J2
            brown_producers = [agent for agent in self.producers_j2 if agent.prod_tech_preference == 'brown']
            selected_producers_b2 = random.sample(brown_producers, self.total_brown_products_j2)
            for prod in selected_producers_b2:
                prod.payoff -= prod.price_brown # only tax for jurisdiction 1
                

        if self.total_green_products_j1 > 0: # check if we have to perform the subtraction for green J1
            green_producers = [agent for agent in self.producers_j1 if agent.prod_tech_preference == 'green']
            selected_producers_g1 = random.sample(green_producers, self.total_green_products_j1)
            for prod in selected_producers_g1:
                prod.payoff -= prod.price_green

        if self.total_green_products_j2 > 0: # check if we have to perform the subtraction for green J2
            green_producers = [agent for agent in self.producers_j2 if agent.prod_tech_preference == 'green']
            selected_producers_g2 = random.sample(green_producers, self.total_green_products_j2)
            for prod in selected_producers_g2:
                prod.payoff -= prod.price_green
      

        # SWITCHING SYSTEM
                
        # Compare payoff to random producer and save data for switching
        prod_probs = {}
        prod_factor_j1_bg = self.perc_brown_prod_j1 * ((self.total_green_producers_j1 + self.beta * self.total_green_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2))
        prod_factor_j1_gb = self.perc_green_prod_j2 * ((self.total_brown_producers_j1 + self.beta * self.total_brown_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2))
        prod_factor_j2_bg = self.perc_brown_prod_j2 * ((self.total_green_producers_j2 + self.beta * self.total_green_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1))
        prod_factor_j2_gb = self.perc_green_prod_j2 * ((self.total_brown_producers_j2 + self.beta * self.total_brown_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1))
        for prod in self.producers:
            #other_producers = [pr for pr in self.producers if pr != prod] 
            if prod.jurisdiction == 1:
                other_prod = random.choice(self.producers_j1)
                if prod.prod_tech_preference == 'brown':
                    factor = prod_factor_j1_bg
                else:
                    factor = prod_factor_j1_gb
            elif prod.jurisdiction == 2:
                other_prod = random.choice(self.producers_j2)
                if prod.prod_tech_preference == 'brown':
                    factor = prod_factor_j2_bg
                else:
                    factor = prod_factor_j2_gb

            prod_probs[prod] = (factor * prod.prod_switch(other_prod), other_prod.prod_tech_preference)  #(prod.payoff - other_prod.payoff, other_prod.prod_tech_preference) # change to probability later

        # Do the actual producer switching
        for prod, probs in prod_probs.items():
            number = random.random()
            if probs[0] > number: 
                prod.prod_tech_preference = probs[1]


        # Compare payoff to random consumer and save data for switching 
        cons_probs = {}
        cons_factor_j1_bg = self.perc_brown_cons_j1 * ((self.total_green_consumers_j1 + self.gamma * self.total_green_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        cons_factor_j1_gb = self.perc_green_cons_j1 * ((self.total_brown_consumers_j1 + self.gamma * self.total_brown_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        cons_factor_j2_bg = self.perc_brown_cons_j2 * ((self.total_green_consumers_j2 + self.gamma * self.total_green_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        cons_factor_j2_gb = self.perc_green_cons_j2 * ((self.total_brown_consumers_j2 + self.gamma * self.total_brown_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        for cons in self.consumers:
            if cons.jurisdiction == 1:
                other_cons = random.choice(self.consumers_j1)
                if cons.cons_tech_preference == 'brown':
                    factor = cons_factor_j1_bg
                else:
                    factor = cons_factor_j1_gb
            elif cons.jurisdiction == 2:
                other_cons = random.choice(self.consumers_j2)
                if cons.cons_tech_preference == 'brown':
                    factor = cons_factor_j2_bg
                else:
                    factor = cons_factor_j2_gb


            cons_probs[cons] = (factor * cons.cons_switch(other_cons), other_cons.cons_tech_preference)
           # cons.cons_switch(other_cons)
            
        # Do the actual consumer switching
        for cons, probs in cons_probs.items():
            number = random.random()
            if probs[0] > number:
                cons.cons_tech_preference = probs[1]
        #super().__init__()


    def jurisdiction_welfare(perc_green_p, perc_green_c, payoffs):

        perc_brown_p = 1 - perc_green_p
        perc_brown_c = 1 - perc_green_c

        welfare = perc_green_c + perc_green_p + perc_brown_c + perc_brown_p
        return welfare



# RUN MODEL AND PRINT OUTPUTS
if __name__ == "__main__":
    model = Jurisdiction(n_consumers=1500, n_producers=1500, tax=0, alpha=0.01, beta=0, gamma=0)
    for i in tqdm(range(1000)):
        model.step()

    # Retrieve and plot data collected by the DataCollector
    model_data = model.datacollector.get_model_vars_dataframe()
    #print(model_data)
    # Separate data for brown and green products
    #brown_data = model_data.filter(like='Brown')
    #green_data = model_data.filter(like='Green')

    # Create separate plots for brown and green products
    plt.figure(figsize=(7, 4))

    # Plot brown products
    # plt.subplot(1, 3, 1)
    # for column in brown_data.columns:
    #     plt.plot(brown_data[column], label=column)
    # plt.title('Brown Products')
    # plt.xlabel('Steps')
    # plt.ylabel('Count')
    # plt.legend()
    # #plt.xticks(range(0, len(brown_data)), map(int, brown_data.index))

    # # Plot green products
    # plt.subplot(1, 3, 2)
    # for column in green_data.columns:
    #     plt.plot(green_data[column], label=column)
    # plt.title('Green Products')
    # plt.xlabel('Steps')
    # plt.ylabel('Count')
    # plt.legend()
    #plt.xticks(range(0, len(green_data)), map(int, green_data.index))

    # adoption rates
    # plt.subplot(1, 3, 3)
    #plt.plot(model_data['Percentage brown Producers'], label='Percentage Brown Producers')
    plt.plot(model_data['Percentage green Producers J1'], label='Percentage Green Producers J1', color='indianred')
    plt.plot(model_data['Percentage green Consumers J1'], label='Percentage Green Consumers J1', color='darkred')
    plt.plot(model_data['Percentage green Producers J2'], label='Percentage Green Producers J2', color='deepskyblue')
    plt.plot(model_data['Percentage green Consumers J2'], label='Percentage Green Consumers J2', color='royalblue')
    plt.title('Adoption of green tech')
    plt.xlabel('Steps')
    plt.ylabel('Percentage')
    plt.legend()
    #plt.xticks(range(0, len(model_data)), map(int, model_data.index))

    plt.tight_layout()
    plt.show()