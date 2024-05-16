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
from scipy import stats

#RandomActivation: Activates each agent once per step, 
#in random order, with the order reshuffled every step.

# CONSUMER CLASS
class Consumer(Agent):
    def __init__(self, unique_id, model, tech_pref, jurisdiction, ext_brown, ext_green, intensity):#, tech_preference):
        super().__init__(unique_id, model)
        self.cons_tech_preference = tech_pref #random.choices(['brown', 'green'], weights=[5, 5])[0]
        self.benefit = 0.5 #random.uniform(0, 1)  # can make it random for heterogeneous agents
        self.ext_brown = ext_brown#random.uniform(0, 1) 
        self.ext_green = ext_green
        self.price_green = 0.5
        self.price_brown = 0.5
        self.payoff = 0
        self.jurisdiction = jurisdiction #random.choice([1,2]) # every agent belongs to either jurisdiction
        self.intensity = intensity

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
            return (1 + np.exp(-self.intensity * (payoff_other_cons - payoff_cons))) ** - 1
                                                       

    def __str__(self):
        return f"Consumer {self.unique_id}"
    


# PRODUCER CLASS
class Producer(Agent):
    def __init__(self, unique_id, model, tech_pref, jurisdiction, cost_brown, cost_green, tax,intensity):#, tech_preference):
        super().__init__(unique_id, model)
        self.prod_tech_preference = tech_pref #random.choices(['brown', 'green'], weights=[5, 5])[0]
        self.cost_brown = cost_brown
        self.cost_green = cost_green
        self.tax = tax
        self.fixed_cost = 0 
        self.price_brown = 0.5
        self.price_green = 0.5
        self.payoff = 0
        self.jurisdiction =  jurisdiction #random.choice([1,2]) # every agent belongs to either jurisdiction
        self.intensity = intensity

    def produce(self):
        return random.choice(['brown', 'green']) # change to self.prod_tech_preference when swithcing is possible
    
    
    def prod_payoff(self, tech_preference, jurisdiction):
        if tech_preference == 'green':
            price = self.price_green
            cost = self.cost_green
        elif tech_preference == 'brown':
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
            return (1 + np.exp(-self.intensity * (payoff_other_prod - payoff_prod))) ** - 1

    
    def __str__(self):
        return f"Producer {self.unique_id}"
    


# JURISDICTION CLASS
class Jurisdiction(Model):

    def __init__(self, n_producers, n_consumers,alpha,beta,gamma, cost_brown, cost_green, ext_brown, ext_green, 
                 tax, intensity_c, intensity_p, init_c1, init_c2, init_p1, init_p2):

        self.tax = tax
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.intensity_c = intensity_c 
        self.intensity_p = intensity_p
        self.init_c1 = init_c1
        self.init_c2 = init_c2
        self.init_p1 = init_p1
        self.init_p2 = init_p2
        
        

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

        # create more controlled initial conditions where first X are green. 
        # Create consumers
        for i in range(n_consumers):
            jurisdiction = 1 if i < (n_consumers * 0.5) else 2
            tech_pref = 'green' if (i <= n_consumers * self.init_c1 or i >= n_consumers * self.init_c2) else 'brown'
            consumer = Consumer(i, self, tech_pref, jurisdiction, ext_brown, ext_green, intensity_c)
            self.schedule.add(consumer)
        
        # Create producers
        for i in range(n_producers):
            jurisdiction = 1 if i < (n_producers * 0.5) else 2
            tech_pref = 'green' if (i <= n_producers * self.init_p1 or i >= n_producers * self.init_p2) else 'brown'
            producer = Producer(n_consumers + i, self, tech_pref, jurisdiction, cost_brown, cost_green, tax, intensity_p)
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
        # print(self.n_consumers_j1, self.n_producers_j1)
        # print(self.n_consumers_j2, self.n_producers_j2)
       
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
                             "Total Green Producers J1": "total_green_producers_j1",
                             "Total Green Producers J2": "total_green_producers_j2",
                             "Total Green Consumers J1": "total_green_consumers_j1",
                             "Total Green Consumers J2": "total_green_consumers_j2",

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

                             "Total adoption producers": "total_adoption_producers",
                             "Total adoption consumers": "total_adoption_consumers",
                             "Total adoption J1": "total_adoption_j1",
                             "Total adoption J2": "total_adoption_j2",

                             "Sales brown J1": "sales_brown_j1",
                             "Sales green J1": "sales_green_j1",
                             "Sales brown J2": "sales_brown_j2",
                             "Sales green J2": "sales_green_j2",

                             "brown payoff producers J1": "J1_brown_payoff",
                             "green payoff producers J1": "J1_green_payoff",

                             "brown payoff consumers J1": "J1_brown_payoff_c",
                             "green payoff consumers J1": "J1_green_payoff_c",

                             "externality":"externality",
                             "welfare jurisdiction 1": "welfare_juris1",
                             "welfare jurisdiction 2": "welfare_juris2"})
        
    
    def step(self):

        self.trading_cycle(self.alpha)
        self.datacollector.collect(self)

    
    def trading_cycle(self,alpha):

        # every time we call the step function, we set the total products in the market to 0
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
        self.sales_brown_j1 = 0
        self.sales_green_j1 = 0
        self.sales_brown_j2 = 0
        self.sales_green_j2 = 0


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


        # producer payoff, later deduce payoff of producers that havent sold
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

        self.total_adoption_producers = (self.total_green_producers_j1 + self.total_green_producers_j2) / self.n_producers
        self.total_adoption_consumers = (self.total_green_consumers_j1 + self.total_green_consumers_j2) / self.n_consumers
        self.total_adoption_j1 = (self.total_green_producers_j1 + self.total_green_consumers_j1) / (self.n_producers_j1 + self.n_consumers_j1)
        self.total_adoption_j2 = (self.total_green_producers_j2 + self.total_green_consumers_j2) / (self.n_producers_j2 + self.n_consumers_j2)


        # print(self.total_brown_consumers_j1)
        # print(self.total_brown_consumers_j2)
        # print(self.total_green_consumers_j1)
        # print(self.total_green_consumers_j2)
        # Consumers have to buy in random order, also random between jurisdictions
        shuffled_consumers = list(self.consumers)
        random.shuffle(shuffled_consumers)

        # Consumers buy one product each if possible
        for agent in shuffled_consumers:
            #print(agent)
            product_color = agent.cons_tech_preference
            juris = agent.jurisdiction

            #Global depletion system
            # if product_color == 'brown' and (self.total_brown_products_j1 != 0 or self.total_brown_products_j2 != 0):
            #     #print('green')
            #     if juris == 1:
            #         #print('brown 1')
            #         prob_j1 = self.total_brown_products_j1 / (self.total_brown_products_j1 + alpha * self.total_brown_products_j2)
            #         #prob_j2 =  alpha * self.total_brown_products_j2 / (self.total_brown_products_j1 + alpha * self.total_brown_products_j2)
            #         if prob_j1 > random.random():
            #             self.total_brown_products_j1 -= 1
            #         else:
            #             self.total_brown_products_j2 -= 1
            #     elif juris == 2:
            #         #print('brown 2')
            #         prob_j2 = self.total_brown_products_j2 / (alpha* self.total_brown_products_j1 + self.total_brown_products_j2)
            #         if prob_j2 > random.random():
            #             self.total_brown_products_j2 -= 1
            #         else:
            #             self.total_brown_products_j1 -= 1

            #     agent.payoff = agent.cons_payoff(agent.cons_tech_preference)

            # elif product_color == 'green' and (self.total_green_products_j1 != 0 or self.total_green_products_j2 != 0):
            #     #print('brown')
            #     if juris == 1:
            #         #print('green 1')
            #         prob_j1 = self.total_green_products_j1 / (self.total_green_products_j1 + alpha * self.total_green_products_j2)
            #         #prob_j2 =  alpha * self.total_brown_products_j2 / (self.total_brown_products_j1 + alpha * self.total_brown_products_j2)
            #         if prob_j1 > random.random():
            #             self.total_green_products_j1 -= 1
            #         else:
            #             self.total_green_products_j2 -= 1
            #     elif juris == 2:
            #         #print('green 2')
            #         prob_j2 = self.total_green_products_j2 / (alpha* self.total_green_products_j1 + self.total_green_products_j2)
            #         if prob_j2 > random.random():
            #             self.total_green_products_j2 -= 1
            #         else:
            #             self.total_green_products_j1 -= 1

            #     agent.payoff = agent.cons_payoff(agent.cons_tech_preference)

            # else:
            #     agent.payoff = 0
                #print('zero payoff')
                

        #     # this code is for first depleting local supply
            if product_color == 'brown':
                if self.total_brown_products_j1 > 0 and juris == 1:
                    #print('i have bought brown in j1')
                    self.total_brown_products_j1 -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer in J1 is able to buy brown
                    self.brown_externality_j1 += agent.ext_brown
                elif self.total_brown_products_j2 > 0 and juris == 2:
                    #print('i have bought brown in j2')
                    self.total_brown_products_j2 -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer in J2 is able to buy brown
                    self.brown_externality_j2 += agent.ext_brown
                else:
                    agent.payoff = 0 # consumer is not able to buy
                    #print('i coudldnt buy')
            elif product_color == 'green':
                #self.total_green_consumers += 1
                if self.total_green_products_j1 > 0 and juris == 1:
                    #print('i have bought green in j1')
                    self.total_green_products_j1 -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer in J1 is able to buy green
                    self.green_externality_j1 += agent.ext_green
                elif self.total_green_products_j2 > 0 and juris == 2:
                    #print('i have bought green in j2')
                    self.total_green_products_j2 -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference) # consumer in J2 is able to buy green
                    self.green_externality_j2 += agent.ext_green
                else:
                    agent.payoff = 0 # consumer not able to buy
                    #print('i coudldnt buy')
           # print('next')
        
        if self.alpha > 0:
            #print('yes')
            #Let global consumers buy what is left in the other jurisdiction
            if self.total_brown_products_j1 > 0:
                poss_cons_j2b = [agent for agent in self.consumers_j2 if agent.payoff == 0 and agent.cons_tech_preference == 'brown']
                amount_j2b = int(alpha * len(poss_cons_j2b))
                self.subset_j2b = random.sample(poss_cons_j2b, amount_j2b)
                if len(self.subset_j2b) != 0:
                    for cons in self.subset_j2b:
                        if self.total_brown_products_j1 == 0:
                            break  
                        self.total_brown_products_j1 -= 1
                        cons.payoff = cons.cons_payoff(cons.cons_tech_preference)
                        
            if self.total_brown_products_j2 > 0:
                poss_cons_j1b = [agent for agent in self.consumers_j1 if agent.payoff == 0 and agent.cons_tech_preference == 'brown']
                amount_j1b = int(alpha * len(poss_cons_j1b))
                self.subset_j1b = random.sample(poss_cons_j1b, amount_j1b)
                if len(self.subset_j1b) != 0:
                    for cons in self.subset_j1b:
                        if self.total_brown_products_j2 == 0:
                            break  
                        self.total_brown_products_j2 -= 1
                        cons.payoff = cons.cons_payoff(cons.cons_tech_preference)

            if self.total_green_products_j1 > 0:
                poss_cons_j2g = [agent for agent in self.consumers_j2 if agent.payoff == 0 and agent.cons_tech_preference == 'green']
                amount_j2g = int(alpha * len(poss_cons_j2g))
                self.subset_j2g = random.sample(poss_cons_j2g, amount_j2g)
                if len(self.subset_j2g) != 0:
                    for cons in self.subset_j2g:
                        if self.total_green_products_j1 == 0:
                            break  
                        self.total_green_products_j1 -= 1
                        cons.payoff = cons.cons_payoff(cons.cons_tech_preference)

            if self.total_green_products_j2 > 0:
                poss_cons_j1g = [agent for agent in self.consumers_j1 if agent.payoff == 0 and agent.cons_tech_preference == 'green']
                amount_j1g = int(alpha * len(poss_cons_j1g))
                self.subset_j1g = random.sample(poss_cons_j1g, amount_j1g)
                if len(self.subset_j1g) != 0:
                    for cons in self.subset_j1g:
                        if self.total_green_products_j2 == 0:
                            break  
                        self.total_green_products_j2 -= 1
                        cons.payoff = cons.cons_payoff(cons.cons_tech_preference)


        # after consumers have bought, subtract from payoff of random producers that havent sold
                    
        if self.total_brown_products_j1 > 0: # check if we have to perform the subtraction for brown J1
            #print('brown J1:', self.total_brown_products_j1)
            brown_producers_j1 = [agent for agent in self.producers_j1 if agent.prod_tech_preference == 'brown']
            selected_producers_b1 = random.sample(brown_producers_j1, self.total_brown_products_j1)
            for prod in selected_producers_b1:
                prod.payoff -= (prod.price_brown - prod.tax) # they dont pay tax if they dont sell the product?

        if self.total_brown_products_j2 > 0: # check if we have to perform the subtraction for brown J2
            #print('brown J2:', self.total_brown_products_j2)
            brown_producers_j2 = [agent for agent in self.producers_j2 if agent.prod_tech_preference == 'brown']
            selected_producers_b2 = random.sample(brown_producers_j2, self.total_brown_products_j2)
            for prod in selected_producers_b2:
                prod.payoff -= prod.price_brown # only tax for jurisdiction 1

        if self.total_green_products_j1 > 0: # check if we have to perform the subtraction for green J1
            #print('green J1:', self.total_green_products_j1)
            green_producers_j1 = [agent for agent in self.producers_j1 if agent.prod_tech_preference == 'green']
            selected_producers_g1 = random.sample(green_producers_j1, self.total_green_products_j1)
            for prod in selected_producers_g1:
                prod.payoff -= prod.price_green

        if self.total_green_products_j2 > 0: # check if we have to perform the subtraction for green J2
            #print('green J2:', self.total_green_products_j2)
            green_producers_j2 = [agent for agent in self.producers_j2 if agent.prod_tech_preference == 'green']
            selected_producers_g2 = random.sample(green_producers_j2, self.total_green_products_j2)
            for prod in selected_producers_g2:
                prod.payoff -= prod.price_green
      


        self.sales_brown_j1 = self.total_brown_producers_j1 - self.total_brown_products_j1
        self.sales_green_j1 = self.total_green_producers_j1 - self.total_green_products_j1
        self.sales_brown_j2 = self.total_brown_producers_j2 - self.total_brown_products_j2
        self.sales_green_j2 = self.total_green_producers_j2 - self.total_green_products_j2 


        # SWITCHING SYSTEM 1 
                
        #Compare payoff to random producer and save data for switching
        #we dont need first term of every factor term
        # prod_probs = {}
        # prod_factor_j1_bg = (self.total_green_producers_j1 + self.beta * self.total_green_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2)
        # prod_factor_j1_gb = ((self.total_brown_producers_j1 + self.beta * self.total_brown_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2))
        # prod_factor_j2_bg = ((self.total_green_producers_j2 + self.beta * self.total_green_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1))
        # prod_factor_j2_gb = ((self.total_brown_producers_j2 + self.beta * self.total_brown_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1))
        # for prod in self.producers:
        #     #other_producers = [pr for pr in self.producers if pr != prod] 
        #     if prod.jurisdiction == 2:
        #         other_prod = random.choice(self.producers_j2)
        #         if prod.prod_tech_preference == 'brown':
        #             factor_p = prod_factor_j2_bg
        #         else:
        #             factor_p = prod_factor_j2_gb

        #     elif prod.jurisdiction == 1:
        #         other_prod = random.choice(self.producers_j1)
        #         if prod.prod_tech_preference == 'brown':
        #             factor_p = prod_factor_j1_bg
        #         else:
        #             factor_p = prod_factor_j1_gb

        #    # print('prod', factor_p, prod.prod_switch(other_prod), factor_p * prod.prod_switch(other_prod))
        #     prod_probs[prod] = (factor_p * prod.prod_switch(other_prod), other_prod.prod_tech_preference)  #(prod.payoff - other_prod.payoff, other_prod.prod_tech_preference) # change to probability later

        # # Do the actual producer switching
        # for prod, probs in prod_probs.items():
        #     number = random.random()
        #     #print(probs[0], number)
        #     if probs[0] > number: 
        #         prod.prod_tech_preference = probs[1]


        # # Compare payoff to random consumer and save data for switching 
        # cons_probs = {}
        # cons_factor_j1_bg = ((self.total_green_consumers_j1 + self.gamma * self.total_green_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        # cons_factor_j1_gb = ((self.total_brown_consumers_j1 + self.gamma * self.total_brown_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        # cons_factor_j2_bg = ((self.total_green_consumers_j2 + self.gamma * self.total_green_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        # cons_factor_j2_gb = ((self.total_brown_consumers_j2 + self.gamma * self.total_brown_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        # for cons in self.consumers:
        #     if cons.jurisdiction == 2:
        #         other_cons = random.choice(self.consumers_j2)
        #         if cons.cons_tech_preference == 'brown':
        #             factor_c = cons_factor_j2_bg
                    
        #         else:
        #             factor_c = cons_factor_j2_gb

        #     elif cons.jurisdiction == 1:
        #         other_cons = random.choice(self.consumers_j1)
        #         if cons.cons_tech_preference == 'brown':
        #             factor_c = cons_factor_j1_bg
                    
        #         else:
        #             factor_c = cons_factor_j1_gb

        #    # print('cons', factor_c, cons.cons_switch(other_cons), factor_c * cons.cons_switch(other_cons))
        #     cons_probs[cons] = (factor_c * cons.cons_switch(other_cons), other_cons.cons_tech_preference)
        #    # cons.cons_switch(other_cons)
            
        # # Do the actual consumer switching
        # for cons, probs in cons_probs.items():
        #     number = random.random()
        #     #print(probs[0], number)
        #     if probs[0] > number:
        #         cons.cons_tech_preference = probs[1]




        # SWITCHING SYSTEM 2
        # Calculate average payoffs for producers for each tech per Jurisdiction
        self.J1_brown_payoff = 0 
        self.J1_green_payoff = 0
        self.J2_brown_payoff = 0
        self.J2_green_payoff = 0

        for prod in self.producers:
            if prod.jurisdiction == 1:
                if prod.prod_tech_preference == 'brown':
                    self.J1_brown_payoff += prod.payoff
                else:
                    self.J1_green_payoff += prod.payoff
            elif prod.jurisdiction ==2:
                if prod.prod_tech_preference == 'brown':
                    self.J2_brown_payoff += prod.payoff
                else:
                    self.J2_green_payoff += prod.payoff

        self.J1_brown_payoff = self.J1_brown_payoff / self.total_brown_producers_j1 if self.total_brown_producers_j1 != 0 else 0
        #print('j1p brown:',self.J1_brown_payoff)
        self.J1_green_payoff = self.J1_green_payoff / self.total_green_producers_j1 if self.total_green_producers_j1 != 0 else 0
        #print('j1p green:',self.J1_green_payoff)
        self.J2_brown_payoff = self.J2_brown_payoff / self.total_brown_producers_j2 if self.total_brown_producers_j2 != 0 else 0
        #print('j2p brown',self.J2_brown_payoff)
        self.J2_green_payoff = self.J2_green_payoff / self.total_green_producers_j2 if self.total_green_producers_j2 != 0 else 0

        # print("J1 brown", self.J1_brown_payoff)
        # print("J1 green", self.J1_green_payoff)
        
        # Producers switch
        prod_factor_j1_bg = (self.total_green_producers_j1 + self.beta * self.total_green_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2)
        prod_factor_j1_gb = (self.total_brown_producers_j1 + self.beta * self.total_brown_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2)
        prod_factor_j2_bg = (self.total_green_producers_j2 + self.beta * self.total_green_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1)
        prod_factor_j2_gb = (self.total_brown_producers_j2 + self.beta * self.total_brown_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1)
        prod_probs = {}
        for prod in self.producers:
            # other_prod = random.choice(self.producers)
            # if prod.prod_tech_preference == other_prod.prod_tech_preference:
            #     continue #prod_probs[prod] = (0, other_prod.prod_tech_preference)

            #else:
            if prod.jurisdiction == 2:
                if prod.prod_tech_preference == 'brown':
                    factor_p = prod_factor_j2_bg
                    payoff_compare = self.J2_green_payoff # You are brown, other prod is green
                    your_group = self.J2_brown_payoff
                else:
                    factor_p = prod_factor_j2_gb
                    payoff_compare = self.J2_brown_payoff 
                    your_group = self.J2_green_payoff

            elif prod.jurisdiction == 1:
                if prod.prod_tech_preference == 'brown':
                    factor_p = prod_factor_j1_bg
                    payoff_compare = self.J1_green_payoff 
                    your_group = self.J1_brown_payoff
                else:
                    factor_p = prod_factor_j1_gb
                    payoff_compare = self.J1_brown_payoff 
                    your_group = self.J1_green_payoff
            
            if factor_p > random.random():
                switch_to = 'brown' if prod.prod_tech_preference == 'green' else 'green'
                prob_p = (1 + np.exp(- self.intensity_p * (payoff_compare - prod.payoff))) ** - 1 # use your_group or prod.payoff
                prod_probs[prod] = (prob_p, switch_to)# (factor_p * prob_p, switch_to) 
            
        # Do the actual producer switching
        for prod, probs in prod_probs.items():
            number = random.random()
            if probs[0] > number: 
                prod.prod_tech_preference = probs[1]


        # Calculate average payoffs for consumers for each tech per Jurisdiction
        self.J1_brown_payoff_c = 0 
        self.J1_green_payoff_c = 0
        self.J2_brown_payoff_c = 0
        self.J2_green_payoff_c = 0

        for cons in self.consumers:
            if cons.jurisdiction == 1:
                if cons.cons_tech_preference == 'brown':
                    self.J1_brown_payoff_c += cons.payoff
                else:
                    self.J1_green_payoff_c += cons.payoff
            elif cons.jurisdiction == 2:
                if cons.cons_tech_preference == 'brown':
                    self.J2_brown_payoff_c += cons.payoff
                else:
                    self.J2_green_payoff_c += cons.payoff

        self.J1_brown_payoff_c = self.J1_brown_payoff_c / self.total_brown_consumers_j1 if self.total_brown_consumers_j1 != 0 else 0
        #print('j1c brown', self.J1_brown_payoff)
        self.J1_green_payoff_c = self.J1_green_payoff_c / self.total_green_consumers_j1 if self.total_green_consumers_j1 != 0 else 0
        #print('j1c green', self.J1_green_payoff)
        self.J2_brown_payoff_c = self.J2_brown_payoff_c / self.total_brown_consumers_j2 if self.total_brown_consumers_j2 != 0 else 0
        self.J2_green_payoff_c = self.J2_green_payoff_c / self.total_green_consumers_j2 if self.total_green_consumers_j2 != 0 else 0

        # Consumers switch
        cons_factor_j1_bg = ((self.total_green_consumers_j1 + self.gamma * self.total_green_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        cons_factor_j1_gb = ((self.total_brown_consumers_j1 + self.gamma * self.total_brown_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        cons_factor_j2_bg = ((self.total_green_consumers_j2 + self.gamma * self.total_green_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        cons_factor_j2_gb = ((self.total_brown_consumers_j2 + self.gamma * self.total_brown_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        
        cons_probs = {}
        for cons in self.consumers:
            # other_cons = random.choice(self.consumers)
            # if cons.cons_tech_preference == other_cons.cons_tech_preference:
            #     cons_probs[cons] = (0, other_cons.cons_tech_preference)

            # else:
            if cons.jurisdiction == 2:
                if cons.cons_tech_preference == 'brown':
                    factor_c = cons_factor_j2_bg
                    payoff_compare = self.J2_green_payoff_c # you are brown, other cons is green
                    your_group = self.J2_brown_payoff_c
                else:
                    factor_c = cons_factor_j2_gb
                    payoff_compare = self.J2_brown_payoff_c
                    your_group = self.J2_green_payoff_c


            elif cons.jurisdiction == 1:
                if cons.cons_tech_preference == 'brown':
                    factor_c = cons_factor_j1_bg
                    payoff_compare = self.J1_green_payoff_c
                    your_group = self.J1_brown_payoff_c
                else:
                    factor_c = cons_factor_j1_gb
                    payoff_compare = self.J1_brown_payoff_c
                    your_group = self.J1_green_payoff_c


            if factor_c > random.random():
                switch_to = 'brown' if cons.cons_tech_preference == 'green' else 'green'
                prob_c = (1 + np.exp(- self.intensity_c * (payoff_compare - cons.payoff))) ** - 1 #use your_group or cons.payoff
                cons_probs[cons] = (prob_c, switch_to)

            # make switching faster?
            # if factor_c < random.random():
            #     continue # 
            # else:
            #     do calculation

           #prob_c = (1 + np.exp(- self.intensity * (payoff_compare - cons.payoff))) ** - 1
            #cons_probs[cons] = (factor_c * prob_c, switch_to) 
            
        # Do the actual consumer switching
        for cons, probs in cons_probs.items():
            number = random.random()
            #print(probs[0], number)
            if probs[0] > number:
                cons.cons_tech_preference = probs[1]


        # WELFARE PER JURISDICTION
        self.welfare_juris1 = (self.total_brown_producers_j1 * self.J1_brown_payoff) + (self.total_green_producers_j1 * self.J1_green_payoff) \
                        + (self.total_brown_consumers_j1 * self.J1_brown_payoff_c) + (self.total_green_consumers_j1 * self.J1_green_payoff_c) \
                        + (self.sales_brown_j1 * self.tax)  #- 0.2 * (self.sales_brown_j1 + len(self.subset_j1b) + self.sales_brown_j1 + len(self.subset_j2b))
       
        #print(self.welfare_juris1)
        
        self.welfare_juris2 = self.total_brown_producers_j2 * self.J2_brown_payoff + self.total_green_producers_j2 * self.J2_green_payoff \
                        + self.total_brown_consumers_j2 * self.J2_brown_payoff_c + self.total_green_consumers_j2 * self.J2_green_payoff_c
                        #- 0.1 * (self.sales_brown_j1 + len(self.subset_j1b) + self.sales_brown_j1 + len(self.subset_j2b))
                        # - global externality
        #print(self.welfare_juris2)


        #random shock factor. Every time step X% of the agents change preference.....
        # shock = 0.01 #change to model parameter later
        # if shock > 0:
        #     change_prod = random.sample(self.producers, int(shock * len(self.producers)))
        #     change_cons = random.sample(self.consumers, int(shock * len(self.consumers)))
        #     for prod in change_prod:
        #         if prod.prod_tech_preference == 'brown':
        #             prod.prod_tech_preference = 'green'
        #         else:
        #             prod.prod_tech_preference = 'brown'

        #     for cons in change_cons:
        #         if cons.cons_tech_preference == 'brown':
        #             cons.cons_tech_preference = 'green'
        #         else:
        #             cons.cons_tech_preference = 'brown'
            

        #super().__init__()


    #def jurisdiction_welfare(brown_prod, green_prod, brown_cons, green_cons, payoff_bp, payoff_gp, ):

       

        #welfare = perc_green_c + perc_green_p + perc_brown_c + perc_brown_p
        #return welfare
    
    # def jurisdiction_sales():
    #     return 



# RUN MODEL AND PRINT OUTPUTS
if __name__ == "__main__":

    ############# SINGLE RUN

    model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=1, gamma=1, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
                         tax=0.2, intensity_c=10, intensity_p=10, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)
    # rcs = []
    # current_adopt = 10
    for i in tqdm(range(200)):
        model.step()
    #     new_adopt = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1] * 100
    #     if current_adopt == 0 and new_adopt == 0:
    #         change = 0
    #     if current_adopt == 0:
    #         change = new_adopt
    #     else:
    #         change = ((new_adopt - current_adopt) / current_adopt) * 100
    #     #print(change)
    #     rcs.append(change)
    #     current_adopt = new_adopt

    # print(np.mean(rcs))


    # Retrieve and plot data collected by the DataCollector
    model_data = model.datacollector.get_model_vars_dataframe()

    plt.figure(figsize=(7, 4))

    #plt.plot(model_data['welfare jurisdiction 1'], label='welfare J1')
    #plt.plot(model_data['welfare jurisdiction 2'], label='welfare J2')

    plt.plot(model_data['Percentage green Producers J1'], label='Producers J1', color='indianred')
    plt.plot(model_data['Percentage green Consumers J1'], label='Consumers J1', color='darkred')
    plt.plot(model_data['Percentage green Producers J2'], label='Producers J2', color='deepskyblue')
    plt.plot(model_data['Percentage green Consumers J2'], label='Consumers J2', color='royalblue')

    # plt.plot(model_data["brown payoff producers J1"], label='producers O', color='deepskyblue')
    # plt.plot(model_data["green payoff producers J1"], label='producers N', color='royalblue')

    # plt.plot(model_data["brown payoff consumers J1"], label='consumers O', color='indianred')
    # plt.plot(model_data["green payoff consumers J1"], label='consumers N', color='darkred')

    #plt.title('Adoption of green tech')
    plt.xlabel('Time')
    #plt.ylim(-0.25, 0.35) 
    #plt.ylabel('Average payoff')
    plt.ylabel("Adoption rate of N")
    plt.legend()
    #plt.xticks(range(0, len(model_data)), map(int, model_data.index))

    plt.tight_layout()
    plt.show()




    ############## SALES VS PREFERENCE

    # model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=0.3, gamma=0.3, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
    #                      tax=0.25, intensity=5)
    # for i in tqdm(range(100)):
    #     model.step()

    # # Retrieve and plot data collected by the DataCollector
    # model_data = model.datacollector.get_model_vars_dataframe()

    # plt.figure(figsize=(7, 4))

    # #plt.plot(model_data['Total Green Producers J1'], label='Green Producers J1')
    # #plt.plot(model_data['Sales green J1'], label='Sales of green J1')
    # #plt.plot(model_data['Total Green Consumers J1'], label='Green Consumers J1')
    # plt.plot((model_data['Total Green Consumers J1'] - model_data['Sales green J1']) / model_data['Total Green Consumers J1'], label='J1')
    # plt.plot((model_data['Total Green Consumers J2'] - model_data['Sales green J2']) / model_data['Total Green Consumers J2'], label='J2')
    # #plt.plot(model_data['Percentage green Consumers J2'], label='Percentage Green Consumers J2', color='royalblue')
    # plt.title('Percentage green consumers unable to buy')
    # plt.xlabel('Time')
    # plt.ylabel('Amount')
    # plt.legend()
    # #plt.xticks(range(0, len(model_data)), map(int, model_data.index))

    # plt.tight_layout()
    # plt.show()



    ########## ALPHA/BETA/GAMMA/tax HEATMAPS

    # beta_vals = np.linspace(0,1,11)
    # gamma_vals = np.linspace(0,1,11)
    # alpha_vals = np.linspace(0,1,11)
    # tax_vals = np.linspace(0,0.5,11)
    
    # adoption_J1P = np.zeros((len(gamma_vals), len(beta_vals)))
    # adoption_J1C = np.zeros((len(gamma_vals), len(beta_vals)))
    # adoption_J2P = np.zeros((len(gamma_vals), len(beta_vals)))
    # adoption_J2C = np.zeros((len(gamma_vals), len(beta_vals)))

    # for i, beta in tqdm(enumerate(beta_vals)):
    #     for j, gamma in enumerate(gamma_vals):
    #         results_J1P = []
    #         results_J1C = []
    #         results_J2P = []
    #         results_J2C = []
    #         for k in range(10):  
    #             model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=beta, gamma=gamma, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
    #                                  tax=0.2, intensity_c=1, intensity_p=1, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)
    #             for l in range(200):  
    #                 model.step()
    #                 current_1 =  model.datacollector.get_model_vars_dataframe()['Percentage green Producers J1'].iloc[-1]
    #                 current_2 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 if (current_1 == 0 or current_1 == 1) and (current_2 == 0 or current_2 == 1) :
    #                     break

    #             model_data =  model.datacollector.get_model_vars_dataframe()
    #             results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
    #             results_J1C.append(model_data['Percentage green Consumers J1'].iloc[-1])
    #             results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])
    #             results_J2C.append(model_data['Percentage green Consumers J2'].iloc[-1])

    #         adoption_J1P[j,i] = np.mean(results_J1P)
    #         adoption_J1C[j,i] = np.mean(results_J1C)
    #         adoption_J2P[j,i] = np.mean(results_J2P)
    #         adoption_J2C[j,i] = np.mean(results_J2C)

    # adoption_J1P = np.flipud(adoption_J1P)
    # adoption_J1C = np.flipud(adoption_J1C)
    # adoption_J2P = np.flipud(adoption_J2P)
    # adoption_J2C = np.flipud(adoption_J2C)
    # fig, axs = plt.subplots(1, 4, figsize=(8, 2))

    # im1 = axs[0].imshow(adoption_J1P, cmap='gray_r', extent=[min(beta_vals), max(beta_vals), min(gamma_vals), max(gamma_vals)],
    #                     vmin=0, vmax=1)  
    # axs[0].set_title('Producers J1')
    # axs[0].set_xlabel('Beta') 
    # axs[0].set_ylabel('Gamma')
  

    # im2 = axs[1].imshow(adoption_J1C, cmap='gray_r', extent=[min(beta_vals), max(beta_vals), min(gamma_vals), max(gamma_vals)],
    #                     vmin=0, vmax=1)  
    # axs[1].set_title('Consumers J1')
    # axs[1].set_xlabel('Beta') 

    # im3 = axs[2].imshow(adoption_J2P, cmap='gray_r', extent=[min(beta_vals), max(beta_vals), min(gamma_vals), max(gamma_vals)],
    #                     vmin=0, vmax=1)  
    # axs[2].set_title('Producers J2')  
    # axs[2].set_xlabel('Beta') 

    # im4 = axs[3].imshow(adoption_J2C, cmap='gray_r', extent=[min(beta_vals), max(beta_vals), min(gamma_vals), max(gamma_vals)],
    #                     vmin=0, vmax=1)  
    # axs[3].set_title('Consumers J2') 
    # axs[3].set_xlabel('Beta') 
    # # Add colorbar axis
    # # cax = fig.add_axes([0.96, 0.15, 0.01, 0.7])  # [left, bottom, width, height]

    # # Add colorbar
    # # cbar = fig.colorbar(im4, cax=cax)

    # # cbar = fig.colorbar(im4, ax=axs, fraction=0.05, pad=0.05, location='right')

    # # Set label for the colorbar
    # # cbar.set_label('Adoption rate')
    # plt.tight_layout()
    # plt.show()



    ################# ADOPTION AS A FUNCTION OF TAX/ALPHA/BETA

    # tax_values = np.linspace(0.05, 0.5, num=20)
    # alpha_vals = np.linspace(0.01,1,num=10)
    # beta_vals = np.linspace(0,1,num=20)
    # gamma_vals = np.linspace(0,1,num=20)
    # # Dictionary to store the results
    # average_results_J1P = {}
    # average_results_J1C = {}
    # average_results_J2P = {}
    # average_results_J2C = {}
    # for tax in tqdm(tax_values):
    #     results_J1P = []
    #     results_J1C = []
    #     results_J2P = []
    #     results_J2C = []
    #     for i in range(30):  
    #         model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=3/3, gamma=3/3, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
    #                              tax=tax, intensity_c=50, intensity_p=50, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)
    #         for j in range(200):  
    #             model.step()

    #         model_data =  model.datacollector.get_model_vars_dataframe()
    #         results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
    #         results_J1C.append(model_data['Percentage green Consumers J1'].iloc[-1])
    #         results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])
    #         results_J2C.append(model_data['Percentage green Consumers J2'].iloc[-1])

    #     average_results_J1P[tax] = np.mean(results_J1P)
    #     average_results_J1C[tax] = np.mean(results_J1C)
    #     average_results_J2P[tax] = np.mean(results_J2P)
    #     average_results_J2C[tax] = np.mean(results_J2C)

    # plt.plot(average_results_J1P.keys(), average_results_J1P.values(), label = 'Jurisdiction 1') 
    # plt.plot(average_results_J2P.keys(), average_results_J2P.values(), label= 'Jurisdiction 2')
    # plt.xlabel('Alpha')
    # plt.ylabel('Adoption Rate')
    # plt.legend()
    # plt.ylim(-0.1, 1.1) 

    # plt.tight_layout()
    # plt.show()


    # fig, axs = plt.subplots(2)
    # axs[0].plot(average_results_J1P.keys(), average_results_J1P.values(), label='J1')
    # axs[0].plot(average_results_J2P.keys(), average_results_J2P.values(), label='J2')
    # axs[0].set_title('Producers')
    # axs[0].set_xlabel('Alpha') 
    # axs[0].set_ylabel('Adoption rate of green')
    # axs[0].set_ylim(-0.1,1.1)
    # axs[0].legend()

    # axs[1].plot(average_results_J1C.keys(), average_results_J1C.values(), label='J1')
    # axs[1].plot(average_results_J2C.keys(), average_results_J2C.values(), label='J2')
    # axs[1].set_title('Consumers')
    # axs[1].set_xlabel('Alpha') 
    # axs[1].set_ylabel('Adoption rate of green')
    # axs[1].set_ylim(-0.1,1.1)
    # axs[1].legend()

    # plt.tight_layout()
    # plt.show()


    #################### Min TAX FOR ADOPTION COST OF GREEN
    #tax_values = np.linspace(0, 0.35, num=15)
#     beta_values = np.linspace(0,1,num=25)
#     cost_b_vals = np.linspace(0.1,0.45,num=15)
#     cost_g_vals = np.linspace(0.2,0.50,num=10)
#     rat_vals = [1,5,10,100]
#     tolerance = 0.005
#     adoption_level = 0.5

#     rat1 = dict()
#     rat5 = dict()
#     rat10 = dict()
#     rat100 = dict()
#     for rat in rat_vals:
#         for cg in tqdm(cost_g_vals):
#             min_tax = 0
#             max_tax = 0.5
#             while abs(max_tax - min_tax) > tolerance:
#                 mid_tax = (min_tax + max_tax) / 2

#                 results_J2P = []
#                 for i in range(10):  
#                     model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=0.5, gamma=0.5, cost_brown=0.25, cost_green=cg, 
#                                          ext_brown=0.1, ext_green=0.3, tax=mid_tax,  intensity_c=rat, intensity_p=rat, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)
#                     for j in range(200):  
#                         model.step()

#                     model_data =  model.datacollector.get_model_vars_dataframe()
#                     results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])

                
#                 if np.mean(results_J2P) < adoption_level:
#                     min_tax = mid_tax
#                 else:
#                     max_tax = mid_tax
            
#             if max_tax != 0.5:
#                 if rat == 1:
#                     rat1[cg] = max_tax
#                 elif rat == 5:
#                     rat5[cg] = max_tax
#                 elif rat == 10:
#                     rat10[cg] = max_tax
#                 else:
#                     rat100[cg] = max_tax
#                 #break # we dont need to check for more tax values for this cost value

#     plt.plot(rat1.keys(), rat1.values(), label='rat1')
#     plt.plot(rat5.keys(), rat5.values(), label='rat5')
#     plt.plot(rat10.keys(), rat10.values(), label='rat10')
#     plt.plot(rat100.keys(), rat100.values(), label='rat100')

#     tax_diagonal = cost_g_vals - 0.25
#     plt.plot(cost_g_vals, tax_diagonal, linestyle='--', color='black', label='costB + tax = costG')
#     #y = [0.25 - x_val for x_val in cost_g_vals]  # Calculate y values based on the sum of x and y equaling 0.45
#     #plt.plot(cost_b_vals, y, color='black', linestyle='--', label='costB + tax = costG')

#     plt.xlabel('Cost Green')
#     plt.ylabel('Tax')
#     plt.ylim(0, 0.5) 
#    # plt.title('Tax Values for Different Cost Brown (rat5 and rat10)')
#     plt.legend()
#     plt.show()


######## Min TAX FOR ADOPTION COST OF GREEN with std dev
#     cost_g_vals = np.linspace(0.2,0.5,num=10)
#     rat_vals = [1,5,10,100]
#     tolerance = 0.005
#     adoption_level = 0.5
#     runs_per_tax = 5

#     rat1 = dict()
#     rat5 = dict()
#     rat10 = dict()
#     rat100 = dict()
#     for rat in rat_vals:
#         for cg in tqdm(cost_g_vals):
#             for run in range(runs_per_tax):
#                 min_tax = 0
#                 max_tax = 0.5
#                 while abs(max_tax - min_tax) > tolerance:
#                     mid_tax = (min_tax + max_tax) / 2

#                     results_J2P = []
#                     for i in range(10):  
#                         model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=1, gamma=1, cost_brown=0.25, cost_green=cg, 
#                                             ext_brown=0.1, ext_green=0.3, tax=mid_tax,  intensity_c=rat, intensity_p=rat, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)
#                         last_10_values = []
#                         for j in range(200):  
#                             model.step()
#                             current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]

#                             last_10_values.append(current)
#                             if len(last_10_values) > 10:
#                                 last_10_values.pop(0)
#                             current_sum = sum(last_10_values)
#                             if current_sum == 0 or current_sum == 10:
#                                 break
                            
#                             # if rat != 100:
#                             #     if current == 0 or current == 1:
#                             #         break
#                             #implement that we need x in a row to be 0 or mean rate of change

#                         model_data =  model.datacollector.get_model_vars_dataframe()
#                         results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])

                    
#                     if np.mean(results_J2P) < adoption_level:
#                         min_tax = mid_tax
#                     else:
#                         max_tax = mid_tax
            
#                 if max_tax != 0.5:
#                     if rat == 1:
#                         if cg not in rat1:
#                             rat1[cg] = [max_tax] 
#                         else:
#                             rat1[cg].append(max_tax) 
#                     elif rat == 5:
#                         if cg not in rat5:
#                             rat5[cg] = [max_tax] 
#                         else:
#                             rat5[cg].append(max_tax) 
#                     elif rat == 10:
#                         if cg not in rat10:
#                             rat10[cg] = [max_tax] 
#                         else:
#                             rat10[cg].append(max_tax) 
#                     else:
#                         if cg not in rat100:
#                             rat100[cg] = [max_tax] 
#                         else:
#                             rat100[cg].append(max_tax) 
#                 #break # we dont need to check for more tax values for this cost value

#     keys = list(rat10.keys())

# # Plot each dictionary separately
#     for dic, label in [(rat1, 'rat1'), (rat5, 'rat5'), (rat10, 'rat10'), (rat100, 'rat100')]:
#     #for dic, label in [(rat5, 'rat5'),(rat10, 'rat10'), (rat100, 'rat100')]:
#         means = [np.mean(dic[key]) for key in keys]
#         std_devs = [np.std(dic[key]) for key in keys]
#         plt.errorbar(keys, means, yerr=std_devs, label=label, fmt='-o', capsize=5)

#     tax_diagonal = cost_g_vals - 0.25
#     plt.plot(cost_g_vals, tax_diagonal, linestyle='--', color='black', label='costB + tax = costG')

#     # Customize plot labels and title
#     plt.xlabel('Cost Green')
#     plt.ylabel('Tax')
#     plt.ylim(0, 0.5)
#     plt.grid(True)
#     #plt.xticks(keys)  # Assuming keys are numeric
#     plt.legend()
#     plt.show()





    # #################### MIN TAX FOR X% ADOPTION

    # # #tax_values = np.linspace(0.15, 0.30, num=15)
    # beta_values = np.linspace(0,1,num=11)
    # #alpha_values = np.linspace(0,1,num=11)
    
    # rat_vals = [1,5,10,100]
    # tolerance = 0.005
    # adoption_level = 0.5

    # #DICTS
    # rat1P1 = dict()
    # rat1P2 = dict()
    # rat5P1 = dict()
    # rat5P2 = dict()
    # rat10P1 = dict()
    # rat10P2 = dict()
    # rat100P1 = dict()
    # rat100P2 = dict()


    # for rat in rat_vals:
    #     for beta in tqdm(beta_values):
    #         min_tax = 0
    #         max_tax = 0.5
    #         while abs(max_tax - min_tax) > tolerance:
    #             mid_tax = (min_tax + max_tax) / 2

    #             results_P1 = []
    #             #results_P2 = []
    #             # results_C1 = []
    #             # results_C2 = []
    #             for i in range(10):  
    #                 model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=beta, gamma=1/3, cost_brown=0.25, cost_green=0.45, 
    #                                      ext_brown=0.1, ext_green=0.3, tax=mid_tax, intensity_c=rat, intensity_p=rat, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)
    #                 for j in range(200):  
    #                     model.step()

    #                 model_data =  model.datacollector.get_model_vars_dataframe()
    #                 results_P1.append(model_data['Percentage green Producers J2'].iloc[-1])
    #                 #results_P2.append(model_data['Percentage green Producers J2'].iloc[-1])
    #                 # results_C1.append(model_data['Percentage green Producers J2'].iloc[-1])
    #                 # results_C2.append(model_data['Percentage green Consumers J2'].iloc[-1])

    #             if np.mean(results_P1) < adoption_level:
    #                 min_tax = mid_tax
    #             else:
    #                 max_tax = mid_tax

    #         if max_tax != 0.5:
    #             if rat == 1:
    #                 if beta not in rat1P1:
    #                     rat1P1[beta] = max_tax
    #             elif rat == 5:
    #                 if beta not in rat5P1:  # Make sure to add only 1 value per beta value and not overwrite
    #                     rat5P1[beta] = max_tax
    #             elif rat == 10:
    #                 if beta not in rat10P1:
    #                     rat10P1[beta] = max_tax
    #             else:
    #                 if beta not in rat100P1:
    #                     rat100P1[beta] = max_tax
                        
    #             #     if rat == 1:
    #             #         if beta not in rat1P2:
    #             #             rat1P2[beta] = tax
    #             #     elif rat == 5:
    #             #         if beta not in rat5P2:  # Make sure to add only 1 value per beta value and not overwrite
    #             #             rat5P2[beta] = tax
    #             #     elif rat == 10:
    #             #         if beta not in rat10P2:
    #             #             rat10P2[beta] = tax
    #             #     else:
    #             #         if beta not in rat100P2:
    #             #             rat100P2[beta] = tax

    #             # if np.mean(results_P1) >= adopt_level and np.mean(results_P2) >= adopt_level:
    #             #     break

    
    # plt.plot(rat1P1.keys(), rat1P1.values(), label='rat1')
    # plt.plot(rat5P1.keys(), rat5P1.values(), label='rat5')
    # plt.plot(rat10P1.keys(), rat10P1.values(), label='rat10')
    # plt.plot(rat100P1.keys(), rat100P1.values(), label='rat100')
    # plt.axhline(y=0.2, color='black', linestyle='--', label='tax + cost brown = cost green')
    # plt.xlabel('Beta')
    # plt.xlim(0,1)
    # plt.ylabel('Tax')
    # plt.ylim(0, 0.5)
    # #plt.title('Min tax for', adoption_level, 'adoption')
    # plt.legend()
    # plt.show()

    # fig, axes = plt.subplots(1, 2)
    # #axes[0].plot(rat1P1.keys(), rat1P1.values(), label='rat1 Cons J1')
    # axes[0].plot(rat5P1.keys(), rat5P1.values(), label='rat5 cons J1')
    # axes[0].plot(rat10P1.keys(), rat10P1.values(), label='rat10 cons J1')
    # #axes[0].plot(rat100P1.keys(), rat100P1.values(), label='rat100 Cons J1')
    # axes[0].set_xlabel('Beta')
    # axes[0].set_ylabel('Tax')
    # axes[0].legend()

    # #axes[1].plot(rat1P2.keys(), rat1P2.values(), label='rat1 Cons J2')
    # axes[1].plot(rat5P2.keys(), rat5P2.values(), label='rat5 cons J2')
    # axes[1].plot(rat10P2.keys(), rat10P2.values(), label='rat10 cons J2')
    # #axes[1].plot(rat100P2.keys(), rat100P2.values(), label='rat100 Cons J2')
    # axes[1].set_xlabel('Beta')
    # axes[1].set_ylabel('Tax')
    # axes[1].legend()

    # plt.tight_layout()
    # plt.show()




    ################### MIN tax for adoption with STD

    beta_values = np.linspace(0,1,num=11)
    #alpha_values = np.linspace(0,1,num=11)
    
    rat_vals = [1,5,10,100]
    tolerance = 0.005
    adoption_level = 0.5
    runs_per_tax = 5

    #DICTS
    rat1P1 = dict()
    rat1P2 = dict()
    rat5P1 = dict()
    rat5P2 = dict()
    rat10P1 = dict()
    rat10P2 = dict()
    rat100P1 = dict()
    rat100P2 = dict()


    for rat in rat_vals:
        for beta in tqdm(beta_values):
            for run in range(runs_per_tax):
                min_tax = 0
                max_tax = 0.5
                while abs(max_tax - min_tax) > tolerance:
                    mid_tax = (min_tax + max_tax) / 2

                    results_P1 = []
                    #results_P2 = []
                    # results_C1 = []
                    # results_C2 = []
                    for i in range(10):  
                        model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=beta, gamma=1/3, cost_brown=0.25, cost_green=0.45, 
                                            ext_brown=0.1, ext_green=0.3, tax=mid_tax, intensity_c=rat, intensity_p=rat, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)
                        last_10_values = []
                        for j in range(200):  
                            model.step()
                            current = model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J2'].iloc[-1]

                            last_10_values.append(current)
                            if len(last_10_values) > 10:
                                last_10_values.pop(0)
                            current_sum = sum(last_10_values)
                            if current_sum == 0 or current_sum == 10:
                                break

                        model_data =  model.datacollector.get_model_vars_dataframe()
                        results_P1.append(model_data['Percentage green Producers J2'].iloc[-1])
                        #results_P2.append(model_data['Percentage green Producers J2'].iloc[-1])
                        # results_C1.append(model_data['Percentage green Producers J2'].iloc[-1])
                        # results_C2.append(model_data['Percentage green Consumers J2'].iloc[-1])

                    if np.mean(results_P1) < adoption_level:
                        min_tax = mid_tax
                    else:
                        max_tax = mid_tax

                if max_tax != 0.5:
                    if rat == 1:
                        if beta not in rat1P1:
                            rat1P1[beta] = [max_tax]
                        else:
                            rat1P1[beta].append(max_tax)
                    elif rat == 5:
                        if beta not in rat5P1:  # Make sure to add only 1 value per beta value and not overwrite
                            rat5P1[beta] = [max_tax]
                        else:
                            rat5P1[beta].append(max_tax)
                    elif rat == 10:
                        if beta not in rat10P1:
                            rat10P1[beta] = [max_tax]
                        else:
                            rat10P1[beta].append(max_tax)
                    else:
                        if beta not in rat100P1:
                            rat100P1[beta] = [max_tax] 
                        else:
                            rat100P1[beta].append(max_tax)
                        

                #     if rat == 1:
                #         if beta not in rat1P2:
                #             rat1P2[beta] = tax
                #     elif rat == 5:
                #         if beta not in rat5P2:  # Make sure to add only 1 value per beta value and not overwrite
                #             rat5P2[beta] = tax
                #     elif rat == 10:
                #         if beta not in rat10P2:
                #             rat10P2[beta] = tax
                #     else:
                #         if beta not in rat100P2:
                #             rat100P2[beta] = tax

                # if np.mean(results_P1) >= adopt_level and np.mean(results_P2) >= adopt_level:
                #     break

    
    keys_rat1P1 = list(rat1P1.keys())
    keys_rat5P1 = list(rat5P1.keys())
    keys_rat10P1 = list(rat10P1.keys())
    keys_rat100P1 = list(rat100P1.keys())

    # Plot each dictionary separately
    plt.errorbar(keys_rat1P1, [np.mean(rat1P1[key]) for key in keys_rat1P1],
                yerr=[np.std(rat1P1[key]) for key in keys_rat1P1],
                label='rat1', fmt='-o', capsize=5)
    plt.errorbar(keys_rat5P1, [np.mean(rat5P1[key]) for key in keys_rat5P1],
                yerr=[np.std(rat5P1[key]) for key in keys_rat5P1],
                label='rat5', fmt='-o', capsize=5)
    plt.errorbar(keys_rat10P1, [np.mean(rat10P1[key]) for key in keys_rat10P1],
                yerr=[np.std(rat10P1[key]) for key in keys_rat10P1],
                label='rat10', fmt='-o', capsize=5)
    plt.errorbar(keys_rat100P1, [np.mean(rat100P1[key]) for key in keys_rat100P1],
             yerr=[np.std(rat100P1[key]) for key in keys_rat100P1],
             label='rat100', fmt='-o', capsize=5)

   
    plt.axhline(y=0.2, color='black', linestyle='--', label='tax + cost brown = cost green')
    # Customize plot labels and title
    plt.xlabel('Beta')
    plt.ylabel('Tax')
    plt.ylim(0, 0.5)
    plt.xlim(0,1)
    plt.grid(True)
    #plt.xticks(keys)  # Assuming keys are numeric
    plt.legend()
    plt.show()



    ########## WELFARE AS A FUNCTION OF TAX

#     tax_values = np.linspace(0, 0.4, num=15)
#     # Dictionary to store the results
#     J1_1 = {}
#     J2_1 = {}
#     J1_5 = {}
#     J2_5 = {}
#     J1_10 = {}
#     J2_10 = {}
#     J1_100 = {}
#     J2_100 = {}
#     #diff_5 = {}
#     #diff_10 = {}
#     rats = [1,5,10,100]
#     for rat in rats:
#         for tax_val in tqdm(tax_values):
#             results_J1 = []
#             results_J2 = []
#             #diffs = []
#             for i in range(30):  
#                 model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=0.8, gamma=0.8, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
#                                     tax=tax_val,intensity=rat)
#                 for j in range(200):  
#                     model.step()

#                 model_data =  model.datacollector.get_model_vars_dataframe()
#                 results_J1.append(model_data['welfare jurisdiction 1'].iloc[-1])
#                 results_J2.append(model_data['welfare jurisdiction 2'].iloc[-1])
#                 #print(model_data['welfare jurisdiction 1'].iloc[-1] / model_data['welfare jurisdiction 2'].iloc[-1])
#                 #print(model_data['welfare jurisdiction 2'].iloc[-1])
#                 #diffs.append(model_data['welfare jurisdiction 1'].iloc[-1] / model_data['welfare jurisdiction 2'].iloc[-1])

#            # print(np.mean(diffs))
#             if rat == 1:
#                 J1_1[tax_val] = np.mean(results_J1)
#                 J2_1[tax_val] = np.mean(results_J2)
#             elif rat == 5:
#                 J1_5[tax_val] = np.mean(results_J1)
#                 J2_5[tax_val] = np.mean(results_J2)
#                 #diff_5[tax_val] = np.mean(diffs)
#             elif rat == 10:
#                 J1_10[tax_val] = np.mean(results_J1)
#                 J2_10[tax_val] = np.mean(results_J2)
#             else:
#                 J1_100[tax_val] = np.mean(results_J1)
#                 J2_100[tax_val] = np.mean(results_J2)
#                 #diff_10[tax_val] = np.mean(diffs)
#             #print(diff_5)
#            # print(diff_10)

# #     plt.plot(diff_5.keys(), diff_5.values(), label='rat5')
# #     plt.plot(diff_10.keys(), diff_10.values(), label='rat10')

# #     plt.xlabel('Tax')
# #     plt.ylabel('Welfare J1 / Welfare J2')
# #    # plt.title('Tax Values for Different Cost Brown (rat5 and rat10)')
# #     plt.legend()
# #     plt.show()
    
#     fig, axs = plt.subplots(2)
#     axs[0].plot(J1_1.keys(), J1_1.values(), label='rat=1')
#     axs[0].plot(J1_5.keys(), J1_5.values(), label='rat=5')
#     axs[0].plot(J1_10.keys(), J1_10.values(), label='rat=10')
#     axs[0].plot(J1_100.keys(), J1_100.values(), label='rat=100')
#     #axs[0].plot(diff_5.keys(), diff_5.values(), label='ra=5')
#     axs[0].set_title('Jurisdiction 1')
#     axs[0].set_xlabel('Tax') 
#     axs[0].set_ylabel('Welfare')
#     axs[0].set_ylim(40, 120) 
#     axs[0].legend()

#     axs[1].plot(J2_1.keys(), J2_1.values(), label='rat=1')
#     axs[1].plot(J2_5.keys(), J2_5.values(), label='rat=5')
#     axs[1].plot(J2_10.keys(), J2_10.values(), label='rat=10')
#     axs[1].plot(J2_100.keys(), J2_100.values(), label='rat=100')
#     axs[1].set_title('Jurisdiction 2')
#     axs[1].set_xlabel('Tax') 
#     axs[1].set_ylabel('Welfare')
#     axs[1].set_ylim(40, 120) 
#     axs[1].legend()

#     plt.tight_layout()
#     plt.show()






    ##################### ADOPTION OF AS A FUNCTION OF COST

    # cost_green_values = np.linspace(0.01, 0.5, num=25)
    # # Dictionary to store the results
    # average_results_J1P = {}
    # average_results_J1C = {}
    # average_results_J2P = {}
    # average_results_J2C = {}
    # ci_J1P = {}
    # ci_J1C = {}
    # ci_J2P = {}
    # ci_J2C = {}
    # for cost_g in tqdm(cost_green_values):
    #     results_J1P = []
    #     results_J1C = []
    #     results_J2P = []
    #     results_J2C = []
    #     for i in range(50):  
    #         model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=0, gamma=0, cost_brown=0.25, cost_green=cost_g, ext_brown=0.2, ext_green=0.3, tax=0.2)
    #         for j in range(100):  
    #             model.step()

    #         model_data =  model.datacollector.get_model_vars_dataframe()
    #         results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
    #         results_J1C.append(model_data['Percentage green Consumers J1'].iloc[-1])
    #         results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])
    #         results_J2C.append(model_data['Percentage green Consumers J2'].iloc[-1])

    #     average_results_J1P[cost_g] = np.mean(results_J1P)
    #     average_results_J1C[cost_g] = np.mean(results_J1C)
    #     average_results_J2P[cost_g] = np.mean(results_J2P)
    #     average_results_J2C[cost_g] = np.mean(results_J2C)

    #     # ci_J1P[ext_g] = stats.t.interval(0.95, len(results_J1P) - 1, loc=np.mean(results_J1P), scale=stats.sem(results_J1P))
    #     # ci_J1C[ext_g] = stats.t.interval(0.95, len(results_J1C) - 1, loc=np.mean(results_J1C), scale=stats.sem(results_J1C))
    #     # ci_J2P[ext_g] = stats.t.interval(0.95, len(results_J2P) - 1, loc=np.mean(results_J2P), scale=stats.sem(results_J2P))
    #     # ci_J2C[ext_g] = stats.t.interval(0.95, len(results_J2C) - 1, loc=np.mean(results_J2C), scale=stats.sem(results_J2C))


    # # ci_values_1 = [((v[1] - v[0]) / 2) for v in ci_J1P.values()]
    # # ci_values_2 = [((v[1] - v[0]) / 2) for v in ci_J1C.values()]
    # # ci_values_3 = [((v[1] - v[0]) / 2) for v in ci_J2P.values()]
    # # ci_values_4 = [((v[1] - v[0]) / 2) for v in ci_J2C.values()]

    # fig, axs = plt.subplots(2)
    # axs[0].plot(average_results_J1P.keys(), average_results_J1P.values(), label='J1')
    # axs[0].plot(average_results_J2P.keys(), average_results_J2P.values(), label='J2')
    # #axs[0].errorbar(average_results_J1P.keys(), average_results_J1P.values(), yerr=ci_values_1, fmt='none', capsize=5)
    # #axs[0].errorbar(average_results_J2P.keys(), average_results_J2P.values(), yerr= ci_values_3, fmt='none', capsize=5)
    # axs[0].set_title('Producers')
    # axs[0].set_xlabel('Cost of Green') 
    # axs[0].set_ylabel('Adoption rate of green')
    # axs[0].legend()

    # axs[1].plot(average_results_J1C.keys(), average_results_J1C.values(), label='J1')
    # axs[1].plot(average_results_J2C.keys(), average_results_J2C.values(), label='J2')
    # #axs[1].errorbar(average_results_J1C.keys(), average_results_J1C.values(), yerr=ci_values_2, fmt='none', capsize=5)
    # #axs[1].errorbar(average_results_J2C.keys(), average_results_J2C.values(), yerr=ci_values_4, fmt='none', capsize=5)
    # axs[1].set_title('Consumers')
    # axs[1].set_xlabel('Cost of Green') 
    # axs[1].set_ylabel('Adoption rate of green')
    # axs[1].legend()

    # plt.tight_layout()
    # plt.show()




# PHASE DIAGRAM FOR INITIAL CONDITIONS
    # j1_vals = np.linspace(0,0.5,11)
    # j2_vals = np.linspace(0,0.5,11)
    
    # adoption_J2P = np.zeros((len(j1_vals), len(j2_vals)))

    # for i, j1_val in tqdm(enumerate(j1_vals)):
    #     for j, j2_val in enumerate(j2_vals):
    #         results_J2P = []
    #         for k in range(10):  
    #             model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=0, gamma=0, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
    #                                  tax=0, intensity_c=10, intensity_p=10, init_c1=j1_val, init_c2= 1-j2_val, init_p1=j1_val, init_p2=1-j2_val)
    #             for l in range(200):  
    #                 model.step()

    #             model_data =  model.datacollector.get_model_vars_dataframe()
    #             results_J2P.append(model_data['Percentage green Producers J1'].iloc[-1])

    #         adoption_J2P[j,i] = np.mean(results_J2P)

    # adoption_J2P = np.flipud(adoption_J2P)

    # plt.imshow(adoption_J2P, cmap='gray_r', extent=[min(j1_vals), max(j1_vals), min(j2_vals), max(j2_vals)],
    #            vmin=0, vmax=1)
    # plt.xticks(ticks=plt.xticks()[0], labels=[f'{2*x:.1f}' for x in plt.xticks()[0]])
    # plt.yticks(ticks=plt.yticks()[0], labels=[f'{2*y:.1f}' for y in plt.yticks()[0]])
    # plt.xlabel('J1')
    # plt.ylabel('J2')
  
    # plt.tight_layout()
    # plt.show()  

