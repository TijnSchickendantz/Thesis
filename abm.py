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
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import warnings
from joblib import Parallel, delayed

#RandomActivation: Activates each agent once per step, 
#in random order, with the order reshuffled every step.

# CONSUMER CLASS
class Consumer(Agent):
    def __init__(self, unique_id, model, tech_pref, jurisdiction, ext_brown, ext_green, intensity):
        super().__init__(unique_id, model)
        self.cons_tech_preference = tech_pref
        self.benefit = 0.5 #random.uniform(0, 1)  # random benefit values for heterogeneous agents
        self.ext_brown = ext_brown
        self.ext_green = ext_green
        self.price_green = 0.5
        self.price_brown = 0.5
        self.payoff = 0
        self.jurisdiction = jurisdiction 
        self.intensity = intensity
    
    
    def cons_payoff(self, tech_preference, jurisdiction):
        """
        Calculate the consumer's payoff

        Parameters:
        - tech_preference (str): The consumer's preference for technology.
        - jurisdiction: 1 or 2 (not currently used in the calculation).

        Returns:
        - float: The calculated payoff
        
        Note: There is commented-out code that suggests future logic for subsidies based on tech preference and jurisdiction.
        """

        if tech_preference == 'green':
            price = self.price_green
            ext = self.ext_green
        elif tech_preference == 'brown':
            price = self.price_brown
            ext = self.ext_brown

        # if tech_preference == 'green' and jurisdiction == 1:
        #     subsidy = 0
        # else:
        #     subsidy = 0

        self.payoff = - price + self.benefit + ext #+ subsidy
        return self.payoff
    

    # def cons_switch(self, other_consumer):
    #     payoff_cons =  self.payoff 
    #     payoff_other_cons = other_consumer.payoff
    #     if self.cons_tech_preference == other_consumer.cons_tech_preference:
    #         return 0
    #     else:
    #         return (1 + np.exp(-self.intensity * (payoff_other_cons - payoff_cons))) ** - 1
                                                       

    def __str__(self):
        return f"Consumer {self.unique_id}"
    


# PRODUCER CLASS
class Producer(Agent):
    def __init__(self, unique_id, model, tech_pref, jurisdiction, cost_brown, cost_green, tax,intensity):#, tech_preference):
        super().__init__(unique_id, model)
        self.prod_tech_preference = tech_pref
        self.cost_brown = cost_brown
        self.cost_green = cost_green
        self.tax = tax
        self.fixed_cost = 0 
        self.price_brown = 0.5
        self.price_green = 0.5
        self.payoff = 0
        self.jurisdiction =  jurisdiction 
        self.intensity = intensity
    
    
    def prod_payoff(self, tech_preference, jurisdiction):
        """
        Calculate the producer's payoff

        Parameters:
        - tech_preference (str): The producer's preference for technology.
        - jurisdiction: 1 or 2

        Returns:
        - float: The calculated payoff
        
        """
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
    
    # def prod_switch(self, other_producer):
    #     payoff_prod = self.payoff 
    #     payoff_other_prod = other_producer.payoff
    #     if self.prod_tech_preference == other_producer.prod_tech_preference:
    #         return 0
    #     else:
    #         return (1 + np.exp(-self.intensity * (payoff_other_prod - payoff_prod))) ** - 1

    
    def __str__(self):
        return f"Producer {self.unique_id}"
    


# JURISDICTION CLASS
class Jurisdiction(Model):
    """
    This class simulates the interactions between producers and consumers within two jurisdictions, 
    tracking the adoption of two technologies.
    """

    def __init__(self, n_producers, n_consumers,alpha,beta,gamma, cost_brown, cost_green, ext_brown, ext_green, 
                 tax, intensity_c, intensity_p, init_c1, init_c2, init_p1, init_p2):
        """
        Initialize the Jurisdiction model with producers and consumers.

        Parameters:
        - n_producers (int): Number of producers in the jurisdiction.
        - n_consumers (int): Number of consumers in the jurisdiction.
        - alpha, beta, gamma (float): Interaction parameters
        - cost_brown, cost_green (float): The costs associated with brown and green technologies, respectively.
        - ext_brown, ext_green (float): Externalities associated with brown and green technologies, respectively.
        - tax (float): Tax rate applied to the brown technology.
        - intensity_c, intensity_p (float): Rationality level affecting consumer and producer behavior.
        - init_c1, init_c2 (float): Initial distribution of consumer adoption rates for green and brown technology.
        - init_p1, init_p2 (float): Initial distribution of producer adoption rates for green and brown technology.
        """

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

       
        # Create consumers
        for i in range(n_consumers):
            jurisdiction = 1 if i < (n_consumers * 0.55) else 2
            tech_pref = 'green' if (i <= n_consumers * self.init_c1 or i >= n_consumers * self.init_c2) else 'brown'
            consumer = Consumer(i, self, tech_pref, jurisdiction, ext_brown, ext_green, intensity_c)
            self.schedule.add(consumer)
        
        # Create producers
        for i in range(n_producers):
            jurisdiction = 1 if i < (n_producers * 0.45) else 2
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

        tot_green_cons_j1 = len([agent for agent in self.consumers_j1 if agent.cons_tech_preference == 'green'])
        tot_green_prods_j1 = len([agent for agent in self.producers_j1 if agent.prod_tech_preference == 'green'])
        tot_green_cons_j2 = len([agent for agent in self.consumers_j2 if agent.cons_tech_preference == 'green'])
        tot_green_prods_j2 = len([agent for agent in self.producers_j2 if agent.prod_tech_preference == 'green'])

        print("initial green consumers in J1:", tot_green_cons_j1)
        print("initial green producers in J1:", tot_green_prods_j1)
        print("initial green consumers in J2:", tot_green_cons_j2)
        print("initial green producers in J2:", tot_green_prods_j2)

        # # Prints to check if we are using the right amount of consumers and producers per jurisdiction for a simulation
        # print('j1:', 'c:', self.n_consumers_j1, 'p:', self.n_producers_j1)
        # print("j2:", 'c', self.n_consumers_j2, 'p:', self.n_producers_j2)
       

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
        """
        Execute one step of the simulation.

        This method performs a single trading and technology switching cycle and collects data for the current state of the model.
        """

        self.trading_cycle(self.alpha)
        self.datacollector.collect(self)

    
    def trading_cycle(self,alpha):
        """
        Execute all processes within a time step.

        This method performs the following steps:
        1. Resets key statistics (e.g., total products, externalities, welfare) to zero.
        2. Calculates the market distribution and number of producers and consumers by technology preference 
        (green or brown) within each jurisdiction.
        3. Calculates the payoff for each producer based on their technology preference and jurisdiction.
        4. Calculates the proportion of producers and consumers using green and brown technologies in each jurisdiction.
        5. Simulates consumer purchasing behavior.
        6. Calculate payoff of each consumer and updates producer payoffs if they are unable to sell their products.
        7. Simulates technology switching behavior for both producers and consumers.


        Note:
        - The method includes logic for both a local first depletion system and a global depletion system, 
        though the global depletion system is currently commented out.
        - This method includes two different switching systems for producers and consumers; 
        the first system is commented out, and the second system is currently used.
        """

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


        # Consumers have to buy in random order, also random between jurisdictions
        shuffled_consumers = list(self.consumers)
        random.shuffle(shuffled_consumers)

        # Consumers buy one product each if possible
        for agent in shuffled_consumers:
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

            #     agent.payoff = agent.cons_payoff(agent.cons_tech_preference, agent.jurisdiction)

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

            #     agent.payoff = agent.cons_payoff(agent.cons_tech_preference, agent.jurisdiction)

            # else:
            #    agent.payoff = 0
                

        #  this code is for first depleting local supply
            if product_color == 'brown':
                if self.total_brown_products_j1 > 0 and juris == 1:
                    self.total_brown_products_j1 -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference, agent.jurisdiction) # consumer in J1 is able to buy brown
                    self.brown_externality_j1 += agent.ext_brown
                elif self.total_brown_products_j2 > 0 and juris == 2:
                    self.total_brown_products_j2 -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference, agent.jurisdiction) # consumer in J2 is able to buy brown
                    self.brown_externality_j2 += agent.ext_brown
                else:
                    agent.payoff = 0 # consumer is not able to buy
            elif product_color == 'green':
                if self.total_green_products_j1 > 0 and juris == 1:
                    self.total_green_products_j1 -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference, agent.jurisdiction) # consumer in J1 is able to buy green
                    self.green_externality_j1 += agent.ext_green
                elif self.total_green_products_j2 > 0 and juris == 2:
                    self.total_green_products_j2 -= 1
                    agent.payoff = agent.cons_payoff(agent.cons_tech_preference, agent.jurisdiction) # consumer in J2 is able to buy green
                    self.green_externality_j2 += agent.ext_green
                else:
                    agent.payoff = 0 # consumer not able to buy
        
        
        # Let global consumers buy what is left in the other jurisdiction. ONLY use this when we use a local market
        if self.alpha > 0:
            if self.total_brown_products_j1 > 0:
                poss_cons_j2b = [agent for agent in self.consumers_j2 if agent.payoff == 0 and agent.cons_tech_preference == 'brown']
                amount_j2b = int(alpha * len(poss_cons_j2b))
                self.subset_j2b = random.sample(poss_cons_j2b, amount_j2b)
                if len(self.subset_j2b) != 0:
                    for cons in self.subset_j2b:
                        if self.total_brown_products_j1 == 0:
                            break  
                        self.total_brown_products_j1 -= 1
                        cons.payoff = cons.cons_payoff(cons.cons_tech_preference, cons.jurisdiction)
                        
            if self.total_brown_products_j2 > 0:
                poss_cons_j1b = [agent for agent in self.consumers_j1 if agent.payoff == 0 and agent.cons_tech_preference == 'brown']
                amount_j1b = int(alpha * len(poss_cons_j1b))
                self.subset_j1b = random.sample(poss_cons_j1b, amount_j1b)
                if len(self.subset_j1b) != 0:
                    for cons in self.subset_j1b:
                        if self.total_brown_products_j2 == 0:
                            break  
                        self.total_brown_products_j2 -= 1
                        cons.payoff = cons.cons_payoff(cons.cons_tech_preference, cons.jurisdiction)

            if self.total_green_products_j1 > 0:
                poss_cons_j2g = [agent for agent in self.consumers_j2 if agent.payoff == 0 and agent.cons_tech_preference == 'green']
                amount_j2g = int(alpha * len(poss_cons_j2g))
                self.subset_j2g = random.sample(poss_cons_j2g, amount_j2g)
                if len(self.subset_j2g) != 0:
                    for cons in self.subset_j2g:
                        if self.total_green_products_j1 == 0:
                            break  
                        self.total_green_products_j1 -= 1
                        cons.payoff = cons.cons_payoff(cons.cons_tech_preference, cons.jurisdiction)

            if self.total_green_products_j2 > 0:
                poss_cons_j1g = [agent for agent in self.consumers_j1 if agent.payoff == 0 and agent.cons_tech_preference == 'green']
                amount_j1g = int(alpha * len(poss_cons_j1g))
                self.subset_j1g = random.sample(poss_cons_j1g, amount_j1g)
                if len(self.subset_j1g) != 0:
                    for cons in self.subset_j1g:
                        if self.total_green_products_j2 == 0:
                            break  
                        self.total_green_products_j2 -= 1
                        cons.payoff = cons.cons_payoff(cons.cons_tech_preference, cons.jurisdiction)



        # After consumers have bought, subtract from payoff of random producers that havent sold
        if self.total_brown_products_j1 > 0: # check if we have to perform the subtraction for brown J1
            brown_producers_j1 = [agent for agent in self.producers_j1 if agent.prod_tech_preference == 'brown']
            selected_producers_b1 = random.sample(brown_producers_j1, self.total_brown_products_j1)
            for prod in selected_producers_b1:
                prod.payoff -= (prod.price_brown - prod.tax) # they dont pay tax if they dont sell the product?

        if self.total_brown_products_j2 > 0: # check if we have to perform the subtraction for brown J2
            brown_producers_j2 = [agent for agent in self.producers_j2 if agent.prod_tech_preference == 'brown']
            selected_producers_b2 = random.sample(brown_producers_j2, self.total_brown_products_j2)
            for prod in selected_producers_b2:
                prod.payoff -= prod.price_brown # only tax for jurisdiction 1

        if self.total_green_products_j1 > 0: # check if we have to perform the subtraction for green J1
            green_producers_j1 = [agent for agent in self.producers_j1 if agent.prod_tech_preference == 'green']
            selected_producers_g1 = random.sample(green_producers_j1, self.total_green_products_j1)
            for prod in selected_producers_g1:
                prod.payoff -= prod.price_green

        if self.total_green_products_j2 > 0: # check if we have to perform the subtraction for green J2
            green_producers_j2 = [agent for agent in self.producers_j2 if agent.prod_tech_preference == 'green']
            selected_producers_g2 = random.sample(green_producers_j2, self.total_green_products_j2)
            for prod in selected_producers_g2:
                prod.payoff -= prod.price_green
      


        self.sales_brown_j1 = self.total_brown_producers_j1 - self.total_brown_products_j1
        self.sales_green_j1 = self.total_green_producers_j1 - self.total_green_products_j1
        self.sales_brown_j2 = self.total_brown_producers_j2 - self.total_brown_products_j2
        self.sales_green_j2 = self.total_green_producers_j2 - self.total_green_products_j2 


        # SWITCHING SYSTEM 1 
                
        # Compare payoff to random producer and save data for switching
    
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
        self.J1_green_payoff = self.J1_green_payoff / self.total_green_producers_j1 if self.total_green_producers_j1 != 0 else 0
        self.J2_brown_payoff = self.J2_brown_payoff / self.total_brown_producers_j2 if self.total_brown_producers_j2 != 0 else 0
        self.J2_green_payoff = self.J2_green_payoff / self.total_green_producers_j2 if self.total_green_producers_j2 != 0 else 0
        
        # Producers switch
        prod_factor_j1_bg = (self.total_green_producers_j1 + self.beta * self.total_green_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2)
        prod_factor_j1_gb = (self.total_brown_producers_j1 + self.beta * self.total_brown_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2)
        prod_factor_j2_bg = (self.total_green_producers_j2 + self.beta * self.total_green_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1)
        prod_factor_j2_gb = (self.total_brown_producers_j2 + self.beta * self.total_brown_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1)
        prod_probs = {}
        for prod in self.producers:
            if random.random() < 0.001: # random technology switching with small probability 
                switch_to = 'brown' if prod.prod_tech_preference == 'green' else 'green'
                prod_probs[prod] = (1, switch_to)
            else:
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
                
                if factor_p > random.random(): # only compute the probability if random numb bigger than factor_p. saves computing time
                    switch_to = 'brown' if prod.prod_tech_preference == 'green' else 'green'
                    prob_p = (1 + np.exp(- self.intensity_p * (payoff_compare - prod.payoff))) ** - 1 # use your_group or prod.payoff
                    prod_probs[prod] = (prob_p, switch_to)  # (factor_p * prob_p, switch_to) this is the old computation where you compute probabilities of all agents
            
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
        self.J1_green_payoff_c = self.J1_green_payoff_c / self.total_green_consumers_j1 if self.total_green_consumers_j1 != 0 else 0
        self.J2_brown_payoff_c = self.J2_brown_payoff_c / self.total_brown_consumers_j2 if self.total_brown_consumers_j2 != 0 else 0
        self.J2_green_payoff_c = self.J2_green_payoff_c / self.total_green_consumers_j2 if self.total_green_consumers_j2 != 0 else 0

        # Consumers switch
        cons_factor_j1_bg = ((self.total_green_consumers_j1 + self.gamma * self.total_green_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        cons_factor_j1_gb = ((self.total_brown_consumers_j1 + self.gamma * self.total_brown_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        cons_factor_j2_bg = ((self.total_green_consumers_j2 + self.gamma * self.total_green_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        cons_factor_j2_gb = ((self.total_brown_consumers_j2 + self.gamma * self.total_brown_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        
        cons_probs = {}
        for cons in self.consumers:
            if random.random() < 0.001: # random technology switching with small probability 
                switch_to = 'brown' if cons.cons_tech_preference == 'green' else 'green'
                cons_probs[cons] = (1, switch_to)
            else:
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
                    prob_c = (1 + np.exp(- self.intensity_c * (payoff_compare - cons.payoff))) ** - 1 # use your_group or cons.payoff
                    cons_probs[cons] = (prob_c, switch_to)

            
        # Do the actual consumer switching
        for cons, probs in cons_probs.items():
            number = random.random()
            if probs[0] > number:
                cons.cons_tech_preference = probs[1]


        # WELFARE PER JURISDICTION.
        # not using this in thesis directly, but I have it in case I ver want to use it

        # self.welfare_juris1 = (self.total_brown_producers_j1 * self.J1_brown_payoff) + (self.total_green_producers_j1 * self.J1_green_payoff) \
        #                 + (self.total_brown_consumers_j1 * self.J1_brown_payoff_c) + (self.total_green_consumers_j1 * self.J1_green_payoff_c) \
        #                 + (self.sales_brown_j1 * self.tax)  #- 0.2 * (self.sales_brown_j1 + len(self.subset_j1b) + self.sales_brown_j1 + len(self.subset_j2b))
        
        # self.welfare_juris2 = self.total_brown_producers_j2 * self.J2_brown_payoff + self.total_green_producers_j2 * self.J2_green_payoff \
        #                 + self.total_brown_consumers_j2 * self.J2_brown_payoff_c + self.total_green_consumers_j2 * self.J2_green_payoff_c
                        #- 0.1 * (self.sales_brown_j1 + len(self.subset_j1b) + self.sales_brown_j1 + len(self.subset_j2b))
                        # - global externality



# RUN MODEL AND PRINT OUTPUTS
if __name__ == "__main__":
    

    #### SINGLE RUN DYNAMICS  
    ### note: the commented out lines serve different desired outputs 

    # initialize list for tracking adoption rates
    all_data_PJ1 = []
    all_data_CJ1 = []
    all_data_PJ2 = []
    all_data_CJ2 = []

    #total_steps = []
    #avg_max_j1 = []
    #avg_max_j2 = []
    
    # do sims amount of simulations
    sims = 1 # if you want sims > 1, can only run a given amount of time steps.
    for i in tqdm(range(sims)):
        model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=0, gamma=0, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
                            tax=0.2, intensity_c=10, intensity_p=10, init_c1=0.055, init_c2=0.955, init_p1=0.045, init_p2=0.945)
        

        # current_adopt = 10
        # max_j1C = []
        # max_j1P = []
        # max_j2P = []
        # max_j2C = []


        # for j in range(1000):
        #     model.step()

        for _ in range(10):
            model.step()
            # max_j1C.append( model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J1'].iloc[-1])
            # max_j2C.append( model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J2'].iloc[-1])


        list_1 = []
        list_2 = []
        list_3 = []
        list_4 = []

        #Fill first list
        for _ in range(30):
            model.step()
           #max_j1C.append( model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J1'].iloc[-1])
           #max_j2C.append( model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J2'].iloc[-1])

            fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J1'].iloc[-1]
            list_1.append(fill_1)
            fill_2 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
            list_3.append(fill_2)

        #step_count = 10 + 30 

        # do model step until simulation has converged
        while True:
            model.step()
            #step_count += 1
           # max_j1C.append( model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J1'].iloc[-1])
           # max_j2C.append( model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J2'].iloc[-1])

            current1 = model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J1'].iloc[-1]
            list_2.append(current1)
            current2 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
            list_4.append(current2)

            if len(list_2) == 30:
                if list_1 == list_2:
                    break

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    t_stat, p_value = ttest_ind(list_1, list_2)
                    t_stat1, p_value1 = ttest_ind(list_3, list_4)

                if p_value > 0.05 and p_value1 > 0.05:
                    break

                list_1 = list_2[:]
                list_2 = []

                list_3 = list_4[:]
                list4 = []

        model_data = model.datacollector.get_model_vars_dataframe()
        all_data_PJ1.append(model_data['Percentage green Producers J1'].values)
        all_data_CJ1.append(model_data['Percentage green Consumers J1'].values)
        all_data_PJ2.append(model_data['Percentage green Producers J2'].values)
        all_data_CJ2.append(model_data['Percentage green Consumers J2'].values)
    
        #print(step_count)
        #total_steps.append(step_count)
        #avg_max_j1.append(np.max(max_j1C))
        #avg_max_j2.append(np.max(max_j2C))
    # print(np.mean(rcs))

    #print(np.mean(total_steps))
    #print(np.mean(avg_max_j1))
    #print(avg_max_j2)
    #print(np.mean(avg_max_j2))

    all_data_PJ1 = np.array(all_data_PJ1)
    all_data_CJ1 = np.array(all_data_CJ1)
    all_data_PJ2 = np.array(all_data_PJ2)
    all_data_CJ2 = np.array(all_data_CJ2)

    # compute means and std devs
    mean_PJ1 = np.mean(all_data_PJ1, axis=0)
    std_PJ1 = np.std(all_data_PJ1, axis=0)
    mean_CJ1 = np.mean(all_data_CJ1, axis=0)
    std_CJ1 = np.std(all_data_CJ1, axis=0)
    mean_PJ2 = np.mean(all_data_PJ2, axis=0)
    std_PJ2 = np.std(all_data_PJ2, axis=0)
    mean_CJ2 = np.mean(all_data_CJ2, axis=0)
    std_CJ2 = np.std(all_data_CJ2, axis=0)
            

    plt.figure(figsize=(7, 4))

    #plt.plot(model_data['welfare jurisdiction 1'], label='welfare J1')
    #plt.plot(model_data['welfare jurisdiction 2'], label='welfare J2')

    # plt.plot(model_data['Percentage green Producers J1'], label='Producers J1', color='indianred')
    # plt.plot(model_data['Percentage green Consumers J1'], label='Consumers J1', color='darkred')
    # plt.plot(model_data['Percentage green Producers J2'], label='Producers J2', color='deepskyblue')
    # plt.plot(model_data['Percentage green Consumers J2'], label='Consumers J2', color='royalblue')

    time_steps = range(len(mean_PJ1))
    #plt.rcParams['figure.dpi'] = 300

    plt.plot(time_steps, mean_PJ1, label='Producers J1', color='indianred')
    plt.fill_between(time_steps, np.clip(mean_PJ1 - std_PJ1, 0, 1), np.clip(mean_PJ1 + std_PJ1, 0, 1), color='indianred', alpha=0.2)

    plt.plot(time_steps, mean_CJ1, label='Consumers J1', color='orange')
    plt.fill_between(time_steps, np.clip(mean_CJ1 - std_CJ1, 0, 1), np.clip(mean_CJ1 + std_CJ1, 0, 1), color='orange', alpha=0.2)

    plt.plot(time_steps, mean_PJ2, label='Producers J2', color='deepskyblue')
    plt.fill_between(time_steps, np.clip(mean_PJ2 - std_PJ2, 0, 1), np.clip(mean_PJ2 + std_PJ2, 0, 1), color='deepskyblue', alpha=0.2)

    plt.plot(time_steps, mean_CJ2, label='Consumers J2', color='royalblue')
    plt.fill_between(time_steps, np.clip(mean_CJ2 - std_CJ2, 0, 1), np.clip(mean_CJ2 + std_CJ2, 0, 1), color='royalblue', alpha=0.2)

    # plt.plot(model_data["brown payoff producers J1"], label='producers O', color='deepskyblue')
    # plt.plot(model_data["green payoff producers J1"], label='producers N', color='royalblue')

    # plt.plot(model_data["brown payoff consumers J1"], label='consumers O', color='indianred')
    # plt.plot(model_data["green payoff consumers J1"], label='consumers N', color='darkred')

    #plt.title('Adoption of green tech')
    plt.xlabel('time steps', fontsize=17)
    plt.ylim(-0.1, 1.1) 
    plt.grid(True)
    #plt.ylabel('Average payoff')
    plt.ylabel("Adoption rate of N", fontsize=16)
    plt.legend(fontsize=15)
    #plt.xticks(range(0, len(model_data)), map(int, model_data.index))

    plt.tight_layout()
    #plt.savefig('D:/tijn2/CS/thesis/result figures/in thesis/single runs/init0.5MIX,dpi=500.png', dpi=500)
    plt.show()



    #### ALPHA/BETA/GAMMA/tax HEATMAPS 1 tax at a time

#     beta_vals = np.linspace(0,1,11)
#     gamma_vals = np.linspace(0,1,11)
#     alpha_vals = np.linspace(0,1,11)
#    # tax_vals = np.linspace(0,0.5,11)
    
#     adoption_J1P = np.zeros((len(gamma_vals), len(beta_vals)))
#     adoption_J1C = np.zeros((len(gamma_vals), len(beta_vals)))
#     adoption_J2P = np.zeros((len(gamma_vals), len(beta_vals)))
#     adoption_J2C = np.zeros((len(gamma_vals), len(beta_vals)))

#     #Iterate over interaction parameters
#     for i, beta in tqdm(enumerate(beta_vals)):
#         for j, gamma in enumerate(alpha_vals):
#             results_J1P = []
#             results_J1C = []
#             results_J2P = []
#             results_J2C = []
            
#             #Do desired amount of simulation per interaction parameter combination
#             for k in range(10):  
#                 model = Jurisdiction(n_consumers=500, n_producers=500, alpha=gamma, beta=beta, gamma=0, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
#                                      tax=0.2, intensity_c=10, intensity_p=10, init_c1=0.04, init_c2=0.94, init_p1=0.06, init_p2=0.96)
                
#                 #let the model run for 10 steps first...
#                 for _ in range(10):
#                     model.step()
                
#                 list_1j1 = []
#                 list_2j1 = []

#                 list_1j2 = []
#                 list_2j2 = []

#                 # fill first list for the first time
#                 for _ in range(30):
#                     model.step()
#                     fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
#                     list_1j1.append(fill_1)
#                     #fill_2 = model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J2'].iloc[-1]
#                     #list_1j2.append(fill_2)

#                 #Model steps until convergence  
#                 while True:  
#                     model.step()
#                     current =  model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
#                     list_2j1.append(current)

#                     #current2 =  model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J2'].iloc[-1]
#                     #list_2j2.append(current2)

#                     if len(list_2j1) == 30:
#                         if list_1j1 == list_2j1: #and list_1j2 == list_2j2:
#                             break

#                         with warnings.catch_warnings():
#                             warnings.simplefilter("ignore", category=RuntimeWarning)
#                             t_stat1, p_value1 = ttest_ind(list_1j1, list_2j1)
#                             #t_stat2, p_value2 = ttest_ind(list_1j2, list_2j2)
#                         #t_stat, p_value = ttest_ind(list_1, list_2)
#                         #u_stat, p_value = mannwhitneyu(list_1, list_2, alternative='two-sided')
            
#                         if p_value1 > 0.05: #and p_value2 > 0.05:  # Means are not statistically different
#                             break
                    
#                         list_1j1 = list_2j1[:]
#                         list_2j1 = []

#                         #list_1j2 = list_2j2[:]
#                         #list_2j2 = []


#                     # current_2 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
#                     # if (current_1 == 0 or current_1 == 1) and (current_2 == 0 or current_2 == 1) :
#                     #     break

#                 model_data =  model.datacollector.get_model_vars_dataframe()
#                 results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
#                 results_J1C.append(model_data['Percentage green Consumers J1'].iloc[-1])
#                 results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])
#                 results_J2C.append(model_data['Percentage green Consumers J2'].iloc[-1])

#             adoption_J1P[j,i] = np.mean(results_J1P)
#             adoption_J1C[j,i] = np.mean(results_J1C)
#             adoption_J2P[j,i] = np.mean(results_J2P)
#             adoption_J2C[j,i] = np.mean(results_J2C)

#     #Flip values to get correct plot
#     adoption_J1P = np.flipud(adoption_J1P)
#     adoption_J1C = np.flipud(adoption_J1C)
#     adoption_J2P = np.flipud(adoption_J2P)
#     adoption_J2C = np.flipud(adoption_J2C)
#     fig, axs = plt.subplots(1, 4, figsize=(10, 2.6))#, dpi=500)

#     im1 = axs[0].imshow(adoption_J1P, cmap='gray_r', extent=[min(beta_vals), max(beta_vals), min(alpha_vals), max(alpha_vals)],
#                         vmin=0, vmax=1)  
#     axs[0].set_title('Producers J1')
#     axs[0].set_xlabel(r'$\beta$',fontsize=13)
#     axs[0].set_ylabel(r'$\gamma$',fontsize=13)
  

#     im2 = axs[1].imshow(adoption_J1C, cmap='gray_r', extent=[min(beta_vals), max(beta_vals), min(alpha_vals), max(alpha_vals)],
#                         vmin=0, vmax=1)  
#     axs[1].set_title('Consumers J1')
#     axs[1].set_xlabel(r'$\beta$',fontsize=13)
#     axs[1].set_yticks([])

#     im3 = axs[2].imshow(adoption_J2P, cmap='gray_r', extent=[min(beta_vals), max(beta_vals), min(alpha_vals), max(alpha_vals)],
#                         vmin=0, vmax=1)  
#     axs[2].set_title('Producers J2')  
#     axs[2].set_xlabel(r'$\beta$',fontsize=13) 
#     axs[2].set_yticks([])

#     im4 = axs[3].imshow(adoption_J2C, cmap='gray_r', extent=[min(beta_vals), max(beta_vals), min(alpha_vals), max(alpha_vals)],
#                         vmin=0, vmax=1)  
#     axs[3].set_title('Consumers J2') 
#     axs[3].set_xlabel(r'$\beta$',fontsize=13)
#     axs[3].set_yticks([]) 

#     #Add colorbar axis
#     #cax = fig.add_axes([0.96, 0.15, 0.01, 0.7])  # [left, bottom, width, height]

#     #Add colorbar
#    #cbar = fig.colorbar(im4, cax=cax)

#     #cbar = fig.colorbar(im4, ax=axs, fraction=0.02, pad=0.04, location='right')

#     #Set label for the colorbar
#     #cbar.set_label('Adoption rate of N')
#     plt.tight_layout()
#     plt.show()



#### Code for running 4x4 ALPHA/BETA/GAMMA/tax HEATMAPS

    # taxes = [0.17, 0.21, 0.25]#, 0.25]    
    # beta_vals = np.linspace(0,1,11)
    # gamma_vals = np.linspace(0,1,11)
    # alpha_vals = np.linspace(0,1,11)

    # #fig, axs = plt.subplots(4, 4, figsize=(9, 14), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})     # settings for 4x4 heatmap
    # fig, axs = plt.subplots(3, 4, figsize=(9, 12), gridspec_kw={'hspace': 0.4, 'wspace': 0.5})      # settings for 3x4 heatmap
    # #fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.1, hspace=0.15, wspace=0.15)

    # for idx, tax in enumerate(taxes):
    #     adoption_J1P = np.zeros((len(gamma_vals), len(beta_vals)))
    #     adoption_J1C = np.zeros((len(gamma_vals), len(beta_vals)))
    #     adoption_J2P = np.zeros((len(gamma_vals), len(beta_vals)))
    #     adoption_J2C = np.zeros((len(gamma_vals), len(beta_vals)))

    #     for i, beta in tqdm(enumerate(beta_vals), total=len(beta_vals)):
    #         for j, alpha in enumerate(alpha_vals):
    #             results_J1P = []
    #             results_J1C = []
    #             results_J2P = []
    #             results_J2C = []
    #             for _ in range(30):
    #                 model = Jurisdiction(n_consumers=500, n_producers=500, alpha=alpha, beta=beta, gamma=0.33,
    #                                     cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3,
    #                                     tax=tax, intensity_c=10, intensity_p=10,
    #                                     init_c1=0.06, init_c2=0.96, init_p1=0.04, init_p2=0.94)

    #                 for _ in range(10):
    #                     model.step()

    #                 list_1j1 = []
    #                 list_2j1 = []

    #                 for _ in range(30):
    #                     model.step()
    #                     fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                     list_1j1.append(fill_1)

                      # Model steps until convergence
    #                 while True:
    #                     model.step()
    #                     current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                     list_2j1.append(current)

    #                     if len(list_2j1) == 30:
    #                         if list_1j1 == list_2j1:
    #                             break

    #                         with warnings.catch_warnings():
    #                             warnings.simplefilter("ignore", category=RuntimeWarning)
    #                             t_stat1, p_value1 = ttest_ind(list_1j1, list_2j1)

    #                         if p_value1 > 0.05:
    #                             break

    #                         list_1j1 = list_2j1[:]
    #                         list_2j1 = []

    #                 model_data = model.datacollector.get_model_vars_dataframe()
    #                 results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
    #                 results_J1C.append(model_data['Percentage green Consumers J1'].iloc[-1])
    #                 results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])
    #                 results_J2C.append(model_data['Percentage green Consumers J2'].iloc[-1])

    #             adoption_J1P[j, i] = np.mean(results_J1P)
    #             adoption_J1C[j, i] = np.mean(results_J1C)
    #             adoption_J2P[j, i] = np.mean(results_J2P)
    #             adoption_J2C[j, i] = np.mean(results_J2C)

    #     adoption_J1P = np.flipud(adoption_J1P)
    #     adoption_J1C = np.flipud(adoption_J1C)
    #     adoption_J2P = np.flipud(adoption_J2P)
    #     adoption_J2C = np.flipud(adoption_J2C)

    #     row = idx
    #     for col, (adoption, title) in enumerate(zip(
    #             [adoption_J1P, adoption_J1C, adoption_J2P, adoption_J2C],
    #             ['prod J1', 'cons J1', 'prod J2', 'cons J2'])):
    #         im = axs[row, col].imshow(adoption, cmap='gray_r',
    #                                 extent=[min(beta_vals), max(beta_vals), min(gamma_vals), max(gamma_vals)],
    #                                 vmin=0, vmax=1)
    #         axs[row, col].set_xticks([0, 0.5, 1]) 
    #         #axs[row, col].set_title(f'{title}', fontsize=8)
    #         #axs[row, col].set_xlabel(r'$\beta$', fontsize=10)
    #         if col == 0:
    #             axs[row, col].set_ylabel(r'$\alpha$', fontsize=10)
    #         if row == 0:
    #             axs[row, col].set_title(f'{title}', fontsize=10)
    #         if row == 2:
    #             axs[row, col].set_xlabel(r'$\beta$', fontsize=10)
    #         #else:
    #             #axs[row, col].set_yticks([])
    #     #fig.text(0.01, 0.88 - (idx * 0.22), f'Tax: {tax}', ha='left', fontsize=9)  # settings for 4x4 heatmap
    #     fig.text(0.01, 0.84 - (idx * 0.29), f'Tax: {tax}', ha='left', fontsize=9)  # settings for 3x4 heatmap

    # # Add a colorbar for the entire figure
    # cbar = fig.colorbar(im, ax=axs, fraction=0.05, pad=0.04)
    # cbar.set_label('Adoption rate of N')

    # plt.tight_layout()#pad=0.5)
    # #plt.savefig('D:/tijn2/CS/thesis/result figures/in thesis/heatmaps/bg heatmap,4 taxes,dpi=500.png', dpi=500)
    # plt.show()



    ################ ADOPTION AS A FUNCTION OF ALPHA/BETA/GAMMA

    #alpha_vals = np.linspace(0.01,1,num=20)
    # gamma_vals = np.linspace(0,1,num=20)
    beta_vals = np.linspace(0,1,num=20)
    # # rat_vals = [1,10,100]


    # # # Dictionary to store the results
    average_results_J1P = {}
    average_results_J1C = {}
    average_results_J2P = {}
    average_results_J2C = {}
    std_results_J1P = {}
    std_results_J1C = {}
    std_results_J2P = {}
    std_results_J2C = {}
    for beta in tqdm(beta_vals):
        results_J1P = []
        results_J1C = []
        results_J2P = []
        results_J2C = []
        for i in range(30):  
            model = Jurisdiction(n_consumers=500, n_producers=500, alpha=beta, beta=0, gamma=0, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
                                 tax=0.2, intensity_c=10, intensity_p=10, init_c1=0.055, init_c2=0.955, init_p1=0.045, init_p2=0.945)

            # for i in range(1000):
            #     model.step()
            for _ in range(10):
                model.step()

            list_1 = []
            list_2 = []
            list_3 = []
            list_4 = []

            # Fill first list
            for _ in range(30):
                model.step()
                fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J1'].iloc[-1]
                list_1.append(fill_1)
                fill_2 = model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J2'].iloc[-1]
                list_3.append(fill_2)

            while True:
                model.step()
                current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J1'].iloc[-1]
                list_2.append(current)
                current2 = model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J2'].iloc[-1]
                list_4.append(current2)

                if len(list_2) == 30:
                    #if list_1 == list_2:
                    #    break

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        t_stat, p_value = ttest_ind(list_1, list_2)
                        t_stat2, p_value2 = ttest_ind(list_3, list_4)

                    if p_value > 0.05 and p_value2 > 0.05:  # Means are not statistically different
                        break

                    list_1 = list_2[:]
                    list_2 = []

                    list_3 = list_4[:]
                    list_4 = []
                    
            model_data =  model.datacollector.get_model_vars_dataframe()
            results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
            results_J1C.append(model_data['Percentage green Consumers J1'].iloc[-1])
            results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])
            results_J2C.append(model_data['Percentage green Consumers J2'].iloc[-1])


        average_results_J1P[beta] = np.mean(results_J1P)
        average_results_J1C[beta] = np.mean(results_J1C)
        average_results_J2P[beta] = np.mean(results_J2P)
        average_results_J2C[beta] = np.mean(results_J2C)
        std_results_J1P[beta] = np.std(results_J1P)
        std_results_J1C[beta] = np.std(results_J1C)
        std_results_J2P[beta] = np.std(results_J2P)
        std_results_J2C[beta] = np.std(results_J2C)


        
    beta_keys = list(average_results_J1P.keys())

    mean_PJ1 = np.array(list(average_results_J1P.values()))
    std_PJ1 = np.array(list(std_results_J1P.values()))
    plt.plot(beta_keys, mean_PJ1, label = 'producers J1')
    plt.fill_between(beta_keys, np.clip(mean_PJ1 - std_PJ1, 0, 1), np.clip(mean_PJ1 + std_PJ1, 0, 1), alpha=0.2)

    mean_CJ1 = np.array(list(average_results_J1C.values()))
    std_CJ1 = np.array(list(std_results_J1C.values()))
    plt.plot(beta_keys, mean_CJ1, label='consumers J1')#, color='blue')
    plt.fill_between(beta_keys, np.clip(mean_CJ1 - std_CJ1, 0, 1), np.clip(mean_CJ1 + std_CJ1, 0, 1), alpha=0.2)
    
    
    mean_PJ2 = np.array(list(average_results_J2P.values()))
    std_PJ2 = np.array(list(std_results_J2P.values()))
    plt.plot(beta_keys, mean_PJ2, label='producers J2')
    plt.fill_between(beta_keys, np.clip(mean_PJ2 - std_PJ2, 0, 1), np.clip(mean_PJ2 + std_PJ2, 0, 1), alpha=0.2)

    # Consumers J2
    mean_CJ2 = np.array(list(average_results_J2C.values()))
    std_CJ2 = np.array(list(std_results_J2C.values()))
    plt.plot(beta_keys, mean_CJ2, label='consumers J2')#, color='green')
    plt.fill_between(beta_keys, np.clip(mean_CJ2 - std_CJ2, 0, 1), np.clip(mean_CJ2 + std_CJ2, 0, 1), alpha=0.2)

    plt.xlabel(r'$\alpha$',fontsize=18)
    #plt.xlabel('tax',fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Adoption Rate of N',fontsize=15)
    plt.legend(fontsize=15)
    plt.ylim(-0.1, 1.1)
    plt.grid(True) 

    plt.tight_layout()
    #plt.savefig('D:/tijn2/CS/thesis/result figures/in thesis/adoption vs alpha/bg=0,tax=0.2,250runs,dpi=500.png', dpi=500)
    plt.show()






    #### TAX VS ADOPTION
    # tax_values = np.linspace(0.12, 0.3, num=20)
    # beta_vals = [0.05, 1]

    # average_results_J1P = {beta: {} for beta in beta_vals}
    # average_results_J1C = {beta: {} for beta in beta_vals}
    # average_results_J2P = {beta: {} for beta in beta_vals}
    # average_results_J2C = {beta: {} for beta in beta_vals}

    # for beta in beta_vals:
    #     for alpha in tqdm(tax_values):
    #         results_J1P = []
    #         results_J1C = []
    #         results_J2P = []
    #         results_J2C = []
    #         for i in range(30):  
    #             model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=beta, gamma=0,
    #                                 cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3,
    #                                 tax=alpha, intensity_c=10, intensity_p=10, init_c1=0.05, init_c2=0.95,
    #                                 init_p1=0.05, init_p2=0.95)

    #             for _ in range(10):
    #                 model.step()

    #             list_1 = []
    #             list_2 = []

    #             for _ in range(30):
    #                 model.step()
    #                 fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_1.append(fill_1)

    #             while True:
    #                 model.step()
    #                 current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_2.append(current)

    #                 if len(list_2) == 30:
    #                     if list_1 == list_2:
    #                         break

    #                     with warnings.catch_warnings():
    #                         warnings.simplefilter("ignore", category=RuntimeWarning)
    #                         t_stat, p_value = ttest_ind(list_1, list_2)

    #                     if p_value > 0.05:  # Means are not statistically different
    #                         break

    #                     list_1 = list_2[:]
    #                     list_2 = []

    #             model_data = model.datacollector.get_model_vars_dataframe()
    #             results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
    #             results_J1C.append(model_data['Percentage green Consumers J1'].iloc[-1])
    #             results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])
    #             results_J2C.append(model_data['Percentage green Consumers J2'].iloc[-1])

    #         average_results_J1P[beta][alpha] = np.mean(results_J1P)
    #         average_results_J1C[beta][alpha] = np.mean(results_J1C)
    #         average_results_J2P[beta][alpha] = np.mean(results_J2P)
    #         average_results_J2C[beta][alpha] = np.mean(results_J2C)

    # fig, axs = plt.subplots(2, figsize=(8, 5))

    # colors = {'J1': 'blue', 'J2': 'orange'}

    # for beta, linestyle in zip(beta_vals, ['-', '--']):
    #     # Producers J1 and J2
    #     axs[0].plot(average_results_J1P[beta].keys(), average_results_J1P[beta].values(), label=f'Producers J1 ($\\beta={beta}$)', linestyle=linestyle, color=colors['J1'])
    #     axs[0].plot(average_results_J2P[beta].keys(), average_results_J2P[beta].values(), label=f'Producers J2 ($\\beta={beta}$)', linestyle=linestyle, color=colors['J2'])
        

    #     # Consumers J1 and J2
    #     axs[1].plot(average_results_J1C[beta].keys(), average_results_J1C[beta].values(), label=f'Consumers J1 ($\\beta={beta}$)', linestyle=linestyle, color=colors['J1'])
    #     axs[1].plot(average_results_J2C[beta].keys(), average_results_J2C[beta].values(), label=f'Consumers J2 ($\\beta={beta}$)', linestyle=linestyle, color=colors['J2'])

    # axs[0].set_title('Producers')
    # axs[0].set_xlabel('Tax')
    # axs[0].set_ylabel('Adoption Rate of N')
    # axs[0].set_ylim(-0.1, 1.1)
    # axs[0].legend()
    # axs[0].grid(True) 

    # axs[1].set_title('Consumers')
    # axs[1].set_xlabel('Tax')
    # axs[1].set_ylabel('Adoption Rate of N')
    # axs[1].set_ylim(-0.1, 1.1)
    # axs[1].legend()
    # axs[1].grid(True) 


    # plt.tight_layout()
    # #plt.savefig('D:/tijn2/CS/thesis/result figures/in thesis/adoption vs tax/b=0.3 and 1,dpi=500.png', dpi=500)
    # plt.show()



    # results_per_rat_val = {rat: {'J1P': {}, 'J1C': {}, 'J2P': {}, 'J2C': {}} for rat in rat_vals}

    # for rat in rat_vals:
    #     #print(f"Processing rat_val: {rat}")
    #     for beta in tqdm(beta_vals):
    #         results_J1P = []
    #         results_J1C = []
    #         results_J2P = []
    #         results_J2C = []
    #         for i in range(100):  
    #             model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=beta, gamma=0.1, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
    #                                 tax=0.2, intensity_c=rat, intensity_p=rat, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)

    #             for _ in range(10):
    #                 model.step()

    #             list_1 = []
    #             list_2 = []

    #             # Fill first list
    #             for _ in range(30):
    #                 model.step()
    #                 fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_1.append(fill_1)

    #             while True:
    #                 model.step()
    #                 current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_2.append(current)

    #                 if len(list_2) == 30:
    #                     if list_1 == list_2:
    #                         break

    #                     with warnings.catch_warnings():
    #                         warnings.simplefilter("ignore", category=RuntimeWarning)
    #                         t_stat, p_value = ttest_ind(list_1, list_2)

    #                     if p_value > 0.05:  # Means are not statistically different
    #                         break

    #                     list_1 = list_2[:]
    #                     list_2 = []
                        
    #             model_data =  model.datacollector.get_model_vars_dataframe()
    #             results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
    #             results_J1C.append(model_data['Percentage green Consumers J1'].iloc[-1])
    #             results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])
    #             results_J2C.append(model_data['Percentage green Consumers J2'].iloc[-1])

    #         results_per_rat_val[rat]['J1P'][beta] = np.mean(results_J1P)
    #         results_per_rat_val[rat]['J1C'][beta] = np.mean(results_J1C)
    #         results_per_rat_val[rat]['J2P'][beta] = np.mean(results_J2P)
    #         results_per_rat_val[rat]['J2C'][beta] = np.mean(results_J2C)

    # # Plotting results for J1P and J2P in a single plot with two subplots
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))

    # # Plotting results for J1P
    # for rat in rat_vals:
    #     ax1.plot(results_per_rat_val[rat]['J1P'].keys(), results_per_rat_val[rat]['J1P'].values(), label=f'J1 - r = {rat}')
    # ax1.set_xlabel(r'$\beta$')
    # ax1.set_ylabel('Adoption Rate')
    # ax1.set_title('Adoption Rate of Producers in Jurisdiction 1')
    # ax1.legend()
    # ax1.set_ylim(-0.1, 1.1)

    # # Plotting results for J2P
    # for rat in rat_vals:
    #     ax2.plot(results_per_rat_val[rat]['J2P'].keys(), results_per_rat_val[rat]['J2P'].values(), label=f'J2 - r = {rat}')
    # ax2.set_xlabel(r'$\beta$')
    # ax2.set_ylabel('Adoption Rate')
    # ax2.set_title('Adoption Rate of Producers in Jurisdiction 2')
    # ax2.legend()
    # ax2.set_ylim(-0.1, 1.1)

    # plt.tight_layout()
    # plt.show()




    ####### Sensativity computation 1, not used in thesis now
                      
    #beta_vals = [0.2, 1]  
    # rat_vals = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30,40,50,75,100]#    np.linspace(1, 100, num=100)
    # rat_sus_j1 = {}
    # rat_sus_j2 = {}
    # beta_vals = np.linspace(0,1,30)
    

    #### SENSATIVITY COMPUTATION 1
    # Dictionary to store the results for each rat_val
   # results_per_rat_val = {rat: {'J1P': {0: 0, 1: 0}, 'J2P': {0: 0, 1: 0}} for rat in rat_vals}

    # for rat in tqdm(rat_vals):
    #     results_J1_lowB = []
    #     results_J1_highB = []
    #     results_J2_lowB = []
    #     results_J2_highB = []
    #     for beta in beta_vals:
    #         for i in range(30):  
    #             model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=beta, gamma=0.3, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
    #                                 tax=0.2, intensity_c=rat, intensity_p=rat, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)

    #             for _ in range(10):
    #                 model.step()

    #             list_1 = []
    #             list_2 = []

    #             # Fill first list
    #             for _ in range(30):
    #                 model.step()
    #                 fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_1.append(fill_1)

    #             while True:
    #                 model.step()
    #                 current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_2.append(current)

    #                 if len(list_2) == 30:
    #                     if list_1 == list_2:
    #                         break

    #                     with warnings.catch_warnings():
    #                         warnings.simplefilter("ignore", category=RuntimeWarning)
    #                         t_stat, p_value = ttest_ind(list_1, list_2)

    #                     if p_value > 0.05:  # Means are not statistically different
    #                         break

    #                     list_1 = list_2[:]
    #                     list_2 = []
                        
    #             model_data =  model.datacollector.get_model_vars_dataframe()

    #             if beta == beta_vals[0]:
    #                 results_J1_lowB.append(model_data['Percentage green Producers J1'].iloc[-1])
    #                 results_J2_lowB.append(model_data['Percentage green Producers J2'].iloc[-1])

    #             if beta == beta_vals[1]:
    #                 results_J1_highB.append(model_data['Percentage green Producers J1'].iloc[-1])
    #                 results_J2_highB.append(model_data['Percentage green Producers J2'].iloc[-1])

    #             #results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
    #             #results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])

    #     rat_sus_j1[rat] = np.abs(np.mean(results_J1_lowB) - np.mean(results_J1_highB))
    #     rat_sus_j2[rat] = np.abs(np.mean(results_J2_lowB) - np.mean(results_J2_highB))

    #         #print(rat, beta, 'j1:', np.mean(results_J1P))
    #         #print(rat, beta, 'j2:', np.mean(results_J2P))
    #         #results_per_rat_val[rat]['J1P'][beta] = np.mean(results_J1P)
    #         #results_per_rat_val[rat]['J2P'][beta] = np.mean(results_J2P)

    # # Calculate susceptibility to beta
    # #susceptibility_J1P = {rat: results_per_rat_val[rat]['J1P'][1] - results_per_rat_val[rat]['J1P'][0] for rat in rat_vals}
    # #susceptibility_J2P = {rat: results_per_rat_val[rat]['J2P'][1] - results_per_rat_val[rat]['J2P'][0] for rat in rat_vals}

    # # Plotting results
    # plt.figure(figsize=(8, 5))
    # plt.plot(rat_sus_j1.keys(), rat_sus_j1.values(), label='Jurisdiction 1')
    # plt.plot(rat_sus_j2.keys(), rat_sus_j2.values(), label='Jurisdiction 2')
    # plt.xlabel('rationality')
    # plt.ylabel(r'Susceptibility to $\beta$')
    # #plt.title('Susceptibility to β for Different rat_vals')
    # plt.legend()
    # plt.ylim(-0.1, 1.1)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()






    #### Sensitivity computation 2, used in thesis
    # variance_J1 = {}
    # variance_J2 = {}

    # for rat in tqdm(rat_vals):
    #     adoption_J1 = []
    #     adoption_J2 = []

    #     for beta in beta_vals:
    #         results_J1 = []
    #         results_J2 = []
    #         for i in range(30):
    #             model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=beta, gamma=0.3,
    #                                 cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3,
    #                                 tax=0.2, intensity_c=rat, intensity_p=rat,
    #                                 init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)

    #             for _ in range(10):
    #                 model.step()

    #             list_1 = []
    #             list_2 = []

    #             for _ in range(30):
    #                 model.step()
    #                 fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_1.append(fill_1)

    #             while True:
    #                 model.step()
    #                 current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_2.append(current)

    #                 if len(list_2) == 30:
    #                     if list_1 == list_2:
    #                         break

    #                     with warnings.catch_warnings():
    #                         warnings.simplefilter("ignore", category=RuntimeWarning)
    #                         t_stat, p_value = ttest_ind(list_1, list_2)

    #                     if p_value > 0.05:
    #                         break

    #                     list_1 = list_2[:]
    #                     list_2 = []

    #             model_data = model.datacollector.get_model_vars_dataframe()
    #             results_J1.append(model_data['Percentage green Producers J1'].iloc[-1])
    #             results_J2.append(model_data['Percentage green Producers J2'].iloc[-1])

    #         adoption_J1.append(np.mean(results_J1))
    #         adoption_J2.append(np.mean(results_J2))

    #     variance_J1[rat] = np.var(adoption_J1)
    #     variance_J2[rat] = np.var(adoption_J2)

    # # Plotting the variance of adoption rates
    # plt.figure(figsize=(8, 5))
    # plt.plot(variance_J1.keys(), variance_J1.values(), label='Jurisdiction 1')
    # plt.plot(variance_J2.keys(), variance_J2.values(), label='Jurisdiction 2')
    # plt.xlabel('Rationality')
    # plt.ylabel(r'Variance of Adoption Rates across $\beta$')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

              


#### Min Tax for adoption with std dev. Not used for graphs in thesis but can check with this if parallel code produces same result as normal code
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
#                         model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=2/4, gamma=2/4, cost_brown=0.25, cost_green=cg, 
#                                             ext_brown=0.1, ext_green=0.3, tax=mid_tax,  intensity_c=rat, intensity_p=rat, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)

#                          #let the model run for X steps first...
#                         for _ in range(10):
#                             model.step()
                        
#                         list_1 = []
#                         list_2 = []

#                         # fill first list for the first time
#                         for _ in range(10):
#                             model.step()
#                             fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
#                             list_1.append(fill_1)

#                         while True:  
#                             model.step()
#                             current =  model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
#                             list_2.append(current)

#                             if len(list_2) == 10:
#                                 if list_1 == list_2:
#                                     break

#                                 with warnings.catch_warnings():
#                                     warnings.simplefilter("ignore", category=RuntimeWarning)
#                                     t_stat, p_value = ttest_ind(list_1, list_2)
#                                 #t_stat, p_value = ttest_ind(list_1, list_2)
#                                 #u_stat, p_value = mannwhitneyu(list_1, list_2, alternative='two-sided')
                    
#                                 if p_value > 0.05:  # Means are not statistically different
#                                     break
                            
#                                 list_1 = list_2[:]
#                                 list_2 = []
    

#                         # last_10_values = []
#                         # for j in range(200):  
#                         #     model.step()
#                             # current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]

#                             # last_10_values.append(current)
#                             # if len(last_10_values) > 10:
#                             #     last_10_values.pop(0)
#                             # current_sum = sum(last_10_values)
#                             # if current_sum == 0 or current_sum == 10:
#                             #     break
                            
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






#### Parallel code for computing MIN tax

# Define the model parameters
    # cost_g_vals = np.linspace(0.2, 0.5, num=10)
    # alpha_vals = np.linspace(0,1,num=11)
    # beta_vals = np.linspace(0,1,num=11)
    # #rat_vals = [1, 5, 10, 100]
    # rat_vals = [5,10]
    # tolerance = 0.01
    # adoption_level = 0.5
    # runs_per_tax = 30

    # # Dictionary to store results
    # results_dict = {1: {}, 5: {}, 10: {}, 100: {}}
    # results_dict = {5: {}, 10: {}}

    # # Function to run a single simulation
    # def run_simulation(rat, cg, mid_tax, n_steps_initial=10, n_steps_list=30):
    #     results_J2P = []
    #     for _ in range(1):
    #         model = Jurisdiction(
    #             n_consumers=500, n_producers=500, alpha=0, beta=3/4, gamma=3/4,
    #             cost_brown=0.25, cost_green=cg, ext_brown=0.1, ext_green=0.3, tax=mid_tax,
    #             intensity_c=rat, intensity_p=rat, init_c1=0.05, init_c2=0.95, init_p1=0.05, init_p2=0.95)

    #         # Let the model run for initial steps
    #         for _ in range(n_steps_initial):
    #             model.step()

    #         list_1 = []
    #         list_2 = []

    #         # Fill first list
    #         for _ in range(n_steps_list):
    #             model.step()
    #             fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #             list_1.append(fill_1)

    #         while True:
    #             model.step()
    #             current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #             list_2.append(current)

    #             if len(list_2) == n_steps_list:
    #                 if list_1 == list_2:
    #                     break

    #                 with warnings.catch_warnings():
    #                     warnings.simplefilter("ignore", category=RuntimeWarning)
    #                     t_stat, p_value = ttest_ind(list_1, list_2)

    #                 if p_value > 0.05:  # Means are not statistically different
    #                     break

    #                 list_1 = list_2[:]
    #                 list_2 = []


    #         # last_10_values = []
    #         # for _ in range(200):  
    #         #     model.step()
    #         #     current = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]

    #         #     last_10_values.append(current)
    #         #     if len(last_10_values) > 10:
    #         #         last_10_values.pop(0)
    #         #     current_sum = sum(last_10_values)
    #         #     if current_sum == 0 or current_sum == 10:
    #         #         break

    #         model_data = model.datacollector.get_model_vars_dataframe()
    #         results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])

    #     return np.mean(results_J2P)

    # # Function to optimize tax for given parameters
    # def optimize_tax(rat, cg):
    #     min_tax = 0
    #     max_tax = 0.5
    #     while abs(max_tax - min_tax) > tolerance:
    #         mid_tax = (min_tax + max_tax) / 2

    #         mean_result = run_simulation(rat, cg, mid_tax)

    #         if mean_result < adoption_level:
    #             min_tax = mid_tax
    #         else:
    #             max_tax = mid_tax

    #     #if max_tax < 0.5:
    #      #   return max_tax

        
    #     if max_tax == 0.5:
    #         return 5 ## A value that we do not show in the graph
    #     else:
    #         return max_tax

    # # Run the optimization for each combination of rat and cg values
    # for rat in rat_vals:
    #     for cg in tqdm(cost_g_vals):
    #         max_taxes = Parallel(n_jobs=-1)(
    #             delayed(optimize_tax)(rat, cg) for _ in range(runs_per_tax))

    #         if cg not in results_dict[rat]:
    #             results_dict[rat][cg] = max_taxes

    # # Plot the results
    # for rat, label in zip(rat_vals, ['r=1', 'r=5', 'r=10', 'r=100']):
    #     keys = list(results_dict[rat].keys())
    #     means = [np.mean(results_dict[rat][key]) for key in keys]
    #     std_devs = [np.std(results_dict[rat][key]) for key in keys]
    #     plt.errorbar(keys, means, yerr=std_devs, label=label, fmt='-o', capsize=2)

    #     #plt.plot(keys, means, label=label, marker='o')
    #     #plt.fill_between(keys, np.array(means) - np.array(std_devs), np.array(means) + np.array(std_devs), alpha=0.2)

    # tax_diagonal = cost_g_vals - 0.25
    # plt.plot(cost_g_vals, tax_diagonal, linestyle='--', color='black', label='cost O + tax = cost N')

    # #plt.axhline(y=0.2, color='black', linestyle='--', label='Cost O + Tax = Cost N')

    # # Customize plot labels and title
    # #plt.xlabel(r'$\alpha$',fontsize=14)
    # plt.xlabel("cost of $N$")
    # plt.ylabel('Tax',fontsize=14)
    # plt.ylim(-0.01, 0.55)
    # plt.grid(True)
    # plt.legend(fontsize=11)
    # #plt.savefig('D:/tijn2/CS/thesis/result figures/in thesis/min tax alpha/PROD j2, bg=0.1, j1200C300P, adopt=0.5,dpi=500.png', dpi=500)
    # plt.show()





    #### MIN tax for adoption with STD. Another check for correctness of parallel code

    # beta_values = np.linspace(0,1,num=11)
    # alpha_values = np.linspace(0,1,num=11)
    
    # rat_vals = [1,5,10,100]
    # tolerance = 0.005
    # adoption_level = 0.5
    # runs_per_tax = 5

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
    #     for alpha in tqdm(alpha_values):
    #         for run in range(runs_per_tax):
    #             min_tax = 0
    #             max_tax = 0.5
    #             while abs(max_tax - min_tax) > tolerance:
    #                 mid_tax = (min_tax + max_tax) / 2

    #                 results_P1 = []
    #                 #results_P2 = []
    #                 # results_C1 = []
    #                 # results_C2 = []
    #                 for i in range(10):  
    #                     model = Jurisdiction(n_consumers=500, n_producers=500, alpha=alpha, beta=2/3, gamma=2/3, cost_brown=0.25, cost_green=0.45, 
    #                                         ext_brown=0.1, ext_green=0.3, tax=mid_tax, intensity_c=rat, intensity_p=rat, init_c1=0.04, init_c2=0.94, init_p1=0.06, init_p2=0.96)
    #                     last_10_values_1 = []
    #                     last_10_values_2 = []

    #                     for j in range(200):  
    #                         model.step()
    #                         current_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J1'].iloc[-1]
    #                         current_2 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]

    #                         last_10_values_1.append(current_1)
    #                         last_10_values_2.append(current_2)

    #                         if len(last_10_values_1) > 10:
    #                             last_10_values_1.pop(0)
    #                         if len(last_10_values_2) > 10:
    #                             last_10_values_2.pop(0)
    #                         current_sum_1 = sum(last_10_values_1)
    #                         current_sum_2 = sum(last_10_values_2)
    #                         if (current_sum_1 == 0 or current_sum_1 == 10) and (current_sum_2 == 0 or current_sum_2 == 10):
    #                             break

    #                     model_data =  model.datacollector.get_model_vars_dataframe()
    #                     results_P1.append(model_data['Percentage green Producers J2'].iloc[-1])
    #                     #results_P2.append(model_data['Percentage green Producers J2'].iloc[-1])
    #                     # results_C1.append(model_data['Percentage green Producers J2'].iloc[-1])
    #                     # results_C2.append(model_data['Percentage green Consumers J2'].iloc[-1])

    #                 if np.mean(results_P1) < adoption_level:
    #                     min_tax = mid_tax
    #                 else:
    #                     max_tax = mid_tax

    #             if max_tax != 0.5:
    #                 if rat == 1:
    #                     if alpha not in rat1P1:
    #                         rat1P1[alpha] = [max_tax]
    #                     else:
    #                         rat1P1[alpha].append(max_tax)
    #                 elif rat == 5:
    #                     if alpha not in rat5P1:  # Make sure to add only 1 value per beta value and not overwrite
    #                         rat5P1[alpha] = [max_tax]
    #                     else:
    #                         rat5P1[alpha].append(max_tax)
    #                 elif rat == 10:
    #                     if alpha not in rat10P1:
    #                         rat10P1[alpha] = [max_tax]
    #                     else:
    #                         rat10P1[alpha].append(max_tax)
    #                 else:
    #                     if alpha not in rat100P1:
    #                         rat100P1[alpha] = [max_tax] 
    #                     else:
    #                         rat100P1[alpha].append(max_tax)
                        

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

    
    # keys_rat1P1 = list(rat1P1.keys())
    # keys_rat5P1 = list(rat5P1.keys())
    # keys_rat10P1 = list(rat10P1.keys())
    # keys_rat100P1 = list(rat100P1.keys())

    # # Plot each dictionary separately
    # plt.errorbar(keys_rat1P1, [np.mean(rat1P1[key]) for key in keys_rat1P1],
    #             yerr=[np.std(rat1P1[key]) for key in keys_rat1P1],
    #             label='rat1', fmt='-o', capsize=5)
    # plt.errorbar(keys_rat5P1, [np.mean(rat5P1[key]) for key in keys_rat5P1],
    #             yerr=[np.std(rat5P1[key]) for key in keys_rat5P1],
    #             label='rat5', fmt='-o', capsize=5)
    # plt.errorbar(keys_rat10P1, [np.mean(rat10P1[key]) for key in keys_rat10P1],
    #             yerr=[np.std(rat10P1[key]) for key in keys_rat10P1],
    #             label='rat10', fmt='-o', capsize=5)
    # plt.errorbar(keys_rat100P1, [np.mean(rat100P1[key]) for key in keys_rat100P1],
    #          yerr=[np.std(rat100P1[key]) for key in keys_rat100P1],
    #          label='rat100', fmt='-o', capsize=5)

   
    # plt.axhline(y=0.2, color='black', linestyle='--', label='tax + cost brown = cost green')
    # # Customize plot labels and title
    # plt.xlabel(r'$\alpha$')
    # plt.ylabel('Tax')
    # plt.ylim(0, 0.5)
    # plt.xlim(-0.1,1.1)
    # plt.grid(True)
    # #plt.xticks(keys)  # Assuming keys are numeric
    # plt.legend()
    # plt.show()



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




#PHASE DIAGRAM FOR INITIAL CONDITIONS
    # j1_vals = np.linspace(0,0.1,11)
    # j2_vals = np.linspace(0,0.1,11)
    # #j1_vals = [0,0.06,0.12,0.18,0.24,0.3,0.36,0.42,0.48,0.54,0.6]

    
    # adoption_J2P = np.zeros((len(j1_vals), len(j2_vals)))

    # for i, j1_val in tqdm(enumerate(j1_vals)):
    #     for j, j2_val in enumerate(j2_vals):
            
    #         results_J2P = []
    #         for k in range(1):  
    #             #model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=0, gamma=0, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
    #             #                     tax=0.2, intensity_c=10, intensity_p=10, init_c1=j1_val, init_c2= 1-j2_val, init_p1=j1_val, init_p2=1-j2_val)
                
    #             model = Jurisdiction(n_consumers=500, n_producers=500, alpha=0, beta=0, gamma=0, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, 
    #                                  tax=0.2, intensity_c=10, intensity_p=10, init_c1=0.05, init_c2= 1-j1_val, init_p1=0.05, init_p2=1-j2_val)
                
    #             # for l in range(100):  
    #             #     model.step()

    #             for _ in range(10):
    #                 model.step()

    #             list_1 = []
    #             list_2 = []
    #             list_3 = []
    #             list_4 = []

    #             #Fill first list
    #             for _ in range(30):
    #                 model.step()

    #                 fill_1 = model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J1'].iloc[-1]
    #                 list_1.append(fill_1)
    #                 fill_2 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_3.append(fill_2)

    #             # do model step until simulation has converged
    #             while True:
    #                 model.step()

    #                 current1 = model.datacollector.get_model_vars_dataframe()['Percentage green Consumers J1'].iloc[-1]
    #                 list_2.append(current1)
    #                 current2 = model.datacollector.get_model_vars_dataframe()['Percentage green Producers J2'].iloc[-1]
    #                 list_4.append(current2)

    #                 if len(list_2) == 30:
    #                     if list_1 == list_2:
    #                         break

    #                     with warnings.catch_warnings():
    #                         warnings.simplefilter("ignore", category=RuntimeWarning)
    #                         t_stat, p_value = ttest_ind(list_1, list_2)
    #                         t_stat1, p_value1 = ttest_ind(list_3, list_4)

    #                     if p_value > 0.05 and p_value1 > 0.05:
    #                         break

    #                     list_1 = list_2[:]
    #                     list_2 = []

    #                     list_3 = list_4[:]
    #                     list4 = []

    #             model_data =  model.datacollector.get_model_vars_dataframe()
    #             results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])

    #         adoption_J2P[j,i] = np.mean(results_J2P)

    # adoption_J2P = np.flipud(adoption_J2P)

    # plt.imshow(adoption_J2P, cmap='gray_r', extent=[min(j1_vals), max(j1_vals), min(j2_vals), max(j2_vals)],
    #            vmin=0, vmax=1)
    # plt.xticks(ticks=plt.xticks()[0], labels=[f'{2*x:.2f}' for x in plt.xticks()[0]], fontsize=11)
    # plt.yticks(ticks=plt.yticks()[0], labels=[f'{2*y:.2f}' for y in plt.yticks()[0]],fontsize=11)
    # plt.xlabel('J1', fontsize=16)
    # plt.ylabel('J2', fontsize=16)

            
    # # plt.annotate('', xy=(0, 0), xytext=(0.1, 0.45), arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=4))
    # # plt.annotate('', xy=(0, 0), xytext=(0.15, 0.25), arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=4))
    # # plt.annotate('', xy=(0, 0), xytext=(0.15, 0.05), arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=4))
             
    # # plt.annotate('', xy=(0.5, 0.5), xytext=(0.3, 0.45), arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=4))
    # # plt.annotate('', xy=(0.5, 0.5), xytext=(0.25, 0.35), arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=4))
    # # plt.annotate('', xy=(0.5, 0.5), xytext=(0.35, 0.05), arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=4))

    # # plt.tight_layout()
    # # ax.set_xlim(0, 1)
    # # ax.set_ylim(0, 1)
    # plt.show()
  

