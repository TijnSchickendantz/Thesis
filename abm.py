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
    def __init__(self, unique_id, model, tech_pref, jurisdiction, ext_brown, ext_green):#, tech_preference):
        super().__init__(unique_id, model)
        self.cons_tech_preference = tech_pref #random.choices(['brown', 'green'], weights=[5, 5])[0]
        self.benefit = 0.5 #random.uniform(0, 1)  # can make it random for heterogeneous agents
        self.ext_brown = ext_brown#random.uniform(0, 1) 
        self.ext_green = ext_green
        self.price_green = 0.5 
        self.price_brown = 0.5
        self.payoff = 0
        self.jurisdiction = jurisdiction #random.choice([1,2]) # every agent belongs to either jurisdiction

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
    def __init__(self, unique_id, model, tech_pref, jurisdiction, cost_brown, cost_green, tax):#, tech_preference):
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

    def __init__(self, n_producers, n_consumers,alpha,beta,gamma, cost_brown, cost_green, ext_brown, ext_green, tax):

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

        # create more controlled initial conditions where first X are green. 
        # Create consumers
        for i in range(n_consumers):
            jurisdiction = 1 if i < (n_consumers / 2) else 2
            tech_pref = 'brown' if (i < n_consumers / 4 or i >= n_consumers * 0.75) else 'green'
            consumer = Consumer(i, self, tech_pref, jurisdiction, ext_brown, ext_green)
            self.schedule.add(consumer)
        
        # Create producers
        for i in range(n_producers):
            jurisdiction = 1 if i < (n_producers / 2) else 2
            tech_pref = 'brown' if (i < n_producers / 4 or i >= n_producers * 0.75) else 'green'
            producer = Producer(n_consumers + i, self, tech_pref, jurisdiction, cost_brown, cost_green, tax)
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
        #     if product_color == 'brown' and (self.total_brown_products_j1 != 0 or self.total_brown_products_j2 != 0):
        #         if juris == 1:
        #             #print('brown 1')
        #             prob_j1 = self.total_brown_products_j1 / (self.total_brown_products_j1 + alpha * self.total_brown_products_j2)
        #             #prob_j2 =  alpha * self.total_brown_products_j2 / (self.total_brown_products_j1 + alpha * self.total_brown_products_j2)
        #             if prob_j1 > random.random():
        #                 self.total_brown_products_j1 -= 1
        #             else:
        #                 self.total_brown_products_j2 -= 1
        #         if juris == 2:
        #             #print('brown 2')
        #             prob_j2 = self.total_brown_products_j2 / (alpha* self.total_brown_products_j1 + self.total_brown_products_j2)
        #             if prob_j2 > random.random():
        #                 self.total_brown_products_j2 -= 1
        #             else:
        #                 self.total_brown_products_j1 -= 1

        #         agent.payoff = agent.cons_payoff(agent.cons_tech_preference)

        #     elif product_color == 'green' and (self.total_green_products_j1 != 0 or self.total_green_products_j2 != 0):
        #         if juris == 1:
        #             #print('green 1')
        #             prob_j1 = self.total_green_products_j1 / (self.total_green_products_j1 + alpha * self.total_green_products_j2)
        #             #prob_j2 =  alpha * self.total_brown_products_j2 / (self.total_brown_products_j1 + alpha * self.total_brown_products_j2)
        #             if prob_j1 > random.random():
        #                 self.total_green_products_j1 -= 1
        #             else:
        #                 self.total_green_products_j2 -= 1
        #         if juris == 2:
        #             #print('green 2')
        #             prob_j2 = self.total_green_products_j2 / (alpha* self.total_green_products_j1 + self.total_green_products_j2)
        #             if prob_j2 > random.random():
        #                 self.total_green_products_j2 -= 1
        #             else:
        #                 self.total_green_products_j1 -= 1

        #         agent.payoff = agent.cons_payoff(agent.cons_tech_preference)

        #     else:
        #         agent.payoff = 0
                

            # this code is for first depleting local supply
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
            if product_color == 'green':
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
            #print('next')
        
        #Let global consumers buy what is left in the other jurisdiction
        if self.total_brown_products_j1 > 0:
            poss_cons_j2b = [agent for agent in self.consumers_j2 if agent.payoff == 0 and agent.cons_tech_preference == 'brown']
            amount_j2b = int(alpha * len(poss_cons_j2b))
            subset_j2b = random.sample(poss_cons_j2b, amount_j2b)
            if len(subset_j2b) != 0:
                for cons in subset_j2b:
                    if self.total_brown_products_j1 == 0:
                        break  
                    self.total_brown_products_j1 -= 1
                    cons.payoff = cons.cons_payoff(cons.cons_tech_preference)
                    
        if self.total_brown_products_j2 > 0:
            poss_cons_j1b = [agent for agent in self.consumers_j1 if agent.payoff == 0 and agent.cons_tech_preference == 'brown']
            amount_j1b = int(alpha * len(poss_cons_j1b))
            subset_j1b = random.sample(poss_cons_j1b, amount_j1b)
            if len(subset_j1b) != 0:
                for cons in subset_j1b:
                    if self.total_brown_products_j2 == 0:
                        break  
                    self.total_brown_products_j2 -= 1
                    cons.payoff = cons.cons_payoff(cons.cons_tech_preference)

        if self.total_green_products_j1 > 0:
            poss_cons_j2g = [agent for agent in self.consumers_j2 if agent.payoff == 0 and agent.cons_tech_preference == 'green']
            amount_j2g = int(alpha * len(poss_cons_j2g))
            subset_j2g = random.sample(poss_cons_j2g, amount_j2g)
            if len(subset_j2g) != 0:
                for cons in subset_j2g:
                    if self.total_green_products_j1 == 0:
                        break  
                    self.total_green_products_j1 -= 1
                    cons.payoff = cons.cons_payoff(cons.cons_tech_preference)

        if self.total_green_products_j2 > 0:
            poss_cons_j1g = [agent for agent in self.consumers_j1 if agent.payoff == 0 and agent.cons_tech_preference == 'green']
            amount_j1g = int(alpha * len(poss_cons_j1g))
            subset_j1g = random.sample(poss_cons_j1g, amount_j1g)
            if len(subset_j1g) != 0:
                for cons in subset_j1g:
                    if self.total_green_products_j2 == 0:
                        break  
                    self.total_green_products_j2 -= 1
                    cons.payoff = cons.cons_payoff(cons.cons_tech_preference)


        # after consumers have bought, subtract from payoff of random producers that havent sold
                    
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
      

        # SWITCHING SYSTEM
                
        # Compare payoff to random producer and save data for switching
        # we dont need first term of every factor term
        prod_probs = {}
        prod_factor_j1_bg = ((self.total_green_producers_j1 + self.beta * self.total_green_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2))
        prod_factor_j1_gb = ((self.total_brown_producers_j1 + self.beta * self.total_brown_producers_j2) / (self.n_producers_j1 + self.beta * self.n_producers_j2))
        prod_factor_j2_bg = ((self.total_green_producers_j2 + self.beta * self.total_green_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1))
        prod_factor_j2_gb = ((self.total_brown_producers_j2 + self.beta * self.total_brown_producers_j1) / (self.n_producers_j2 + self.beta * self.n_producers_j1))
        for prod in self.producers:
            #other_producers = [pr for pr in self.producers if pr != prod] 
            if prod.jurisdiction == 2:
                other_prod = random.choice(self.producers_j2)
                if prod.prod_tech_preference == 'brown':
                    factor = prod_factor_j2_bg
                else:
                    factor = prod_factor_j2_gb

            elif prod.jurisdiction == 1:
                other_prod = random.choice(self.producers_j1)
                if prod.prod_tech_preference == 'brown':
                    factor = prod_factor_j1_bg
                else:
                    factor = prod_factor_j1_gb

            prod_probs[prod] = (factor * prod.prod_switch(other_prod), other_prod.prod_tech_preference)  #(prod.payoff - other_prod.payoff, other_prod.prod_tech_preference) # change to probability later

        # Do the actual producer switching
        for prod, probs in prod_probs.items():
            number = random.random()
            if probs[0] > number: 
                prod.prod_tech_preference = probs[1]


        # Compare payoff to random consumer and save data for switching 
        cons_probs = {}
        cons_factor_j1_bg = ((self.total_green_consumers_j1 + self.gamma * self.total_green_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        cons_factor_j1_gb = ((self.total_brown_consumers_j1 + self.gamma * self.total_brown_consumers_j2) / (self.n_consumers_j1 + self.gamma * self.n_consumers_j2))
        cons_factor_j2_bg = ((self.total_green_consumers_j2 + self.gamma * self.total_green_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        cons_factor_j2_gb = ((self.total_brown_consumers_j2 + self.gamma * self.total_brown_consumers_j1) / (self.n_consumers_j2 + self.gamma * self.n_consumers_j1))
        for cons in self.consumers:
            if cons.jurisdiction == 2:
                other_cons = random.choice(self.consumers_j2)
                if cons.cons_tech_preference == 'brown':
                    factor = cons_factor_j2_bg
                else:
                    factor = cons_factor_j2_gb

            elif cons.jurisdiction == 1:
                other_cons = random.choice(self.consumers_j1)
                if cons.cons_tech_preference == 'brown':
                    factor = cons_factor_j1_bg
                else:
                    factor = cons_factor_j1_gb


            cons_probs[cons] = (cons.cons_switch(other_cons), other_cons.cons_tech_preference)
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

    # tax_levels = [0.1,0.2,0.3,0.35]
    beta_vals = np.linspace(0,1,11)
    gamma_vals = np.linspace(0,1,11)
    # for tax in tax_levels:
    # strategy: save 4 matrices of data per agent/Jurisdiction combination.
    adoption_J1P = np.zeros((len(gamma_vals), len(beta_vals)))
    adoption_J1C = np.zeros((len(gamma_vals), len(beta_vals)))
    adoption_J2P = np.zeros((len(gamma_vals), len(beta_vals)))
    adoption_J2C = np.zeros((len(gamma_vals), len(beta_vals)))

    for i, beta in tqdm(enumerate(beta_vals)):
        for j, gamma in enumerate(gamma_vals):
            results_J1P = []
            results_J1C = []
            results_J2P = []
            results_J2C = []
            for k in range(30):  
                model = Jurisdiction(n_consumers=1000, n_producers=1000, alpha=0, beta=beta, gamma=gamma, cost_brown=0.25, cost_green=0.25, ext_brown=0.25, ext_green=0.25, tax=0)
                for l in range(100):  
                    model.step()

                model_data =  model.datacollector.get_model_vars_dataframe()
                results_J1P.append(model_data['Percentage green Producers J1'].iloc[-1])
                results_J1C.append(model_data['Percentage green Consumers J1'].iloc[-1])
                results_J2P.append(model_data['Percentage green Producers J2'].iloc[-1])
                results_J2C.append(model_data['Percentage green Consumers J2'].iloc[-1])

            adoption_J1P[j,i] = np.mean(results_J1P)
            adoption_J1C[j,i] = np.mean(results_J1C)
            adoption_J2P[j,i] = np.mean(results_J2P)
            adoption_J2C[j,i] = np.mean(results_J2C)

    fig, axs = plt.subplots(1, 4, figsize=(8, 2))

    im1 = axs[0].imshow(adoption_J1P, cmap='gray', extent=[min(beta_vals), max(beta_vals), min(gamma_vals), max(gamma_vals)])  
    axs[0].set_title('Producers J1')
    axs[0].set_xlabel('Beta')  # Set x-axis label
    axs[0].set_ylabel('Gamma')
  

    im2 = axs[1].imshow(adoption_J1C, cmap='gray', extent=[min(beta_vals), max(beta_vals), min(gamma_vals), max(gamma_vals)])  
    axs[1].set_title('Consumers J1')

    im3 = axs[2].imshow(adoption_J2P, cmap='gray', extent=[min(beta_vals), max(beta_vals), min(gamma_vals), max(gamma_vals)])  
    axs[2].set_title('Producers J2')  

    im4 = axs[3].imshow(adoption_J2C, cmap='gray', extent=[min(beta_vals), max(beta_vals), min(gamma_vals), max(gamma_vals)])  
    axs[3].set_title('Consumers J2') 


    plt.tight_layout()
    plt.show()


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
    #         model = Jurisdiction(n_consumers=1000, n_producers=1000, alpha=0, beta=0, gamma=0, cost_brown=0.25, cost_green=cost_g, ext_brown=0.25, ext_green=0.25, tax=0.1)
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



    
    # model = Jurisdiction(n_consumers=1000, n_producers=1000, alpha=0, beta=0.5, gamma=0.5, cost_brown=0.25, cost_green=0.45, ext_brown=0.1, ext_green=0.3, tax=0.30)
    # for i in tqdm(range(200)):
    #     model.step()

    # # Retrieve and plot data collected by the DataCollector
    # model_data = model.datacollector.get_model_vars_dataframe()

    # plt.figure(figsize=(7, 4))

    # plt.plot(model_data['Percentage green Producers J1'], label='Percentage Green Producers J1', color='indianred')
    # plt.plot(model_data['Percentage green Consumers J1'], label='Percentage Green Consumers J1', color='darkred')
    # plt.plot(model_data['Percentage green Producers J2'], label='Percentage Green Producers J2', color='deepskyblue')
    # plt.plot(model_data['Percentage green Consumers J2'], label='Percentage Green Consumers J2', color='royalblue')
    # plt.title('Adoption of green tech')
    # plt.xlabel('Steps')
    # plt.ylabel('Percentage')
    # plt.legend()
    # #plt.xticks(range(0, len(model_data)), map(int, model_data.index))

    # plt.tight_layout()
    # plt.show()