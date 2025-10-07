import numpy as np
from customer import Customer
import random

class Simulation:
    def __init__(self, store, num_customers = 5):
        self.store = store
        self.customers = []
        self.traffic = np.zeros((store.rows, store.cols), dtype = int)
        self.create_customers(num_customers)
    def create_customers(self, n):
        sections = list(self.store.sections.keys())

        for i in range(n):
            section = random.choice(sections)
            customer = Customer(self.store, section)
            self.customers.append(customer)
    def update(self):
        for c in self.customers:
            if not c.finished:
                c.update()
                x, y = c.position
                self.traffic[x, y] += 1
   
