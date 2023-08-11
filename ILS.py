# -*- coding: utf-8 -*-import random
"""
Created on Tue May 30 10:59:09 2023

@author: gopal
"""

import random
import numpy 

def allot(position, num_workers):
    values = [[] for i in range(num_workers)]
    for task, worker in enumerate(position):
        values[worker-1].append(task)
    graph = dict.fromkeys([i for i in range (num_tasks)])
    for i in range (num_workers):
        graph[i] = values[i]
    return graph
    

class Particle:
    def __init__(self, num_tasks, num_workers, solution):
        self.position = solution
        self.velocity = solution
        self.best_position = self.position.copy()
        self.best_cost = float('inf')
        self.w_graph = allot(self.position, num_workers)
        self.count = 0

def evaluate_solution(solution, costs, comm_costs):
    total_cost = 0
    for task, worker in enumerate(solution):
        if(worker != -1):
            total_cost += costs[task][worker-1]
        
    for i in range (len(solution)):
        for j in range (len(solution)):
            if comm_costs[i][j] != 0:
                if solution[i] != solution[j]:
                    total_cost += comm_costs[i][j]
                    
    return total_cost

def viable(solution, pro_req, pro_limit):
    p_lim = pro_limit.copy()
    flag = True
    for task, worker in enumerate(solution):
        p_lim[worker-1] -= pro_req[task] 
        if (p_lim[worker-1] < 0):
            flag = False
            break
    return flag

def initialize_swarm(num_particles, num_tasks, num_workers, pro_req, pro_limit, costs, comm_costs):
    swarm = []
    for _ in range(num_particles):
        task = [i for i in range(num_tasks)]
        solution = [-1 for _ in range(num_tasks)]
        lim = pro_limit.copy()
        while task != []:
            t = random.choice(task)
            cost = [float('inf') for _ in range(num_workers)]
            for i in range(num_workers):
                if (lim[i] - pro_req[t]) >= 0:
                    solution[t] = i+1
                    cost[i] = evaluate_solution(solution, costs, comm_costs)
            min_cost = min(cost)
            min_index = cost.index(min_cost)
            lim[min_index] = lim[min_index] - pro_req[t]
            solution[t] = min_index+1
            task.remove(t)
        particle = Particle(num_tasks, num_workers, solution)
        swarm.append(particle)
            
    
    return swarm

def repair(particle, pro_req, pro_limit, costs, comm_costs, num_workers, num_tasks):
    task = [i for i in range(num_tasks)]
    solution = particle.velocity
    lim = pro_limit.copy()
    for i in range(num_tasks):
        if solution[i] != -1:
            lim[solution[i]-1] -= pro_req[i]
    
    while task != []:
        t = random.choice(task)
        if solution[t] == -1:
            cost = [float('inf') for _ in range(num_workers)]
            for i in range(num_workers):
                if (lim[i] - pro_req[t]) >= 0:
                    solution[t] = i+1
                    cost[i] = evaluate_solution(solution, costs, comm_costs)
            min_cost = min(cost)
            min_index = cost.index(min_cost)
            lim[min_index] = lim[min_index] - pro_req[t]
            solution[t] = min_index+1
        task.remove(t)
    return solution
    
    
def update_velocity(particle, choice, pro_req, pro_limit, costs, comm_costs, num_workers, num_tasks):
    
    for i in choice:
        particle.velocity[i] = -1
    
    particle.velocity = repair(particle, pro_req, pro_limit, costs, comm_costs, num_workers, num_tasks)
    
        

def update_position(particle, pro_req, pro_limit, costs, comm_costs, num_workers, num_tasks):
    
    workers = [i for i in range(num_workers)]
    particle.count = 0
    workers = random.choices([i for i in range(num_workers)],k=num_workers//3)
    choice = []
    for i in workers:
        for j in particle.w_graph[i]:
            choice.append(j)
    update_velocity(particle, choice, pro_req, pro_limit, costs, comm_costs, num_workers, num_tasks)
    particle.position = particle.velocity
    

def pso(num_tasks, num_workers, costs, comm_costs, pro_req, pro_limit, num_particles, max_iterations):
    swarm = initialize_swarm(num_particles, num_tasks, num_workers, pro_req, pro_limit, costs, comm_costs)
    print('initialized all solution')
    global_best_position = None
    global_best_cost = float('inf')
    count = 0

    for i in range(max_iterations):
        for particle in swarm:
            cost = evaluate_solution(particle.position, costs, comm_costs)
            flag = viable(particle.position, pro_req, pro_limit)
            if flag == False:
                cost = float('inf')
            if cost < particle.best_cost:
                particle.best_position = particle.position.copy()
                particle.best_cost = cost
                if(i>0):
                    count = 0
                    print('here')
                
            if cost < global_best_cost:
                count = 0
                print(i,global_best_cost,cost)
                global_best_position = particle.position.copy()
                global_best_cost = cost
                
            count += 1
        
        for particle in swarm:
           
            update_position(particle, pro_req, pro_limit, costs, comm_costs, num_workers, num_tasks)
            
            
        if count > 2*num_particles:
            print(i)
            break
        
    
    return global_best_position, global_best_cost
#%%


def assign(graph, comm_costs, num_tasks):
    for i in range (num_tasks):
        for j in range (num_tasks):
                if j not in graph[i] or j>i :
                    comm_costs[i][j] = 0
               
    return comm_costs

# for creating files 

l = [[5,3]]
ld = [0.3,0.5,0.8]
for i in l:
    num_tasks = i[0]
    num_workers = i[1]
    costs = [[random.randint(1, 200) for y in range(num_workers)] for x in range(num_tasks)]
    pro_req = [random.randint(1, 50) for x in range(num_tasks)]
    pro_limit = [random.randint(50, 250) for x in range(num_workers)]
    for j in ld:
        d = j
        array = numpy.array(costs)
        array1 = numpy.array(pro_req)
        array2 = numpy.array(pro_limit)
        numpy.savetxt(r"C:\Users\gopal\OneDrive\Desktop\project\{}_{}_{}_cost.txt".format(num_tasks,num_workers,d), array)
        numpy.savetxt(r"C:\Users\gopal\OneDrive\Desktop\project\{}_{}_{}_pr.txt".format(num_tasks,num_workers,d), array1)
        numpy.savetxt(r"C:\Users\gopal\OneDrive\Desktop\project\{}_{}_{}_pl.txt".format(num_tasks,num_workers,d), array2)
        edges = int(round(d * (num_tasks * (num_tasks-1))/2))
        
        values = [[] for i in range(num_tasks)]
        for i in range (edges):
            while True:
                a,b = random.randint(1, num_tasks)-1,random.randint(1, num_tasks)-1
        
                if b not in values[a] and a != b:
                    values[a].append(b)
                    values[b].append(a)
                    break
        
        graph = dict.fromkeys([i for i in range (num_tasks)])
        for i in range (num_tasks):
            graph[i] = values[i]
            
            
        comm_costs = assign(graph,[[random.randint(1,50) for j in range(num_tasks)] for i in range(num_tasks)],num_tasks) 
        array3 = numpy.array(comm_costs)
        numpy.savetxt(r"C:\Users\gopal\OneDrive\Desktop\project\{}_{}_{}_c_cost.txt".format(num_tasks,num_workers,d), array3)
#%%
import time 
import numpy

start = time.time()
num_particles = 3
max_iterations = 1000 
  
num_tasks = 125
num_workers = 75
d = 0.3
costs = numpy.loadtxt(r"C:\Users\gopal\OneDrive\Desktop\project\{}_{}_{}_cost.txt".format(num_tasks,num_workers,d)).tolist()
pro_req = numpy.loadtxt(r"C:\Users\gopal\OneDrive\Desktop\project\{}_{}_{}_pr.txt".format(num_tasks,num_workers,d)).tolist()
pro_limit = numpy.loadtxt(r"C:\Users\gopal\OneDrive\Desktop\project\{}_{}_{}_pl.txt".format(num_tasks,num_workers,d)).tolist()
comm_costs = numpy.loadtxt(r"C:\Users\gopal\OneDrive\Desktop\project\{}_{}_{}_c_cost.txt".format(num_tasks,num_workers,d)).tolist()
best_solution, best_cost = pso(num_tasks, num_workers, costs, comm_costs, pro_req, pro_limit, num_particles, max_iterations)

#print("Best solution:", best_solution)
print("Best cost:", best_cost)
end = time.time()
print(end - start)



