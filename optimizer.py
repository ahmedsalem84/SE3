import numpy as np
import random
import math

class Optimizer:
    """Base class for all Meta-heuristic algorithms."""
    def __init__(self, name, pop_size, dim, max_iter, lb=0.0, ub=1.0):
        self.name = name
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.loss_history = []
        self.best_score = -float('inf')
        self.best_pos = None

    def optimize(self, func_wrapper):
        raise NotImplementedError

# ======================================================
# 1. CIRCLE SEARCH ALGORITHM (CSA) - PROPOSED METHOD
# ======================================================
class CSA(Optimizer):
    def optimize(self, func_wrapper):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func_wrapper(x) for x in X])
        
        # Sort to find center
        idx = np.argsort(fitness)[::-1]
        X = X[idx]
        fitness = fitness[idx]
        
        self.best_pos = X[0].copy()
        self.best_score = fitness[0]
        self.loss_history.append(self.best_score)
        
        for it in range(self.max_iter):
            Xc = self.best_pos
            radius = 1 - (it / self.max_iter) # Shrinking radius
            
            for i in range(self.pop_size):
                if i == 0: continue # Elitism
                angle = 2 * np.pi * random.random()
                X[i] = Xc + (radius * np.cos(angle)) * (X[i] - Xc)
                X[i] = np.clip(X[i], self.lb, self.ub)
                
                f_new = func_wrapper(X[i])
                if f_new > fitness[i]:
                    fitness[i] = f_new
                    if f_new > self.best_score:
                        self.best_score = f_new
                        self.best_pos = X[i].copy()
            
            self.loss_history.append(self.best_score)
        return self.best_score, self.best_pos

# ======================================================
# 2. GENETIC ALGORITHM (GA)
# ======================================================
class GA(Optimizer):
    def optimize(self, func_wrapper):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func_wrapper(x) for x in X])
        self.best_score = np.max(fitness)
        self.best_pos = X[np.argmax(fitness)].copy()
        
        for it in range(self.max_iter):
            parents = []
            # Tournament Selection
            for _ in range(self.pop_size):
                candidates_idx = np.random.randint(0, self.pop_size, 3)
                best_cand = candidates_idx[np.argmax(fitness[candidates_idx])]
                parents.append(X[best_cand])
            parents = np.array(parents)
            
            offspring = []
            for i in range(0, self.pop_size, 2):
                p1 = parents[i]
                p2 = parents[i+1] if i+1 < self.pop_size else parents[0]
                # Crossover
                if random.random() < 0.8:
                    pt = random.randint(1, self.dim-1)
                    child1 = np.concatenate([p1[:pt], p2[pt:]])
                    child2 = np.concatenate([p2[:pt], p1[pt:]])
                else:
                    child1, child2 = p1.copy(), p2.copy()
                # Mutation
                for c in [child1, child2]:
                    if random.random() < 0.1:
                        m = random.randint(0, self.dim-1)
                        c[m] = random.uniform(self.lb, self.ub)
                    offspring.append(c)
            
            X = np.array(offspring[:self.pop_size])
            fitness = np.array([func_wrapper(x) for x in X])
            
            curr_max = np.max(fitness)
            if curr_max > self.best_score:
                self.best_score = curr_max
                self.best_pos = X[np.argmax(fitness)].copy()
            self.loss_history.append(self.best_score)
        return self.best_score, self.best_pos

# ======================================================
# 3. PARTICLE SWARM OPTIMIZATION (PSO)
# ======================================================
class PSO(Optimizer):
    def optimize(self, func_wrapper):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        V = np.zeros_like(X)
        Pbest = X.copy()
        Pbest_fit = np.array([-float('inf')]*self.pop_size)
        Gbest = np.zeros(self.dim)
        Gbest_fit = -float('inf')
        
        for i in range(self.pop_size):
            f = func_wrapper(X[i])
            Pbest_fit[i] = f
            if f > Gbest_fit:
                Gbest_fit = f
                Gbest = X[i].copy()
                
        for it in range(self.max_iter):
            w = 0.9 - it * (0.5 / self.max_iter) # Linear decay
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                V[i] = w*V[i] + 2.0*r1*(Pbest[i]-X[i]) + 2.0*r2*(Gbest-X[i])
                X[i] = np.clip(X[i] + V[i], self.lb, self.ub)
                
                f = func_wrapper(X[i])
                if f > Pbest_fit[i]:
                    Pbest_fit[i] = f
                    Pbest[i] = X[i].copy()
                    if f > Gbest_fit:
                        Gbest_fit = f
                        Gbest = X[i].copy()
            
            self.loss_history.append(Gbest_fit)
            self.best_score = Gbest_fit
            self.best_pos = Gbest.copy()
        return self.best_score, self.best_pos

# ======================================================
# 4. WHALE OPTIMIZATION ALGORITHM (WOA)
# ======================================================
class WOA(Optimizer):
    def optimize(self, func_wrapper):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        Leader_pos = np.zeros(self.dim)
        Leader_score = -float('inf')
        
        # Initial search
        for i in range(self.pop_size):
            f = func_wrapper(X[i])
            if f > Leader_score:
                Leader_score = f
                Leader_pos = X[i].copy()
        
        for it in range(self.max_iter):
            a = 2 - 2 * (it/self.max_iter)
            for i in range(self.pop_size):
                r = random.random()
                A = 2*a*r - a
                C = 2*r
                l = random.uniform(-1,1)
                p = random.random()
                
                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * Leader_pos - X[i])
                        X[i] = Leader_pos - A * D
                    else:
                        rand_idx = random.randint(0, self.pop_size-1)
                        X_rand = X[rand_idx]
                        D = abs(C * X_rand - X[i])
                        X[i] = X_rand - A * D
                else:
                    dist = abs(Leader_pos - X[i])
                    X[i] = dist * np.exp(l) * np.cos(2*np.pi*l) + Leader_pos
                
                X[i] = np.clip(X[i], self.lb, self.ub)
                
                # Check if new leader
                f = func_wrapper(X[i])
                if f > Leader_score:
                    Leader_score = f
                    Leader_pos = X[i].copy()
            
            self.loss_history.append(Leader_score)
            self.best_score = Leader_score
            self.best_pos = Leader_pos
        return self.best_score, self.best_pos

# ======================================================
# 5. GREY WOLF OPTIMIZER (GWO)
# ======================================================
class GWO(Optimizer):
    def optimize(self, func_wrapper):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        Alpha_pos, Beta_pos, Delta_pos = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        Alpha_score, Beta_score, Delta_score = -float('inf'), -float('inf'), -float('inf')
        
        for it in range(self.max_iter):
            # Evaluate
            for i in range(self.pop_size):
                f = func_wrapper(X[i])
                if f > Alpha_score:
                    Alpha_score = f; Alpha_pos = X[i].copy()
                if f > Beta_score and f < Alpha_score:
                    Beta_score = f; Beta_pos = X[i].copy()
                if f > Delta_score and f < Beta_score:
                    Delta_score = f; Delta_pos = X[i].copy()
            
            a = 2 - 2 * (it/self.max_iter)
            for i in range(self.pop_size):
                for j in range(self.dim):
                    r1, r2 = random.random(), random.random()
                    A1, C1 = 2*a*r1 - a, 2*r2
                    D_alpha = abs(C1 * Alpha_pos[j] - X[i,j])
                    X1 = Alpha_pos[j] - A1 * D_alpha
                    
                    r1, r2 = random.random(), random.random()
                    A2, C2 = 2*a*r1 - a, 2*r2
                    D_beta = abs(C2 * Beta_pos[j] - X[i,j])
                    X2 = Beta_pos[j] - A2 * D_beta
                    
                    r1, r2 = random.random(), random.random()
                    A3, C3 = 2*a*r1 - a, 2*r2
                    D_delta = abs(C3 * Delta_pos[j] - X[i,j])
                    X3 = Delta_pos[j] - A3 * D_delta
                    
                    X[i,j] = (X1 + X2 + X3) / 3
                X[i] = np.clip(X[i], self.lb, self.ub)
                
            self.loss_history.append(Alpha_score)
            self.best_score = Alpha_score
            self.best_pos = Alpha_pos
        return self.best_score, self.best_pos

# ======================================================
# 6. SPARROW SEARCH ALGORITHM (SSA)
# ======================================================
class SSA(Optimizer):
    def optimize(self, func_wrapper):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func_wrapper(x) for x in X])
        
        # Sort
        idx = np.argsort(fitness)[::-1]
        X = X[idx]
        fitness = fitness[idx]
        
        self.best_score = fitness[0]
        self.best_pos = X[0].copy()
        
        # Params
        PD = 0.2 # Producers
        SD = 0.1 # Sparrows who perceive danger
        ST = 0.8 # Safety Threshold
        
        for it in range(self.max_iter):
            r2 = random.random()
            
            # 1. Update Producers (Top 20%)
            pd_idx = int(self.pop_size * PD)
            for i in range(pd_idx):
                if r2 < ST:
                    X[i] = X[i] * np.exp(-i / (random.random() * self.max_iter))
                else:
                    X[i] = X[i] + np.random.normal(0, 1)
                X[i] = np.clip(X[i], self.lb, self.ub)

            # 2. Update Scroungers (Remaining 80%)
            # Need to re-evaluate producers to get current best/worst? 
            # Simplified: Use current best from previous iter
            x_best = self.best_pos
            x_worst = X[-1]
            
            for i in range(pd_idx, self.pop_size):
                if i > self.pop_size / 2:
                    # Hungry, fly to random place
                    X[i] = np.random.normal(0, 1) * np.exp((x_worst - X[i]) / (i**2))
                else:
                    # Follow best
                    L = np.ones(self.dim)
                    A = np.random.choice([1, -1], self.dim)
                    # A+ = A^T(AA^T)^-1  (Approximation)
                    X[i] = x_best + np.abs(X[i] - x_best) * 0.5 # Simplified A+
                X[i] = np.clip(X[i], self.lb, self.ub)
            
            # 3. Update Scouters (Random 10%)
            sd_indices = np.random.choice(self.pop_size, int(self.pop_size * SD), replace=False)
            for i in sd_indices:
                f_i = func_wrapper(X[i])
                if f_i > self.best_score:
                    X[i] = self.best_pos + np.random.normal(0,1) * (np.abs(X[i] - self.best_pos))
                else:
                    # e is epsilon
                    X[i] = X[i] + (np.random.rand() * 2 - 1) * (np.abs(X[i] - x_worst) / (0.0001 + (f_i - fitness[-1])))
                X[i] = np.clip(X[i], self.lb, self.ub)
            
            # Re-evaluate all
            fitness = np.array([func_wrapper(x) for x in X])
            current_best = np.max(fitness)
            if current_best > self.best_score:
                self.best_score = current_best
                self.best_pos = X[np.argmax(fitness)].copy()
            
            self.loss_history.append(self.best_score)
        
        return self.best_score, self.best_pos

# ======================================================
# 7. FIREFLY ALGORITHM (FA)
# ======================================================
class FA(Optimizer):
    def optimize(self, func_wrapper):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func_wrapper(x) for x in X])
        
        self.best_score = np.max(fitness)
        self.best_pos = X[np.argmax(fitness)].copy()
        
        # Params
        alpha = 0.2  # Randomness
        beta0 = 1.0  # Attractiveness at r=0
        gamma = 1.0  # Absorption
        
        for it in range(self.max_iter):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] > fitness[i]: # Move i towards j if j is brighter
                        r = np.linalg.norm(X[i] - X[j])
                        beta = beta0 * np.exp(-gamma * r**2)
                        
                        # Move
                        rand_step = alpha * (np.random.rand(self.dim) - 0.5)
                        X[i] = X[i] + beta * (X[j] - X[i]) + rand_step
                        X[i] = np.clip(X[i], self.lb, self.ub)
                        
                        # Evaluate
                        f_new = func_wrapper(X[i])
                        if f_new > fitness[i]:
                            fitness[i] = f_new
                            if f_new > self.best_score:
                                self.best_score = f_new
                                self.best_pos = X[i].copy()
            
            self.loss_history.append(self.best_score)
        return self.best_score, self.best_pos

# ======================================================
# 8. DIFFERENTIAL EVOLUTION (DE)
# ======================================================
class DE(Optimizer):
    def optimize(self, func_wrapper):
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func_wrapper(x) for x in X])
        
        self.best_score = np.max(fitness)
        self.best_pos = X[np.argmax(fitness)].copy()
        
        F = 0.5 # Mutation
        CR = 0.7 # Crossover
        
        for it in range(self.max_iter):
            new_X = []
            for i in range(self.pop_size):
                # Pick 3 random distinct
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = X[np.random.choice(idxs, 3, replace=False)]
                
                # Mutation
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points): 
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, X[i])
                
                # Selection
                f_trial = func_wrapper(trial)
                if f_trial > fitness[i]:
                    new_X.append(trial)
                    fitness[i] = f_trial
                    if f_trial > self.best_score:
                        self.best_score = f_trial
                        self.best_pos = trial.copy()
                else:
                    new_X.append(X[i])
                    
            X = np.array(new_X)
            self.loss_history.append(self.best_score)
            
        return self.best_score, self.best_pos