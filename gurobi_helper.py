import numpy as np
import gurobi as gb
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

GUROBI_TIMEOUT = 5 # in second

# Helper Functions

class Gurobi_Model():
    '''
        Class Method I made to help me encapsulate the whole Gurobi model creation, 
        Finding Slack Variable and perform sensitivity analysis
    '''
    
    def __init__(self, obj_linear, cons, sense, b, variable_types = 'C',
                lower_bound = None, upper_bound = None,
                optimisation_type = gb.GRB.MAXIMIZE, obj_quadratic = None) -> None:
        '''
            Initialising the Model
        '''
        if (obj_quadratic is None) and (obj_linear is None):
            raise Exception("No objective defined")

        constraints, variables = cons.shape

        ## Optimisation Model at work (Maximise)
        try:
            self.model = gb.Model()
            self.model_X = self.model.addMVar(variables,
                                        vtype=variable_types,
                                        lb= lower_bound, ub= upper_bound)

            self.model_constraints = self.model.addMConstrs(cons, self.model_X, sense, b)
            self.model.setMObjective(obj_quadratic, obj_linear, 0, sense=optimisation_type)
            self.model.Params.OutputFlag = 0
            self.model.Params.TimeLimit = GUROBI_TIMEOUT

            self.model.optimize()

        except Exception:
            print("Error in optimising")
            raise Exception
        
        ## Assigning stuff for other use cases
        try:
            self.obj_q = obj_quadratic
            self.obj = obj_linear
            self.constraint = cons
            self.sense = sense
            self.b = b
        except Exception:
            print("Error in Storing equations")
            raise Exception
        return

    @property
    def optimal_obj(self): return self.model.objVal

    @property
    def optimal_x(self): return self.model_X.x

    @property
    def output(self):
        return {'objVal': self.optimal_obj,
                'x': self.optimal_x}

    def print_equations(self)->None:
        '''
            Print Objective Functions and Constraint Equations
        '''
        char = "a"
        print("Optimise System of equations:")
        for item in self.obj:
            print(str(item)+char,end=" + ")
            char = chr(ord(char) + 1)
        print("\b\b")
        print("Subject to:")#, end=" ")
        for i in range(self.constraint.shape[0]):
            char = "a"
            print("\t"+str(i)+")",end=" ")
            for j in range(self.constraint.shape[1]):
                print(str(self.constraint[i,j])+char,end=" + ")
                char = chr(ord(char) + 1)
            print("\b\b "+self.sense[i]+"= "+str(self.b[i]))
        return

    def print_slack(self)->None:
        '''
            Print Slacks for each constraints
        '''
        print("\nConstraint values at Optimal solution:\n",self.constraint@self.optimal_x)
        print("Binding values:", self.b)
        print("Slack: ", abs((self.constraint@self.optimal_x)-self.b))
        return

    def sensitivity_analysis(self, verbose = False)->None:
        '''
            Sensitivity Analysis
        '''
        shadow_cost = {index: con.Pi for index, con in enumerate(self.model_constraints)}
        shadow_cost_bound = {index: (con.SARHSLow, con.SARHSUp) for index, con in enumerate(self.model_constraints)}
        if verbose:
            try:
                print("\nShadow Cost:")
                # shadow_cost = [con.Pi for con in self.model_constraints]
                print("\t", shadow_cost)
            except:
                print("\tCan not fetch value for Shadow Cost")

            try:
                print("\nUpper and Lower Bound of Shadow Cost")
                # shadow_cost_bound = [(con.SARHSLow, con.SARHSUp) for con in self.model_constraints]
                print("\t", shadow_cost_bound)
            except:
                print("\tCan not fetch value for Upper and Lower")

            try:
                print("\nRange of Objective where Optimal Corner remains the same")
                print("\tLower Bound:",self.model_X.SAObjLow)# <- Range of Slope for which optimal Corner remains same
                print("\tUpper Bound:",self.model_X.SAObjUP) # <- Range of Slope for which optimal Corner remains same
            except:
                print("\tCan not fetch value for Optimal Corner")        

        return {
            'shadow_cost': shadow_cost,
            'shadow_cost_bound': shadow_cost_bound,
            'optimal_corner_range': {"low": self.model_X.SAObjLow,
                                    'high': self.model_X.SAObjUP}
        }
# # Questions

# ############################
# ## Problem - 1 (GYM)
# ############################

# health = np.array([10, 8, 3]*3) # day wise
# variables = health.shape[0]
# constraints = 10

# A = np.zeros((constraints, variables))
# b = np.array([1, 2, 1.5, 1.2, 0.25, 0.15]+[0]*4)
# sense = np.array(['=']*3 + [">"]+['<']*2 + ['>']*4)

# print(A.shape, b.shape, sense.shape)
# A[0, :3] = [1]*3
# A[1, 3:6] = [1]*3
# A[2, 6:] = [1]*3
# A[3, 3] = 1
# A[4, 0] = 1
# A[5, -2] = 1
# A[6, :3] = [-0.1, -0.15, 1]
# A[7, 3:6] = [-0.1, -0.15, 1]
# A[8, 6:] = [-0.1, -0.15, 1]
# A[9,:] = [-1,2,0]*3


# mdl_gym = Gurobi_Model(obj_linear= health,
#                     cons = A,
#                     sense = sense,
#                     b = b)

# # mdl_gym.print_equations()
# print(f"Optimum time spent in gym -> Cardio Wednesday\n\t{round(mdl_gym.output['x'][3], 2)} hours")
# sensitivity = mdl_gym.sensitivity_analysis()

# print(f'Healthiness increass for each additional hour spent in gym on Friday: {round(sensitivity["shadow_cost"][2],1)}')
