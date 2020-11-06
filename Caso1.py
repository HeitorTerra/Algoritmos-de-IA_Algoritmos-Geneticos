"""Import dos comandos"""
import numpy as np

from geneticalgorithm import geneticalgorithm as ga
import control as co
def f(X):
    return np.sum(X)
"""Declaração da variável  s de  controle  para a  FT"""
s=co.tf('s')
"""Utiliza uma array"""
varbound=np.array([[0,10]]*3)

"""Implementação dos parametros """
algorithm_param = {'max_num_iteration': 3000,\
                   'population_size':100,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

model=ga(function=f,\
            dimension=3,\
            variable_type='real',\
            variable_boundaries=varbound,\
            algorithm_parameters=algorithm_param)
'executa o modelo'
model.run()

""" Melhor indivíduo(s) aplicados na FT, onde estes representam respectivamente Kp, Ki, Kd
na matris res[]
"""
res=model.best_variable
Kp, Ki, Kd= res[0], res[1], res[2]
"""FT representada """
K_ = Kp+ Ki*4/s + Kd*4*s

print(f'A função de transferencia: {K_}')