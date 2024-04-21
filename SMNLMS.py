
import numpy as np
import pandas as pd
from pysr import *
import sys

def evaluateEmpiricalQuantitiesSMNLMS(tauVector , betaVector , NVector , sigmanu2Vector , sigmax2Vector , numberOfIterations , numberOfLastIterations , numberOfRepeats):
	
	MSE = np.zeros((len(tauVector),len(betaVector),len(NVector),len(sigmanu2Vector),len(sigmax2Vector)))
	MSD = np.zeros((len(tauVector),len(betaVector),len(NVector),len(sigmanu2Vector),len(sigmax2Vector)))
	Pup = np.zeros((len(tauVector),len(betaVector),len(NVector),len(sigmanu2Vector),len(sigmax2Vector)))

	for tauIndex in range(len(tauVector)):
		for betaIndex in range(len(betaVector)):
			for NIndex in range(len(NVector)):
				for sigmanu2Index in range(len(sigmanu2Vector)):
					for sigmax2Index in range(len(sigmax2Vector)):
						print(str(tauIndex + 1) + '/' + str(len(tauVector)) + ' - ' + str(betaIndex + 1) + '/' + str(len(betaVector)) + ' - ' + str(NIndex + 1) + '/' + str(len(NVector)) + ' - ' + str(sigmanu2Index + 1) + '/' + str(len(sigmanu2Vector)) + ' - ' + str(sigmax2Index + 1) +  '/' + str(len(sigmax2Vector)))
						tau = tauVector[tauIndex]
						beta = betaVector[betaIndex]
						N = NVector[NIndex]
						sigmanu2 = sigmanu2Vector[sigmanu2Index]
						sigmax2 = sigmax2Vector[sigmax2Index]

						for repeat in range(numberOfRepeats):
							wk = np.zeros((N,1))
							w0 = np.random.randn(N,1)
							x  = np.sqrt(sigmax2) * np.random.randn( numberOfIterations + N - 1, 1 )
							d  = np.convolve(w0[:,0], x[:,0])
							d  += np.sqrt( sigmanu2 ) * np.random.randn(len(d))
							gamma = np.sqrt(tau * sigmanu2)

							for k in range(N, numberOfIterations + N - 1):
								iteration = k - N + 1
								xk = x[k:k-N:-1]
								yk = np.dot(wk.T, xk)
								ek = d[k] - yk
								
								if abs(ek) > gamma:
									mu = 1 - gamma/abs(ek)
									wk = wk + beta * mu / (np.dot(xk.T, xk)) * ek * xk

									if iteration > numberOfIterations - numberOfLastIterations + 1:
										Pup[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index] += 1 / (numberOfRepeats * numberOfLastIterations)

								if iteration > numberOfIterations - numberOfLastIterations + 1:
									MSD[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index] += np.linalg.norm(wk - w0) ** 2 / (numberOfRepeats * numberOfLastIterations)
									MSE[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index] += ek.item() ** 2 / (numberOfRepeats * numberOfLastIterations)
	return MSE,MSD,Pup


def RegressaoSimbolica(X,y):
	model = PySRRegressor(
		parsimony = 0.0001, # padrão = 0.0032
		progress=False,
		niterations=100,
		weight_randomize=0.001, # default: 0.00023
		populations=300,
		population_size=100,
		model_selection='score', # 'accuracy', 'best', or 'score'
		nested_constraints={"exp":{"exp": 0},"log10":{"log10": 0},"erf":{"erf": 0},"erfc":{"erfc": 0}},
		binary_operators=["+", "*","-","^"],
		unary_operators=[
			"exp",
			"inv(x) = 1/x",
			"log10",
			"erf",
			"erfc",
		],
		extra_sympy_mappings={"inv": lambda x: 1 / x})

	model.fit(X, y)
	print(model)

	best_idx = model.equations_.query(
		f"loss < {2 * model.equations_.loss.min()}"
	).score.idxmax()
	model.sympy(best_idx)

	model.get_best().equation



tauVector = np.arange(0, 6)
betaVector = np.arange(0.1, 1.1, 0.1)
NVector = [10, 20, 50]
sigmanu2Vector = np.power(10,np.arange(-6, -1), dtype=float)
sigmax2Vector = np.power(10,np.arange(-2, 2), dtype=float)
numberOfIterations = 50000
numberOfLastIterations = 1000
numberOfRepeats = 100

tauVector = np.arange(0, 2)
betaVector = np.arange(0.1, 1.1, 0.1)
NVector = [10]
sigmanu2Vector = np.power(10,np.arange(-6, -1), dtype=float)
sigmax2Vector = np.power(10,np.arange(1, 2), dtype=float)
numberOfIterations = 50
numberOfLastIterations = 1000
numberOfRepeats = 100

print('>> Geradando dados SMNLMS...')
MSE,MSD,Pup = evaluateEmpiricalQuantitiesSMNLMS(tauVector,betaVector,NVector,sigmanu2Vector,sigmax2Vector,numberOfIterations,numberOfLastIterations,numberOfRepeats)

data = {
    'tau': [],
    'beta': [],
    'N': [],
    'sigmanu2': [],
    'sigmax2': [],
	'MSE': [],
    'MSD': [],
    'Pup': [],
}


for tauIndex in range(len(tauVector)):
		for betaIndex in range(len(betaVector)):
			for NIndex in range(len(NVector)):
				for sigmanu2Index in range(len(sigmanu2Vector)):
					for sigmax2Index in range(len(sigmax2Vector)):
						data['tau'].append(tauVector[tauIndex])
						data['beta'].append(betaVector[betaIndex])
						data['N'].append(NVector[NIndex])
						data['sigmanu2'].append(sigmanu2Vector[sigmanu2Index])
						data['sigmax2'].append(sigmax2Vector[sigmax2Index])
						data['MSE'].append(MSE[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index])
						data['MSD'].append(MSD[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index])
						data['Pup'].append(Pup[tauIndex , betaIndex , NIndex , sigmanu2Index , sigmax2Index])
						
file = './resultsSMNLMS.csv'

print('>> Gerando arquivo csv gerado SMNLMS')
df = pd.DataFrame(data)
df.to_csv(file, index=False, header=False)

print('>> Iniciando Regração Simbólica...')

arr = np.loadtxt( file , delimiter = ',' , dtype = float)

#tau beta N sigmanu2 sigmax2
X = arr[ : ,  [ 0 , 1 , 2 , 3 , 4 ] ]
#/usr/local/Cellar/python@3.11/3.11.4_1/Frameworks/Python.framework/Versions/3.11/Python
#/usr/local/Cellar/python@3.11/3.11.6_1/Frameworks/Python.framework/Versions/3.11/Python

print('>> MSE')
RegressaoSimbolica(X, arr[ : , 5 ])
print('>> MSD')
RegressaoSimbolica(X, arr[ : , 6 ])
print('>> Pup')
RegressaoSimbolica(X, arr[ : , 7 ])

print('>> FIM!')














