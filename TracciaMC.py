import random
import math 
import matplotlib.pyplot as plt
import numpy as np

R = 300 # numero righe
C = 300 # numero colonne
PREC = 10 # precisione dei float

def genMatrixZeros():
	# inizializzo la matrice
	matrix = [] 
	  
	for i in range(R):
		a = [] 

		for j in range(C): 
			a.append(0) # appendi zero

		matrix.append(a) 

	return matrix


def genVectorZeros():
	# inizializzo il vettore
	vector = [] 
	  
	for j in range(R): 
		vector.append(0)

	return vector


def genVerticalVectorZeros():
	# inizializzo il vettore
	vector = [] 
	  
	for j in range(R): 
		tmp = []
		tmp.append(0)
		vector.append(tmp)

	return vector


def multiplyMatrices(X, Y):
	# dichiaro una nuova matrice e la inizializzo 
	result = genMatrixZeros()
	# itero sulle righe di X
	for i in range(len(X)):
		# itero sulle colonne di una riga di Y
		for j in range(len(Y[0])):
			for k in range(len(Y)):
				result[i][j] = result[i][j] + (X[i][k] * Y[k][j])

	return result


def vectPerMatrix(vect, matrix):
	result = genVectorZeros() # vettore risultato

	# itero sulle colonne del vettore
	for i in range(len(matrix[0])):
		for j in range(len(vect)):
			result[i] = result[i] + vect[j] * matrix[j][i]

	return result


def vectPerVect(vect1, vect2):
	result = 0

	# itero sulle colonne del vettore
	for i in range(len(vect1)):
		result += vect1[i] * vect2[i][0]

	return result

def transposeMatrix(toTransp):
	# dichiaro una nuova matrice e la inizializzo
	transposed = genMatrixZeros()

	for i in range(len(toTransp)):
		for j in range(len(toTransp[0])):
			transposed[j][i] = toTransp[i][j] # traspongo l'elemento

	return transposed


def transposeVector(toTransp):
	transposed = [] # dichiaro un vettore vuoto

	for i in range(len(toTransp)):
		transposed.append(toTransp[i][0])	

	return transposed


# def is_int(v):
#    return type(v) is int


# Stampa una matrice
def printMatrix(matrix):	  
	# scorro e stampo la matrice
	for i in range(len(matrix)): 
		for j in range(len(matrix[0])): 
			print(matrix[i][j], end = "\t") 
		print() 


# stampa un vettore
def printVector(vector):
	for i in range(len(vector)): 
		print(vector[i], end = "\t") 
	print() 

# stampa un vettore
def printVerticalVector(vector):
	for i in range(len(vector)): 
		print(vector[i][0], end = "\n") 
	print() 



# Genera una matrice B con B_ij campionato uniformemente nell'intervallo [0,1]. 
def genMatrix():
	# inizializzo la matrice
	matrix = [] 
	  
	for i in range(R):
		a = [] 

		for j in range(C): 
			a.append(random.uniform(0, 1)) # appendi random

		matrix.append(a) 

	return matrix


# Calcola (||A||_2F)^2 dalla definizione. 
def computeFrobeniusQuad(A):
	res = 0

	# itero sulle righe
	for i in range(R):
		# itero sulle colonne
		for j in range(C): 
			# incremento res dell'elemento al quadrato	
			res += (A[i][j])**2

	# restituisco la radice quadrata di res
	return res


# Calcola Tr(A) dalla definizione. 
def computeTrace(A):
	trace = 0

	# itero sulle righe
	for i in range(R):
		# incremento trace dell'elemento sulla diagonale	
		trace += (A[i][i])

	# restituisco la radice quadrata di res
	return trace


# Calcola Var(A) dalla definizione. 
def computeVar(A, M):
	var = 0

	# itero sulle righe
	for i in range(R):
		for r in range(0, i):
			# incremento trace dell'elemento sulla diagonale	
			var += (A[i][r])**2

	# restituisco la radice quadrata di res
	return (4*var/M)


# campiona u = [u1, . . . , un] vettore di Rademacher
def genRademacher():
	u = []

	possValues = [-1, 1]

	for i in range(R):
		tmp = []
		tmp.append(random.choice(possValues))
		u.append(tmp)

	# print("\nRademacher: ") # DEBUG
	# printVerticalVector(u) # DEBUG
	return u


# Stima Tr(A) 
def estimateTrace(A, M):
	trace = 0
	var = 0

	# definisci un vettore X
	X = []
	X.append(0)		# poni X[0] a 0

	# definisci un vettore X_camp
	X_camp = []
	X_camp.append(0) 	# poni X_camp[0] a 0

	# itera da 1 a M
	for m in range(1, M+1):
		# 1. campiona u = [u1, . . . , un] vettore di Rademacher
		u = genRademacher()
		# 2. ottieni X_m = u_T*A*u dall’oracolo
		tmp = vectPerMatrix(transposeVector(u), A)

		# print("Vettore: ", end="") # DEBUG
		# printVector(tmp) # DEBUG

		tmp2 = vectPerVect(tmp, u)
		# print("Scalare: " + str(tmp2)) # DEBUG

		X.append(vectPerVect(tmp, u))
	
		# calcola X_camp[m]
		X_camp.append(X_camp[m-1] + ((X[m] - X_camp[m-1]) / m))

		# l'ultimo calcolato sarà la traccia
		trace = X_camp[m]

	# calcola la varianza
	# itera da 1 a M
	for m in range(1, M+1):
		var = (((X[m] - trace)**2)/(M-1))

	# print("\nTraccia: " + str(trace)) # DEBUG

	# print("Varianza: " + str(var), end="\n--------------------------------\n") # DEBUG

	return trace, var


# stimare runs volte Tr(A) con M uguale a quello passato alla funzione;
def MonteCarloTrace(A, M, runs, tracesMC, varMC):
	
	for run in range(runs):
		
		# stima trace e var
		trace, var = estimateTrace(A, M)

		tracesMC.append(trace)

		varMC.append(var)


def main():
	print("Stima della Traccia - Lorenzo La Corte")
		
	# Genera una matrice 300x300 B con B_ij campionato uniformemente nell'intervallo [0,1]. 
	B = genMatrix()

	# print("\nstampo B:") # DEBUG
	# printMatrix(B) # DEBUG

	B_T = transposeMatrix(B)
	# print("\nstampo B Trasposta:") # DEBUG
	# printMatrix(B_T) # DEBUG

	# La matrice A = B^T*B è semidefinita positiva. 
	A = multiplyMatrices(B_T, B)
	# print("\nstampo A = B^T*B:") # DEBUG
	# printMatrix(A) # DEBUG

	# Calcola ||A||_2F e Tr(A) dalle definizioni. 
	frobQuadReal = computeFrobeniusQuad(A)
	print("\nLa norma di Frobenius (||A||_F)^2 della matrice A è: " + str("{:.5e}".format(frobQuadReal)))
	traceReal = computeTrace(A)
	print("\nLa traccia Tr(A) della matrice A è: " + str("{:.5e}".format(traceReal)))

	# Usa MonteCarloTrace per stimare 100 volte Tr(A) con M=5,10,25 e 100. 

	RUNS = 100
	listOfM = [5, 10, 25, 100]

	for M in listOfM:
		print("\nSto usando M=" + str(M)) # DEBUG

		tracesMC = []
		varMC = []

		MonteCarloTrace(A, M, RUNS, tracesMC, varMC) # la funzione gira RUNS volte e riempie il dizionario con le traccie

		# print("Tracce: " + str(tracesMC)) # DEBUG
		# print("Varianze: " + str(varMC)) # DEBUG

		espMed = (sum(tracesMC) / len(tracesMC))
		varMed = (sum(varMC) / len(varMC)) # round(varMC[int(RUNS/2)], PREC)

		print("\nTraccia Media Empirica: " + str("{:.5e}".format(espMed))) # DEBUG
		# print("Varianza Media: " + str(varMed)) # DEBUG


		# Costruisci un istogramma con le stime ottenute e commenta il significato delle posizioni nell'istogramma occupate da Tr(A) e Tr(A) ± σM. 

		# ------- Istogramma ------- #
		ROUNDING = -3

		tracesMC_Rounded = []
	
		for x in tracesMC:
			tmp = int(round(x, ROUNDING)/1000)
			tracesMC_Rounded.append(tmp)


		binwidth = 1
		n, bins, patches = plt.hist(tracesMC_Rounded, bins=np.arange(0, max(tracesMC_Rounded) + binwidth, binwidth))
		plt.title("Istogramma - Stima di una Traccia")
		plt.xlabel("Traccia Stimata (approssimata nell'ordine di 10^3)")
		plt.ylabel("Frequenza")

		# marco Tr(A)
		plt.axvline(int(round(traceReal, ROUNDING)/1000), color='y', linestyle='dashed', linewidth=5)

		# marco Tr(A)-sigma e Tr(A)+sigma
		plt.axvline(int(round((traceReal-math.sqrt(varMed)), ROUNDING)/1000), color='r', linestyle='dashed', linewidth=3)
		plt.axvline(int(round((traceReal+math.sqrt(varMed)), ROUNDING)/1000), color='r', linestyle='dashed', linewidth=3)

		plt.savefig(f'Istogramma_M=' + str(M) + '.png')
		plt.close()

		# Confronta varMed con 2(||A||_F)^2/M.
		print("\nConfronto Tra:")
		varReal = computeVar(A, M)
		print("Varianza Media Empirica: " + str("{:.5e}".format(varMed))) # DEBUG
		print("La varianza reale della matrice A è: " + str("{:.5e}".format(varReal)))
		print("2(||A||_F)^2/M: " + str("{:.5e}".format(2*frobQuadReal/M))) # DEBUG

		print("--------------------------------------------------------------------")				
		

main()