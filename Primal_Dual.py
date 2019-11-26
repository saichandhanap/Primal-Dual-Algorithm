import numpy as np

R = int(input("Enter the number of rows in matrix A:"))
C = int(input("Enter the number of columns in matrix A:"))

print("Enter the entries in a single line separated by space: ")

entries = list(map(int, input().split()))

matrix_A = np.array(entries).reshape(R,C) # matrix A

A = np.copy(matrix_A)

max_or_min = int(input("Enter maximize(1) or minimize(0):"))#1 for maximize and 0 for minimize

print("Enter cost coefficients for objective function:")

cost_coeff = []
dummy=[]

for i in range(0,R+1):
    dummy.append(0)

dummy = np.array(dummy)

for i in range(0,C):
    cost_coeff.append(int(input()))

cost_coeff_m = np.array(cost_coeff)
if(max_or_min==1):
    cost_coeff_m = (-1)*(cost_coeff_m)
cost_coeff_m = np.append(cost_coeff_m, dummy)
cost_coeff_m = np.matrix(cost_coeff_m)

vector_b = []

print("Enter b vector values:")

for i in range(0,R):
    vector_b.append(int(input()))
vector_b_m = np.matrix(vector_b)
vector_b_m = vector_b_m.transpose()

print("Enter L if constraints have < symbol or R if they have > symbol")

eq = input()

if(eq == "L" or eq == "l" or eq == "E" or eq == "e"):
    I = np.identity(R,dtype=int)
    matrix_A = np.concatenate((matrix_A,I),axis=1)
elif(eq == "R" or eq == 'r'):
    matrix_A = (-1)*(matrix_A)
    I = np.identity(R,dtype=int)
    matrix_A = np.concatenate((matrix_A,I),axis=1)

matrix_A = np.concatenate((matrix_A, vector_b_m), axis = 1)

matrix_arp = np.copy(matrix_A)

matrix_A = np.concatenate((matrix_A, cost_coeff_m), axis = 0)

#print(cost_coeff)#print(vector_b)#print(vector_b_m)

#print(matrix_A)

def simplex(matrix_A,q):
	h=0
	while (h<1):
		h+=1
		p_list=[]
		#print(q)
		for i in range(0,np.size(matrix_A,0)-2):
			if matrix_A[i,q]>0:
				p_list.append(np.true_divide(matrix_A[i,-1],matrix_A[i,q]))
		p = int(np.argmin(p_list))
		#print("Value of p is: ")
		#print(p)
		s = np.shape(matrix_A)
		new_matrix_A = np.zeros(s)
		for i in range(0,s[0]):
			for j in range(0,s[1]):
				if(i!=p):
					temp1 = matrix_A[i,q]*matrix_A[p,j]
					temp2 = temp1/matrix_A[p,q]
					new_matrix_A[i,j] = matrix_A[i,j] - temp2
				elif(i==p):
					temp1 = matrix_A[p,j]/matrix_A[p,q]
					new_matrix_A[i,j] = temp1
		matrix_A = np.copy(new_matrix_A)
		#print(new_matrix_A)
		#print("************")
	return new_matrix_A

def initial_simplex(matrix_A):
	while ((matrix_A[-1,:]<0).any()):
	    q = np.argmin(matrix_A[-1,:])
	    q=int(q)
	    print("Value of q is: ")
	    #print(q)

	    p_list=[]
	    for i in range(0,np.size(matrix_A,0)-1):
	        if matrix_A[i,q]>0 :
	            p_list.append(np.true_divide(matrix_A[i,-1],matrix_A[i,q]))

	    #p_list = [np.true_divide(matrix_A[:-1,-1],matrix_A[:-1,q]) for i in matrix_A[:-1,q] if i>0] 
	    p = int(np.argmin(p_list))
	    #print("Value of p is: ")
	    #print(p)
	    s = np.shape(matrix_A)
	    new_matrix_A = np.zeros(s)
	    #print(new_matrix_A)

	    for i in range(0,s[0]):
	        for j in range(0,s[1]):
	            if(i!=p):
	                temp1 = matrix_A[i,q]*matrix_A[p,j]
	                temp2 = temp1/matrix_A[p,q]
	                new_matrix_A[i,j] = matrix_A[i,j] - temp2
	            elif(i==p):
	                temp1 = matrix_A[p,j]/matrix_A[p,q]
	                new_matrix_A[i,j] = temp1
	    matrix_A = np.copy(new_matrix_A)
    
	    #print(new_matrix_A)
	    #print("---------")
	return new_matrix_A
'''
#find solutions:x1,x2,x3,...
def check(i,matrix_A):
    for k in range(0,np.size(matrix_A,0)-1):
        if matrix_A[k,i]==1:
            return matrix_A[k,-1]

final =  np.zeros(np.size(matrix_A,1))

for i in range(0,np.size(matrix_A,1)):
    if matrix_A[-1,i]==0:
        final[i]=check(i,matrix_A)
for i in range(0,np.size(final)-1):
    if final[i]!=0:
        print("x"+str(i+1)+"="+str(final[i]))

if(max_or_min==1):
    x = (-1)*(matrix_A[-1,-1])
    print("Objective Function: " + str(x))
else:
    x = matrix_A[-1,-1]
    print("Objective Function: " + str(x)) 
#Here we got for primal...
'''
C_j = np.zeros(C)
C_j_1 = np.ones(R)
C_j = np.concatenate((C_j,C_j_1),axis = None)
C_j = np.concatenate((C_j,np.zeros(1)),axis = None)
#print(C_j)
#print("Before")
#print(matrix_arp)
z_j = np.sum(matrix_arp,axis=0)
r_d = C_j - z_j #r_d

#print(r_d)
flag=0
feasible_lambda = np.zeros(R)

index=[]
for i in range(0,R):
	index.append(C+i)

if not(eq == "E" or eq == "e"): #since simplex doesnot work for E we take lambda =0
	print("------------------------")
	new_matrix_A=initial_simplex(matrix_A)
	for i in range(0,R):
		feasible_lambda[i] = new_matrix_A[-1,index[i]]
#print(feasible_lambda)
#print(A) A in notes
x = np.matmul(feasible_lambda,A) #last row calculation
#print(x)

last_row = cost_coeff - x
last_row = np.append(last_row,dummy) #final last row
#print(last_row)

r_d = np.matrix(r_d)
last_row = np.matrix(last_row)

matrix_arp = np.concatenate((matrix_arp, r_d), axis = 0)

matrix_arp = np.concatenate((matrix_arp, last_row), axis = 0)

#print("After")
#print(matrix_arp)

u_dual = np.ones(R)
#print(u_dual)
q_new=-1
while(matrix_arp[-2,-1]!=0):
	epsilon_list=[]
	t = q_new
	for i in range(0,np.size(z_j)):
		if z_j[i]!=0 :
			epsilon_list.append(np.true_divide(last_row[0,i],z_j[i]))
	    	#p_list = [np.true_divide(matrix_A[:-1,-1],matrix_A[:-1,q]) for i in matrix_A[:-1,q] if i>0]
	#print(epsilon_list)
	epsilon = np.min(epsilon_list[0:C])
	#print(epsilon)

	matrix_arp[-1,:] = (epsilon)*(matrix_arp[-2,:])+matrix_arp[-1,:]
	for i in range(0,C):
		if matrix_arp[-1,i]==0 and t!=i:
			q_new = i
			break
	print(q_new)
	
	matrix_arp = simplex(matrix_arp,q_new)
	if matrix_arp[-2,-1]>0:
		print(" The ARP is unbounded")
		flag=1
		break
#print(matrix_arp)
if flag!=1:
	def check(i,matrix_A):
	    for k in range(0,np.size(matrix_arp,0)-1):
	        if matrix_A[k,i]==1:
	            return matrix_arp[k,-1]

	final =  np.zeros(C)

	for i in range(0,C):
	    if matrix_arp[-1,i]==0:
	        final[i]=check(i,matrix_arp)
	for i in range(0,np.size(final)):
	    if final[i]!=0:
	        print("x"+str(i+1)+"="+str(final[i]))
	cost_coeff = np.matrix(cost_coeff)
	cost_coeff = cost_coeff.transpose()
	print("Objective Value:"+ str(int(np.matmul(final,cost_coeff))))
