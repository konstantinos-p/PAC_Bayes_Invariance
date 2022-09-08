import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant

"""
In this scripts we estimate the spectral norm of a concatenation of convolutional or locally connected matrices. We see that the mean is approximately the same
while the variance is greater for the convolutional matrices. We can compute the results for different numbers of input and output channels.
"""



def gen_mat_uncor(N,k,sigma):
    '''
    Get uncorrelated matrix.
    '''
    mask = np.hstack( (np.ones((1,k) ), np.zeros( (1,N-k) )) )
    mask = circulant(mask)
    mat = np.random.normal(0,sigma,(N,N))
    mat_uncor = mat*mask
    return mat_uncor

def gen_mat_cor(N,k,sigma):
    '''
    Get correlated matrix.
    '''
    filters = np.hstack((np.random.normal(0,sigma,(1,k)),np.zeros((1,N-k))))
    mat_cor =  circulant(filters)
    return mat_cor

def create_cor_tot(a,b,N,k,sigma):
    '''
    Create the complete correlated matrix.
    '''
    mat1 = np.zeros((N*b,N*a))

    for i in range(0,b):
        st = gen_mat_cor(N,k,sigma)
        tmp = st
        for j in range(0,a-1):

            app = gen_mat_cor(N,k,sigma)

            tmp = np.hstack((tmp, app))

        mat1[i*N:(i+1)*N,:] = tmp


    return mat1

def create_uncor_tot(a,b,N,k,sigma):
    '''
    Create the complete uncorrelated matrix.
    '''
    mat2 = np.zeros((N * b, N * a))

    for i in range(0, b):
        st = gen_mat_cor(N, k, sigma)
        tmp = st
        for j in range(0, a - 1):
            app = gen_mat_uncor(N, k, sigma)

            tmp = np.hstack((tmp, app))

        mat2[i * N:(i + 1) * N, :] = tmp

    return mat2

def measure_total_matrix(a,b,N,k,sigma,num_trials):
    '''
    Estimate the metrics for the matrix of interest.
    '''

    sum_l2_mat1 = 0
    sum_l2_mat2 = 0

    norm_mat1 = np.zeros((num_trials,1))
    norm_mat2 = np.zeros((num_trials,1))

    max_mat1 = 0
    max_mat2 = 0

    min_mat1 = 0
    min_mat2 = 0

    for i in range(0, num_trials):

        if i%100 == 0:
            print('Inner Iteration: ', i)

        mat1 = create_cor_tot(a, b, N, k, sigma)
        mat2 = create_uncor_tot(a, b, N, k, sigma)

        norm_mat1[i] = np.linalg.norm(mat1, 2)
        norm_mat2[i] = np.linalg.norm(mat2, 2)




    avg_l2_cor = np.average(norm_mat1)
    avg_l2_uncor = np.average(norm_mat2)

    max_l2_cor = np.max(norm_mat1)
    max_l2_uncor = np.max(norm_mat2)

    min_l2_cor = np.min(norm_mat1)
    min_l2_uncor = np.min(norm_mat2)

    return avg_l2_cor,max_l2_cor,min_l2_cor,avg_l2_uncor,max_l2_uncor,min_l2_uncor

def theory_uncor(a,b,k,epsilon,N):
    '''
    Estimate the spectral norm for an uncorrelated matrix.
    '''

    uncor = (1+epsilon)*(np.sqrt(k*a)+np.sqrt(k*b)+5*np.sqrt(np.log(np.maximum(N*a,N*b))/np.log(1+epsilon)))

    return uncor

def theory_cor(a,b,k,N):
    '''
    Estimate the spectral norm for a correlated matrix.
    '''

    #cor = (1.4*np.sqrt(k))*(np.sqrt(a)+np.sqrt(b)+np.sqrt(2*np.log(4*N)))
    cor = 1.4*(np.sqrt(k))*(np.sqrt(a)+np.sqrt(b))

    return cor

'''
Set hyperparameters
'''
N = 100
k = 9
sigma = 1
num_trials = 100
ab_max = 5
epsilon = 0.2

avg_l2_cor = np.zeros((ab_max,1))
avg_l2_uncor = np.zeros((ab_max,1))

max_l2_cor = np.zeros((ab_max,1))
max_l2_uncor = np.zeros((ab_max,1))

min_l2_cor = np.zeros((ab_max,1))
min_l2_uncor = np.zeros((ab_max,1))

theory_l2_cor = np.zeros((ab_max,1))
theory_l2_uncor = np.zeros((ab_max,1))

for i in range(0,ab_max):

    print('Outer Iteration: ', i)

    a = i+2
    b = i+2

    avg_l2_cor[i],max_l2_cor[i],min_l2_cor[i],avg_l2_uncor[i],max_l2_uncor[i],min_l2_uncor[i] =  measure_total_matrix(a, b, N, k, sigma, num_trials)

    theory_l2_cor[i] = theory_cor(a,b,k,N)
    theory_l2_uncor[i] = theory_uncor(a,b,k,epsilon,N)


'''
Plot results
'''
fig, ax = plt.subplots()
ax.grid(linestyle='--',linewidth=1.5,alpha=0.5,zorder=0)

x_points = np.arange(0,ab_max)+2
ax.plot(x_points,avg_l2_cor,label = "Empirical l2 Dep",linestyle = '-',color = 'xkcd:brick red',linewidth=4.0)
ax.plot(x_points,avg_l2_uncor,label = "Empirical l2 Indep",linestyle = '-',color = 'xkcd:royal blue',linewidth=4.0)
ax.plot(x_points,theory_l2_cor,label = "Theoretical l2 Dep",linestyle = '--',color = 'xkcd:red',linewidth=4.0)
ax.plot(x_points,theory_l2_uncor,label = "Theoretical l2 Indep",linestyle = '--',color = 'xkcd:bright blue',linewidth=4.0)


a = x_points[:]
b =  np.reshape(max_l2_cor,(-1))
c = np.reshape(min_l2_cor,(-1))
ax.fill_between(a,b,c,interpolate=True,facecolor='xkcd:brick red',alpha=0.3)

a = x_points[:]
b =  np.reshape(max_l2_uncor,(-1))
c = np.reshape(min_l2_uncor,(-1))
ax.fill_between(a,b,c,interpolate=True,facecolor='xkcd:royal blue',alpha=0.3)

plt.ylabel('E[ ||.||_2 ]')
plt.xlabel('a,b size')

plt.legend(loc = 3)
