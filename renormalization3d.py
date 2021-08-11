import numpy as np
import scipy.signal
import copy
import warnings
from scipy import stats
from functools import lru_cache
import multiprocessing as mp
from functools import partial
import itertools
from matplotlib import pyplot as plt


t_map = 1
L = 5  
J = 5 
lowH = 1e-11   
upH = 1e10    
upr = 250    
upk = 100   
lowU = 1e-12  
neighborhood = 1e-8  
Sigma = 0.6 
Kappa = 0.1
abstol = 1e-2
reltol = 1e-3
precision = np.float64

mode_1 = [1,0,0]  
mode_2 = [0,1,0]
mode_3 = [0,0,1]
Omega = np.array([1, 1, -1])
N = np.array([[ 0, 0, 1],  
              [ 1, 0, 0],
              [ 0, 1,-1]], dtype =int)

sigma_rot_number = 1.324717957
w = np.array( [sigma_rot_number, sigma_rot_number**2, 1], dtype = precision)
eigenvalue = 1/(sigma_rot_number)
sign_eigen = np.sign(eigenvalue)

grid_length = 10
y = np.arange( 0, 0.4, 0.4/grid_length)
y = y.tolist()
x = np.arange( 0, 0.15, 0.15/grid_length)
x = x.tolist()
main_flag = []
lie_flags = []
convergence_flags = []


#### Dimensions of hamiltonian.
f_dim = np.zeros( (J+1,2*L+1,2*L+1,2*L+1), dtype = precision )
index = np.hstack( (np.arange(0,L+1), np.arange(-L,0) ) )
v2, v1, v3 = np.meshgrid(index,index,index) 
[NTv1,NTv2,NTv3] = np.einsum( 'ij, jklm -> iklm', N.T, np.stack( (v1,v2,v3) ) )   
mask = ((abs(NTv1)<=L) & (abs(NTv2)<=L) & (abs(NTv3)<=L))
Norm_nu = np.sqrt(v1**2 + v2**2 + v3**2, dtype= precision).reshape(1, 2*L+1, 2*L+1, 2*L+1)
Vec_J = np.arange(J+1, dtype = precision).reshape(J+1, 1, 1, 1)
CompIm = Sigma * np.repeat(Norm_nu, J+1, axis=0) + Kappa * Vec_J
om0_nu = (w[0] * v1 + w[1] * v2 + w[2]*v3).reshape(1, 2*L+1, 2*L+1, 2*L+1) 
rom0_nu = np.repeat(np.abs(om0_nu), J+1, axis=0) / np.sqrt((w ** 2).sum())
iplus = rom0_nu <= CompIm 
iminus = np.logical_not(iplus) 


class Hamil:
    def __init__(self, f_0, Omega_0, w_0, flag_0 , counter_0, counter_1):
        self.f = f_0
        self.Omega = Omega_0
        self.w = w_0
        self.flag = flag_0
        self.Lie_counter = counter_0
        self.Convergence_counter = counter_1


def Hmu(mu_1,mu_2): ###will give the hamiltonian H_eps
    h_new = Hamil( f_dim.copy(), Omega.copy(), w.copy(), None, None, None)
    h_new.f = f_dim.copy()
    h_new.f[0][ mode_1[0], mode_1[1], mode_1[2]] = (0.5)*mu_1
    h_new.f[0][-mode_1[0],-mode_1[1],-mode_1[2]] = (0.5)*mu_1
    h_new.f[0][ mode_2[0], mode_2[1], mode_2[2]] = (0.5)*mu_2
    h_new.f[0][-mode_2[0],-mode_2[1],-mode_2[2]] = (0.5)*mu_2
    h_new.f[0][ mode_3[0], mode_3[1], mode_3[2]] = (0.5)*0.1
    h_new.f[0][-mode_3[0],-mode_3[1],-mode_3[2]] = (0.5)*0.1
    h_new.f[2][0,0,0] = 0.5
    h_new.Omega = Omega
    return h_new


def norm(f_0): 
    return np.abs(f_0).sum()

def LAMB(H):
    return (2.0)*H.f[2][0,0,0]*((np.matmul(N,H.Omega)**2).sum())*(sign_eigen/eigenvalue)

         
def productsum(f,g):
    ff = np.roll(f, [L,L,L], axis = (1,2,3))
    gg = np.roll(g, [L,L,L], axis = (1,2,3))
    zz = scipy.signal.convolve(ff, gg, 'full', 'auto')
    return np.roll(zz[:J+1, L:3*L+1, L:3*L+1, L:3*L+1] ,[-L, -L, -L], axis = (1,2,3))


def vectimesderiv(vec,f_0):
    prod = np.einsum( 'i,iklm -> klm', vec, np.stack( (v1,v2, v3) ) )
    return prod[np.newaxis]*f_0

def exponential(H_0, t_0):
    h_new = copy.deepcopy(H_0)
    z = h_new.f.copy()
    lin_reg_coef = np.zeros(5)
    y = np.zeros( h_new.f.shape, dtype = h_new.f.dtype)
    a = -(z[1][0,0,0])/(2*z[2][0,0,0])
    y[0][iminus[0]] = (z[0][iminus[0]])/(om0_nu[0][iminus[0]])       
    for m in range(1,J+1):
        y[m][iminus[m]] = (z[m][iminus[m]]-2*z[2][0,0,0]*((h_new.Omega[0]*v1 + h_new.Omega[1]*v2 + h_new.Omega[2]*v3  )[iminus[m]])*y[m-1][iminus[m]] )/(om0_nu[0][iminus[m]])
    ny = np.roll(Vec_J*y, -1, axis = 0)
    oy = vectimesderiv( h_new.Omega, y)
    nf = np.roll( Vec_J*z, -1, axis = 0)
    of = vectimesderiv( h_new.Omega, z)
    g = a*nf - vectimesderiv(h_new.w,y) - productsum(nf,oy) + productsum(ny,of) 
    old_g = z.copy()  
    g = (t_0)*g
    z += g 
    k = 2
    while( (k < upk) & ( norm(g+ old_g) > lowU ) & (norm(z)< upH) ):
        old_g = g.copy()
        ng = np.roll( Vec_J*g, -1, axis = 0)
        og = vectimesderiv( h_new.Omega, g)
        g = ( a*ng  - productsum(ng,oy) + productsum(ny,og))/precision(k) 
        g = (t_0)*g
        if k<5:
            if norm(g) != 0:
                lin_reg_coef[k] = np.log(norm(g))
            else:
                return h_new
        if k>= 5:
            lin_reg_coef = np.roll(lin_reg_coef, -1)
            lin_reg_coef[-1] = np.log(norm(g))
        z = z + g 
        k += 1
    h_new.f = z
    slope, intercept, rvalue, pvalue, stderr = stats.linregress( [0,1,2,3,4], lin_reg_coef)
    if (slope >= 0):
        h_new.flag = 'there are not exponential decay'
        h_new.Lie_counter = -1
        return h_new
    if (k == upk) &  (norm(z) < upH )  :
        h_new.flag = 'Need to add more terms k! while compute one step of Lie transform, need larger upk'
        h_new.Lie_counter = -2
        return h_new
    elif (k < upk) & ( norm(z) >= upH ):
        h_new.flag = 'Series in Lie transform is diverges'
        h_new.Lie_counter = -3
        return h_new
    else:
        h_new.Lie_counter = -4
        return h_new


def exp_adaptative(H,step):
    h_new = copy.deepcopy(H)
    if step < 5e-2:  
        h_new.flag = 'need smaller steps'
        h_new.Lie_counter = -7
        return h_new
    res1 = exponential(h_new, step)
    res2 = exponential( exponential( h_new, 0.5*step), 0.5*step)
    if norm( res1.f - res2.f) < abstol + reltol*norm(res1.f):
        h_new.f = 0.75*res1.f + 0.25*res2.f
        return h_new
    else:
        return exp_adaptative(exp_adaptative(h_new,0.5*step), 0.5*step)


def U_adaptive(H_0, t_0):  
    h_new = copy.deepcopy(H_0)
    I = copy.deepcopy(H_0)
    h_new.flag = None
    h_new.Convergence_counter = None
    z = h_new.f.copy()
    r = 0
    if (norm(z[iminus]) <= lowH) | ( norm(z) >= upH) :
        Lie_counter = -10
        return h_new
    while( ( upH> norm(I.f[iminus]) > lowH ) & ( r< upr) & (I.flag == None) ):
        I = exp_adaptative(I, t_0)
        r += 1
        I.f[:,v1,v2, v3] = 0.5*(I.f[:,v1,v2,v3] + I.f[:, -v1, -v2, -v3])
        I.f[0][0,0,0] = 0.0
    h_new = I
    if (h_new.flag != None):
        return h_new
    elif (r == upr) & (upH> norm(h_new.f[iminus]) > lowH):
        h_new.flag = 'non-resonant modes were not fully eliminated, need larger upr'
        h_new.Lie_counter = -5
        return h_new
    elif (r < upr) & ( norm(h_new.f[iminus]) >= upH ):
        h_new.flag = 'non-resonant modes are too large, at order r = {}'.format(r)
        h_new.Lie_counter = -6
        return h_new
    else:
        h_new.Lie_counter = 'r = {}'.format(r)
        return h_new


def U_time1(H_0, t_0): 
    h_new = copy.deepcopy(H_0)
    I = copy.deepcopy(H_0)
    h_new.flag = None
    h_new.Convergence_counter = None
    z = h_new.f.copy()
    r = 0
    if (norm(z[iminus]) <= lowH) | ( norm(z) >= upH) :
        Lie_counter = -10
        return h_new
    while( ( upH> norm(I.f[iminus]) > lowH ) & ( r< upr) & (I.flag == None) ):
        I = exponential(I, t_0)
        r += 1
        I.f[:,v1,v2, v3] = 0.5*(I.f[:,v1,v2,v3] + I.f[:, -v1, -v2, -v3])
        I.f[0][0,0,0] = 0.0
    h_new = I
    if (h_new.flag != None):
        return h_new
    elif (r == upr) & (upH> norm(h_new.f[iminus]) > lowH):
        h_new.flag = 'non-resonant modes were not fully eliminated, need larger upr'
        h_new.Lie_counter = -5
        return h_new
    elif (r < upr) & ( norm(h_new.f[iminus]) >= upH ):
        h_new.flag = 'non-resonant modes are too large, at order r = {}'.format(r)
        h_new.Lie_counter = -6
        return h_new
    else:
        h_new.Lie_counter = 'r = {}'.format(r)
        return h_new


def RENORM(H_0):
    h_new = copy.deepcopy(H_0)
    lamb = LAMB(h_new)
    N_Omega = np.matmul(N, h_new.Omega)
    num = sign_eigen*(np.linalg.norm(N_Omega))/lamb 
    c = np.zeros( h_new.f.shape, dtype = h_new.f.dtype )
    c[:,v1[mask], v2[mask], v3[mask]] = h_new.f[:, NTv1[mask], NTv2[mask], NTv3[mask]] 
    c[:,v1,v2,v3] = (lamb * (sign_eigen/eigenvalue) ) * c[:,int(sign_eigen)*v1,int(sign_eigen)*v2, int(sign_eigen)*v3] 
    h_new.f = np.power(num*np.ones(J+1), np.arange(J+1)).reshape(J+1,1,1,1) * c 
    h_new.Omega = N_Omega / np.linalg.norm( N_Omega )
    #h_new = U_adaptive(h_new, t_map)
    h_new = U_time1(h_new, t_map)
    h_new.f[ abs(h_new.f) <= lowH] = 0 
    return h_new


def convergence_mu(mu_1, mu_2):
    z = np.zeros( f_dim.shape, dtype = f_dim.dtype )
    z[2][0,0,0] = 0.5
    I = Hamil(f_dim, Omega, w, None, None, None)
    I = Hmu(mu_1, mu_2)
    r = 0
    while( ( upH > norm(I.f-z) > lowH) & (r < upr) & (I.flag == None) ): 
        I = RENORM(I)
        r += 1
    if I.flag != None:
        return 1, r, I.Lie_counter
    elif norm(I.f-z) <= lowH:
        return -1, 0, I.Lie_counter
    elif norm(I.f -z) >= upH:
        return 1, r, I.Lie_counter
    else:
        return 1, r, -7



if __name__ == '__main__':
    pool = mp.Pool(5)
    main_flag, convergence_flags, lie_flags = zip(* pool.starmap( convergence_mu, itertools.product(x, y) ) )
    pool.close()

main_flag = np.asarray(main_flag).reshape( (len(x), len(y) ) )
lie_flags = np.asarray(lie_flags).reshape( (len(x), len(y) ) )
convergence_flags = np.asarray(convergence_flags).reshape( (len(x),len(y)) )


#np.save('dataflags3dw1-LJ{}-sig{}-grid{}-adapt'.format(L, Sigma, grid_length), main_flag)
np.save('convflags3dw1-LJ{}-sig{}-grid{}-adapt'.format(L, Sigma, grid_length), convergence_flags)
#np.save('lieflags3dw1-LJ{}-sig{}-grid{}-adapt'.format(L, Sigma, grid_length), lie_flags)




plt.xlabel('$\mu_2$',fontsize =15)
plt.ylabel('$\mu_1$', fontsize =15)
plt.title('$\omega =[{},{},{}]$, $V(\phi) = \mu_1\cos(\phi_1) + \mu_2\cos(\phi_2) + 0.1\cos(\phi_3)$, $\Omega =$[{},{},{}]'.format(w[0],w[1],w[2], Omega[0], Omega[1], Omega[2]) )
plt.imshow(convergence_flags, extent= (np.amin(y), np.amax(y),np.amin(x), np.amax(x)), aspect='auto', origin = 'lower')
plt.colorbar()
plt.grid()
plt.show()
#plt.savefig('b3dw1-LJ{}-sig{}-grid{}-adapt-flags.pdf'.format(L, Sigma, grid_length) )
plt.clf()

