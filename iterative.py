import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import time
from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

font = {'size'   : 22}

plt.rc('font', **font)

#Domain
L = 1
n = 101
h = d_x = d_y = L/(n-1)

x = y = np.linspace(0,L,n)
X,Y = np.meshgrid(x,y)

#Source Term
f = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        f[i,j] = 2*x[i]*(x[i]-1) + 2*y[j]*(y[j]-1)


#Analytical Solution
u_a = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        u_a[i,j] = x[i] * y[j] * (x[i-1]-1) * (y[j-1]-1)


# Initial guess
u0 = np.ones((n,n))/20
u0[0,:] = u0[-1,:] = u0[:,0] = u0[:,-1] = 0


# Compute maximum absolute error between computed & analytical solution
def error(u):
    error = u-u_a
    err = (abs(error)).max()
    #print(err)    
    return err

# Plot analytical solution & Actual solution
def plt3d(u_a,X,Y,u_n,title):
    fig = plt.figure(figsize=(35,15), dpi=50)
    ax = fig.add_subplot(121, projection='3d')
    p = ax.plot_surface(X,Y,u_a,rstride=5,cstride=5,cmap='viridis')
    p.set_clim(0, 0.07)
    plt.title('Analytical solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_zlim3d(bottom=0,top=0.07)


    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(X,Y,u_n,rstride=5,cstride=5,cmap='viridis')
    plt.title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    ax.set_zlim3d(bottom=0,top=0.07)
    plt.tight_layout()
    fig.colorbar(p,shrink=0.8, aspect=15)
    plt.show()
    plt.show()


# Plot 2D graph of maximum error vs. iteration
def plt2d(max_err,iteration,title):
    fig = plt.figure(figsize=(15,13))
    fig.add_subplot(111)
    plt.plot(iteration,max_err)
    plt.title(title)
    plt.xlabel('iterations')
    plt.ylabel('maximum error')
    plt.show()
    

# Create tridiagonal elements
def CM(N):
    # Build coefficient 'matrix', [A]
    d = -4*np.ones(N) # main diagonal
    l = np.ones(N-1) # lower diagonal
    up = np.ones(N-1) # upper diagonal
    
    d_n = d.copy() # main diagonal - after elimination of lower diagonal
    for i in range(1,N):
        d_n[i] = d[i] - (up[i-1]*l[i-1])/d_n[i-1]

        
    return l, d_n, up


# Create tridiagonal elements for relaxation methods
def CMR(N,w):
    # Build coefficient 'matrix', [A]
    d = -4*np.ones(N) # main diagonal
    l = np.ones(N-1)*w # lower diagonal
    up = np.ones(N-1)*w # upper diagonal
    
    d_n = d.copy() # main diagonal - after elimination of lower diagonal
    # Forward elimination of lower-diagonal elements
    for i in range(1,N):
        d_n[i] = d[i] - up[i-1]*l[i-1]/d_n[i-1]
                
    return l, d_n, up


# THOMAS ALGORITHM - Tridiagonal matrix system solver, solve [A][X]=[B]
def TDMA(B,l,d_n,up):
    N = np.size(B)   
    
    # Forward elimination of lower-diagonal elements
    for i in range(1,N):
        B[i] = B[i] - B[i-1]*l[i-1]/d_n[i-1]
    
    X = np.zeros_like(B)

    # Backward substitution
    X[-1] = B[-1]/d_n[-1]
    d_n[:-1] - d_n[1:]*up
    for i in range (N-2,-1,-1):
        X[i] = (B[i] - up[i]*X[i+1])/d_n[i]
    #print(np.size(X))
    #print(X)

    return X


#Jacobi Iteration Method
def jacobi(u,f,h,max_er,max_it):
    t = time.time()
    u_n = u.copy()
    it = 0
    max_err = []
    iteration = []
    while True:
        it = it + 1
        u_n[1:-1,1:-1] = 0.25*(u_n[2:,1:-1] + u_n[:-2,1:-1] + u_n[1:-1,2:] + u_n[1:-1,:-2] - f[1:-1,1:-1]*h*h)
        err = error(u_n)
        max_err.__iadd__([err])
        iteration.__iadd__([it])

        if err < max_er:
            break
        if it > max_it:
            break
    
    t = time.time() - t

    print(' ----------- Jacobi method ---------------')
    print('computational time = ' + ('%.6f' %t) + 's') 
    print( 'Iterations =', it)
    print('Maximum error = ' + ('%.6f' %err))
    #plt2d(max_err,iteration,'Jacobi iteration method')
    #plt3d(u_a,X,Y,u_n,'Jacobi iteration method')
    return u_n, iteration, max_err, t

#Gauss-Seidel Iteration Method
def gauss_seidel(u,f,h,max_er,max_it):
    t = time.time()
    u_n = u.copy()
    it = 0
    max_err = []
    iteration = []
    while True:
        it = it + 1
        for j in range(1,n-1):
            for i in range(1,n-1):
                u_n[i,j] = 0.25*(u_n[i+1,j] + u_n[i-1,j] + u_n[i,j+1] + u_n[i,j-1] - f[i,j]*h*h)

        err = error(u_n)
        max_err.__iadd__([err])
        iteration.__iadd__([it])

        if err < max_er:
            break
        if it > max_it:
            break
    
    t = time.time() - t

    print(' ----------- Gauss_Seidel method ---------------')
    print('computational time = ' + ('%.6f' %t) + 's') 
    print( 'Iterations =', it)
    print('Maximum error = ' + ('%.6f' %err))
    #plt2d(max_err,iteration,'Gauss_Seidel method')
    #plt3d(u_a,X,Y,u_n,'Gauss_Seidel method')
    return u_n, iteration, max_err, t

#SOR Iteration Method
def sor(u,f,h,max_er,max_it,w):
    t = time.time()
    u_n = u.copy()
    it = 0
    max_err = []
    iteration = []
    while True:
        it = it + 1
        for j in range(1,n-1):
            for i in range(1,n-1):
                u_n[i,j] = (1-w)*u_n[i,j] +  w*0.25*(u_n[i+1,j] + u_n[i-1,j] + u_n[i,j+1] + u_n[i,j-1] - f[i,j]*h*h)

        err = error(u_n)
        max_err.__iadd__([err])
        iteration.__iadd__([it])

        if err < max_er:
            break
        if it > max_it:
            break
    
    t = time.time() - t

    print(' ----------- SOR-Gauss_Seidel method ---------------')
    print('computational time = ' + ('%.6f' %t) + 's') 
    print( 'Iterations =', it)
    print('Maximum error = ' + ('%.6f' %err))
    #plt2d(max_err,iteration,'SOR-Gauss_Seidel method (w=1.93)')
    #plt3d(u_a,X,Y,u_n,'SOR-Gauss_Seidel method (w=1.93)')
    return u_n, iteration, max_err, t

#Line Gauss-Seidel Iteration Method
def line_gauss_seidel(u,f,h,max_er,max_it,method):
    t = time.time()
    u_n = u.copy()
    it = 0
    l , d_n, up = CM(n-2)
    B = np.zeros(n)
    max_err = []
    iteration = []
    if method == 'column':
        title = 'Line Gauss-Seidel iteration method [column]'
        while True:
            it = it + 1
            for j in range(1,n-1):
                B[1:-1] = -u_n[2:,j] - u_n[:-2,j] + f[1:-1,j]*h*h
                u_n[1:-1,j] = TDMA(B[1:-1],l,d_n,up)
            
            err = error(u_n)
            max_err.__iadd__([err])
            iteration.__iadd__([it])

            if err < max_er:
                break
            if it > max_it:
                break
    
    if method == 'row':
        title = 'Line Gauss-Seidel iteration method [row]'
        while True:
            it = it + 1
            for i in range(1,n-1):
                B[1:-1] = -u_n[i,2:] - u_n[i,:-2] + f[i,1:-1]*h*h
                u_n[i,1:-1] = TDMA(B[1:-1],l,d_n,up)
            
            err = error(u_n)
            max_err.__iadd__([err])
            iteration.__iadd__([it])

            if err < max_er:
                break
            if it > max_it:
                break
    
    if method == 'adi':
        title = 'Line Gauss-Seidel iteration method [adi]'
        while True:
            it = it + 1

            for j in range(1,n-1):
                B[1:-1] = -u_n[2:,j] - u_n[:-2,j] + f[1:-1,j]*h*h
                u_n[1:-1,j] = TDMA(B[1:-1],l,d_n,up)

            
            for i in range(1,n-1):
                B[1:-1] = -u_n[i,2:] - u_n[i,:-2] + f[i,1:-1]*h*h
                u_n[i,1:-1] = TDMA(B[1:-1],l,d_n,up)                
            
            err = error(u_n)
            max_err.__iadd__([err])
            iteration.__iadd__([it])

            if err < max_er:
                break
            if it > max_it:
                break

    t = time.time() - t

    print(' ----------- ' + title + ' ---------------')
    print('computational time = ' + ('%.6f' %t) + 's') 
    print( 'Iterations =', it)
    print('Maximum error = ' + ('%.6f' %err))
    #plt2d(max_err,iteration,title)
    #plt3d(u_a,X,Y,u_n,title)
    return u_n, iteration, max_err, t

#Line Gauss-Seidel Iteration Method
def line_sor(u,f,h,max_er,max_it,w,method):
    t = time.time()
    u_n = u.copy()
    it = 0
    l , d_n, up = CMR(n-2,w)
    B = np.zeros(n)
    max_err = []
    iteration = []
    if method == 'column':
        title = 'Line SOR iteration method [column]'
        while True:
            it = it + 1
            for j in range(1,n-1):
                B[1:-1] = -4*(1-w)*u_n[1:-1,j]-w*(u_n[2:,j] + u_n[:-2,j] - f[1:-1,j]*h*h)
                u_n[1:-1,j] = TDMA(B[1:-1],l,d_n,up)
            
            err = error(u_n)
            max_err.__iadd__([err])
            iteration.__iadd__([it])

            if err < max_er:
                break
            if it > max_it:
                break
    
    if method == 'row':
        title = 'Line SOR iteration method [row]'
        while True:
            it = it + 1
            for i in range(1,n-1):
                B[1:-1] = -4*(1-w)*u_n[i,1:-1]-w*(u_n[i,2:] + u_n[i,:-2] - f[i,1:-1]*h*h)
                u_n[i,1:-1] = TDMA(B[1:-1],l,d_n,up)
            
            err = error(u_n)
            max_err.__iadd__([err])
            iteration.__iadd__([it])

            if err < max_er:
                break
            if it > max_it:
                break
    
    if method == 'adi':
        title = 'Line SOR iteration method [adi]'
        while True:
            it = it + 1

            for j in range(1,n-1):
                B[1:-1] = -4*(1-w)*u_n[1:-1,j]-w*(u_n[2:,j] + u_n[:-2,j] - f[1:-1,j]*h*h)
                u_n[1:-1,j] = TDMA(B[1:-1],l,d_n,up)

            for i in range(1,n-1):
                B[1:-1] = -4*(1-w)*u_n[i,1:-1]-w*(u_n[i,2:] + u_n[i,:-2] - f[i,1:-1]*h*h)
                u_n[i,1:-1] = TDMA(B[1:-1],l,d_n,up)            
            
            err = error(u_n)
            max_err.__iadd__([err])
            iteration.__iadd__([it])

            if err < max_er:
                break
            if it > max_it:
                break

    t = time.time() - t

    print(' ----------- ' + title + ' w = ' + str(w) + ' ---------------')
    print('computational time = ' + ('%.6f' %t) + 's') 
    print( 'Iterations =', it)
    print('Maximum error = ' + ('%.6f' %err))
    #plt2d(max_err,iteration,title)
    #plt3d(u_a,X,Y,u_n,title)
    return u_n, iteration, max_err, t

# Red-Black Gauss-Seidel Iteration Method
def rb_gauss_seidel(u,f,h,max_er,max_it):
    t = time.time()
    u_n = u.copy()
    it = 0
    max_err = []
    iteration = []

    # Step 1: create *integer* array the same size as u 
    x = np.zeros_like(u,dtype=np.int)

    # Step 2: populate all non-boundary cells with running numbers from 1 to (n-2)^2
    x[1:-1,1:-1] = np.arange(1,(n-2)**2+1).reshape(n-2,n-2)

    # Step 3: get indices of even (red) and odd (black) points
    ir, jr = np.where((x>0) & (x%2 == 0)) # indices of red pts = indices of even numbers
    ib, jb = np.where((x>0) & (x%2 == 1)) # indices of black pts = indices of odd numbers
    while True:
        it = it + 1
        
        # Red point update
        u_n[ir,jr] = 0.25*(u_n[ir+1,jr] + u_n[ir-1,jr] + u_n[ir,jr+1] + u_n[ir,jr-1] - f[ir,jr]*h*h)
        
        # Black point update
        u_n[ib,jb] = 0.25*(u_n[ib+1,jb] + u_n[ib-1,jb] + u_n[ib,jb+1] + u_n[ib,jb-1] - f[ib,jb]*h*h)
                
        err = error(u_n)
        max_err.__iadd__([err])
        iteration.__iadd__([it])

        if err < max_er:
            break
        if it > max_it:
            break
    
    t = time.time() - t

    print(' ----------- Red-black Gasuss-Seidel method ---------------')
    print('computational time = ' + ('%.6f' %t) + 's') 
    print( 'Iterations =', it)
    print('Maximum error = ' + ('%.6f' %err))
    #plt2d(max_err,iteration,'Red-black Gasuss-Seidel method')
    #plt3d(u_a,X,Y,u_n,'Red-black Gasuss-Seidel method')
    return u_n, iteration, max_err, t

# Red-Black SOR Iteration Method
def rb_sor(u,f,h,max_er,max_it,w):
    t = time.time()
    u_n = u.copy()
    it = 0
    max_err = []
    iteration = []

    # Step 1: create *integer* array the same size as u 
    x = np.zeros_like(u,dtype=np.int)

    # Step 2: populate all non-boundary cells with running numbers from 1 to (n-2)^2
    x[1:-1,1:-1] = np.arange(1,(n-2)**2+1).reshape(n-2,n-2)

    # Step 3: get indices of even (red) and odd (black) points
    ir, jr = np.where((x>0) & (x%2 == 0)) # indices of red pts = indices of even numbers
    ib, jb = np.where((x>0) & (x%2 == 1)) # indices of black pts = indices of odd numbers
    while True:
        it = it + 1
        
        # Red point update
        u_n[ir,jr] = (1-w)*u_n[ir,jr] + 0.25*w*(u_n[ir+1,jr] + u_n[ir-1,jr] + u_n[ir,jr+1] \
                                                  + u_n[ir,jr-1] - f[ir,jr]*h*h)
        
        # Black point update
        u_n[ib,jb] = (1-w)*u_n[ib,jb] + 0.25*w*(u_n[ib+1,jb] + u_n[ib-1,jb] + u_n[ib,jb+1] \
                                                    + u_n[ib,jb-1] - f[ib,jb]*h*h)
                
        err = error(u_n)
        max_err.__iadd__([err])
        iteration.__iadd__([it])

        if err < max_er:
            break
        if it > max_it:
            break
    
    t = time.time() - t

    print(' ----------- Red-black SOR method ---------------')
    print('computational time = ' + ('%.6f' %t) + 's') 
    print( 'Iterations =', it)
    print('Maximum error = ' + ('%.6f' %err))
    #plt2d(max_err,iteration,'Red-black SOR method')
    #plt3d(u_a,X,Y,u_n,'Red-black SOR method')
    return u_n, iteration, max_err, t

#plt3d(u_a,X,Y,u0,'Initital-Gauss')

u_j, it_j, conv_j, t_j = jacobi(u0,f,h,0.01,1000)

u_gs, it_gs, conv_gs, t_gs = gauss_seidel(u0,f,h,0.01,1000)

u_sor, it_sor, conv_sor, t_sor = sor(u0,f,h,0.01,1000,1.93)

u_lgs_c, it_lgs_c, conv_lgs_c, t_lgs_c = line_gauss_seidel(u0,f,h,0.01,1000,'column')
u_lgs_r, it_lgs_r, conv_lgs_r, t_lgs_r = line_gauss_seidel(u0,f,h,0.01,1000,'row')
u_lgs_adi, it_lgs_adi, conv_lgs_adi, t_lgs_adi = line_gauss_seidel(u0,f,h,0.01,1000,'adi')

u_lsor_c, it_lsor_c, conv_lsor_c, t_lsor_c = line_sor(u0,f,h,0.01,1000,1.7,'column')
u_lsor_r, it_lsor_r, conv_lsor_r, t_lsor_r = line_sor(u0,f,h,0.01,1000,1.7,'row')
u_lsor_adi, it_lsor_adi, conv_lsor_adi, t_lsor_adi = line_sor(u0,f,h,0.01,1000,1.7,'adi')

u_rbgs, it_rbgs, conv_rbgs, t_rbgs = rb_gauss_seidel(u0,f,h,0.01,1000)

u_rbsor, it_rbsor, conv_rbsor, t_rbsor = rb_sor(u0,f,h,0.01,1000,1.87)


#Results
Iteration = [it_j, it_gs, it_sor, it_lgs_c, it_lgs_r, it_lgs_adi, it_lsor_c, it_lsor_r, it_lsor_adi, it_rbgs, it_rbsor]
Error = np.array([conv_j, conv_gs, conv_sor, conv_lgs_c, conv_lgs_r, conv_lgs_adi,  conv_lsor_c, conv_lsor_r, conv_lsor_adi, conv_rbgs, conv_rbsor])

Time = [t_j, t_gs, t_sor, t_lgs_c, t_lgs_r, t_lgs_adi, t_lsor_c, t_lsor_r, t_lsor_adi, t_rbgs, t_rbsor]
label = ['JAC','GS','SOR','Line_GS-Col','Line_GS-Raw','Line_GS-ADI','Line_SOR-Col', 'Line_SOR-Raw', 'Line_SOR-SOR', 'RB-GS', 'RB-SOR']
s = [      0,   1,   2,         3,          4,              5,           6,               7,              8,           9,       10,  ] #select data to plot
p = np.size(s) #no. of plots

Iteration_list = [max(p) for p in Iteration] 
Error_list = [max(p) for p in Error]


# Plot of maximum error vs. iteration
fig = plt.figure(figsize=(35,28))
for i in s:
    plt.plot(Iteration[i],Error[i], linewidth=2 , label= " - " + str(label[i]))
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Maximum error')
plt.xlim([0,700])
plt.ylim([0,0.06])
plt.show()


# Scatter plot of computation time vs. iteration
fig = plt.figure(figsize=(35,28))
for i in s:
    plt.scatter(Iteration_list[i], Time[i], s=40 , label= " - " + str(label[i]))
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Computational Time')
plt.show()


file1 = open("main_output_1.txt","a") 
for i in (s):
    file1.write("Iteration : %.5f" % Iteration_list[i] + ' : ' + "Error : %.7f" % Error_list[i] + ' : ' + "Time : %.7f" % Time[i] + ' : ' + "Label : %.2f " % s[i] + "\n" )


fig = plt.figure(figsize=(35,28))

# Bar chart of computation time
fig.add_subplot(121)
plt.bar(s,Time)
plt.title('Computation time')

# Bar chart of iterations
fig.add_subplot(122)
plt.bar(s,Iteration_list)
plt.title('Iterations')

plt.show()
