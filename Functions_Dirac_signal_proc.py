"""
Functions used in the notebook

"""


####MATRICES

def get_B1(elist): 
    """
    creates the matrix B1 from edge list.
    Assumes the edge list is ordered, i.e. (n1,n2) with n1<n2.
    Assumes node index starts at 1.
    """
    num_edges = len(elist)
    data = [-1] * num_edges + [1] * num_edges
    row_ind = [e[0]-1 for e in elist] + [e[1]-1 for e in elist]
    col_ind = [i for i in range(len(elist))] * 2
    B1 = csc_matrix(
        (np.array(data), (np.array(row_ind), np.array(col_ind))), dtype=np.int8)
    return B1.toarray()

def get_B2(elist, tlist):
    """
    creates the matrix B2 from edge lists and triangle lists
    Assumes the edge list is ordered, i.e. (n1,n2) with n1<n2.
    Assumes node index starts at 1.
    """
    if len(tlist) == 0:
        return csc_matrix([], shape=(len(elist), 0), dtype=np.int8)

    elist_dict = {tuple(sorted(j)): i for i, j in enumerate(elist)}

    data = []
    row_ind = []
    col_ind = []
    for i, t in enumerate(tlist):
        e1 = t[[0, 1]]
        e2 = t[[1, 2]]
        e3 = t[[0, 2]]

        data.append(1)
        row_ind.append(elist_dict[tuple(e1)])
        col_ind.append(i)

        data.append(1)
        row_ind.append(elist_dict[tuple(e2)])
        col_ind.append(i)

        data.append(-1)
        row_ind.append(elist_dict[tuple(e3)])
        col_ind.append(i)

    B2 = csc_matrix((np.array(data), (np.array(row_ind), np.array(
        col_ind))), shape=(len(elist), len(tlist)), dtype=np.int8)
    return B2.toarray()


def get_D(B1,B2):
    """
    generates the unnormalised Dirac operators D1 and D2 from the boundary matrices B1 and B2
    """
    N=B1.shape[0]
    L=B1.shape[1]
    T=B2.shape[1]
    
    D1 = np.block([[np.zeros((N,N)),B1,np.zeros((N,T))],
                [B1.transpose(),         np.zeros((L,L)),np.zeros((L,T))],
                [np.zeros((T,N)),         np.zeros((T,L)),np.zeros((T,T))]])
    
    if T==0:
        D2 = np.zeros((N+L,N+L))
    else:
        D2 = np.block([[np.zeros((N,N)),np.zeros((N,L)),np.zeros((N,T))],
                    [np.zeros((L,N)),         np.zeros((L,L)),B2],
                    [np.zeros((T,N)),        B2.transpose(),np.zeros((T,T))]])
    
    return D1,D2

def get_wvd(D_one,D1,D2):
    #return the eigenvalues and eigenvectors of D1 or D2 depending on the flag D_one
    w1, v1 = np.linalg.eigh(D1) # eigenvalues and eigenvectors
    w2, v2 = np.linalg.eigh(D2)
    w = w1 if D_one==True else w2
    v = v1 if D_one==True else v2
    D = D1 if D_one==True else D2
    vinv = np.linalg.inv(v)
    
    return w,v,D,vinv



###SIGNAL DEFINITIONS
def get_signal(s_gen,D):
    """
    projects a signal (s_gen) onto the image of D1 or D2 (D)
    """
    s = D.dot(np.linalg.pinv(D)).dot(s_gen)
    s = s/np.linalg.norm(s)
    
    return s

def add_noise(s,w,v,D,beta):
    """
    Adds noise to the signal (s)

    inputs:
    s: pure signal
    w: eigenvalues of D
    v: eigenvectors of D
    D: the Dirac operator
    beta: controls the amount of noise added.

    outputs:
    true signal, noisy signal, mass of the true signal, signal to noise ratio
    """
    mask_nonzero = (~np.isclose(w, 0))
    v_nonzero = v[:, mask_nonzero] #all non-zero eigenvectors
    w_nonzero = w[mask_nonzero] #all non-zero eigenvalues
    
    noise = np.random.normal(0,1,s.shape[0])/np.sqrt(len(w_nonzero))
    noise = ((v_nonzero).dot(np.transpose(v_nonzero))).dot(noise) #ensures the noise is non-harmonic
    s_noisy = s+beta*noise
    s_true = s
    snr = np.linalg.norm(s)**2/np.linalg.norm(beta*noise)**2
    m_target = ((np.transpose(s)).dot(D)).dot(s)/((np.transpose(s)).dot(s)) #True mass
    
    return s_true, s_noisy, m_target, snr

def build_signals(w,v,D,x_min,beta=1, gaussian=False, m_true=1, sigma_m=0.2):
    #########################
    #This function builds the true signal and noise to fit a few different scenarios. The key parameters are:
    #beta: sets the amount of noise
    #x_min: True or False to take the lowest or highest eigenvalue mode.
    #Gaussian: True or False depending on whether the true signal is a Gaussian superposition of modes around m_true with std sigma_m
    
    #########################
    
    #Parsing of the eigenvectors

    w_real = w

    w_min =w_real[(w_real>0) & (~np.isclose(w_real, 0))].min() #min eigenvalue
    m_min_index = np.where(w_real==w_min)[0] #corresponding index

    w_max =w_real[(w_real>0) & (~np.isclose(w_real, 0))].max() #max eigenvalue
    m_max_index = np.where(w_real==w_max)[0] #corresponding index

    mask_neg_and_nonzero = (w_real<0) & (~np.isclose(w_real, 0)) #index of negative eigenvalues
    v_neg = v[:, mask_neg_and_nonzero] #negative eigenvectors

    mask_nonzero = (~np.isclose(w_real, 0))

    v_nonzero = v[:, mask_nonzero] #all non-zero eigenvectors
    w_nonzero = w_real[mask_nonzero] #all non-zero eigenvalues

    
    s = np.real(v[:,m_min_index[0]]) if x_min==True else np.real(v[:,m_max_index[0]]) 
    
    x = (w_nonzero-m_true*np.ones(w_nonzero.shape))/sigma_m if gaussian==True else 0
    
    s = v_nonzero.dot(np.exp(-x*x/2))/(np.sqrt(2*np.pi*sigma_m**2)) if gaussian==True else s
    s = s/np.linalg.norm(s)
    
    noise = np.random.normal(0,1,s.shape[0])/np.sqrt(len(w_nonzero))
    #noise = ((v_nonzero).dot(np.transpose(v_nonzero))).dot(noise) #ensures the noise is non-harmonic
    
    #noise = np.random.normal(0,1,s.shape[0]) if antialigned==False else v_neg.dot(np.random.randn(v_neg.shape[1]))
    #noise = (noise-s*((s.T).dot(noise))) #ensures the noise is orthogonal to the signal
    noise = ((v_nonzero).dot(np.transpose(v_nonzero))).dot(noise) #ensures the noise is non-harmonic
    #noise = noise/np.linalg.norm(noise)
    
    m_target = ((np.transpose(s)).dot(D)).dot(s)/((np.transpose(s)).dot(s)) #True mass
    
    s_noisy = s + beta*noise
    s_true = s
    snr = np.linalg.norm(s)**2/np.linalg.norm(beta*noise)**2
    return s_true, s_noisy, m_target, snr


def build_pure_signal(x_min, gaussian, w, v, D, m_true=1,sigma_m=0.2 ):
    #########################
    #This function builds the true signal to fit a few different scenarios. The key parameters are:
    #x_min: True or False to take the lowest or highest eigenvalue mode.
    #Gaussian: True or False depending on whether the true signal is a Gaussian superposition of modes around m_true with std sigma_m
    
    #########################
    
    #Parsing of the eigenvectors
    w_min =w[(w>0) & (~np.isclose(w, 0))].min() #min eigenvalue
    m_min_index = np.where(w==w_min)[0] #corresponding index

    w_max =w[(w>0) & (~np.isclose(w, 0))].max() #max eigenvalue
    m_max_index = np.where(w==w_max)[0] #corresponding index

    mask_neg_and_nonzero = (w<0) & (~np.isclose(w, 0)) #index of negative eigenvalues
    v_neg = v[:, mask_neg_and_nonzero] #negative eigenvectors

    mask_nonzero = (~np.isclose(w, 0))

    v_nonzero = v[:, mask_nonzero] #all non-zero eigenvectors
    w_nonzero = w[mask_nonzero] #all non-zero eigenvalues

    s = np.real(v[:,m_min_index[0]]) if x_min==True else np.real(v[:,m_max_index[0]])     

    x = (w_nonzero-m_true*np.ones(w_nonzero.shape))/sigma_m if gaussian==True else 0  

    s = v_nonzero.dot(np.exp(-x*x/2))/(np.sqrt(2*np.pi*sigma_m**2)) if gaussian==True else s
    s = s/np.linalg.norm(s)
    
    return s

def get_noise(s,w,v,D):
    """
    returns noise of the same size as the signal (s), in the non-harmonic image of D
    """
    mask_nonzero = (~np.isclose(w, 0))
    v_nonzero = v[:, mask_nonzero] #all non-zero eigenvectors
    w_nonzero = w[mask_nonzero] #all non-zero eigenvalues
    
    noise = np.random.normal(0,1,s.shape[0])/np.sqrt(len(w_nonzero))
    noise = ((v_nonzero).dot(np.transpose(v_nonzero))).dot(noise) #ensures the noise is non-harmonic
    
    return noise


####PROCESSING FUNCTIONS:
def process(w,v,vinv,s_input,m,gamma):
    ##################
    #Given a measured signal, a mass and gamma, this returns the estimated signal that minimises the error.
    
    #D is the Dirac operator: either D1 or D2
    #s_input is the measured signal
    #m is the mass
    #gamma controls the balance of the regularizer and the norm error.
    ##################
    
    Q_l = 1/(1+gamma*(w-m)**2)
    Q_op = v.dot(np.diag(Q_l)).dot(vinv)
    return Q_op.dot(s_input)
   
    
def optimize_m(w,v,vinv,D,s_noisy,s_true,m_0,gamma,tol, epsilon=0.3):
    ##################
    #This function learns the mass and returns 
    #list_m (list of mass iterations), 
    #list_error_m (list of error in the signal at each iteration)
    #list_it (counter of iterations) 
    #m (final mass) 
    #s_est_m (final signal)
    
    #inputs:
    #w: eigenvalues of D
    #v: eigenvectors of D
    #vinv: inverse of the matrix of eigenvectors of D
    #D: the Dirac operator
    #s_noisy, s_true: noisy and true signals
    #m_0: initial guess for the mass
    #gamma: balances the regulariser and the norm error
    #tol: parameter controls the convergence threshold
    #epsilon: learning rate
    ##################
    
    list_m = []
    list_error_m = []
    list_it =[]
    it = 0
    m_old = 1000
    m=m_0
    while (np.abs(m_old-m))>tol :
        m_old = m
        it = it+1
        s_est_m = process(w,v,vinv,s_noisy, m, gamma)
        m=(1-epsilon)*m_old+epsilon*((s_est_m.transpose()).dot(D)).dot(s_est_m)/(s_est_m.transpose().dot(s_est_m)) #updates the mass
        
        error_m = np.linalg.norm(s_est_m - s_true,2) #error in the signal
        
        #iteration updates in the lists
        list_m.append(m)
        list_error_m.append(error_m)
        list_it.append(it)
        
    return list_m, list_error_m, list_it, m, s_est_m


######Visualisation:
def plot_NL(s,list_edges, list_triangle, pos, ax, norm, the_map = mpl.cm.get_cmap('coolwarm'), edge_width_factor = 20):
    """
    plots signal located on nodes and links (encoded in topological signal s)
    list_edges, list_triangles: defines the underlying space
    pos: positions of the nodes
    ax: axes to plot
    norm: colormap normalisation
    the_map: color map to be used
    edge_width_factor: multiplication factor of the signal into an edge width. 
    the thickness of edges, and node size is proportional to the signal value. 
    """
    N=len(pos)
    L=len(list_edges)
    T =len(list_triangle)
    
    G = nx.Graph()
    for j in range(0,N):
        G.add_node(j+1,weight = s[j]) #add nodes with weight
        
    for j in range(0, len(list_edges)):
        edge = list_edges[j]
        n1 = edge[0]
        n2 = edge[1]
        w12 = s[j+N] #skip the nodes to obtain the signal on edges
        G.add_edge(n1, n2, weight = w12)

    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    
    edge_width = [edge_width_factor*abs(G[u][v]['weight'])+0.2 for u,v in edges]
    nodesize = [500*abs(nx.get_node_attributes(G, 'weight')[u])+3 for u in G.nodes()]
    
    patches = []
    patches_edges = []

    
    #Colour map set up
    color_lookup_nodes = {} #Builds dictionary for nodes weights
    for k in G.nodes():
        t = nx.get_node_attributes(G, 'weight')[k]
        color_lookup_nodes[k] = t
    
    
    color_lookup_edges = {} #Builds dictionary for edges weights
    for k in G.edges():
        n1 = k[0]
        n2 = k[1]
        t = G[n1][n2]['weight']
        color_lookup_edges[(n1, n2)] = t

    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=the_map) #will take values to colour later
    
    nodecolors = [mapper.to_rgba(i) for i in color_lookup_nodes.values()]
    
    for i in range(T):
        triangle = list_triangle[i]
        polygoni = matplotlib.patches.Polygon(np.array([list(pos[triangle[0]]), list(pos[triangle[1]]), list(
            pos[triangle[2]])]), color='lightgrey', closed=True, ec = 'lightgrey', alpha =0.08)

        patches.append(polygoni)
        
    p = PatchCollection(patches, match_original=True)

    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=nodesize, node_color=nodecolors, ax = ax)
    edges = nx.draw_networkx_edges(
        G,
        pos,
        edge_color=[mapper.to_rgba(i) for i in color_lookup_edges.values()],
        width=edge_width,
        ax = ax)
    
    ax.axis('off')
    return mapper

def plot_LTv1(s,list_edges, list_triangle, pos, ax, norm, the_map, edge_width_factor = 1):

    """
    plots signal located on links and triangles (encoded in topological signal s)
    list_edges, list_triangles: defines the underlying space
    pos: positions of the nodes
    ax: axes to plot
    norm: colormap normalisation
    the_map: color map to be used
    edge_width_factor = 1, not used.

    """
    N=len(pos)
    L=len(list_edges)
    T =len(list_triangle)
    
    G = nx.Graph()
    for j in range(0,N):
        G.add_node(j+1,weight = s[j]) #add nodes with weight
        
    for j in range(0, len(list_edges)):
        edge = list_edges[j]
        n1 = edge[0]
        n2 = edge[1]
        w12 = s[j+N] #skip the nodes to obtain the signal on edges
        G.add_edge(n1, n2, weight = w12)

    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    
    patches = []
    
    #Colour map set up
    
    color_lookup_nodes = {} #Builds dictionary for nodes weights
    for k in G.nodes():
        t = nx.get_node_attributes(G, 'weight')[k]
        color_lookup_nodes[k] = t
    
    
    color_lookup_edges = {} #Builds dictionary for edges weights
    for k in G.edges():
        n1 = k[0]
        n2 = k[1]
        t = G[n1][n2]['weight']
        color_lookup_edges[(n1, n2)] = t
    
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=the_map) #will take values to colour later
    
    nodecolors = [mapper.to_rgba(i) for i in color_lookup_nodes.values()]
    
    triangle_colors = [mapper.to_rgba(i) for i in s[-T:]]
    for i in range(T):
        triangle = list_triangle[i]
        polygoni = matplotlib.patches.Polygon(np.array([list(pos[triangle[0]]), list(pos[triangle[1]]), list(
            pos[triangle[2]])]), color=triangle_colors[i], closed=True, linewidth = 0)
        patches.append(polygoni)
    
    p = PatchCollection(patches, alpha=1, match_original=True)
    ax.add_collection(p)
   
    nodes = nx.draw_networkx_nodes(G, pos, node_size=0, node_color=nodecolors, ax=ax)
    edges = nx.draw_networkx_edges(
        G,
        pos,
        edge_color=[mapper.to_rgba(i) for i in color_lookup_edges.values()],
        width=3,
        ax=ax
        )
    ax.axis('off')
    return mapper

def xy_to_long_lattitude(a):
    """
    return longitude and latitude coordinates from x and y 

    input:
    a = [x,y]
    return:
    [long, lat]
    """
    x = a[0]
    y = a[1]
    long = x*180/np.pi
    lat = np.arcsin(y)*180/np.pi
    return [long ,lat]


### Mass sweep functions:
def mass_sweep_avg(s, beta, gamma, Nit, w, v, D, vinv, mmin=0, mmax=3):
    """
    conducts a mass sweep and returns the average over different iterations of noise

    inputs:
    s: true signal
    beta: amount of noise to be added
    gamma: regulariser parameter
    Nit: number of iterations
    w,v,D, vinv: eigenvalues, eigenvectors, Dirac, and inverse of eigenvectors
    mmin: mass min to start the sweep from
    mmax: mass max where to stop the sweep

    returns:
    error_avg: average error for each mass, 
    error_std: standard deviation of the error for each mass, 
    error0_avg: average error with decoupled processing (m=0), 
    error0_std: std of the error with decoupled processing(m=0)  
    mass_avg: average mass of the estimated signal for each mass of the sweep
    mass_std: std mass of the estimated signal for each mass of the sweep
    list_mass: mass swept over
    """
    list_mass = np.arange(mmin, mmax, 0.01)
    list_error = np.zeros((Nit, len(list_mass)))
    list_error0 = np.zeros((Nit))
    mass_est = np.zeros((Nit,len(list_mass)))
    list_snr = np.zeros((Nit))
    
    s = get_signal(s_gen, D)
    
    for i in range(Nit):
        s_true, s_noisy, m_target, snr = add_noise(s, w,v,D,beta)
        s_est = process(w,v,vinv, s_noisy, 0, gamma)
        error = np.linalg.norm(s_est - s_true,2)
        list_error0[i] = error 
        list_snr[i] = snr
        list_error_m = []
        list_mass_est = []
        
        for m in list_mass:
            mass = m
            s_est_m = process(w,v,vinv, s_noisy, m, gamma)
            error_m = np.linalg.norm(s_est_m - s_true,2)
            list_error_m.append(error_m)
            list_mass_est.append(((np.transpose(s_est_m)).dot(D)).dot(s_est_m)/((np.transpose(s_est_m)).dot(s_est_m)))
        list_error[i:,] = list_error_m
        mass_est[i:,] = list_mass_est
    
    error_avg = np.average(list_error,axis = 0)
    error0_avg = np.average(list_error0)
    mass_avg = np.average(mass_est,axis = 0)
    error_std =np.std(list_error,axis = 0)
    error0_std = np.std(list_error0)
    mass_std = np.std(mass_est,axis = 0)
    
    return error_avg, error_std, error0_avg, error0_std, mass_avg, mass_std, list_mass

def mass_sweep_avg_ratio(x_min, gamma, beta, Nit, w, v, vinv, D, gaussian = False):
    """
    conducts a mass sweep and returns the average relative error over different iterations of noise
    the signal is built from the eigenvectors of the Dirac directly using build_signals.

    inputs:
    x_min: flag: true for the eigenvector of the Dirac with the minimum eigenvalue, False for the max.
    gamma: regulariser parameter
    beta: amount of noise to be added
    Nit: number of iterations
    w,v, vinv: eigenvalues, eigenvectors, and inverse of eigenvectors
    gaussian: False for signals built from single eigenvectors of the Dirac, True for a Gaussian superposition
    
    returns:
    error_avg: relative average error for each mass, 
    error_std: standard deviation of the relative error for each mass, 
    mass_avg: average mass of the estimated signal for each mass of the sweep
    mass_std: std mass of the estimated signal for each mass of the sweep
    list_mass: mass swept over
    """
    
    list_error = np.zeros((Nit, 500))
    list_mass = np.linspace(-5,5, 500)
    mass_est = np.zeros((Nit,500))

    for i in range(Nit):
        s_true, s_noisy, m_target, snr = build_signals(w,v,D,x_min,beta = beta, gaussian = gaussian)
        s_est = process(w,v,vinv, s_noisy, 0, gamma)
        error = np.linalg.norm(s_est - s_true,2)
        list_error_m = []
        list_mass_est = []
        for m in list_mass:
            mass = m
            s_est_m = process(w,v,vinv, s_noisy, m, gamma)
            error_m = np.linalg.norm(s_est_m - s_true,2)
            list_error_m.append(error_m/error)
            list_mass_est.append(((np.transpose(s_est_m)).dot(D)).dot(s_est_m)/((np.transpose(s_est_m)).dot(s_est_m)))
        list_error[i:,] = list_error_m
        mass_est[i:,] = list_mass_est
    
    error_avg = np.average(list_error,axis = 0)
    mass_avg = np.average(mass_est,axis = 0)
    error_std =np.std(list_error,axis = 0)
    mass_std = np.std(mass_est,axis = 0)

    return error_avg, error_std, mass_avg, mass_std, list_mass


#####GRID ANALYSIS
def get_griderror(s,w,v,vinv,D,list_gamma, list_beta, m_0 = 0.3, tol = 0.0001):
    """
    Obtains the error in processing over a grid in beta and gamma
    
    inputs:
    s: pure signal (noise is generated in the function)
    w,v,vinv,D: eigenvector decomposition and inverse of eigenvectors, Dirac
    list_gamma: range of gamma to explore (regularisation parameter)
    list_beta: range of beta to explore (amount of noise)
    m_0: initial mass
    tol: tolerance

    returns the matrix of error 
    """
    noise = get_noise(s, w, v, D)
    list_error_m = np.empty([len(list_beta),len(list_gamma)])
    i = 0
    tic = time.perf_counter()
    for gamma in list_gamma:
        k=0
        for beta_i in list_beta:
            s_noisy = s + beta_i*noise
            s_true = s
            list_m, list_error, list_it, m, s_est_m = optimize_m(w,v,vinv,D,s_noisy,s_true,m_0,gamma,tol)
            error_m = np.linalg.norm(s_true-s_est_m)
        
            list_error_m[k,i]=error_m
            k+=1
        i+=1
    toc = time.perf_counter()
    snr = np.linalg.norm(s)**2/np.linalg.norm(max(list_beta)*noise)**2
    print('time',toc-tic, 'snr at beta = 2',snr)

    return list_error_m

def grid_error_avg(D_one, x_min, gaussian, list_gamma, list_beta, D1, D2, Nit, m_0, tol = 0.0001):
    
    """
    Obtains the average of the error in processing over a grid in beta and gamma
    
    inputs:
    D_one: True or False for nodes-links or links-triangles signals
    x_min: True or False depending on whether the signal is aligned with minimum or max eigenvector of the Dirac
    list_gamma: range of gamma to explore (regularisation parameter)
    list_beta: range of beta to explore (amount of noise)
    D1, D2: Dirac operators
    Nit: number of iterations to consider
    m_0: initial mass
    tol: tolerance

    returns the matrix of average error 
    """

    w,v,D,vinv = get_wvd(D_one,D1,D2)
    s = build_pure_signal(x_min, gaussian, w, v, D)
    
    avg_error = np.zeros([len(list_beta),len(list_gamma)])
    
    for i in np.arange(Nit):
        list_error_m = get_griderror(s,w,v,vinv,D,list_gamma,list_beta, m_0, tol)
        avg_error+=list_error_m/Nit

    return avg_error
        