import numpy as np
import scipy as sp
import scipy.io
import scipy.signal

def music(patt, L, grid):
    """Performs a MUSIC-scan for a given pattern in the leadfield

    Parameters
    ----------
    patt : (N,M) ndarray for N channels; 
        each column in patt represents a spatial pattern;
        (only the span(patt) matters; mixing of the patterns has no effect)
    L : (N,M,3) ndarray for m grid points; L(:,i,j) is the potential
        of a unit dipole at point i in direction j
    grid : (M,3) ndarray denoting locations of grid points
    
    Returns
    -------
    s : (M, K) 
        s(i,k) indicates fit-quality (from 0 (worst) to 1 (best)) at grid-point
        i of k.th dipole (i.e. preceeding k-1 dipoles are projected out 
        at each location); the first column is the 'usual' MUSIC-scan
    vmax : (N, ) ndarray
        the field of the best dipole
    imax : int
        denotes grid-index of best dipole
    dip_mom : (3, ) ndarray
        the moment of the  best dipole
    dip_loc : (3, ) ndarray
        the location of the  best dipole
               
    """
    if np.ndim(patt)==1:
        patt=np.expand_dims(patt,1)
    nchan, nx = patt.shape
    
    dims=np.ones(4,dtype=int)
    dims[:len(L.shape)]=L.shape
    nchan, ng, ndum, nsubj =    dims
    
    data = patt /np.linalg.norm(patt)
    nd = np.minimum(nx,ndum)
    
    [s,vmax,imax]=calc_spacecorr_all(L,data,nd)
    
    dip_mom=vmax2dipmom(L,imax,vmax)
    dip_loc = grid[imax,:]
    
    return s, vmax, imax, dip_mom, dip_loc

def calc_spacecorr_all(V, data, nd):

    dims=np.ones(4,dtype=int)
    dims[:len(V.shape)]=V.shape
    nchan, ng, ndum, nsubj =    dims                            # [nchan,ng,ndum]=size(V);
    s = np.zeros((ng, nd))                                      # s=zeros(ng,nd);

    for i in range(ng):                                         # for i=1:ng;
        Vortholoc = sp.linalg.orth(np.squeeze(V[:, i, :]))      # Vortholoc=orth(squeeze(V(:,i,:)));
        s[i, :] = calc_spacecorr(Vortholoc, data, nd)           # s(i,:)=calc_spacecorr(Vortholoc,data,nd);

    imax = np.argmax(s[:, 0])                                   # [smax,imax]=max(s(:,1));
    Vortholoc = sp.linalg.orth(np.squeeze(V[:, imax, :]))       # Vortholoc=orth(squeeze(V(:,imax,:)));
    vmax , sbest= calc_bestdir(Vortholoc, data)                        # vmax=calc_bestdir(Vortholoc,data);
    return s, vmax, imax

def calc_spacecorr(Vloc, data_pats, nd):

    A = np.matmul(data_pats.T, Vloc)                                      # A=data_pats'*Vloc;
    s = np.sqrt(np.abs(np.linalg.eigvals(np.outer(A, A))[:nd]))                    # s=sd(1:nd);

    return s

def vmax2dipmom(V, imax_all, vmax_all):
    if np.isscalar(imax_all):
        ns = 1 
        vmax_all=np.expand_dims(vmax_all,1)
        imax_all=np.expand_dims(imax_all,0)
    else: 
        ns =len(imax_all)
        
    dips_mom_all = np.zeros((ns, 3))                                # dips_mom_all = zeros(ns, 3);

    for i in range(ns):                                             # for i=1:ns
        Vloc = np.squeeze(V[:, imax_all[i], :])                     # Vloc = squeeze(V(:, imax_all(i),:));

        v = vmax_all[:,i]                                         # v = vmax_all(:, i);
        dip = np.linalg.inv(Vloc.T @ Vloc) @ Vloc.T @ v             # dip = inv(Vloc'*Vloc)*Vloc' * v;
        dips_mom_all[i, :] = dip.T / np.linalg.norm(dip)            # dips_mom_all(i,:)=dip'/norm(dip);

    return dips_mom_all

def calc_bestdir(Vloc, data_pats, proj_pats={}):

    if len(proj_pats) == 0:                                                 # if nargin==2
        A = data_pats.T @ Vloc                                              # A=data_pats'*Vloc;
        u, s, v = sp.linalg.svd(A)                                          # [u s, v]=svd(A);
        vmax = Vloc @ v[:, 0]                                               # vmax=Vloc*v(:,1);
        vmax = vmax / np.linalg.norm(vmax)                                  # vmax=vmax/norm(vmax);
        s = s[0]                                                            # s=s(1,1);
    else:
        n, m = Vloc.shape                                                   # [n m]=size(Vloc);
        a = proj_pats.T @ Vloc
        V_proj = sp.linalg.orth(Vloc - proj_pats @ a)                       # V_proj=orth(Vloc-proj_pats*(proj_pats'*Vloc));
        A = data_pats.T @ V_proj                                            # A=data_pats'*V_proj;
        u, s, v = sp.linalg.svd(A)                                          # [u, s v]=svd(A);
        BB = Vloc.T @ proj_pats                                             # BB=(Vloc'*proj_pats);
        Q = np.linalg.inv(np.identity(m) - BB @ BB.T + np.sqrt(np.finfo(float).eps))   # Q=inv(eye(m)-BB*BB'+sqrt(eps));
        vmax = Vloc @ (Q @ Vloc.T @ (V_proj @ v[:, 0]))                     # vmax=Vloc*(Q*Vloc'*(V_proj*v(:,1)));
        vmax = vmax / np.linalg.norm(vmax)                                             # vmax=vmax/norm(vmax);
        s = s[0, 0]                                                         # s=s(1,1);

    return vmax, s

