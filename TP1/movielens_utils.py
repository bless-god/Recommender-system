import numpy as np
from scipy import sparse


def load_movielens(filename, minidata=False):
    """
    Cette fonction lit le fichier filename de la base de donnees
    Movielens, par exemple 
    filename = '~/datasets/ml-100k/u.data'
    Elle retourne 
    R : une matrice utilisateur-item contenant les scores
    mask : une matrice valant 1 si il y a un score et 0 sinon
    """

    data = np.loadtxt(filename, dtype=int)

    R = sparse.coo_matrix((data[:, 2], (data[:, 0]-1, data[:, 1]-1)),
                          dtype=float)
    R = R.toarray()  # not optimized for big data

    # code la fonction 1_K
    mask = sparse.coo_matrix((np.ones(data[:, 2].shape),
                              (data[:, 0]-1, data[:, 1]-1)), dtype=bool )
    mask = mask.toarray()  # not optimized for big data

    if minidata is True:
        R = R[0:100, 0:200].copy()
        mask = mask[0:100, 0:200].copy()

    return R, mask


def objective(P, Q0, R, mask, rho):
    """
    La fonction objectif du probleme simplifie.
    Prend en entree 
    P : la variable matricielle de taille C x I
    Q0 : une matrice de taille U x C
    R : une matrice de taille U x I
    mask : une matrice 0-1 de taille U x I
    rho : un reel positif ou nul

    Sorties :
    val : la valeur de la fonction
    grad_P : le gradient par rapport a P
    """

    tmp = (R - Q0.dot(P)) * mask

    val = np.sum(tmp ** 2)/2. + rho/2. * (np.sum(Q0 ** 2) + np.sum(P ** 2))

    grad_P = Q0.T.dot(-tmp) + rho * P # todo

    return val, grad_P


def total_objective(P, Q, R, mask, rho):
    """
    La fonction objectif du probleme complet.
    Prend en entree 
    P : la variable matricielle de taille C x I
    Q : la variable matricielle de taille U x C
    R : une matrice de taille U x I
    mask : une matrice 0-1 de taille U x I
    rho : un reel positif ou nul

    Sorties :
    val : la valeur de la fonction
    grad_P : le gradient par rapport a P
    grad_Q : le gradient par rapport a Q
    """

    tmp = (R - Q.dot(P)) * mask

    val = np.sum(tmp ** 2)/2. + rho/2. * (np.sum(Q ** 2) + np.sum(P ** 2))

    grad_P = Q.T.dot(-tmp) + rho * P  # todo

    grad_Q = (-tmp).dot(P.T) + rho * Q  # todo

    return val, grad_P, grad_Q


def total_objective_vectorized(PQvec, R, mask, rho):
    """
    Vectorisation de la fonction precedente de maniere a ne pas
    recoder la fonction gradient
    return vectorized gradient and value
    PQvec : 1 x (U*I) vector
    """
    
    # reconstruction de P et Q
    n_items = R.shape[1]
    n_users = R.shape[0]
    F = PQvec.shape[0] // (n_items + n_users)
    Pvec = PQvec[0:n_items*F]
    Qvec = PQvec[n_items*F:]
    P = np.reshape(Pvec, (F, n_items))
    Q = np.reshape(Qvec, (n_users, F))

    val, grad_P, grad_Q = total_objective(P, Q, R, mask, rho)
    PQvec_grad = np.concatenate([grad_P.ravel(), grad_Q.ravel()])
    
    return val, PQvec_grad



def find_gamma(x, gamma0, a, bcoef, R, mask, rho, Q=None):
    """
    Function based on Talyor based line search to find the next step in optimization
    x : current position. F-by-N matrix when Q is not None; Vector of size (U+I)*F when Q is None
    gamma0 : scalar, current step size
    a, bcoef : scalar, parameters for line search. gamma' = ba^l
    Q : None when P is the only variable; Otherwise it's a U-by-F matrix
    """
    b = bcoef * gamma0
    
    for l in range(0,100):
        
        gamma = b * a**l     
        
        if Q is None: 
            # Take both P,Q as variable
            val, x_grad = total_objective_vectorized(x, R, mask, rho)
            
            x_new = x - gamma * x_grad
            val_new, x_new_grad = total_objective_vectorized(x_new, R, mask, rho)            
            
        else:
            # Q fixed, find next step and gradient
            val, x_grad = objective(x, Q, R, mask, rho)
            
            x_new = x - gamma * x_grad            
            val_new, x_new_grad = objective(x_new, Q, R, mask, rho)
                    
        measure = val - val_new - 1/2 * gamma * np.linalg.norm(x_grad)**2

        if measure > 0:
            break

    return x_new, x_grad