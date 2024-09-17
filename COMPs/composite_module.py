from PENGoLINS.nonmatching_coupling import *

#class Composite_material():


#no difference between 90 and -90
# [[0, 1, 0.0], [1, 0, 0.0], [0, 0, -1]]
# [[3.749399456654644e-33, 1.0, -6.123233995736766e-17], [1.0, 3.749399456654644e-33, 6.123233995736766e-17], [1.2246467991473532e-16, -1.2246467991473532e-16, -1.0]]
def rotational_mat_for_strain(x):
    """
    T^-T
    """
    x_rad = Constant(x*np.pi/180)

    if x == 0:
        x1, x2, x3 = 1, 0, 0
    elif x == 90:
        x1, x2, x3 = 0, 1, 0
    else:
        x1 = cos(x_rad)**2
        x2 = sin(x_rad)**2
        x3 = sin(2*x_rad)
    return as_matrix([[x1, x2, x3/2], [x2, x1, -x3/2], [-x3, x3, x1-x2]])

def rotational_mat_for_strain_not_transposed(x):
    """
    T^-T
    T used for YX systems like in plate with a hole example, where for some patches rotation is done clockwise, due to change of coordinates
    """
    x_rad = Constant(x*np.pi/180)

    if x == 0:
        x1, x2, x3 = 1, 0, 0
    elif x == 90:
        x1, x2, x3 = 0, 1, 0
    else:
        x1 = cos(x_rad)**2
        x2 = sin(x_rad)**2
        x3 = sin(2*x_rad)
    return as_matrix([[x1, x2, x3/2], [x2, x1, -x3/2], [-x3, x3, x2-x1]])

def Compliance_matrix(E1, E2, nu12, G12):
    """
    Strain = Comp_mat*stress for lamina (pC)
    """

    nu21 = nu12*E2/E1

    pC = as_matrix([[1/E1, -nu12/E1, 0],
                   [-nu21/E2, 1/E2, 0],
                   [0, 0, 1/G12]])
    return pC

def Stiffness_matrix_lamina(pC):
    """
    Stress = Stiff_mat*strain for lamina (pS)
    """
    return inv(pC)

# coordinates
def layer_cord(pT, offset=0):
    """
    pT - thickness of each layer
    """
    tT = sum(pT)
    z0 = [sum(pT[:k])-tT/2+offset for k in range(len(pT))]
    z1 = [sum(pT[:k+1])-tT/2+offset for k in range(len(pT))]
    return z0, z1

def Compliance_matrix_lamina_array(prop_array):
    """
    Compute relative stiffness for stresses

    Returns
    -------
    (B_) : relative compliance created to split the stress between layers
    """
    pC = [[] for i in range(len(prop_array))]

    for i in range(len(prop_array)):
        pC[i] = Compliance_matrix(prop_array[i][0], prop_array[i][1], prop_array[i][2], prop_array[i][3])

    return pC

def Stiffness_matrix_lamina_array(pC):
    """
    Compute relative stiffness for stresses

    Returns
    -------
    (B_) : relative stiffness created to split the stress between layers
    """
    pS = [[] for i in range(len(pC))]

    for i in range(len(pC)):
        pS[i] = Stiffness_matrix_lamina(pC[i])

    return pS

def Stiffness_matrix_laminate(pR, pS):
    """
    Matrix to change general strain of the whole structure to stresses for each layer
    p0 - ufl vector, array of angles
    pS - ufl vector, array of stiffness matrices for through angles
    """
    # Q_ =  T^-1 * Q * T^-T
    return [dot(pR[i].T, dot(pS[i], pR[i])) for i in range(len(pR))]

def complex_laminate_ABD_mat(pQ, pT):
    """
    Compute laminated A,B and D material matrices for complex composites (not only orthotropic) with multiple materials in array, different thickness
    p0 - ufl vector, list of float, fiber angles for all layers in degree
    pT - ufl vector, list of float, layer thickness for all layers
    prop_array - list of E1, E2, nu12, G12 for each layer

    Returns
    -------
    (A, B, D) : Extensional, extensional-bending coupling and
                bending material matrices
    """

    z0, z1 = layer_cord(pT)

    mA = sum([x*y for x, y in zip(pQ, pT)])
    mB = sum([x*(top**2-bot**2) for x, top, bot in zip(pQ, z1, z0)])/2
    mD = sum([x*(top**3-bot**3) for x, top, bot in zip(pQ, z1, z0)])/3

    return (mA, mB, mD)

def complex_laminate_A_mat(pQ, pT):

    return sum([x*y for x, y in zip(pQ, pT)])


def complex_laminate_B_mat(pQ, z0, z1):

    return sum([x*(top**2-bot**2) for x, top, bot in zip(pQ, z1, z0)])/2


def complex_laminate_D_mat(pQ, z0, z1):

    return sum([x*(top**3-bot**3) for x, top, bot in zip(pQ, z1, z0)])/3



def initialize_composite_array(num_srfs, n_ply = None, nh = None, kh = None):

    ### n_ply must be a list or a scalar not a np.array since isinstance won't work for numpy arrays. eg.

    array = []
    for i in range(num_srfs):
        array += [[],]
        if n_ply != None:
            if isinstance(n_ply, list):
                for j in range(n_ply[i]):
                    array[i] += [[],]
                    if nh != None:
                        for k in range(nh):
                            array[i][j] += [[],]
                            if kh != None:
                                for u in range(kh):
                                    array[i][j][k] += [[],]
            else:
                for j in range(n_ply):
                    array[i] += [[],]
                    if nh != None:
                        for k in range(nh):
                            array[i][j] += [[],]
                            if kh != None:
                                for u in range(kh):
                                    array[i][j][k] += [[],]
    return array

def composite_gather(function_to_gather, comm):
    for i in range(len(function_to_gather)):
        for k in range(len(function_to_gather[i])):
            for gi in range(len(function_to_gather[i][k])):
                vec_local = function_to_gather[i][k][gi].vector().get_local()
                array_gathered = comm.gather(array, root=0)
