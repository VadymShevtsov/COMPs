from tIGAr.NURBS import *
from dolfin import *
from PENGoLINS.igakit_utils import *

import COMPs.composite_module as cm

from numpy.polynomial import Polynomial as P
import numpy as np
import math


import copy

# if False dofs will be reordered to compute_vertex_values method, but will crash the program
parameters["reorder_dofs_serial"] = True


def utility_function_check_prop_output(A, exampl_array):
    """
    used to project the whole UFL matrix A, exampl_array is used to form nodal output (DOF type)
    """    
    if hasattr(A[0, 0], 'vector'):
        A_temp_0_0 = (A[0, 0].vector().get_local())
        A_temp_0_1 = (A[0, 1].vector().get_local())
        A_temp_0_2 = (A[0, 2].vector().get_local())
        A_temp_1_0 = (A[1, 0].vector().get_local())
        A_temp_1_1 = (A[1, 1].vector().get_local())
        A_temp_1_2 = (A[1, 2].vector().get_local())
        A_temp_2_0 = (A[2, 0].vector().get_local())
        A_temp_2_1 = (A[2, 1].vector().get_local())
        A_temp_2_2 = (A[2, 2].vector().get_local())
    else:
        A_temp_0_0 = [float(A[0, 0]) for gi in range(len(exampl_array))]
        A_temp_0_1 = [float(A[0, 1]) for gi in range(len(exampl_array))]
        A_temp_0_2 = [float(A[0, 2]) for gi in range(len(exampl_array))]
        A_temp_1_0 = [float(A[1, 0]) for gi in range(len(exampl_array))]
        A_temp_1_1 = [float(A[1, 1]) for gi in range(len(exampl_array))]
        A_temp_1_2 = [float(A[1, 2]) for gi in range(len(exampl_array))]
        A_temp_2_0 = [float(A[2, 0]) for gi in range(len(exampl_array))]
        A_temp_2_1 = [float(A[2, 1]) for gi in range(len(exampl_array))]
        A_temp_2_2 = [float(A[2, 2]) for gi in range(len(exampl_array))]

    return A_temp_0_0, A_temp_0_1, A_temp_0_2, A_temp_1_0, A_temp_1_1, A_temp_1_2, A_temp_2_0, A_temp_2_1, A_temp_2_2


#only 2 projections per surface, but has bottleneck in the amount of computations, way faster then others in parallel due to decreased number of computations for each processor
def stress_strain_output_full_numpy(splines, spline_funct, D_stif_, R_, h_tot, z_middle, n_ply, print_out = False):

    """
    Function used to form stress/strain output in the serial run
    
    splines - IGA surfaces on which the output is done
    spline_funct - UFL variable that stores deformations
    D_stif_ - constitutive matrix for each surface and each layer
    Rot_str - rotational matrix
    h_tot - an array of total thicknesses
    z_middle - an array of middle coordinates of each ply
    n_ply - number of each plies
    
    """    


    # Stress_strain_output(problem.splines[i], problem.spline_funcs[i], D_stif_new[i][k], R_[i][k], h_total[i], z_mid[i][k])

    num_spl = len(splines)

    SPK_stress_lay = cm.initialize_composite_array(num_spl, n_ply, 3)
    strain_arr = cm.initialize_composite_array(num_spl, n_ply, 3)

    epsilonBar_arr = cm.initialize_composite_array(num_spl, 3)
    kappaBar_arr = cm.initialize_composite_array(num_spl, 3)

    for i in range(num_spl):

        spline_strain = ShellForceSVK(splines[i], spline_funct[i], 1, 0.5, 1,  linearize=False)

        #voigt2D strains = True by def
        epsilonBar_temp = voigt2D(spline_strain.membraneStrain())
        kappaBar_temp = voigt2D(spline_strain.curvatureChange())

        for gi in range(3):
            epsilonBar_arr[i][gi] = splines[i].projectScalarOntoLinears(\
            epsilonBar_temp[gi], lumpMass=True)
            kappaBar_arr[i][gi] = splines[i].projectScalarOntoLinears(\
            kappaBar_temp[gi], lumpMass=True)

        epsilonBar_temp_0 = (epsilonBar_arr[i][0].vector().get_local())
        epsilonBar_temp_1 = (epsilonBar_arr[i][1].vector().get_local())
        epsilonBar_temp_2 = (epsilonBar_arr[i][2].vector().get_local())

        kappaBar_temp_0 = (kappaBar_arr[i][0].vector().get_local())
        kappaBar_temp_1 = (kappaBar_arr[i][1].vector().get_local())
        kappaBar_temp_2 = (kappaBar_arr[i][2].vector().get_local())

        ref_array = epsilonBar_arr[i][0]

        if print_out == True and i == 0:
            print("epsilonBar_temp_1")
            print(epsilonBar_temp_1)
            print("kappaBar_temp_1")
            print(kappaBar_temp_1)

        for k in range(n_ply[i]):

            D_stif_temp_0_0, D_stif_temp_0_1, D_stif_temp_0_2,\
            D_stif_temp_1_0, D_stif_temp_1_1, D_stif_temp_1_2,\
            D_stif_temp_2_0, D_stif_temp_2_1, D_stif_temp_2_2 = utility_function_check_prop_output(D_stif_[i][k], epsilonBar_temp_0)

            Rot_str_np = np.array([[1.1 for ig in range(3)] for gi in range(3)])

            pR_temp_0_0, pR_temp_0_1, pR_temp_0_2,\
            pR_temp_1_0, pR_temp_1_1, pR_temp_1_2,\
            pR_temp_2_0, pR_temp_2_1, pR_temp_2_2 = utility_function_check_prop_output(R_[i][k], epsilonBar_temp_0)

            strain_array_0, strain_array_1, strain_array_2 = [], [], []
            stress_array_0, stress_array_1, stress_array_2 = [], [], []

            for strain_i_count in range(len(epsilonBar_temp_0)):
                # local strain

                Green_Lagrange_strain_temp = np.array([epsilonBar_temp_0[strain_i_count], epsilonBar_temp_1[strain_i_count], epsilonBar_temp_2[strain_i_count]]) + \
                                            float(z_middle[i][k])*np.array([kappaBar_temp_0[strain_i_count], kappaBar_temp_1[strain_i_count], kappaBar_temp_2[strain_i_count]])

                # local stiffness

                D_stif_temp = np.array([[D_stif_temp_0_0[strain_i_count], D_stif_temp_0_1[strain_i_count], D_stif_temp_0_2[strain_i_count]],
                               [D_stif_temp_1_0[strain_i_count], D_stif_temp_1_1[strain_i_count], D_stif_temp_1_2[strain_i_count]],
                               [D_stif_temp_2_0[strain_i_count], D_stif_temp_2_1[strain_i_count], D_stif_temp_2_2[strain_i_count]]])

                Rot_str_np = np.array([[pR_temp_0_0[strain_i_count],pR_temp_0_1[strain_i_count],pR_temp_0_2[strain_i_count]],
                                 [pR_temp_1_0[strain_i_count],pR_temp_1_1[strain_i_count],pR_temp_1_2[strain_i_count]],
                                 [pR_temp_2_0[strain_i_count],pR_temp_2_1[strain_i_count],pR_temp_2_2[strain_i_count]]])

                strain_rotated_temp = np.dot(Rot_str_np, Green_Lagrange_strain_temp)

                strain_array_0.append(strain_rotated_temp[0])
                strain_array_1.append(strain_rotated_temp[1])
                strain_array_2.append(strain_rotated_temp[2])

                SPK_stress_each_layer_local_temp = np.dot(D_stif_temp, strain_rotated_temp)

                stress_array_0.append(SPK_stress_each_layer_local_temp[0])
                stress_array_1.append(SPK_stress_each_layer_local_temp[1])
                stress_array_2.append(SPK_stress_each_layer_local_temp[2])

                # if print_out == True and SPK_stress_each_layer_local_temp[1] < 0 and strain_rotated_temp[1] > 0:
                #     print("original strains")
                #     print(strain_matr_temp, "\n")
                #     print("strains rotation matrix")
                #     print(Rot_str_np, "\n")
                #     print("local strains")
                #     print(strain_rotated_temp, "\n")
                #     print("stress-strains transition matrix")
                #     print(D_stif_temp, "\n")
                #     print("local stress")
                #     print(SPK_stress_each_layer_local_temp, "\n")
                #     print("\n\n\n")

            strain_return_array = [strain_array_0, strain_array_1, strain_array_2]
            stress_return_array = [stress_array_0, stress_array_1, stress_array_2]

            for gi in range(3):
                SPK_stress_lay[i][k][gi] = ref_array.copy(deepcopy=True)
                strain_arr[i][k][gi] = ref_array.copy(deepcopy=True)

                SPK_stress_lay[i][k][gi].vector()[:] = stress_return_array[gi]
                strain_arr[i][k][gi].vector()[:] = strain_return_array[gi]

            if print_out == True and i == 0 and k == 0:
                print(strain_arr[i][k][1].vector().get_local())

    return SPK_stress_lay, strain_arr


###stress connectivity

###########################check the work just in case

class stress_connectivity:
    """
    stress connectivity class for a serial run
    
    """
    def __init__(self, problem, el_nm_array, degree_param = 2, print_steps = False):
        """
        problem - used to define NonMatchingCouplingLaminate class
        el_nm_array - number of elements in each patch
        degree_param - artificial parameter used to calculate the degree for each curve used for curve fitting

        """
        self.map_list = problem.mapping_list
        self.map_locs = problem.mortar_parametric_coords
        self.el_nm_array = el_nm_array
        self.print_steps = print_steps
        self.degree_param = degree_param
        self.splines = problem.splines

    # watch out for a proper number of elements in each direction

    def parametric_cord_to_dir(self, array_of_loc):
        """
        function to analyse the location of each connection (given in parametric space)
        
        """
    
        if array_of_loc[0][0] == array_of_loc[1][0]:
            main_dir = 0 #all points in question would be in Y dir, so our dir is x
            if array_of_loc[0][0] == 1:
                side_dir = 1
            else:
                side_dir = 0
            #[[1., 1.], [1, 0.5]]
            start_curv = array_of_loc[0][1] #1
            finish_curv = array_of_loc[1][1] #0.5

        if array_of_loc[0][1] == array_of_loc[1][1]:
            main_dir = 1 #all points in question would be in X dir, so our dir is Y
            if array_of_loc[0][1] == 1:
                side_dir = 1
            else:
                side_dir = 0

            #in case [[0., 1.], [0.5, 1.]]
            start_curv = array_of_loc[0][0] #0
            finish_curv = array_of_loc[1][0] #0.5

        return main_dir, side_dir, start_curv, finish_curv

    def param_to_array(self, array, dirr, side, start, finish, reverse):
        """
        data from the analysis is used to form an edge array
        
        """
        

        # think that node arrays are the same in each line
        array_max_dim_0 = len(array[0])
        array_max_dim_1 = len(array)

        if dirr == 0 and side == 0:
            temp0 = [array[i][0] for i in range(array_max_dim_1)]
        elif dirr == 0 and side == 1:
            temp0 = [array[i][array_max_dim_0-1] for i in range(array_max_dim_1)]
        elif dirr == 1 and side == 0:
            temp0 = array[0]
        elif dirr == 1 and side == 1:
            temp0 = array[array_max_dim_1-1]

        temp1 = temp0[math.floor(start*len(temp0)):math.ceil(finish*len(temp0))]

        if reverse == True:
            temp1 = temp1[::-1]

        return temp1

    def array_to_param(self, array, side_array, dirr, side, start, finish, reverse):
        """
        edge array is used to form updated DOF output
        
        """        

        # think that node arrays are the same in each line
        array_max_dim_0 = len(array[0])
        array_max_dim_1 = len(array)

        if reverse == True:
            side_array = side_array[::-1]

        if dirr == 0 and side == 0:
            tem_i = 0
            # for i in range(len(side_array)):
            for i in range(math.floor(start*len(array)),math.ceil(finish*len(array))):
                array[i][0] = side_array[tem_i]
                tem_i += 1
        elif dirr == 0 and side == 1:
            tem_i = 0
            # for i in range(len(side_array)):
            for i in range(math.floor(start*len(array)),math.ceil(finish*len(array))):
                array[i][array_max_dim_0-1] = side_array[tem_i]
                tem_i += 1
        elif dirr == 1 and side == 0:
            tem_i = 0
            # for i in range(len(side_array)):
            for i in range(math.floor(start*len(array[0])),math.ceil(finish*len(array[0]))):
                array[0][i] = side_array[tem_i]
                tem_i += 1
        elif dirr == 1 and side == 1:
            tem_i = 0
            # for i in range(len(side_array)):
            # array[0] since dim should be the same as array[array_max_dim_1-1]
            for i in range(math.floor(start*len(array[0])),math.ceil(finish*len(array[0]))):
                array[array_max_dim_1-1][i] = side_array[tem_i]
                tem_i += 1

        return array

    def connect_stress(self, cauchy_stress_layer):
        
        """
        function that is used to smooth the function
        """
        
        num_of_surfs = len(self.el_nm_array)
        n_ply = [len(cauchy_stress_layer[i]) for i in range(num_of_surfs)]

        cauchy_stress_layer_return = cm.initialize_composite_array(num_of_surfs, n_ply, 3)

        for i in range(num_of_surfs):
            for k in range(n_ply[i]):
                for j in range(3):
                    cauchy_stress_layer_return[i][k][j] = cauchy_stress_layer[i][k][j].copy(deepcopy=True)

        for i in range(len(self.map_list)):
            s_ind0, s_ind1 = self.map_list[i]

            inter_loc_s_ind0 = self.map_locs[i][0] #[[1., 0.],  [1., 1.]]
            inter_loc_s_ind1 = self.map_locs[i][1] #[[0., 0.],  [0., 1.]]



            dir_A_mat_0, side_A_mat_0, start_curv_temp0, finish_curv_temp0 = self.parametric_cord_to_dir(inter_loc_s_ind0)
            start_curv_0 = min(start_curv_temp0, finish_curv_temp0)
            finish_curv_0 = max(start_curv_temp0, finish_curv_temp0)

            if start_curv_temp0 > finish_curv_temp0:
                reverse_0 = True
            else:
                reverse_0 = False

            dir_A_mat_1, side_A_mat_1, start_curv_temp1, finish_curv_temp1 = self.parametric_cord_to_dir(inter_loc_s_ind1)
            start_curv_1 = min(start_curv_temp1, finish_curv_temp1)
            finish_curv_1 = max(start_curv_temp1, finish_curv_temp1)

            if start_curv_temp1 > finish_curv_temp1:
                reverse_1 = True
            else:
                reverse_1 = False


            #number of elements in each direction for each of connected patches
            num_el0_s0, num_el1_s0  = self.el_nm_array[s_ind0]
            num_el0_s1, num_el1_s1  = self.el_nm_array[s_ind1]

            min_n_ply = min(len(cauchy_stress_layer_return[s_ind0]),len(cauchy_stress_layer_return[s_ind1]))

            id_stress_s_ind0_reverse = vertex_to_dof_map(cauchy_stress_layer_return[s_ind0][0][0].function_space())
            id_stress_s_ind1_reverse = vertex_to_dof_map(cauchy_stress_layer_return[s_ind1][0][0].function_space())

            for k in range(min_n_ply):
                for j in range(3):
                    ### https://fenicsproject.org/pub/tutorial/html/._ftut1019.html about compute_vertex_values and their order
                    stress_s0_array_j_V = np.array(cauchy_stress_layer_return[s_ind0][k][j].compute_vertex_values())
                    stress_s0_array_j_dof = np.array(cauchy_stress_layer_return[s_ind0][k][j].vector().get_local())

                    stress_s1_array_j_V = np.array(cauchy_stress_layer_return[s_ind1][k][j].compute_vertex_values())
                    stress_s1_array_j_dof = np.array(cauchy_stress_layer_return[s_ind1][k][j].vector().get_local())

                    ###### by running this test we can see that map always stays constant and is only dependent on the nodes

                    # # id_stress_s_ind0 and id_stress_s_ind1 wont be the same since its 2 different surf
                    # id_stress_s_check_0 = v_to_dof_simple_func(cauchy_stress_layer_return[s_ind0][k][j])
                    # id_stress_s_ind0 = self.V_to_dof(stress_s0_array_j_V, stress_s0_array_j_dof)
                    # if id_stress_s_check_0 == id_stress_s_ind0:
                    #     print(s_ind0, k, j, "yes")
                    # else:
                    #     print(k, j, "\nno\n")

                    # form num_el0_s0/ num_el1_s0 and num_el0_s1/ num_el1_s1 arrays out stress_0_array_j and stress_1_array_j

                    proper_arrays0 = [[stress_s0_array_j_V[num0+num1*num_el0_s0] for num0 in range(num_el0_s0)] for num1 in range(num_el1_s0)]
                    proper_arrays1 = [[stress_s1_array_j_V[num0+num1*num_el0_s1] for num0 in range(num_el0_s1)] for num1 in range(num_el1_s1)]

                    # printing test to see value output stress_s0_array_j_V outputs acording to ._ftut1019.html order
                    # test is done on 1 layer Stress12

                    # and s_ind1 == 9
                    if self.print_steps == True and k == 0 and j == 2:
                        # for ed in range(num_el1_s0):
                        #     print(np.round(proper_arrays0[ed], 3))
                        print("\n\n\n\n\n", s_ind0, s_ind1)
                        print("stress_s0_array_j_V")
                        print(np.round(stress_s0_array_j_V, 6))
                        print("stress_s1_array_j_V")
                        print(np.round(stress_s1_array_j_V, 6))
                        # print("id_stress_s_ind0")
                        # print(np.round(id_stress_s_ind0, 6))


                    # using dir_A_mat_0, side_A_mat_0 dir_A_mat_1, side_A_mat_1 take the needed values

                    stress_s_ind0_side_array = self.param_to_array(proper_arrays0, dir_A_mat_0, side_A_mat_0, start_curv_0, finish_curv_0, reverse_0)
                    stress_s_ind1_side_array = self.param_to_array(proper_arrays1, dir_A_mat_1, side_A_mat_1, start_curv_1, finish_curv_1, reverse_1)

                    if self.print_steps == True and k == 0 and j == 2:
                        print("proper_arrays0")
                        for ed in range(num_el1_s0):
                            #printing acording to geometry
                            print(np.round(proper_arrays0[(num_el1_s0)-1-ed], 6))
                        print("dir_A_mat_0, side_A_mat_0, start_curv_0, finish_curv_0")
                        print(dir_A_mat_0, side_A_mat_0, start_curv_0, finish_curv_0)
                        print("stress_s_ind0_side_array")
                        print(stress_s_ind0_side_array)
                        print("proper_arrays1")
                        for ed in range(num_el1_s1):
                            #printing acording to geometry
                            print(np.round(proper_arrays1[(num_el1_s1)-1-ed], 6))
                        print("dir_A_mat_1, side_A_mat_1, start_curv_1, finish_curv_1")
                        print(dir_A_mat_1, side_A_mat_1, start_curv_1, finish_curv_1)
                        print("stress_s_ind1_side_array")
                        print(stress_s_ind1_side_array)


                    # [-238.343 -259.081 -290.855 -310.239]
                    # [-241.806 -266.232 -297.403 -313.274]
                    # [-243.947 -271.912 -303.955 -318.593]
                    # [-250.131 -284.227 -315.294 -324.343]
                    # [-267.377 -312.505 -325.534 -327.63 ]
                    # [-315.181 -335.206 -326.616 -328.874]
                    # 0 1
                    # [-328.87416643969976, -327.63042583731846, -324.34262445666513, -318.59296758824075, -313.2735181738468, -310.2385463184358]
                    # [-238.343 -259.081 -290.855 -310.239]
                    # [-241.806 -266.232 -297.403 -313.274]
                    # [-243.947 -271.912 -303.955 -318.593]
                    # [-250.131 -284.227 -315.294 -324.343]
                    # [-267.377 -312.505 -325.534 -327.63 ]
                    # [-315.181 -335.206 -326.616 -328.874]
                    # 1 1
                    # [-238.3427163197952, -259.08106836861003, -290.85519679059126, -310.2385463184358]

                    # curve fit in case we have different number of nodes

                    x_cord0 = [ii/(len(stress_s_ind0_side_array)-1) for ii in range(len(stress_s_ind0_side_array))]
                    order0 = max(3, int(len(stress_s_ind0_side_array)/self.degree_param))
                    stress_s_ind0_poly = np.poly1d(np.polyfit(x_cord0, stress_s_ind0_side_array, order0))

                    x_cord1 = [ii/(len(stress_s_ind1_side_array)-1) for ii in range(len(stress_s_ind1_side_array))]
                    order1 = max(3, int(len(stress_s_ind1_side_array)/self.degree_param))
                    stress_s_ind1_poly = np.poly1d(np.polyfit(x_cord1, stress_s_ind1_side_array, order1))

                    x_cord_refine = [(i)/(1001-1) for i in range(1001)]

                    stress_s_ind0_poly_to_array = [stress_s_ind0_poly(i) for i in x_cord_refine]
                    stress_s_ind1_poly_to_array = [stress_s_ind1_poly(i) for i in x_cord_refine]

                    # make the avarage curve out of it

                    proper_stress_array = [(stress_s_ind0_poly_to_array[ii]+stress_s_ind1_poly_to_array[ii])/2 for ii in range(len(x_cord_refine))]
                    proper_stress_array_poly = np.poly1d(np.polyfit(x_cord_refine, proper_stress_array, 8))

                    # put the values back to the nodes

                    stress_s_ind0_new = [proper_stress_array_poly(i) for i in x_cord0]
                    stress_s_ind1_new = [proper_stress_array_poly(i) for i in x_cord1]

                    proper_arrays0_new = self.array_to_param(proper_arrays0, stress_s_ind0_new, dir_A_mat_0, side_A_mat_0, start_curv_0, finish_curv_0, reverse_0)
                    proper_arrays1_new = self.array_to_param(proper_arrays1, stress_s_ind1_new, dir_A_mat_1, side_A_mat_1, start_curv_1, finish_curv_1, reverse_1)

                    if self.print_steps == True and k == 0 and j == 2:
                        print("proper_arrays0_new")
                        for ed in range(num_el1_s0):
                            #printing acording to geometry
                            print(np.round(proper_arrays0_new[(num_el1_s0)-1-ed], 6))
                        print("proper_arrays1_new")
                        for ed in range(num_el1_s1):
                            #printing acording to geometry
                            print(np.round(proper_arrays1_new[(num_el1_s1)-1-ed], 6))


                    stress_s0_array_j_V_new = []
                    for num1 in range(num_el1_s0):
                        for num0 in range(num_el0_s0):
                            stress_s0_array_j_V_new.append(proper_arrays0_new[num1][num0])

                    stress_s1_array_j_V_new = []
                    for num1 in range(num_el1_s1):
                        for num0 in range(num_el0_s1):
                            stress_s1_array_j_V_new.append(proper_arrays1_new[num1][num0])

                    stress_ar_0 = [1.1 for ii in range(len(stress_s0_array_j_dof))]
                    for ii in range(len(stress_s0_array_j_dof)):
                        stress_ar_0[id_stress_s_ind0_reverse[ii]] = stress_s0_array_j_V_new[ii]

                    stress_ar_1 = [1.1 for i in range(len(stress_s1_array_j_dof))]
                    for ii in range(len(stress_s1_array_j_dof)):
                        stress_ar_1[id_stress_s_ind1_reverse[ii]] = stress_s1_array_j_V_new[ii]

                    cauchy_stress_layer_return[s_ind0][k][j].vector()[:] = stress_ar_0
                    cauchy_stress_layer_return[s_ind1][k][j].vector()[:] = stress_ar_1

        return cauchy_stress_layer_return

class stress_connectivity_parallel(stress_connectivity):
    """
    additional class that is used to smooth the functions is parallel
    """
    def __init__(self, problem, el_nm_array, degree_param = 2, print_steps = False):

        self.comm = problem.comm

        super().__init__(problem, el_nm_array, degree_param, print_steps)

    def connect_stress_parallel(self, cauchy_stress_layer):
        num_of_surfs = len(self.el_nm_array)
        n_ply = [len(cauchy_stress_layer[i]) for i in range(num_of_surfs)]

        if MPI.rank(self.comm) == 0:
            print("\tparallel save...")

        #write initial array on all processors
        for i in range(num_of_surfs):

            ####################### saving mesh from parallel breaks the connectivity in-between the partition, save should only be done in serial

            # filename_mesh = "mesh_surface_" + str(i)
            #
            # full_mesh_name = "./temp/"+str(filename_mesh)+".xdmf"
            #
            # fFile_mesh = XDMFFile(worldcomm, full_mesh_name)
            #
            # #deepcopy of the mesh
            # mesh_to_send = Mesh(self.splines[i].V_linear.mesh())
            #
            # fFile_mesh.write(mesh_to_send)
            # fFile_mesh.close()

            for k in range(n_ply[i]):
                for j in range(3):

                    filename = "Stress_" + str(j) + "_surface_" + str(i) + "_ply_" + str(k)

                    full_name = "./temp/"+str(filename)+".xdmf"

                    fFile = XDMFFile(self.comm, full_name)

                    fFile.write_checkpoint(cauchy_stress_layer[i][k][j], filename, 0)
                    fFile.close()

        self.comm.barrier()

        if MPI.rank(self.comm) == 0:

            print("\t\tsingle read...")

            cauchy_stress_layer_return = cm.initialize_composite_array(num_of_surfs, n_ply, 3)

            #read initial array on single processor
            for i in range(num_of_surfs):

                filename_mesh = "mesh_surface_" + str(i)

                full_mesh_name = "./temp/"+str(filename_mesh)+".xdmf"

                fFile_mesh = XDMFFile(selfcomm, full_mesh_name)

                #### this is the problem
                mesh_temp = Mesh(selfcomm)

                fFile_mesh.read(mesh_temp)

                V_temp = FunctionSpace(mesh_temp,"CG",1)

                fFile_mesh.close()

                for k in range(n_ply[i]):
                    for j in range(3):

                        filename = "Stress_" + str(j) + "_surface_" + str(i) + "_ply_" + str(k)
                        full_name = "./temp/"+str(filename)+".xdmf"

                        fFile = XDMFFile(selfcomm, full_name)

                        temp_funct = Function(V_temp)

                        fFile.read_checkpoint(temp_funct, filename, 0)

                        cauchy_stress_layer_return[i][k][j] = temp_funct

                        fFile.close()

##################################################################################################### stress_connectivity_parallel(stress_connectivity)

            print("\t\t\tconnecting...")
            cauchy_stress_layer_return = self.connect_stress(cauchy_stress_layer_return)

#################################################################################################################

            print("\t\tsingle write...")

            #write modifed array on a single processor
            for i in range(num_of_surfs):
                for k in range(n_ply[i]):
                    for j in range(3):

                        filename = "Stress_" + str(j) + "_surface_" + str(i) + "_ply_" + str(k)

                        full_name = "./temp/"+str(filename)+".xdmf"

                        fFile = XDMFFile(selfcomm, full_name)

                        fFile.write_checkpoint(cauchy_stress_layer_return[i][k][j], filename, 0)
                        fFile.close()

        self.comm.barrier()

        if MPI.rank(self.comm) == 0:
            print("\tparallel read...")

        #read modifed array on all processors
        for i in range(num_of_surfs):

            V_temp = self.splines[i].V_linear

            for k in range(n_ply[i]):
                for j in range(3):

                    filename = "Stress_" + str(j) + "_surface_" + str(i) + "_ply_" + str(k)
                    full_name = "./temp/"+str(filename)+".xdmf"

                    fFile = XDMFFile(worldcomm, full_name)

                    temp_funct = Function(V_temp)

                    fFile.read_checkpoint(temp_funct, filename, 0)

                    cauchy_stress_layer[i][k][j] = temp_funct

                    fFile.close()

        return cauchy_stress_layer


###composite damage part

### using only vector().get_local() because
### here we don't care about nodes order but only the order being the same
### https://fenicsproject.org/pub/tutorial/html/._ftut1019.html

class damage_coef:
    """
    Damage_coef class
    """
    def __init__(self, z0, z1, mas_pT_list, R_, mas_prop_array, mas_max_sigma_array,
    mas_displ_ratio_array, mesh_size_CG1, print_steps=False):
        """
        z0, z1 - arrays that contain top and bottom coordinates
        mas_pT_list - array that contains layer thicknesses
        R_ - rotational arrays
        mas_prop_array - arrays that contain properties for each layer
        mas_max_sigma_array - arrays contain additional properties (stress limits)
        mas_displ_ratio_array - damage parameters
        mesh_size_CG1 - reference array
        
        """
        self.z0 = z0
        self.z1 = z1
        self.mas_pT_list = mas_pT_list
        self.R_ = R_
        self.mas_prop_array = mas_prop_array
        self.mas_max_sigma_array = mas_max_sigma_array
        self.mas_displ_ratio_array = mas_displ_ratio_array
        self.mesh_size_CG1 = mesh_size_CG1

        self.print_steps = print_steps


    #array_initial is used outside of the module
    #initialization of some arrays that hold DMG parameters
    def array_initial(self, num_srfs, n_ply, reference_array):

        damage_coef = cm.initialize_composite_array(num_srfs, n_ply, 3)
        damage_coef_tens = cm.initialize_composite_array(num_srfs, n_ply, 2)
        damage_coef_comp = cm.initialize_composite_array(num_srfs, n_ply, 2)

        NRB_delta_FT_0 = cm.initialize_composite_array(num_srfs, n_ply)
        NRB_delta_FC_0 = cm.initialize_composite_array(num_srfs, n_ply)
        NRB_delta_MT_0 = cm.initialize_composite_array(num_srfs, n_ply)
        NRB_delta_MC_0 = cm.initialize_composite_array(num_srfs, n_ply)

        NRB_delta_FT_f = cm.initialize_composite_array(num_srfs, n_ply)
        NRB_delta_FC_f = cm.initialize_composite_array(num_srfs, n_ply)
        NRB_delta_MT_f = cm.initialize_composite_array(num_srfs, n_ply)
        NRB_delta_MC_f = cm.initialize_composite_array(num_srfs, n_ply)

        for i in range(num_srfs):
            for k in range(n_ply[i]):
                for j in range(3):

                    #damage is not seperated into positive and negative
                    damage_coef[i][k][j] = reference_array[i].copy(deepcopy=True)
                    damage_coef[i][k][j].vector()[:] = np.float64(0)
                    if j != 2:
                        damage_coef_tens[i][k][j] = reference_array[i].copy(deepcopy=True)
                        damage_coef_tens[i][k][j].vector()[:] = np.float64(0)
                        damage_coef_comp[i][k][j] = reference_array[i].copy(deepcopy=True)
                        damage_coef_comp[i][k][j].vector()[:] = np.float64(0)

                NRB_delta_FT_0[i][k] =  reference_array[i].copy(deepcopy=True)
                NRB_delta_FT_0[i][k].vector()[:] = np.float64(0)
                NRB_delta_FC_0[i][k] =  reference_array[i].copy(deepcopy=True)
                NRB_delta_FC_0[i][k].vector()[:] = np.float64(0)
                NRB_delta_MT_0[i][k] =  reference_array[i].copy(deepcopy=True)
                NRB_delta_MT_0[i][k].vector()[:] = np.float64(0)
                NRB_delta_MC_0[i][k] =  reference_array[i].copy(deepcopy=True)
                NRB_delta_MC_0[i][k].vector()[:] = np.float64(0)

                NRB_delta_FT_f[i][k] =  reference_array[i].copy(deepcopy=True)
                NRB_delta_FT_f[i][k].vector()[:] = np.float64(0)
                NRB_delta_FC_f[i][k] =  reference_array[i].copy(deepcopy=True)
                NRB_delta_FC_f[i][k].vector()[:] = np.float64(0)
                NRB_delta_MT_f[i][k] =  reference_array[i].copy(deepcopy=True)
                NRB_delta_MT_f[i][k].vector()[:] = np.float64(0)
                NRB_delta_MC_f[i][k] =  reference_array[i].copy(deepcopy=True)
                NRB_delta_MC_f[i][k].vector()[:] = np.float64(0)

        return damage_coef, damage_coef_tens, damage_coef_comp,\
               NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0,\
               NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f


    #used to calculate effective stress
    def form_effect_stress(self, dmg_coef, eff_stress):
        return_array = [[],[],[]]

        for i in range(3):
            return_array[i] =  eff_stress[i].copy(deepcopy=True)

        dmg_coef_array_0 = (dmg_coef[0].vector().get_local())
        dmg_coef_array_1 = (dmg_coef[1].vector().get_local())
        dmg_coef_array_2 = (dmg_coef[2].vector().get_local())
        stress_array_0 = (eff_stress[0].vector().get_local())
        stress_array_1 = (eff_stress[1].vector().get_local())
        stress_array_2 = (eff_stress[2].vector().get_local())

        new_array_0, new_array_1, new_array_2 = [], [], []

        for i in range(len(stress_array_0)):

            temp_mat_dmg = [[1/(1-dmg_coef_array_0[i]), 0, 0],
                           [0, 1/(1-dmg_coef_array_1[i]), 0],
                           [0, 0, 1/(1-dmg_coef_array_2[i])]]

            temp_stress_array = [stress_array_0[i], stress_array_1[i], stress_array_2[i]]

            eff_stress_result = np.dot(temp_mat_dmg, temp_stress_array)
            new_array_0.append(eff_stress_result[0])
            new_array_1.append(eff_stress_result[1])
            new_array_2.append(eff_stress_result[2])

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1
        return_array[2].vector()[:] = new_array_2

        return return_array

    #used to properly form array with stress limits used in further calculations
    def stress_lim_func(self, eff_stress, F1_ma, F1_mi, F2_ma, F2_mi, F12_ma):

        # from [1000.         1000.         1000.         1000.         1000.
        #         1000.         1000.         1000.         1005.03448218 1343.40229769
        #         1000.         1000.         1027.17755855 1644.65663853 2000.
        #         1000.         1000.         1032.21204073 1744.40636959 2000.
        #         2000.         1000.         1000.         1143.90283557 1806.28882302
        #         2000.         2000.         2000.         1000.         1010.06896435
        #         1303.39682414 1917.97961786 2000.         2000.         2000.
        #         2000.         1484.89655347 1460.1275747  1751.44599944 1989.93103565
        #         2000.         2000.         2000.         2000.         2000.
        #         1989.93103565 2000.         2000.         2000.         2000.
        #         2000.         2000.         2000.         1989.93103565 1994.96551782
        #         2000.         2000.         2000.         2000.         2000.
        #         1710.68672293 1972.82244145 2000.         2000.         2000.
        #         2000.         1511.18726081 1967.78795927 2000.         2000.
        #         2000.         1511.18726081 1967.78795927 2000.         2000.
        #         1387.42235395 1856.09716443 2000.         1164.04076427 1646.79483445
        #         1030.20689306]
        #
        # to [1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000. 1000.
        #      1000. 2000. 2000. 1000. 1000. 1000. 2000. 2000. 2000. 1000. 1000. 1000.
        #      2000. 2000. 2000. 2000. 1000. 1000. 1000. 2000. 2000. 2000. 2000. 2000.
        #      2000. 1000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000.
        #      2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000. 2000.
        #      1000. 2000. 2000. 2000. 2000. 2000. 1000. 2000. 2000. 2000. 2000. 1000.
        #      2000. 2000. 2000. 1000. 2000. 2000. 1000. 2000. 1000.]


        return_array = [[],[],[]]

        for i in range(3):
            return_array[i] =  eff_stress[i].copy(deepcopy=True)

        stress_array_0 = (eff_stress[0].vector().get_local())
        stress_array_1 = (eff_stress[1].vector().get_local())

        new_array_0, new_array_1, new_array_2 = [], [], []

        for i in range(len(stress_array_0)):
            if stress_array_0[i]>= 0:
                new_array_0.append(F1_ma)
            else:
                new_array_0.append(F1_mi)
            if stress_array_1[i]>= 0:
                new_array_1.append(F2_ma)
            else:
                new_array_1.append(F2_mi)

            new_array_2.append(F12_ma)

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1
        return_array[2].vector()[:] = new_array_2

        return return_array


    # used to calculate the current equivalent strains
    def strain_mc_func(self, eff_stress, strain):
        return_array = [[],[],[]]

        for i in range(3):
            return_array[i] =  strain[i].copy(deepcopy=True)

        strain_array_0 = (strain[0].vector().get_local())
        strain_array_1 = (strain[1].vector().get_local())
        strain_array_2 = (strain[2].vector().get_local())
        stress_array_0 = (eff_stress[0].vector().get_local())
        stress_array_1 = (eff_stress[1].vector().get_local())

        new_array_0, new_array_1, new_array_2 = [], [], []

        for i in range(len(stress_array_0)):
            if stress_array_0[i]>= 0:
                new_array_0.append((strain_array_0[i]+abs(strain_array_0[i]))/2)
            else:
                new_array_0.append((-strain_array_0[i]+abs(-strain_array_0[i]))/2)
            if stress_array_1[i]>= 0:
                new_array_1.append((strain_array_1[i]+(strain_array_1[i]))/2)
            else:
                new_array_1.append((-strain_array_1[i]+abs(-strain_array_1[i]))/2)

            new_array_2.append(strain_array_2[i])

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1
        return_array[2].vector()[:] = new_array_2

        return return_array

    # check for the degradation initialization
    def hashin_failure_func(self, eff_stress, Stress_lim):

        return_array = [[],[]]

        for i in range(2):
            return_array[i] =  eff_stress[i].copy(deepcopy=True)

        Stress_lim_array_0 = (Stress_lim[0].vector().get_local())
        Stress_lim_array_1 = (Stress_lim[1].vector().get_local())
        Stress_lim_array_2 = (Stress_lim[2].vector().get_local())

        stress_array_0 = (eff_stress[0].vector().get_local())
        stress_array_1 = (eff_stress[1].vector().get_local())
        stress_array_2 = (eff_stress[2].vector().get_local())

        new_array_0, new_array_1 = [], []

        for i in range(len(stress_array_0)):
            new_array_0.append((stress_array_0[i]/Stress_lim_array_0[i])**2)

            new_array_1.append(((stress_array_1[i]/Stress_lim_array_1[i])**2.0)  +  ((stress_array_2[i]/Stress_lim_array_2[i])**2.0))

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1

        return return_array

    #not in use, as using the native panga function
    def lC_func(self, displacement, strain):

        return_array =  strain[0].copy(deepcopy=True)

        displ_array_0 = (displacement[0].vector().get_local())
        displ_array_1 = (displacement[1].vector().get_local())

        strain_array_0 = (strain[0].vector().get_local())
        strain_array_1 = (strain[1].vector().get_local())

        new_array_0 = []

        for i in range(len(displ_array_0)):
            Lc1 = displ_array_0[i]/strain_array_0[i]
            Lc2 = displ_array_1[i]/strain_array_1[i]
            new_array_0.append(np.sqrt(((Lc1**2)+(Lc2**2))/2))

        return_array.vector()[:] = new_array_0

        return return_array

    # used to calculate the current equivalent displacement
    def delta_crit_func(self, length_c, strain_mc):

        return_array = [[],[]]

        for i in range(2):
            return_array[i] =  strain_mc[i].copy(deepcopy=True)

        length_c_array_0 = (length_c.vector().get_local())

        strain_mc_array_0 = (strain_mc[0].vector().get_local())
        strain_mc_array_1 = (strain_mc[1].vector().get_local())
        strain_mc_array_2 = (strain_mc[2].vector().get_local())

        new_array_0, new_array_1 = [], []

        for i in range(len(length_c_array_0)):
            new_array_0.append(length_c_array_0[i]*strain_mc_array_0[i])
            new_array_1.append(length_c_array_0[i]*np.sqrt(((strain_mc_array_1[i])**2.0)  +  (strain_mc_array_2[i])**2.0))

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1

        return return_array

    # not used
    def sigma_F_M_func(self, Hashin_fail, Delta_F_M_inp, eff_stress, strain_mc, length_c):

        return_array = [[],[]]

        for i in range(2):
            return_array[i] =  Delta_F_M_inp[i].copy(deepcopy=True)

        length_c_array_0 = (length_c.vector().get_local())

        strain_mc_array_1 = (strain_mc[1].vector().get_local())
        strain_mc_array_2 = (strain_mc[2].vector().get_local())

        stress_array_0 = (eff_stress[0].vector().get_local())
        stress_array_1 = (eff_stress[1].vector().get_local())
        stress_array_2 = (eff_stress[2].vector().get_local())

        Delta_F_M_array_0 = (Delta_F_M_inp[0].vector().get_local())
        Delta_F_M_array_1 = (Delta_F_M_inp[1].vector().get_local())

        Hashin_fail_array_0 = (Hashin_fail[0].vector().get_local())
        Hashin_fail_array_1 = (Hashin_fail[1].vector().get_local())

        new_array_0, new_array_1 = [], []

        for i in range(len(length_c_array_0)):
            #fiber
            if Hashin_fail_array_0[i] >= 1 and Delta_F_M_array_0[i] >= 1e-12:
                if stress_array_0[i] >= 0:

                    new_array_0.append(((stress_array_0[i] + np.abs(stress_array_0[i]))/2.0))
                else:
                    new_array_0.append(((-stress_array_0[i] + np.abs(-stress_array_0[i]))/2.0))
            else:
                new_array_0.append(np.float64(0))
            #matrix
            if Hashin_fail_array_1[i] >= 1 and Delta_F_M_array_1[i] >= 1e-12:
                if stress_array_1[i] >= 0:
                    new_array_1.append((((stress_array_1[i] + np.abs(stress_array_1[i]))/2.0)*strain_mc_array_1[i]\
                    +stress_array_2[i]*strain_mc_array_2[i])/(Delta_F_M_array_1[i]/length_c_array_0[i]))
                else:
                    new_array_1.append((((-stress_array_1[i] + np.abs(-stress_array_1[i]))/2.0)*strain_mc_array_1[i]\
                    +stress_array_2[i]*strain_mc_array_2[i])/(Delta_F_M_array_1[i]/length_c_array_0[i]))
            else:
                new_array_1.append(np.float64(0))

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1

        return return_array

    # array used to format the input
    def form_NRB_delta_array_func(self, eff_stress, NRB_delta_FT, NRB_delta_FC, NRB_delta_MT, NRB_delta_MC):

        return_array = [[],[]]

        for i in range(2):
            return_array[i] =  eff_stress[i].copy(deepcopy=True)

        stress_array_0 = (eff_stress[0].vector().get_local())
        stress_array_1 = (eff_stress[1].vector().get_local())

        NRB_delta_FT_array_0 = (NRB_delta_FT.vector().get_local())
        NRB_delta_FC_array_0 = (NRB_delta_FC.vector().get_local())
        NRB_delta_MT_array_0 = (NRB_delta_MT.vector().get_local())
        NRB_delta_MC_array_0 = (NRB_delta_MC.vector().get_local())

        new_array_0, new_array_1 = [], []

        for i in range(len(stress_array_0)):

            if stress_array_0[i] >= 0:
                new_array_0.append(NRB_delta_FT_array_0[i])
            else:
                new_array_0.append(NRB_delta_FC_array_0[i])

            if stress_array_1[i] >= 0:
                new_array_1.append(NRB_delta_MT_array_0[i])
            else:
                new_array_1.append(NRB_delta_MC_array_0[i])

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1

        return return_array

    # array used to update the displacement array
    def update_NRB_delta_array_func(self, eff_stress, NRB_delta_0_new, NRB_delta_FT, NRB_delta_FC, NRB_delta_MT, NRB_delta_MC):

        return_array_0 = NRB_delta_FT.copy(deepcopy=True)
        return_array_1 = NRB_delta_FC.copy(deepcopy=True)
        return_array_2 = NRB_delta_MT.copy(deepcopy=True)
        return_array_3 = NRB_delta_MC.copy(deepcopy=True)

        stress_array_0 = (eff_stress[0].vector().get_local())
        stress_array_1 = (eff_stress[1].vector().get_local())

        NRB_delta_0_new_array_0 = (NRB_delta_0_new[0].vector().get_local())
        NRB_delta_0_new_array_1 = (NRB_delta_0_new[1].vector().get_local())

        NRB_delta_FT_array_0 = (NRB_delta_FT.vector().get_local())
        NRB_delta_FC_array_0 = (NRB_delta_FC.vector().get_local())
        NRB_delta_MT_array_0 = (NRB_delta_MT.vector().get_local())
        NRB_delta_MC_array_0 = (NRB_delta_MC.vector().get_local())

        new_array_0, new_array_1, new_array_2, new_array_3 = [], [], [], []

        for i in range(len(stress_array_0)):

            if stress_array_0[i] >= 0:
                new_array_0.append(NRB_delta_0_new_array_0[i])
            else:
                new_array_0.append(NRB_delta_FT_array_0[i])

            if stress_array_0[i] < 0:
                new_array_1.append(NRB_delta_0_new_array_0[i])
            else:
                new_array_1.append(NRB_delta_FC_array_0[i])

            if stress_array_1[i] >= 0:
                new_array_2.append(NRB_delta_0_new_array_1[i])
            else:
                new_array_2.append(NRB_delta_MT_array_0[i])

            if stress_array_1[i] < 0:
                new_array_3.append(NRB_delta_0_new_array_1[i])
            else:
                new_array_3.append(NRB_delta_MC_array_0[i])

        return_array_0.vector()[:] = new_array_0
        return_array_1.vector()[:] = new_array_1
        return_array_2.vector()[:] = new_array_2
        return_array_3.vector()[:] = new_array_3

        return return_array_0, return_array_1, return_array_2, return_array_3

    # new displacement at the start of degradation
    def new_NRB_delta_F_M_0_func(self, Hashin_fail, Delta_F_M_inp, delta_f_m_0_old):

        return_array = [[],[]]

        for i in range(2):
            return_array[i] =  delta_f_m_0_old[i].copy(deepcopy=True)

        Hashin_fail_array_0 = (Hashin_fail[0].vector().get_local())
        Hashin_fail_array_1 = (Hashin_fail[1].vector().get_local())

        Delta_F_M_inp_array_0 = (Delta_F_M_inp[0].vector().get_local())
        Delta_F_M_inp_array_1 = (Delta_F_M_inp[1].vector().get_local())

        delta_f_m_0_old_array_0 = (delta_f_m_0_old[0].vector().get_local())
        delta_f_m_0_old_array_1 = (delta_f_m_0_old[1].vector().get_local())

        new_array_0, new_array_1 = [], []

        for i in range(len(delta_f_m_0_old_array_0)):
            if Hashin_fail_array_0[i] >= 1 and Delta_F_M_inp_array_0[i] >= 1e-12:
                if delta_f_m_0_old_array_0[i] < 1e-12:
                    new_array_0.append(Delta_F_M_inp_array_0[i]/np.sqrt(Hashin_fail_array_0[i]))
                else:
                    new_array_0.append(delta_f_m_0_old_array_0[i])
            else:
                new_array_0.append(np.float64(0))

            if Hashin_fail_array_1[i] >= 1 and Delta_F_M_inp_array_1[i] >= 1e-12:
                if delta_f_m_0_old_array_1[i] < 1e-12:
                    new_array_1.append(Delta_F_M_inp_array_1[i]/np.sqrt(Hashin_fail_array_1[i]))
                else:
                    new_array_1.append(delta_f_m_0_old_array_1[i])
            else:
                new_array_1.append(np.float64(0))

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1

        return return_array

    # new current equivalent displacement                                                                          ### this mistake would not be found without print() output
    def new_NRB_delta_F_M_F_func(self, eff_stress, Hashin_fail, Delta_F_M_inp, delta_F_M_F_old, delta_f_m_0_old, delta_f_m_0_new, ratio_0F_ft, ratio_0F_fc, ratio_0F_mt, ratio_0F_mc):

        return_array = [[],[]]

        for i in range(2):
            return_array[i] =  delta_f_m_0_old[i].copy(deepcopy=True)

        stress_array_0 = (eff_stress[0].vector().get_local())
        stress_array_1 = (eff_stress[1].vector().get_local())

        Hashin_fail_array_0 = (Hashin_fail[0].vector().get_local())
        Hashin_fail_array_1 = (Hashin_fail[1].vector().get_local())

        Delta_F_M_inp_array_0 = (Delta_F_M_inp[0].vector().get_local())
        Delta_F_M_inp_array_1 = (Delta_F_M_inp[1].vector().get_local())

        delta_F_M_F_old_array_0 = (delta_F_M_F_old[0].vector().get_local())
        delta_F_M_F_old_array_1 = (delta_F_M_F_old[1].vector().get_local())

        delta_f_m_0_old_array_0 = (delta_f_m_0_old[0].vector().get_local())
        delta_f_m_0_old_array_1 = (delta_f_m_0_old[1].vector().get_local())

        delta_f_m_0_new_array_0 = (delta_f_m_0_new[0].vector().get_local())
        delta_f_m_0_new_array_1 = (delta_f_m_0_new[1].vector().get_local())


        new_array_0, new_array_1 = [], []

        for i in range(len(delta_F_M_F_old_array_0)):
            if Hashin_fail_array_0[i] >= 1 and Delta_F_M_inp_array_0[i] >= 1e-12:
                if delta_f_m_0_old_array_0[i] < 1e-12:
                    if stress_array_0[i] >= 0:
                        new_array_0.append(ratio_0F_ft*delta_f_m_0_new_array_0[i])
                    else:
                        new_array_0.append(ratio_0F_fc*delta_f_m_0_new_array_0[i])
                else:
                    new_array_0.append(delta_F_M_F_old_array_0[i])
            else:
                new_array_0.append(np.float64(0))

            if Hashin_fail_array_1[i] >= 1 and Delta_F_M_inp_array_1[i] >= 1e-12:
                if delta_f_m_0_old_array_1[i] < 1e-12:
                    if stress_array_1[i] >= 0:
                        new_array_1.append(ratio_0F_mt*delta_f_m_0_new_array_1[i])
                    else:
                        new_array_1.append(ratio_0F_mc*delta_f_m_0_new_array_1[i])
                else:
                    new_array_1.append(delta_F_M_F_old_array_1[i])
            else:
                new_array_1.append(np.float64(0))

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1

        return return_array

    # calculating new damage 1 and damage 2 coef.
    def form_damage_coef_12_func(self, Hashin_fail, Delta_F_M_inp, damage_coef_p, delta_F_M_F_new, delta_f_m_0_new):

        return_array = [[],[]]

        for i in range(2):
            return_array[i] =  damage_coef_p[i].copy(deepcopy=True)

        Hashin_fail_array_0 = (Hashin_fail[0].vector().get_local())
        Hashin_fail_array_1 = (Hashin_fail[1].vector().get_local())

        Delta_F_M_inp_array_0 = (Delta_F_M_inp[0].vector().get_local())
        Delta_F_M_inp_array_1 = (Delta_F_M_inp[1].vector().get_local())

        damage_coef_p_array_0 = (damage_coef_p[0].vector().get_local())
        damage_coef_p_array_1 = (damage_coef_p[1].vector().get_local())

        delta_F_M_F_new_array_0 = (delta_F_M_F_new[0].vector().get_local())
        delta_F_M_F_new_array_1 = (delta_F_M_F_new[1].vector().get_local())

        delta_f_m_0_new_array_0 = (delta_f_m_0_new[0].vector().get_local())
        delta_f_m_0_new_array_1 = (delta_f_m_0_new[1].vector().get_local())

        new_array_0, new_array_1 = [], []

        for i in range(len(damage_coef_p_array_0)):

            if Hashin_fail_array_0[i] >= 1 and Delta_F_M_inp_array_0[i] >= 1e-12:

                if delta_F_M_F_new_array_0[i] != delta_f_m_0_new_array_0[i]:
                    new_array_0.append(max(damage_coef_p_array_0[i], (delta_F_M_F_new_array_0[i]/Delta_F_M_inp_array_0[i])* \
                     (Delta_F_M_inp_array_0[i]-delta_f_m_0_new_array_0[i])/(delta_F_M_F_new_array_0[i]-delta_f_m_0_new_array_0[i])))
                else:
                    new_array_0.append(damage_coef_p_array_0[i])
            else:
                new_array_0.append(damage_coef_p_array_0[i])

            if Hashin_fail_array_1[i] >= 1 and Delta_F_M_inp_array_1[i] >= 1e-12:
                if delta_F_M_F_new_array_1[i] != delta_f_m_0_new_array_1[i]:
                    new_array_1.append(max(damage_coef_p_array_1[i], (delta_F_M_F_new_array_1[i]/Delta_F_M_inp_array_1[i])* \
                     (Delta_F_M_inp_array_1[i]-delta_f_m_0_new_array_1[i])/(delta_F_M_F_new_array_1[i]-delta_f_m_0_new_array_1[i])))
                else:
                    new_array_1.append(damage_coef_p_array_1[i])
            else:
                new_array_1.append(damage_coef_p_array_1[i])

        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1

        return return_array

    # form a damage array
    def update_damage_coef_TC_func(self, eff_stress, damage_coef_new, damage_coef_tens, damage_coef_comp):

        return_array_0 = [[],[]]
        return_array_1 = [[],[]]

        for i in range(2):
            return_array_0[i] =  damage_coef_tens[i].copy(deepcopy=True)

        for i in range(2):
            return_array_1[i] =  damage_coef_comp[i].copy(deepcopy=True)

        stress_array_0 = (eff_stress[0].vector().get_local())
        stress_array_1 = (eff_stress[1].vector().get_local())

        damage_coef_new_array_0 = (damage_coef_new[0].vector().get_local())
        damage_coef_new_array_1 = (damage_coef_new[1].vector().get_local())

        damage_coef_tens_array_0 = (damage_coef_tens[0].vector().get_local())
        damage_coef_tens_array_1 = (damage_coef_tens[1].vector().get_local())
        damage_coef_comp_array_0 = (damage_coef_comp[0].vector().get_local())
        damage_coef_comp_array_1 = (damage_coef_comp[1].vector().get_local())

        new_array_0, new_array_1, new_array_2, new_array_3 = [], [], [], []

        for i in range(len(stress_array_0)):

            if stress_array_0[i] >= 0:
                new_array_0.append(damage_coef_new_array_0[i])
            else:
                new_array_0.append(damage_coef_tens_array_0[i])

            if stress_array_1[i] >= 0:
                new_array_1.append(damage_coef_new_array_1[i])
            else:
                new_array_1.append(damage_coef_tens_array_1[i])

            if stress_array_0[i] < 0:
                new_array_2.append(damage_coef_new_array_0[i])
            else:
                new_array_2.append(damage_coef_comp_array_0[i])

            if stress_array_1[i] < 0:
                new_array_3.append(damage_coef_new_array_1[i])
            else:
                new_array_3.append(damage_coef_comp_array_1[i])

        # stress >= 0
        return_array_0[0].vector()[:] = new_array_0
        return_array_0[1].vector()[:] = new_array_1
        # stress < 0
        return_array_1[0].vector()[:] = new_array_2
        return_array_1[1].vector()[:] = new_array_3


        return return_array_0, return_array_1

    # using the damage array form new d_1, d_2, d_3
    def damage_coef_full_func(self, damage_coef, damage_coef_tens, damage_coef_comp):
        return_array = [[],[],[]]

        for i in range(3):
            if i != 2:
                return_array[i] =  damage_coef[i].copy(deepcopy=True)
            else:
                return_array[2] =  damage_coef[1].copy(deepcopy=True)

        damage_coef_array_0 = (damage_coef[0].vector().get_local())
        damage_coef_array_1 = (damage_coef[1].vector().get_local())

        damage_coef_tens_array_0 = (damage_coef_tens[0].vector().get_local())
        damage_coef_tens_array_1 = (damage_coef_tens[1].vector().get_local())

        damage_coef_comp_array_0 = (damage_coef_comp[0].vector().get_local())
        damage_coef_comp_array_1 = (damage_coef_comp[1].vector().get_local())

        new_array_0, new_array_1, new_array_2 = [], [], []

        for i in range(len(damage_coef_array_0)):

            if damage_coef_array_0[i] >= 1:
                damage_coef_array_0[i] = 0.99999
            if damage_coef_array_1[i] >= 1:
                damage_coef_array_1[i] = 0.99999


            damage_coef_tens_array_0_temp = damage_coef_tens_array_0[i]
            damage_coef_comp_array_0_temp = damage_coef_comp_array_0[i]
            damage_coef_tens_array_1_temp = damage_coef_tens_array_1[i]
            damage_coef_comp_array_1_temp = damage_coef_comp_array_1[i]

            if damage_coef_tens_array_0[i] >= 1:
                damage_coef_tens_array_0_temp = 0.99999
            if damage_coef_comp_array_0[i] >= 1:
                damage_coef_comp_array_0_temp = 0.99999
            if damage_coef_tens_array_1[i] >= 1:
                damage_coef_tens_array_1_temp = 0.99999
            if damage_coef_comp_array_1[i] >= 1:
                damage_coef_comp_array_1_temp = 0.99999

            damage_coef_array_2 = 1.0 - (1.0 - damage_coef_tens_array_0_temp)*(1.0 - damage_coef_comp_array_0_temp) * \
                                           (1.0 - damage_coef_tens_array_1_temp)*(1.0 - damage_coef_comp_array_1_temp)

            if damage_coef_array_2 >= 1:
                damage_coef_array_2 = 0.99999
                
            new_array_0.append(damage_coef_array_0[i])
            new_array_1.append(damage_coef_array_1[i])
            new_array_2.append(damage_coef_array_2)

        # stress >= 0
        return_array[0].vector()[:] = new_array_0
        return_array[1].vector()[:] = new_array_1
        return_array[2].vector()[:] = new_array_2

        return return_array
    
    
    #function used to form a compliance matrix on node-level
    def form_comp_dolph(self, dmg_coef, D_comp_temp):
        return_array = [[[],[],[]],
                        [[],[],[]],
                        [[],[],[]]]

        for i in range(3):
            for j in range(3):
                return_array[i][j] =  dmg_coef[i].copy(deepcopy=True)

        dmg_coef_array_0 = (dmg_coef[0].vector().get_local())
        dmg_coef_array_1 = (dmg_coef[1].vector().get_local())
        dmg_coef_array_2 = (dmg_coef[2].vector().get_local())

        #complience given as ufl Constant
        D_comp_0_0 = (float(D_comp_temp[0,0]))
        D_comp_0_1 = (float(D_comp_temp[0,1]))
        D_comp_0_2 = (float(D_comp_temp[0,2]))
        D_comp_1_0 = (float(D_comp_temp[1,0]))
        D_comp_1_1 = (float(D_comp_temp[1,1]))
        D_comp_1_2 = (float(D_comp_temp[1,2]))
        D_comp_2_0 = (float(D_comp_temp[2,0]))
        D_comp_2_1 = (float(D_comp_temp[2,1]))
        D_comp_2_2 = (float(D_comp_temp[2,2]))

        D_comp_np_array = np.array([[D_comp_0_0,D_comp_0_1,D_comp_0_2],
                                    [D_comp_1_0,D_comp_1_1,D_comp_1_2],
                                    [D_comp_2_0,D_comp_2_1,D_comp_2_2]])

        new_array_0, new_array_1, new_array_2 = [], [], []
        new_array_3, new_array_4, new_array_5 = [], [], []
        new_array_6, new_array_7, new_array_8 = [], [], []

        for i in range(len(dmg_coef_array_0)):

            temp_compliance_array = np.array([[D_comp_np_array[0][0]/(1-dmg_coef_array_0[i]), D_comp_np_array[0][1], D_comp_np_array[0][2]],
                                              [D_comp_np_array[1][0], D_comp_np_array[1][1]/(1-dmg_coef_array_1[i]), D_comp_np_array[1][2]],
                                              [D_comp_np_array[2][0], D_comp_np_array[2][1], D_comp_np_array[2][2]/(1-dmg_coef_array_2[i])]])

            new_array_0.append(temp_compliance_array[0][0])
            new_array_1.append(temp_compliance_array[0][1])
            new_array_2.append(temp_compliance_array[0][2])
            new_array_3.append(temp_compliance_array[1][0])
            new_array_4.append(temp_compliance_array[1][1])
            new_array_5.append(temp_compliance_array[1][2])
            new_array_6.append(temp_compliance_array[2][0])
            new_array_7.append(temp_compliance_array[2][1])
            new_array_8.append(temp_compliance_array[2][2])


        return_array[0][0].vector()[:] = new_array_0
        return_array[0][1].vector()[:] = new_array_1
        return_array[0][2].vector()[:] = new_array_2
        return_array[1][0].vector()[:] = new_array_3
        return_array[1][1].vector()[:] = new_array_4
        return_array[1][2].vector()[:] = new_array_5
        return_array[2][0].vector()[:] = new_array_6
        return_array[2][1].vector()[:] = new_array_7
        return_array[2][2].vector()[:] = new_array_8

        return as_matrix(return_array)

    #function used to form a stiffness matrix on node-level
    def Stiffness_matrix_lamina_dolph(self, pC, ref_array = None):

        return_array = [[[],[],[]],
                        [[],[],[]],
                        [[],[],[]]]

        if ref_array == None:
            for i in range(3):
                for j in range(3):
                    return_array[i][j] =  pC[i,j].copy(deepcopy=True)
        else:
            for i in range(3):
                for j in range(3):
                    return_array[i][j] =  ref_array.copy(deepcopy=True)

        ref_array_temp = return_array[0][0].vector().get_local()

        pC_array_0_0, pC_array_0_1, pC_array_0_2,\
        pC_array_1_0, pC_array_1_1, pC_array_1_2,\
        pC_array_2_0, pC_array_2_1, pC_array_2_2 = utility_function_check_prop_output(pC, ref_array_temp)

        new_array_0, new_array_1, new_array_2 = [], [], []
        new_array_3, new_array_4, new_array_5 = [], [], []
        new_array_6, new_array_7, new_array_8 = [], [], []

        for i in range(len(pC_array_0_0)):

            temp_compliance_array = np.array([[pC_array_0_0[i], pC_array_0_1[i], pC_array_0_2[i]],
                                              [pC_array_1_0[i], pC_array_1_1[i], pC_array_1_2[i]],
                                              [pC_array_2_0[i], pC_array_2_1[i], pC_array_2_2[i]]])

            temp_stiff_array = np.linalg.inv(temp_compliance_array)

            ###change to single array with for for cycle
            new_array_0.append(temp_stiff_array[0][0])
            new_array_1.append(temp_stiff_array[0][1])
            new_array_2.append(temp_stiff_array[0][2])
            new_array_3.append(temp_stiff_array[1][0])
            new_array_4.append(temp_stiff_array[1][1])
            new_array_5.append(temp_stiff_array[1][2])
            new_array_6.append(temp_stiff_array[2][0])
            new_array_7.append(temp_stiff_array[2][1])
            new_array_8.append(temp_stiff_array[2][2])


        return_array[0][0].vector()[:] = new_array_0
        return_array[0][1].vector()[:] = new_array_1
        return_array[0][2].vector()[:] = new_array_2
        return_array[1][0].vector()[:] = new_array_3
        return_array[1][1].vector()[:] = new_array_4
        return_array[1][2].vector()[:] = new_array_5
        return_array[2][0].vector()[:] = new_array_6
        return_array[2][1].vector()[:] = new_array_7
        return_array[2][2].vector()[:] = new_array_8

        return as_matrix(return_array)

    def Stiffness_matrix_lamina_array_dolph(self, pC, ref_array = None):
        pS = [[] for i in range(len(pC))]

        for i in range(len(pC)):
            pS[i] = self.Stiffness_matrix_lamina_dolph(pC[i], ref_array)

        return pS

    def Stiffness_matrix_laminate_dolph(self, pR, pS, ref_array = None):

        return_array = []

        for k in range(len(pR)):

            return_array_temp = [[[],[],[]],
                                 [[],[],[]],
                                 [[],[],[]]]

            if ref_array == None:
                for i in range(3):
                    for j in range(3):
                        return_array_temp[i][j] =  pS[0][i,j].copy(deepcopy=True)
            else:
                for i in range(3):
                    for j in range(3):
                        return_array_temp[i][j] =  ref_array.copy(deepcopy=True)

            ref_array_temp = return_array_temp[0][0].vector().get_local()

            #in case pS is already a function ref_arrray wont be useds

            pS_array_0_0, pS_array_0_1, pS_array_0_2,\
            pS_array_1_0, pS_array_1_1, pS_array_1_2,\
            pS_array_2_0, pS_array_2_1, pS_array_2_2 = utility_function_check_prop_output(pS[k], ref_array_temp)

            #in case pR is non-constant

            pR_temp_0_0, pR_temp_0_1, pR_temp_0_2,\
            pR_temp_1_0, pR_temp_1_1, pR_temp_1_2,\
            pR_temp_2_0, pR_temp_2_1, pR_temp_2_2 = utility_function_check_prop_output(pR[k], ref_array_temp)

            new_array_0, new_array_1, new_array_2 = [], [], []
            new_array_3, new_array_4, new_array_5 = [], [], []
            new_array_6, new_array_7, new_array_8 = [], [], []

            for i in range(len(pS_array_0_0)):

                temp_stiff_array = np.array([[pS_array_0_0[i], pS_array_0_1[i], pS_array_0_2[i]],
                                             [pS_array_1_0[i], pS_array_1_1[i], pS_array_1_2[i]],
                                             [pS_array_2_0[i], pS_array_2_1[i], pS_array_2_2[i]]])

                pR_k = np.array([[pR_temp_0_0[i],pR_temp_0_1[i],pR_temp_0_2[i]],
                                 [pR_temp_1_0[i],pR_temp_1_1[i],pR_temp_1_2[i]],
                                 [pR_temp_2_0[i],pR_temp_2_1[i],pR_temp_2_2[i]]])

                temp_stiff_array = np.dot(np.transpose(pR_k), np.dot(temp_stiff_array, pR_k))

                ###change to single array with for for cycle
                new_array_0.append(temp_stiff_array[0][0])
                new_array_1.append(temp_stiff_array[0][1])
                new_array_2.append(temp_stiff_array[0][2])
                new_array_3.append(temp_stiff_array[1][0])
                new_array_4.append(temp_stiff_array[1][1])
                new_array_5.append(temp_stiff_array[1][2])
                new_array_6.append(temp_stiff_array[2][0])
                new_array_7.append(temp_stiff_array[2][1])
                new_array_8.append(temp_stiff_array[2][2])


            return_array_temp[0][0].vector()[:] = new_array_0
            return_array_temp[0][1].vector()[:] = new_array_1
            return_array_temp[0][2].vector()[:] = new_array_2
            return_array_temp[1][0].vector()[:] = new_array_3
            return_array_temp[1][1].vector()[:] = new_array_4
            return_array_temp[1][2].vector()[:] = new_array_5
            return_array_temp[2][0].vector()[:] = new_array_6
            return_array_temp[2][1].vector()[:] = new_array_7
            return_array_temp[2][2].vector()[:] = new_array_8

            return_array.append(as_matrix(return_array_temp))

        return return_array

    # , k_param = 0, i_param = 1 are ply and surf for printing
    def damaged_outer_loop(self, cauchy_stress_layer, strain_array, D_stif_new, A_mat_new, B_mat_new, D_mat_new, damage_coef, damage_coef_tens, damage_coef_comp, NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0, NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f, k_param = 0, i_param = 1):

        if self.print_steps == True:
            print("computing damage")

        num_srfs = len(cauchy_stress_layer)
        n_ply = [len(cauchy_stress_layer[i]) for i in range(num_srfs)]

        #intialize as function of in a function space for the first cycle
        if self.print_steps == True:
            print("intializing parameters")

        for i in range(num_srfs):
            # fresh complience matrix
            D_comp_temp = cm.Compliance_matrix_lamina_array(self.mas_prop_array[i])
            for k in range(n_ply[i]):
                if self.print_steps == True and k == k_param and i == i_param:
                    print("surface", i+1, "layer", k +1)

                F1_max, F1_min, F2_max, F2_min, F12_max = self.mas_max_sigma_array[i][k]

                ratio_0F_ft, ratio_0F_fc, ratio_0F_mt, ratio_0F_mc = self.mas_displ_ratio_array[i][k]
                if self.print_steps == True and k == k_param and i == i_param:
                    print("cauchy_stress_layer 2")
                    print(np.array(cauchy_stress_layer[i][k][1].vector().get_local()))
                    print()

                    print("cauchy_stress_layer 12")
                    print(np.array(cauchy_stress_layer[i][k][2].vector().get_local()))
                    print()

                #computes with old coef, from the privious cycle for this point, on this surf and layer
                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing cauchy_eff_stress_layer 2")

                cauchy_eff_stress_layer = self.form_effect_stress(damage_coef[i][k], cauchy_stress_layer[i][k])
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(cauchy_eff_stress_layer[1].vector().get_local()))
                    print()
                    print("computing cauchy_eff_stress_layer 12")
                    print(np.array(cauchy_eff_stress_layer[2].vector().get_local()))
                    print()
                    print("computing strain_array")
                    print(np.array(strain_array[i][k][1].vector().get_local()))
                    print()


                #define stress limit function as a field, can define all like that
                if self.print_steps == True and k == k_param and i == i_param:
                    print("defining Stress_limit 2")

                Stress_limit = self.stress_lim_func(cauchy_eff_stress_layer, F1_max, F1_min, F2_max, F2_min, F12_max)
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(Stress_limit[1].vector().get_local()))
                    print()
                    print("defining Stress_limit 12")
                    print(np.array(Stress_limit[2].vector().get_local()))
                    print()

                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing strain_MC 1")

                strain_MC = self.strain_mc_func(cauchy_eff_stress_layer, strain_array[i][k])
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(strain_MC[0].vector().get_local()))
                    print()
                    print("computing strain_MC 2")
                    print(np.array(strain_MC[1].vector().get_local()))
                    print()
                    print("computing strain_MC 12")
                    print(np.array(strain_MC[2].vector().get_local()))
                    print()

                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing Hashin_f 1")

                Hashin_f = self.hashin_failure_func(cauchy_eff_stress_layer, Stress_limit)
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(Hashin_f[0].vector().get_local()))
                    print()
                    print("computing Hashin_f 2")
                    print(np.array(Hashin_f[1].vector().get_local()))
                    print()

                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing Lc")

                Lc = self.mesh_size_CG1[i]
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(Lc.vector().get_local()))
                    print()

                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing Delta_F_M")

                Delta_F_M = self.delta_crit_func(Lc, strain_MC)
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(Delta_F_M[1].vector().get_local()))
                    print()

                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing Sigma_F_M")

                Sigma_F_M = self.sigma_F_M_func(Hashin_f, Delta_F_M, cauchy_eff_stress_layer, strain_MC, Lc)
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(Sigma_F_M[1].vector().get_local()))
                    print()
                    
                #form a propper array
                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing NRB_delta_F_M_0_old")

                NRB_delta_F_M_0 = self.form_NRB_delta_array_func(cauchy_eff_stress_layer, NRB_delta_FT_0[i][k], NRB_delta_FC_0[i][k], \
                                                                             NRB_delta_MT_0[i][k], NRB_delta_MC_0[i][k])
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(NRB_delta_F_M_0[1].vector().get_local()))
                    print()

                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing NRB_delta_F_M_0_new")

                NRB_delta_F_M_0_new = self.new_NRB_delta_F_M_0_func(Hashin_f, Delta_F_M, NRB_delta_F_M_0)
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(NRB_delta_F_M_0_new[1].vector().get_local()))
                    print()

                #push new NRB_delta_F_M_0 to propper arrays based on eff_stress
                NRB_delta_FT_0[i][k], NRB_delta_FC_0[i][k], NRB_delta_MT_0[i][k],  NRB_delta_MC_0[i][k] = self.update_NRB_delta_array_func(cauchy_eff_stress_layer, \
                NRB_delta_F_M_0_new, NRB_delta_FT_0[i][k], NRB_delta_FC_0[i][k], NRB_delta_MT_0[i][k], NRB_delta_MC_0[i][k])

                #form a propper array
                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing NRB_delta_F_M_F_old")

                NRB_delta_F_M_F = self.form_NRB_delta_array_func(cauchy_eff_stress_layer, NRB_delta_FT_f[i][k], NRB_delta_FC_f[i][k],
                                                                              NRB_delta_MT_f[i][k], NRB_delta_MC_f[i][k])
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(NRB_delta_F_M_F[1].vector().get_local()))
                    print()

                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing NRB_delta_F_M_F_new")

                NRB_delta_F_M_F_new = self.new_NRB_delta_F_M_F_func(cauchy_eff_stress_layer, Hashin_f, Delta_F_M, NRB_delta_F_M_F, NRB_delta_F_M_0, NRB_delta_F_M_0_new,\
                ratio_0F_ft, ratio_0F_fc, ratio_0F_mt, ratio_0F_mc)
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(NRB_delta_F_M_F_new[1].vector().get_local()))
                    print()

                #push new NRB_delta to propper arrays based on eff_stress
                NRB_delta_FT_f[i][k], NRB_delta_FC_f[i][k], NRB_delta_MT_f[i][k], NRB_delta_MC_f[i][k] = self.update_NRB_delta_array_func(cauchy_eff_stress_layer, \
                NRB_delta_F_M_F_new, NRB_delta_FT_f[i][k], NRB_delta_FC_f[i][k], NRB_delta_MT_f[i][k], NRB_delta_MC_f[i][k])

                # form new damage 1 2
                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing damage_coef_new_1")
                                                                                                     #here was an error NRB_delta_F_M_F_new
                damage_coef_new_1_2 = self.form_damage_coef_12_func(Hashin_f, Delta_F_M, damage_coef[i][k], NRB_delta_F_M_F_new, NRB_delta_F_M_0_new)
                if self.print_steps == True and k == k_param and i == i_param:
                    print(np.array(damage_coef_new_1_2[0].vector().get_local()))
                    print()

                    print("computing damage_coef_new_2")
                    print(np.array(damage_coef_new_1_2[1].vector().get_local()))
                    print()

                if self.print_steps == True and k == k_param and i == i_param:
                    print("computing new damage_coef")

                damage_coef_tens[i][k], damage_coef_comp[i][k] = self.update_damage_coef_TC_func(cauchy_eff_stress_layer, \
                damage_coef_new_1_2, damage_coef_tens[i][k], damage_coef_comp[i][k])

                damage_coef[i][k] = self.damage_coef_full_func(damage_coef_new_1_2, damage_coef_tens[i][k], damage_coef_comp[i][k])

                #Computation of compliance for this layer

                D_comp_temp[k] = self.form_comp_dolph(damage_coef[i][k], D_comp_temp[k])

            D_stif_temp = self.Stiffness_matrix_lamina_array_dolph(D_comp_temp)
            Q_stif_temp = self.Stiffness_matrix_laminate_dolph(self.R_[i], D_stif_temp)
            #ABD matries for this surface
            D_stif_new[i] = D_stif_temp

            A_mat_new[i] = cm.complex_laminate_A_mat(Q_stif_temp, self.mas_pT_list[i])
            B_mat_new[i] = cm.complex_laminate_B_mat(Q_stif_temp, self.z0[i], self.z1[i])
            D_mat_new[i] = cm.complex_laminate_D_mat(Q_stif_temp, self.z0[i], self.z1[i])

        return D_stif_new, A_mat_new, B_mat_new, D_mat_new, \
               damage_coef, damage_coef_tens, damage_coef_comp, \
               NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0, \
               NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f

    def array_initial_NEWTON(self, num_srfs, n_ply, damage_coef, damage_coef_tens, damage_coef_comp):

        damage_coef_NEWTON = cm.initialize_composite_array(num_srfs, n_ply, 3)
        damage_coef_tens_NEWTON = cm.initialize_composite_array(num_srfs, n_ply, 2)
        damage_coef_comp_NEWTON = cm.initialize_composite_array(num_srfs, n_ply, 2)

        D_stif_new_NEWTON = cm.initialize_composite_array(num_srfs)
        A_mat_new_NEWTON = cm.initialize_composite_array(num_srfs)
        B_mat_new_NEWTON = cm.initialize_composite_array(num_srfs)
        D_mat_new_NEWTON = cm.initialize_composite_array(num_srfs)

        for i in range(num_srfs):
            D_comp_temp = cm.Compliance_matrix_lamina_array(self.mas_prop_array[i])
            for k in range(n_ply[i]):
                for j in range(3):
                    
                    #damage is not seperated into positive and negative
                    damage_coef_NEWTON[i][k][j] = damage_coef[i][k][j].copy(deepcopy=True)
                    if j != 2:
                        damage_coef_tens_NEWTON[i][k][j] = damage_coef_tens[i][k][j].copy(deepcopy=True)
                        damage_coef_comp_NEWTON[i][k][j] = damage_coef_comp[i][k][j].copy(deepcopy=True)

                D_comp_temp[k] = self.form_comp_dolph(damage_coef_NEWTON[i][k], D_comp_temp[k])

            D_stif_temp = self.Stiffness_matrix_lamina_array_dolph(D_comp_temp)
            Q_stif_temp = self.Stiffness_matrix_laminate_dolph(self.R_[i], D_stif_temp)
            #ABD matries for this surface
            D_stif_new_NEWTON[i] = D_stif_temp

            A_mat_new_NEWTON[i] = cm.complex_laminate_A_mat(Q_stif_temp, self.mas_pT_list[i])
            B_mat_new_NEWTON[i] = cm.complex_laminate_B_mat(Q_stif_temp, self.z0[i], self.z1[i])
            D_mat_new_NEWTON[i] = cm.complex_laminate_D_mat(Q_stif_temp, self.z0[i], self.z1[i])

        return damage_coef_NEWTON, damage_coef_tens_NEWTON, damage_coef_comp_NEWTON, D_stif_new_NEWTON, A_mat_new_NEWTON, B_mat_new_NEWTON, D_mat_new_NEWTON

    def damaged_inner_loop(self, cauchy_stress_layer, strain_array, D_stif_new_NEWTON, A_mat_new_NEWTON, B_mat_new_NEWTON, D_mat_new_NEWTON, damage_coef_NEWTON, damage_coef_tens_NEWTON, damage_coef_comp_NEWTON, NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0, NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f, print_inner_steps = False):

            if print_inner_steps == True:
                print("computing damage NEWTON")

            num_srfs = len(cauchy_stress_layer)
            n_ply = [len(cauchy_stress_layer[i]) for i in range(num_srfs)]
            for i in range(num_srfs):

                # fresh complience matrix
                D_comp_temp = cm.Compliance_matrix_lamina_array(self.mas_prop_array[i])
                for k in range(n_ply[i]):
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("NEWTON surface", i+1, "layer", k +1)

                    F1_max, F1_min, F2_max, F2_min, F12_max = self.mas_max_sigma_array[i][k]

                    ratio_0F_ft, ratio_0F_fc, ratio_0F_mt, ratio_0F_mc = self.mas_displ_ratio_array[i][k]
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("cauchy_stress_layer NEWTON")
                        print(np.array(cauchy_stress_layer[i][k][1].vector().get_local()))
                        print()

                    #computes with old coef, from the privious cycle for this point, on this surf and layer
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing cauchy_eff_stress_layer NEWTON")

                    cauchy_eff_stress_layer = self.form_effect_stress(damage_coef_NEWTON[i][k], cauchy_stress_layer[i][k])
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(cauchy_eff_stress_layer[1].vector().get_local()))
                        print()


                    #define stress limit function as a field
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("defining Stress_limit NEWTON")

                    Stress_limit = self.stress_lim_func(cauchy_eff_stress_layer, F1_max, F1_min, F2_max, F2_min, F12_max)
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(Stress_limit[1].vector().get_local()))
                        print()

                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing strain_MC NEWTON")

                    strain_MC = self.strain_mc_func(cauchy_eff_stress_layer, strain_array[i][k])
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(strain_MC[1].vector().get_local()))
                        print()

                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing Hashin_f NEWTON")

                    Hashin_f = self.hashin_failure_func(cauchy_eff_stress_layer, Stress_limit)
                    if print_inner_steps == True and k == k_param and i == i_param:
                        # print(np.array(Hashin_f[0].vector().get_local()))
                        # print()
                        print(np.array(Hashin_f[1].vector().get_local()))
                        print()

                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing Lc NEWTON")

                    Lc = self.mesh_size_CG1[i]
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(Lc.vector().get_local()))
                        print()

                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing Delta_F_M NEWTON")

                    Delta_F_M = self.delta_crit_func(Lc, strain_MC)
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(Delta_F_M[1].vector().get_local()))
                        print()

                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing Sigma_F_M NEWTON")

                    Sigma_F_M = self.sigma_F_M_func(Hashin_f, Delta_F_M, cauchy_eff_stress_layer, strain_MC, Lc)
                    if print_inner_steps == True and k == 0 and i == 0:
                        print(np.array(Sigma_F_M[1].vector().get_local()))
                        print()

                    #form a propper array
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing NRB_delta_F_M_0 NEWTON")

                    NRB_delta_F_M_0 = self.form_NRB_delta_array_func(cauchy_eff_stress_layer, NRB_delta_FT_0[i][k], NRB_delta_FC_0[i][k], \
                                                                                 NRB_delta_MT_0[i][k], NRB_delta_MC_0[i][k])
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(NRB_delta_F_M_0[1].vector().get_local()))
                        print()

                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing NRB_delta_F_M_0_new NEWTON")

                    NRB_delta_F_M_0_new = self.new_NRB_delta_F_M_0_func(Hashin_f, Delta_F_M, NRB_delta_F_M_0)
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(NRB_delta_F_M_0_new[1].vector().get_local()))
                        print()

                    #form a propper array
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing NRB_delta_F_M_F_old NEWTON")

                    NRB_delta_F_M_F = self.form_NRB_delta_array_func(cauchy_eff_stress_layer, NRB_delta_FT_f[i][k], NRB_delta_FC_f[i][k],
                                                                                  NRB_delta_MT_f[i][k], NRB_delta_MC_f[i][k])
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(NRB_delta_F_M_F[1].vector().get_local()))
                        print()

                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing NRB_delta_F_M_F_new NEWTON")

                    NRB_delta_F_M_F_new = self.new_NRB_delta_F_M_F_func(cauchy_eff_stress_layer, Hashin_f, Delta_F_M, NRB_delta_F_M_F, NRB_delta_F_M_0, NRB_delta_F_M_0_new,\
                    ratio_0F_ft, ratio_0F_fc, ratio_0F_mt, ratio_0F_mc)
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(NRB_delta_F_M_F_new[1].vector().get_local()))
                        print()

                    # form new damage 1 2
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing damage_coef_new_1_2 NEWTON")
                                                                                                         #here was an error NRB_delta_F_M_F_new
                    damage_coef_new_1_2 = self.form_damage_coef_12_func(Hashin_f, Delta_F_M, damage_coef_NEWTON[i][k], NRB_delta_F_M_F_new, NRB_delta_F_M_0_new)
                    if print_inner_steps == True and k == k_param and i == i_param:
                        print(np.array(damage_coef_new_1_2[1].vector().get_local()))
                        print()

                    if print_inner_steps == True and k == k_param and i == i_param:
                        print("computing new damage_coef NEWTON")

                    damage_coef_tens_NEWTON[i][k], damage_coef_comp_NEWTON[i][k] = self.update_damage_coef_TC_func(cauchy_eff_stress_layer, \
                    damage_coef_new_1_2, damage_coef_tens_NEWTON[i][k], damage_coef_comp_NEWTON[i][k])

                    damage_coef_NEWTON[i][k] = self.damage_coef_full_func(damage_coef_new_1_2, damage_coef_tens_NEWTON[i][k], damage_coef_comp_NEWTON[i][k])

                    #Computation of compliance for this layer
                    D_comp_temp[k] = self.form_comp_dolph(damage_coef_NEWTON[i][k], D_comp_temp[k])

                if print_inner_steps == True and i == i_param:
                    print(np.array(damage_coef_NEWTON[i][0][0].vector().get_local()))
                    print()
                    print(np.array(damage_coef_NEWTON[i][0][1].vector().get_local()))
                    print()
                    print(np.array(damage_coef_NEWTON[i][0][2].vector().get_local()))
                    print()

                if print_inner_steps == True and i == i_param:
                    print("computing new ABD NEWTON")

                D_stif_temp = self.Stiffness_matrix_lamina_array_dolph(D_comp_temp)
                Q_stif_temp = self.Stiffness_matrix_laminate_dolph(self.R_[i], D_stif_temp)
                #ABD matries for this surface
                D_stif_new_NEWTON[i] = D_stif_temp

                A_mat_new_NEWTON[i] = cm.complex_laminate_A_mat(Q_stif_temp, self.mas_pT_list[i])
                B_mat_new_NEWTON[i] = cm.complex_laminate_B_mat(Q_stif_temp, self.z0[i], self.z1[i])
                D_mat_new_NEWTON[i] = cm.complex_laminate_D_mat(Q_stif_temp, self.z0[i], self.z1[i])

            return D_stif_new_NEWTON, A_mat_new_NEWTON, B_mat_new_NEWTON, D_mat_new_NEWTON, \
                   damage_coef_NEWTON, damage_coef_tens_NEWTON, damage_coef_comp_NEWTON


from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.nonmatching_coupling_laminate import *

def get_residuals_ABD(splines, spline_funcs, spline_test_funcs, h_total, A_mat_new_NEWTON, B_mat_new_NEWTON, D_mat_new_NEWTON, source_terms):
    #setting and not updating since update didn't work
    residuals = []
    for i in range(len(splines)):
        residuals += [SVK_residual_laminate(splines[i],
                      spline_funcs[i],
                      spline_test_funcs[i],
                      #ABD will be new
                      h_total[i], A_mat_new_NEWTON[i], B_mat_new_NEWTON[i], D_mat_new_NEWTON[i],
                      source_terms[i])]

    return residuals

def solve_nonlinear_nonmatching_degradation_problem(problem,
                            num_srfs, n_ply,
                            stress_class, damage_lib,
                            h_total, R_, z_mid, source_terms,
                            NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0,
                            NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f,
                            damage_coef, damage_coef_tens, damage_coef_comp,
                            \
                            solver="direct",
                            ref_error=None, rtol=1e-3, max_it=20,
                            zero_mortar_funcs=True,
                            ksp_type=PETSc.KSP.Type.CG,
                            pc_type=PETSc.PC.Type.FIELDSPLIT,
                            fieldsplit_type="additive",
                            fieldsplit_ksp_type=PETSc.KSP.Type.PREONLY,
                            fieldsplit_pc_type=PETSc.PC.Type.LU,
                            ksp_rtol=1e-15, ksp_max_it=100000,
                            ksp_view=False, ksp_monitor_residual=False,
                            iga_dofs=False,
                            print_inner_steps = False,
                            connect_str = True):
    
    '''
    function completely developed by PENGoLINS, all additions are marked with ### ### fields
    https://github.com/hanzhao2020/PENGoLINS/blob/main/PENGoLINS/nonmatching_coupling.py#L729
    '''
    

    if zero_mortar_funcs:
        for i in range(len(problem.mortar_funcs)):
            for j in range(len(problem.mortar_funcs[i])):
                for k in range(len(problem.mortar_funcs[i][j])):
                        problem.mortar_funcs[i][j][k].interpolate(Constant(
                            (0.,)*len(problem.mortar_funcs[i][j][k])))
    if iga_dofs:
        u_iga_list = []
        for i in range(problem.num_splines):
            u_FE_temp = Function(problem.splines[i].V)
            u_iga_list += [v2p(multTranspose(problem.splines[i].M,
                               u_FE_temp.vector())),]
            problem.spline_funcs[i].interpolate(Constant((0.,0.,0.)))
        u_iga = create_nest_PETScVec(u_iga_list, comm=problem.comm)

########################################################################################################################################################
    damage_coef_NEWTON, damage_coef_tens_NEWTON, damage_coef_comp_NEWTON, D_stif_new_NEWTON, A_mat_new_NEWTON, B_mat_new_NEWTON, D_mat_new_NEWTON\
     = damage_lib.array_initial_NEWTON(num_srfs, n_ply, damage_coef, damage_coef_tens, damage_coef_comp)
########################################################################################################################################################
    for newton_iter in range(max_it+1):

########################################################################################################################################################
        #update stiffness matrix on the left based on new displacement
        if newton_iter != 0:
            if MPI.rank(problem.comm) == 0:
                print("NEWTON updating residuals... ")

            residuals = get_residuals_ABD(problem.splines, problem.spline_funcs, problem.spline_test_funcs,
                                          h_total, A_mat_new_NEWTON, B_mat_new_NEWTON, D_mat_new_NEWTON, source_terms)
########################################################################################################################################################
        dRt_dut_FE, Rt_FE = problem.assemble_nonmatching()

        problem.extract_nonmatching_system(Rt_FE, dRt_dut_FE)

        if solver == "direct":
            if MPI.size(problem.comm) == 1:
                problem.A.convert("seqaij")
            else:
                problem.A = create_aijmat_from_nestmat(problem.A, problem.A_list,
                                                    comm=problem.comm)

        if solver == "ksp" and pc_type != PETSc.PC.Type.FIELDSPLIT:
            problem.A = create_aijmat_from_nestmat(problem.A, problem.A_list,
                                                comm=problem.comm)

        current_norm = problem.b.norm()

        if newton_iter==0 and ref_error is None:
            ref_error = current_norm

        rel_norm = current_norm/ref_error
        if newton_iter >= 0:
            if MPI.rank(problem.comm) == 0:
                print("Solver iteration: {}, relative norm: {:.12}."
                      .format(newton_iter, rel_norm))
            sys.stdout.flush()

        if rel_norm < rtol:
            if MPI.rank(problem.comm) == 0:
                print("Newton's iteration finished in {} "
                      "iterations (relative tolerance: {}).\n"
                      .format(newton_iter, rtol))
            break

        if newton_iter == max_it:
            if MPI.rank(problem.comm) == 0:
                raise StopIteration("Nonlinear solver failed to "
                      "converge in {} iterations.\n".format(max_it))

        du_list = []
        du_IGA_list = []
        for i in range(problem.num_splines):
            du_list += [Function(problem.splines[i].V),]
            du_IGA_list += [zero_petsc_vec(problem.splines[i].M.size(1),
                                           comm=problem.splines[i].comm)]
        du = create_nest_PETScVec(du_IGA_list, comm=problem.comm)

        solve_nonmatching_mat(problem.A, du, -problem.b, solver=solver,
                              ksp_type=ksp_type, pc_type=pc_type,
                              fieldsplit_type=fieldsplit_type,
                              fieldsplit_ksp_type=fieldsplit_ksp_type,
                              fieldsplit_pc_type=fieldsplit_pc_type,
                              rtol=ksp_rtol, max_it=ksp_max_it,
                              ksp_view=ksp_view,
                              monitor_residual=ksp_monitor_residual)

        if iga_dofs:
            u_iga += du

        for i in range(problem.num_splines):
            problem.splines[i].M.mat().mult(du_IGA_list[i],
                                         du_list[i].vector().vec())
            problem.spline_funcs[i].assign(problem.spline_funcs[i]+du_list[i])
            v2p(du_list[i].vector()).ghostUpdate()

        problem.update_mortar_funcs()
        ########################################################################################################################################################
        if MPI.rank(problem.comm) == 0:
            print("NEWTON projection")
                                            # stress_strain_output
                                            # stress_strain_output_full_numpy
        cauchy_stress_layer, strain_array = stress_strain_output_full_numpy(problem.splines, problem.spline_funcs, D_stif_new_NEWTON,
                                                                                R_, h_total, z_mid, n_ply, print_out = False)
        
        if connect_str == True:
            if MPI.rank(problem.comm) == 0:
                print("NEWTON connecting patches...")

            cauchy_stress_layer = stress_class.connect_stress_parallel(cauchy_stress_layer)
            strain_array = stress_class.connect_stress_parallel(strain_array)

        #update properties and inside damage_coef
        D_stif_new_NEWTON, A_mat_new_NEWTON, B_mat_new_NEWTON, D_mat_new_NEWTON, \
        damage_coef_NEWTON, damage_coef_tens_NEWTON, damage_coef_comp_NEWTON = damage_lib.damaged_inner_loop(cauchy_stress_layer, strain_array,  D_stif_new_NEWTON, A_mat_new_NEWTON, B_mat_new_NEWTON, D_mat_new_NEWTON, \
                                                                                                       damage_coef_NEWTON, damage_coef_tens_NEWTON, damage_coef_comp_NEWTON, \
                                                                                                       #using this since if new will be bigger they will update _NEWTON var,
                                                                                                       #if not they will take a value from outside given in _0/_f
                                                                                                       NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0, \
                                                                                                       NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f,
                                                                                                       print_inner_steps)
        ########################################################################################################################################################

    if iga_dofs:
        return problem.spline_funcs, u_iga
    else:
        return problem.spline_funcs
