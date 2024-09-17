from tIGAr.NURBS import *
from PENGoLINS.nonmatching_coupling import *
from PENGoLINS.nonmatching_coupling_laminate import *
from PENGoLINS.igakit_utils import *
import numpy as np


# #import test module:
import COMPs.composite_module as cm
from COMPs.composite_damage_V6 import *

import time
import dolfin as df
import sys
np.set_printoptions(threshold=sys.maxsize)



# creation must be done by horizontal lines from bottom to top
def create_srf(num_el_r, num_el_alpha, pts_t, p=2, Ri=6, angle_lim=None, param=None):

    """

                          o 2
                        / |
               C[1,1]  /  |           v
                      /   | C[0,1]    ^
                     /    |           |
             C[0,0] |     |           +------> u
           arc info o-----o 1
                     C[1,0]
    """

    if angle_lim != None:
        if MPI.rank(worldcomm) == 0:
            print("surface, angles: ", angle_lim)
        angle = (math.radians(angle_lim[0]), math.radians(angle_lim[1]))
        Ci = circle(center=[0,0,0], radius=Ri, angle=angle)
        L1 = line(pts_t[0],pts_t[1])
        if param == "arc2line":
            S = ruled(Ci,L1)
        elif param == "line2arc":
            S = ruled(L1, Ci)
    else:

        """
                    C[1,1]
             3 o--------------o 4
               |  v           |
               |  ^           |
        C[0,0] |  |           | C[0,1]
               |  |           |
               |  +------> u  |
               o--------------o
             1      C[1,0]      2
        """

        if MPI.rank(worldcomm) == 0:
            print("surface, angles: ", angle_lim)
        L1 = line(pts_t[0],pts_t[1])
        L2 = line(pts_t[2],pts_t[3])
        S = ruled(L1,L2)

    deg1, deg2 = S.degree
    S.elevate(0,p-deg1)
    S.elevate(1,p-deg2)
    new_knots_r = np.linspace(0,1,num_el_r)[1:-1]
    new_knots_alpha = np.linspace(0,1,num_el_alpha)[1:-1]
    S.refine(0,new_knots_r)
    S.refine(1,new_knots_alpha)

    return S



def create_spline(srf, num_field=3, BCs=[], zero_domain=[], comm_in = worldcomm):
    spline_mesh = NURBSControlMesh(srf, useRect=False)
    spline_generator = EqualOrderSpline(comm_in, num_field, spline_mesh)
    # for each spline define bc
    if MPI.rank(comm_in) == 0:
        print(BCs)
    if BCs != None:
        for i in range(len(BCs)):
            BC = BCs[i]
            parametric_direction = BC[0]
            side = BC[1]
            #xyz dir
            field = BC[2]
            scalar_spline = spline_generator.getScalarSpline(field)

            side_dofs = scalar_spline.getSideDofs(parametric_direction,
                                                  side, nLayers=2)

            spline_generator.addZeroDofs(field, side_dofs)

# BC list should be equal to zero domain list

    if zero_domain != None:
        # zero_domain = [[ZeroDomain0(), [0,1,2]],]
        for h in range(len(zero_domain)):
            # zero_domain[h] = [ZeroDomain0(), [0,1,2]]
            for f in range(len(zero_domain[h][1])):
                # zero_domain[h][1] = [0,1,2]
                # zero_domain[h][1][f] = 0;1;2
                # zero_domain[h][0] = ZeroDomain0()
                field_zero = zero_domain[h][1][f]
                spline_generator.addZeroDofsByLocation(zero_domain[h][0], field_zero)

    quad_deg = 2*srf.degree[0]
    spline = ExtractedSpline(spline_generator, quad_deg, comm = comm_in)
    return spline

def distr_loading(problem, f, bdry, load_srf_ind):

    xi = problem.splines[load_srf_ind].parametricCoordinates()

    return inner(f, problem.splines[load_srf_ind].rationalize(
        problem.spline_test_funcs[load_srf_ind]))*bdry(xi)\
        *problem.splines[load_srf_ind].ds


#!composite formulation

h_total = 3.43e-3

EL_1 = Constant(23.6e+9)
ET_1 = Constant(10.0e+9)
nuLT_1 = Constant(0.23)
nuLT_2 = nuLT_1*ET_1/EL_1
GLT_1 = Constant(1e+9)

proper_ar1 = [EL_1, ET_1, nuLT_1, GLT_1]

F1_max_1 = np.float64(735e+6)
F1_min_1 = np.float64(600e+6)
F2_max_1 = np.float64(45e+6)
F2_min_1 = np.float64(100e+6)
F12_max_1 = np.float64(45e+6)

max_sigma_ar_1 = [F1_max_1, F1_min_1, F2_max_1, F2_min_1, F12_max_1]

ratio_0F_ft_1 = np.float64(1.005)
ratio_0F_fc_1 = np.float64(1.005)
ratio_0F_mt_1 = np.float64(1.005)
ratio_0F_mc_1 = np.float64(1.005)

ratio_ar_1 = [ratio_0F_ft_1, ratio_0F_fc_1, ratio_0F_mt_1, ratio_0F_mc_1]

alpha_list = [0, 45, 90, -45, 0, 0, 45, 90, -45, 0, 0, 45, 90, -45, 0, 0, 45, 90, -45, 0, 0, 45, 90, -45, 0]

h_th_float = h_total/len(alpha_list)
h_th = Constant(h_th_float)

pT_list = [h_th for i in range(len(alpha_list))]
prop_array = [proper_ar1 for i in range(len(alpha_list))]
max_sigma_array = [max_sigma_ar_1 for i in range(len(alpha_list))]
displ_ratio_array = [ratio_ar_1 for i in range(len(alpha_list))]

#loading (H/m2)
num_stress = 50
loading_val_Pa = np.array([(0.6038461538*(10**6)*i/(num_stress-1))+0.1 for i in range(num_stress)])

# loading_val_stress = np.array([0.1, 47.1e+6, 86e+6])

Save_iteration = True

# geometry and general staff
L = 600e-3
w = 600e-3

#more elements higher load with a dmg modle
num_el0 = 10
num_el1 = 11

loading_val = loading_val_Pa

penalty_coefficient = 1.0e3

# p = 3
p = 2

if MPI.rank(worldcomm) == 0:
    #small check-output
    print("\ncomposite layup")
    for i in range(len(pT_list)):
        print(float(pT_list[i]), alpha_list[i])

    h_total = sum(pT_list)

    print(float(h_total))

#dir 1-0
pts0 = [[[0., 0., 0.], [L/2, 0., 0.],
#dir 1-1
        [0., w/2, 0.], [L/2, w/2, 0.]],None,None]
pts1 = [[[L/2, 0., 0.], [L, 0., 0.],
        [L/2, w/2, 0.], [L, w/2, 0.]],None,None]
pts2 = [[[0., w/2, 0.], [L/2, w/2, 0.],
        [0., w, 0.], [L/2, w, 0.]],None,None]
pts3 = [[[L/2, w/2, 0.], [L, w/2, 0.],
        [L/2, w, 0.], [L, w, 0.]],None,None]

pts_list = [pts0, pts1, pts2, pts3]
num_srfs = len(pts_list)

if MPI.rank(worldcomm) == 0:
    print("\nPenalty coefficient:", penalty_coefficient)

if MPI.rank(worldcomm) == 0:
    print("Creating geometry...")

#for now
n_ply = [len(alpha_list) for i in range(num_srfs)]

BC_1_surf = [[0,0, 0], [0,0, 1], [0,0, 2], [1,0, 0], [1,0, 1], [1,0, 2]]
BC_2_surf = [[0,1, 0], [0,1, 1], [0,1, 2], [1,0, 0], [1,0, 1], [1,0, 2]]
BC_3_surf = [[0,0, 0], [0,0, 1], [0,0, 2], [1,1, 0], [1,1, 1], [1,1, 2]]
BC_4_surf = [[0,1, 0], [0,1, 1], [0,1, 2], [1,1, 0], [1,1, 1], [1,1, 2]]

#full Z zero

BCs_list = [BC_1_surf, BC_2_surf, BC_3_surf, BC_4_surf]

# By XY system point BC

# changed to e-4 due to scale of the problem

class ZeroDomain0(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 1e-4 and x[1] < 1e-4
class ZeroDomain1(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > L-1e-4 and x[1] < 1e-4

#propper point BC example

# p_BC_1_surf = [[ZeroDomain0(), [0,1,2]],]
# p_BC_2_surf = [[ZeroDomain1(), [1,]],]

p_BC_1_surf = None
p_BC_2_surf = None
p_BC_3_surf = None
p_BC_4_surf = None

# for each spline define bc
zero_domain_list = [p_BC_1_surf, p_BC_2_surf, p_BC_3_surf, p_BC_4_surf]

nurbs_srfs = []
splines = []
single_proc_splines = []

num_el = [[num_el0, num_el1] for i in range(len(pts_list))]

print()

for i in range(len(pts_list)):
    #clockvise creation
    nurbs_srfs += [create_srf(num_el[i][0], num_el[i][1],
                              pts_t = pts_list[i][0],
                              p=p,
                              angle_lim = pts_list[i][1],
                              param=pts_list[i][2])]

    splines += [create_spline(nurbs_srfs[i], BCs=BCs_list[i], zero_domain=zero_domain_list[i])]

    if worldcomm.size == 1:
        #########################################################################################################
        #for single run we only need mesh
        #########################################################################################################
        loading_val = [0.1]
        print("saving mesh for parallel runs")
        single_proc_splines += [create_spline(nurbs_srfs[i], BCs=BCs_list[i], zero_domain=zero_domain_list[i])]

        filename_mesh = "mesh_surface_" + str(i)

        full_mesh_name = "./temp/"+str(filename_mesh)+".xdmf"

        fFile_mesh = XDMFFile(worldcomm, full_mesh_name)

        #deepcopy of the mesh
        mesh_to_send = Mesh(single_proc_splines[-1].V_linear.mesh())

        fFile_mesh.write(mesh_to_send)
        fFile_mesh.close()


mesh_size_CG1 = cm.initialize_composite_array(num_srfs)

for i in range(num_srfs):
    mesh_size = spline_mesh_size(splines[i])
    mesh_size_CG1[i] = splines[i].projectScalarOntoLinears(mesh_size, lumpMass=True)

# to see Characteristic length
print(np.array(mesh_size_CG1[0].vector().get_local())[0]*1000)

#!composite formulation

mas_prop_array = [prop_array for i in range(num_srfs)]
mas_alpha_list = [alpha_list for i in range(num_srfs)]
mas_pT_list = [pT_list for i in range(num_srfs)]

# #offset test
# test1 = cm.layer_cord(mas_pT_list[0])[0]
# test2 = cm.layer_cord(mas_pT_list[0], -1)[0]
# for i in range(len(test1)):
#     print(float(test1[i]), float(test2[i]))

z0 = [cm.layer_cord(mas_pT_list[i])[0] for i in range(len(mas_pT_list))]
z1 = [cm.layer_cord(mas_pT_list[i])[1] for i in range(len(mas_pT_list))]

mas_max_sigma_array = [max_sigma_array for i in range(num_srfs)]
mas_displ_ratio_array = [displ_ratio_array for i in range(num_srfs)]

h_total = [sum(mas_pT_list[i]) for i in range(num_srfs)]

R_ = [[cm.rotational_mat_for_strain(mas_alpha_list[i][j]) for j in range(len(mas_alpha_list[i]))] for i in range(num_srfs)]

R_non_constant = R_

####### experiment with non-constant angle at each point of the plate #######

# #new tilt from the main axis pushed into R_
# #tilt should follow the natural order of nodes in FE space (.compute_vertex_values)
# new_angles_list = [[45-90*i/(num_el[j][0]-1) for i in range(num_el[j][0])] for j in range(num_srfs)]
# new_angles_list[-1] = [0 for i in range(num_el[-1][0])]
# new_angles_list[-2] = [0 for i in range(num_el[-2][0])]
# print(new_angles_list)
# #rotation based on IGA mesh
# # R_non_constant = angle_field(splines, n_ply, mas_alpha_list, num_el, mesh_size_CG1, new_angles_list, spline_special_rule_invert)

# initializing properties for composite libs
damage_lib = damage_coef(z0, z1, mas_pT_list, R_, mas_prop_array, mas_max_sigma_array,\
                                              mas_displ_ratio_array, mesh_size_CG1, \
                                              print_steps = False)

D_comp = [cm.Compliance_matrix_lamina_array(mas_prop_array[i]) for i in range(num_srfs)]
D_stif = [cm.Stiffness_matrix_lamina_array(D_comp[i]) for i in range(num_srfs)]

#prop for panga
if MPI.rank(worldcomm) == 0:
    print("\nprop ufl, fast initialization, slow computation")

Q_stif_panga_connection = [cm.Stiffness_matrix_laminate(R_[i], D_stif[i]) for i in range(num_srfs)]
A_mat_panga_connection = [cm.complex_laminate_A_mat(Q_stif_panga_connection[i], mas_pT_list[i]) for i in range(num_srfs)]
B_mat_panga_connection = [cm.complex_laminate_B_mat(Q_stif_panga_connection[i], z0[i], z1[i]) for i in range(num_srfs)]
D_mat_panga_connection = [cm.complex_laminate_D_mat(Q_stif_panga_connection[i], z0[i], z1[i]) for i in range(num_srfs)]


z_mid = [[(z0[i][k]+z1[i][k])/2 for k in range(len(z0[i]))] for i in range(len(mas_pT_list))]

if MPI.rank(worldcomm) == 0:
    print("\nprop dolfin, slow initialization, fast computation")

#props for dolf
D_stif_new = [damage_lib.Stiffness_matrix_lamina_array_dolph(D_comp[i], mesh_size_CG1[i]) for i in range(num_srfs)]
Q_stif = [damage_lib.Stiffness_matrix_laminate_dolph(R_non_constant[i], D_stif_new[i], mesh_size_CG1[i]) for i in range(num_srfs)]
A_mat_new = [cm.complex_laminate_A_mat(Q_stif[i], mas_pT_list[i]) for i in range(num_srfs)]
B_mat_new = [cm.complex_laminate_B_mat(Q_stif[i], z0[i], z1[i]) for i in range(num_srfs)]
D_mat_new = [cm.complex_laminate_D_mat(Q_stif[i], z0[i], z1[i]) for i in range(num_srfs)]


problem = NonMatchingCouplingLaminate(splines, h_total[0], A_mat_panga_connection[0], B_mat_panga_connection[0], D_mat_panga_connection[0],
                                      comm=worldcomm)


mapping_list = [[0,1], [2,3], [0,2], [1,3]]
num_mortar_mesh = len(mapping_list)

mortar_nels = []
mortar_mesh_locations = []
			#first line for the first patch
v_mortar_locs = [np.array([[1., 0.], [1., 1.]]),
			#second line for the second patch
                 np.array([[0., 0.], [0., 1.]])]
h_mortar_locs = [np.array([[0., 1.], [1., 1.]]),
                 np.array([[0., 0.], [1., 0.]])]

num_el_max = max(num_el0, num_el1)

for i in range(num_mortar_mesh):
    mortar_nels += [(num_el_max+i+2)*2,]
    if i < 2:
        mortar_mesh_locations += [v_mortar_locs,]
    else:
        mortar_mesh_locations += [h_mortar_locs,]

problem.create_mortar_meshes(mortar_nels)
problem.mortar_meshes_setup(mapping_list, mortar_mesh_locations,
                            penalty_coefficient)

######################################################################## preprocessor to genarate mortar mesh data (analysis example - mortar mesh info from preprocessor.txt)

# ###################################################
# # [HZ] Automate intersection computation
# # Save NURBS surfs to an IGS file
# from PENGoLINS.occ_preprocessing import *
# from PENGoLINS.igakit_utils import *
# occ_surfs = []
# num_srfs = len(pts_list)
# for i in range(num_srfs):
#     occ_surfs += [ikNURBS2BSpline_surface(nurbs_srfs[i])]
# igs_filename = "./geom.igs"
# write_geom_file(occ_surfs, igs_filename)
#
# # Load IGS file and compute intersections automatically
# print("Importing geometry...")
# igs_shapes = read_igs_file(igs_filename, as_compound=False)
# occ_surfaces = [topoface2surface(face, BSpline=True)
#                 for face in igs_shapes]
#
# preprocessor = OCCPreprocessing(occ_surfaces, reparametrize=False,
#                                 refine=False)
# preprocessor.compute_intersections(mortar_refine=2)
#
# print(preprocessor.mapping_list)
# print(preprocessor.intersections_para_coords)
#
# # [HZ] Use data of mortar mesh from preprocessor
# problem.create_mortar_meshes(preprocessor.mortar_nels)
# problem.mortar_meshes_setup(preprocessor.mapping_list,
#                             preprocessor.intersections_para_coords,
#                             penalty_coefficient)
# ########################################################


num_el_stress = num_el
stress_class = stress_connectivity_parallel(problem, num_el_stress, degree_param = 6, print_steps = False)

# preparing files for export

SAVE_PATH = "./"
u_file_names = []
u_files = []
F_file_names = []
F_files = []
Stress_output_names = []
Stress_output = []
damage_params_names = []
damage_params = []

for i in range(num_srfs):
    u_file_names += [[],]
    u_files += [[],]
    F_file_names += [[],]
    F_files += [[],]
    Stress_output_names += [[],]
    Stress_output += [[],]
    damage_params_names += [[],]
    damage_params += [[],]

    for j in range(3):
        u_file_names[i] += [SAVE_PATH+"results/"+"u"+str(i)
                            +"_"+str(j)+"_file.xdmf",]
        u_files[i] += [XDMFFile(problem.comm, u_file_names[i][j]),]

        u_files[i][-1].parameters["flush_output"] = True
        u_files[i][-1].parameters["functions_share_mesh"] = True
        u_files[i][-1].parameters["rewrite_function_mesh"] = False

        F_file_names[i] += [SAVE_PATH+"results/"+"F"+str(i)
                            +"_"+str(j)+"_file.xdmf",]
        F_files[i] += [XDMFFile(problem.comm, F_file_names[i][j]),]

        if j == 2:
            F_file_names[i] += [SAVE_PATH+"results/"+"F"
                                +str(i)+"_3_file.xdmf",]
            F_files[i] += [XDMFFile(problem.comm, F_file_names[i][3]),]

        F_files[i][-1].parameters["flush_output"] = True
        F_files[i][-1].parameters["functions_share_mesh"] = True
        F_files[i][-1].parameters["rewrite_function_mesh"] = False

    for k in range(n_ply[i]):
        Stress_output_names[i] += [[],]
        Stress_output[i] += [[],]

        damage_params_names[i] += [[],]
        damage_params[i] += [[],]

        for j in range(3):

            ###############change name to second Piola–Kirchhoff

            Stress_output_names[i][k] += [SAVE_PATH+"results/SPK_"+str(j)+"_"+str(i)+"_srf_"+str(k)+"_layer.xdmf",]
            Stress_output[i][k] += [XDMFFile(problem.comm, Stress_output_names[i][k][j]),]

            Stress_output[i][k][-1].parameters["flush_output"] = True
            Stress_output[i][k][-1].parameters["functions_share_mesh"] = True
            Stress_output[i][k][-1].parameters["rewrite_function_mesh"] = False

            damage_params_names[i][k] += [SAVE_PATH+"results/damage_coef_"+str(j)+"_"+str(i)+"_srf_"+str(k)+"_layer.xdmf",]
            damage_params[i][k] += [XDMFFile(problem.comm, damage_params_names[i][k][j]),]

            damage_params[i][k][-1].parameters["flush_output"] = True
            damage_params[i][k][-1].parameters["functions_share_mesh"] = True
            damage_params[i][k][-1].parameters["rewrite_function_mesh"] = False

# all possible boundaries

left_bdry = lambda x : conditional(le(x[0], 1e-3),
                        Constant(1.), Constant(0.))

right_bdry = lambda x : conditional(gt(x[0], 1.-1e-3),
                        Constant(1.), Constant(0.))

top_bdry = lambda x : conditional(gt(x[1], 1.-1e-3),
                        Constant(1.), Constant(0.))

bot_bdry = lambda x : conditional(le(x[1], 1e-3),
                        Constant(1.), Constant(0.))

# initialization of dmg variables

damage_coef, damage_coef_tens, damage_coef_comp, \
NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0, \
NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f = damage_lib.array_initial(num_srfs, n_ply, mesh_size_CG1)

break_param = False

max_displ = []

for load_cycle in range(len(loading_val)):

    f0 = as_vector([Constant(0.), Constant(0.), Constant(-loading_val[load_cycle])])

    if MPI.rank(worldcomm) == 0:
        print("\n\nload_cycle: ", load_cycle+1, np.round(float(loading_val_Pa[load_cycle]), 4),"\n\n")

    source_terms = []

    for i in range(len(splines)):
        # setting up loading
        source_terms += [inner(f0, problem.splines[i].rationalize(\
        problem.spline_test_funcs[i]))*problem.splines[i].dx,]

    residuals = get_residuals_ABD(problem.splines, problem.spline_funcs, problem.spline_test_funcs,
                                  h_total, A_mat_new, B_mat_new, D_mat_new, source_terms)
    if MPI.rank(worldcomm) == 0:
        print("setting residuals")

    problem.set_residuals(residuals)

    if MPI.rank(worldcomm) == 0:
        print("solving")

    #clear nonliniar solver
    soln = problem.solve_nonlinear_nonmatching_problem(rtol=1e-2, max_it=15)


    #nonliniar with degradation solver
    soln = solve_nonlinear_nonmatching_degradation_problem(problem,
                                    num_srfs, n_ply,
                                    stress_class, damage_lib,
                                    h_total, R_, z_mid, source_terms,
                                    NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0,
                                    NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f,
                                    damage_coef, damage_coef_tens, damage_coef_comp,
                                    rtol=1e-2, max_it=25,
                                    print_inner_steps = False,
                                    connect_str = True)

    # #clear liniar solver
    # soln = problem.solve_linear_nonmatching_problem()

    if MPI.rank(worldcomm) == 0:
        print("Computing stresses...")

        start = time.perf_counter()

    SPK_stress_layer, strain_array = stress_strain_output_full_numpy(problem.splines, problem.spline_funcs, D_stif_new,
                                                                            R_, h_total, z_mid, n_ply, print_out = False)

    if MPI.rank(worldcomm) == 0:
        finish = time.perf_counter()
        print("regular python time: ", finish-start)

    #connect stress and strains
    if MPI.rank(worldcomm) == 0:
        print("connecting stresses...")
    SPK_stress_layer = stress_class.connect_stress_parallel(SPK_stress_layer)

    if MPI.rank(worldcomm) == 0:
        print("connecting strains...")
    strain_array = stress_class.connect_stress_parallel(strain_array)


    if MPI.rank(worldcomm) == 0:
        max_defl_temp = 0

    for i in range(num_srfs):
        defl_temp = MPI.min(worldcomm, float(np.min(problem.spline_funcs[i].split()[2].vector().get_local())))

        worldcomm.barrier()

        if MPI.rank(worldcomm) == 0:
            if defl_temp < max_defl_temp:
                max_defl_temp = defl_temp

    worldcomm.barrier()

    if MPI.rank(worldcomm) == 0:
        max_displ.append(max_defl_temp)

    if Save_iteration == True:
        if MPI.rank(worldcomm) == 0:
            print("saving stress...")
        for i in range(problem.num_splines):
            for k in range(n_ply[i]):
                for j in range(3):

                    SPK_stress_layer_proj = SPK_stress_layer[i][k][j]

                    SPK_stress_layer_proj.rename("SPK_"+str(j)+"_"+str(i)+"_srf_"+str(k)+"_layer",
                                                    "SPK_"+str(j)+"_"+str(i)+"_srf_"+str(k)+"_layer")
                    Stress_output[i][k][j].write(SPK_stress_layer_proj, float(load_cycle))

        if MPI.rank(worldcomm) == 0:
            print("saving displacement...")
        for i in range(problem.num_splines):
            soln_split = problem.spline_funcs[i].split()
            for j in range(3):
                soln_split[j].rename("u"+str(i)+"_"+str(j),
                                     "u"+str(i)+"_"+str(j))

                u_files[i][j].write(soln_split[j], float(load_cycle))

                problem.splines[i].cpFuncs[j].rename("F"+str(i)+"_"+str(j),
                                                     "F"+str(i)+"_"+str(j))

                F_files[i][j].write(problem.splines[i].cpFuncs[j], float(load_cycle))

                if j == 2:
                    problem.splines[i].cpFuncs[3].rename("F"+str(i)+"_3",
                                                         "F"+str(i)+"_3")

                    F_files[i][3].write(problem.splines[i].cpFuncs[3], float(load_cycle))


    if load_cycle != len(loading_val)-1:

        if MPI.rank(worldcomm) == 0:
            print("Computing dmg...")

        D_stif_new, A_mat_new, B_mat_new, D_mat_new, \
        damage_coef, damage_coef_tens, damage_coef_comp, \
        NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0, \
        NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f = damage_lib.damaged_outer_loop(SPK_stress_layer, strain_array,  D_stif_new, A_mat_new, B_mat_new, D_mat_new, \
                                                                                                       damage_coef, damage_coef_tens, damage_coef_comp, \
                                                                                                       NRB_delta_FT_0, NRB_delta_FC_0, NRB_delta_MT_0, NRB_delta_MC_0, \
                                                                                                       NRB_delta_FT_f, NRB_delta_FC_f, NRB_delta_MT_f, NRB_delta_MC_f, k_param = 2)


    if Save_iteration == True and load_cycle != len(loading_val)-1:
        if MPI.rank(worldcomm) == 0:
            print("saving damage_params...")
        for i in range(problem.num_splines):
            for k in range(n_ply[i]):
                for j in range(3):

                    damage_param_proj = damage_coef[i][k][j]
                    damage_param_proj.rename("dmg_coef_"+str(j)+"_"+str(i)+"_srf_"+str(k)+"_layer",
                                                    "dmg_coef_"+str(j)+"_"+str(i)+"_srf_"+str(k)+"_layer")

                    damage_params[i][k][j].write(damage_param_proj, float(load_cycle))

    if break_param == True:
        break

if MPI.rank(worldcomm) == 0:

    mine_deflec = np.array(max_displ)*1000*(-1)

    print(loading_val_Pa)
    print()
    print(mine_deflec)


    correct_displ_Z = [-7.11E-15, 7.264573991, 12.5560538117, 16.2331838565, 18.8340807175, 21.3452914798, 23.6771300448, 25.8295964126, 27.4439461883, 29.3273542601, 31.0313901345, 32.466367713,
                       33.9013452915, 35.4260089686, 36.5022421525, 37.5784753363, 38.2062780269, 38.9237668161, 39.730941704, 40.4484304933, 41.2556053812, 41.6143497758]


    correct_loading_val_Newtons = np.array([0.1/(10**6), 0.02, 0.0553846154, 0.0930769231, 0.1238461538, 0.1584615385, 0.1938461538, 0.2292307692, 0.2592307692, 0.2961538462, 0.3338461538, 0.3661538462, 0.4038461538, 0.4376923077,
                      0.4715384615, 0.4984615385, 0.5153846154, 0.5407692308, 0.5615384615, 0.5776923077, 0.5992307692, 0.6038461538])*10**6


    import matplotlib.pyplot as plt

    plt.plot(correct_displ_Z, correct_loading_val_Newtons, color='cyan', linewidth=1, marker='x',
    markersize=5, linestyle='-')
    plt.plot(mine_deflec, loading_val_Pa, color='red', linewidth=1, marker='o',
    markersize=5, linestyle='-')

    # plt.title("Line Chart")
    plt.ylabel('Load(N)')
    plt.xlabel('Deflection(mm)')
    plt.legend(labels = ('E22 experiment', 'mine'))

    plt.show()
