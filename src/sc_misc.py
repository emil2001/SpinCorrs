import vector
import numpy as np


def calculate_cosine(pred, X):
    '''
    Calculates cos(lep, d) in top-quark reference frame using predictied momenta of neutrino and mediator
    """
    Parameters:
    pred(torch.tensor) - predicted values of neutrino and mediator momenta 

    X(torch.tensor) - tensor of momenta of charged particles
    '''
    #Neutrino
    p_nu_x = np.array(pred[:, 0])
    p_nu_y = np.array(pred[:, 1])
    p_nu_z = np.array(pred[:, 2])
    p4nm = vector.arr({"px": p_nu_x, "py": p_nu_y, "pz": p_nu_z, "M": np.zeros_like(p_nu_x)})

    #Lepton
    p_l_x = np.array(X['p_l_x'])
    p_l_y = np.array(X['p_l_y'])
    p_l_z = np.array(X['p_l_z'])
    E = np.sqrt(p_l_x**2 + p_l_y**2 + p_l_z**2)
    p4Mu = vector.arr({"px": p_l_x, "py": p_l_y, "pz": p_l_z, "E": E})

    #B-quark
    p_b_x = np.array(X['p_b_x'])
    p_b_y = np.array(X['p_b_y'])
    p_b_z = np.array(X['p_b_z'])
    E = np.sqrt(p_b_x**2 + p_b_y**2 + p_b_z**2)
    p4j1 = vector.arr({"px": p_b_x, "py": p_b_y, "pz": p_b_z, "E": E})

    #Light quark
    p_q_x = np.array(X['p_q_x'])
    p_q_y = np.array(X['p_q_y'])
    p_q_z = np.array(X['p_q_z'])
    E = np.sqrt(p_q_x**2 + p_q_y**2 + p_q_z**2)
    p4j2 = vector.arr({"px": p_q_x, "py": p_q_y, "pz": p_q_z, "E": E})

    #Mediator
    pt_phi = np.array(pred[:, 3])
    eta_phi = np.array(pred[:, 4])
    phi_phi = np.array(pred[:, 5])
    M = 400 * np.ones_like(pt_phi)
    p4phi = vector.arr({"pt": pt_phi, "eta": eta_phi, "phi": phi_phi, "M": M})

    p4W = p4Mu + p4nm
    p4top = p4W + p4j1 + p4phi

    boostedLepton = p4Mu.boostCM_of_p4(-p4top)
    boostedLJet = p4j2.boostCM_of_p4(-p4top)
    # Get the spatial components as 3D vectors
    p3Lepton = boostedLepton.to_beta3()
    p3LJet = boostedLJet.to_beta3()

    # Calculate the cosine of the angle
    Cos_lep_light_pred = p3Lepton.dot(p3LJet) / (p3Lepton.mag * p3LJet.mag)
    p4top = p4W + p4j1

    boostedLepton = p4Mu.boostCM_of_p4(-p4top)
    boostedLJet = p4j2.boostCM_of_p4(-p4top)
    # Get the spatial components as 3D vectors
    p3Lepton = boostedLepton.to_beta3()
    p3LJet = boostedLJet.to_beta3()

    # Calculate the cosine of the angle
    Cos_lep_light_nophi_pred = p3Lepton.dot(p3LJet) / (p3Lepton.mag * p3LJet.mag)
    return Cos_lep_light_pred, Cos_lep_light_nophi_pred


def calculate_cosine_theory(X, y):
    '''
    Calculates cos(lep, d) in top-quark reference frame using predictied momenta of neutrino and mediator
    """
    Parameters:
    X(torch.tensor) - tensor of momenta of charged particles

    y(torch.tensor) - tensor of momenta of netrino and mediator
    '''
    #Neutrino
    p_nu_x = np.array(y['p_nu_x'])
    p_nu_y = np.array(y['p_nu_y'])
    p_nu_z = np.array(y['p_nu_z'])
    p4nm = vector.arr({"px": p_nu_x, "py": p_nu_y, "pz": p_nu_z, "M": np.zeros_like(pt_nu_rec)})

    #Lepton
    p_l_x = np.array(X['p_l_x'])
    p_l_y = np.array(X['p_l_y'])
    p_l_z = np.array(X['p_l_z'])
    E = np.sqrt(p_l_x**2 + p_l_y**2 + p_l_z**2)
    p4Mu = vector.arr({"px": p_l_x, "py": p_l_y, "pz": p_l_z, "E": E})

    #b-quark
    p_b_x = np.array(X['p_b_x'])
    p_b_y = np.array(X['p_b_y'])
    p_b_z = np.array(X['p_b_z'])
    E = np.sqrt(p_b_x**2 + p_b_y**2 + p_b_z**2)
    p4j1 = vector.arr({"px": p_b_x, "py": p_b_y, "pz": p_b_z, "E": E})

    #light quark
    p_q_x = np.array(X['p_q_x'])
    p_q_y = np.array(X['p_q_y'])
    p_q_z = np.array(X['p_q_z'])
    E = np.sqrt(p_q_x**2 + p_q_y**2 + p_q_z**2)
    p4j2 = vector.arr({"px": p_q_x, "py": p_q_y, "pz": p_q_z, "E": E})

    #Mediator
    pt_phi = np.array(y['pt_phi'])
    eta_phi = np.array(y['eta_phi'])
    phi_phi = np.array(y['phi_phi'])
    M = 400 * np.ones_like(pt_phi_x)
    p4phi = vector.arr({"pt": pt_phi, "eta": eta_phi, "phi": phi_phi, "M": M})

    p4W = p4Mu + p4nm
    p4top = p4W + p4j1 + p4phi

    boostedLepton = p4Mu.boostCM_of_p4(-p4top)
    boostedLJet = p4j2.boostCM_of_p4(-p4top)
    # Get the spatial components as 3D vectors
    p3Lepton = boostedLepton.to_beta3()
    p3LJet = boostedLJet.to_beta3()

    # Calculate the cosine of the angle
    Cos_lep_light_true = p3Lepton.dot(p3LJet) / (p3Lepton.mag * p3LJet.mag)
    
    p4top = p4W + p4j1

    boostedLepton = p4Mu.boostCM_of_p4(-p4top)
    boostedLJet = p4j2.boostCM_of_p4(-p4top)
    # Get the spatial components as 3D vectors
    p3Lepton = boostedLepton.to_beta3()
    p3LJet = boostedLJet.to_beta3()

    # Calculate the cosine of the angle
    Cos_lep_light_nophi_true = p3Lepton.dot(p3LJet) / (p3Lepton.mag * p3LJet.mag)
    return Cos_lep_light_true, Cos_lep_light_nophi_true