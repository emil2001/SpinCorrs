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
    
    p_nu_x = np.array(pred[:, 0])
    p_nu_y = np.array(pred[:, 1])
    p_nu_z = np.array(pred[:, 2])
    p4nm = vector.arr({"px": p_nu_x, "py": p_nu_y, "pz": p_nu_z, "M": np.zeros_like(p_nu_x)})

    pt_l = np.array(X['Pt_Lep'])
    eta_l = np.array(X['Eta_Lep'])
    phi_l = np.array(X['Phi_Lep'])
    M = np.array(X['M_Lep'])
    p4Mu = vector.arr({"pt": pt_l, "eta": eta_l, "phi": phi_l, "M": M})

    pt_b = np.array(X['Pt_BJ1'])
    eta_b = np.array(X['Eta_BJ1'])
    phi_b = np.array(X['Phi_BJ1'])
    M = np.array(X['M_BJ1'])
    p4j1 = vector.arr({"pt": pt_b, "eta": eta_b, "phi": phi_b, "M": M})

    pt_q = np.array(X['Pt_LJ'])
    eta_q = np.array(X['Eta_LJ'])
    phi_q = np.array(X['Phi_LJ'])
    M = np.array(X['M_LJ'])
    p4j2 = vector.arr({"pt": pt_q, "eta": eta_q, "phi": phi_q, "M": M})

    p_phi_x = np.array(pred[:, 3])
    p_phi_y = np.array(pred[:, 4])
    p_phi_z = np.array(pred[:, 5])
    M = 400 * np.ones_like(p_phi_x)
    p4phi = vector.arr({"px": p_phi_x, "py": p_phi_y, "pz": p_phi_z, "M": M})

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
    
    p_nu_x = np.array(y['Px_NuGen'])
    p_nu_y = np.array(y['Py_NuGen'])
    p_nu_z = np.array(y['Pz_NuGen'])
    p4nm = vector.arr({"px": p_nu_x, "py": p_nu_y, "pz": p_nu_z, "M": np.zeros_like(p_nu_x)})

    pt_l = np.array(X['Pt_Lep'])
    eta_l = np.array(X['Eta_Lep'])
    phi_l = np.array(X['Phi_Lep'])
    M = np.array(X['M_Lep'])
    p4Mu = vector.arr({"pt": pt_l, "eta": eta_l, "phi": phi_l, "M": M})

    pt_b = np.array(X['Pt_BJ1'])
    eta_b = np.array(X['Eta_BJ1'])
    phi_b = np.array(X['Phi_BJ1'])
    M = np.array(X['M_BJ1'])
    p4j1 = vector.arr({"pt": pt_b, "eta": eta_b, "phi": phi_b, "M": M})

    pt_q = np.array(X['Pt_LJ'])
    eta_q = np.array(X['Eta_LJ'])
    phi_q = np.array(X['Phi_LJ'])
    M = np.array(X['M_LJ'])
    p4j2 = vector.arr({"pt": pt_q, "eta": eta_q, "phi": phi_q, "M": M})

    p_phi_x = np.array(y['Px_DMGen'])
    p_phi_y = np.array(y['Py_DMGen'])
    p_phi_z = np.array(y['Pz_DMGen'])
    M = 400 * np.ones_like(p_phi_x)
    p4phi = vector.arr({"px": p_phi_x, "py": p_phi_y, "pz": p_phi_z, "M": M})

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