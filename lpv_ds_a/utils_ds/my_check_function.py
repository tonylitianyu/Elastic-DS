import numpy as np


def my_check_function(Data, P):
    d = int(len(Data) / 2)
    x = Data[:d, :]
    xd = Data[d:, :]
    return object_function(P, x, xd, 0.0001)



def object_function(P, x, xd, w):
    J_total = 0
    for i in np.arange(len(x[0])):
        dlyap_dx, dlyap_dt = compute_Energy_Single(x[:, i], xd[:, i], P)
        norm_vx = np.linalg.norm(dlyap_dx, 2)
        norm_xd = np.linalg.norm(xd[:, i], 2)
        if norm_xd == 0 or norm_vx == 0:
            J = 0
        else:
            J = dlyap_dt / (norm_vx * norm_xd)
        J_total += (1 + w) / 2 * J**2 * np.sign(J) + (1 - w) / 2 * J**2

    return J_total


def compute_Energy_Single(x, xd, p):
    # lyap resp to x (P + P.T) @ X : shape: 3
    if len(x) == 3:
        dlyap_dx_1 = 2 * (p[0] * x[0] + p[1] * x[1] + p[2] * x[2])
        dlyap_dx_2 = 2 * (p[1] * x[0] + p[3] * x[1] + p[4] * x[2])
        dlyap_dx_3 = 2 * (p[2] * x[0] + p[4] * x[1] + p[5] * x[2])
        # lyap resp to y
        v_dot = xd[0] * dlyap_dx_1 + xd[1] * dlyap_dx_2 + xd[2] * dlyap_dx_3
        # derivative of x
        dv = np.array([dlyap_dx_1, dlyap_dx_2, dlyap_dx_3])
    else:
        dlyap_dx_1 = 2 * (p[0] * x[0] + p[1] * x[1])
        dlyap_dx_2 = 2 * (p[1] * x[0] + p[2] * x[1])
        v_dot = xd[0] * dlyap_dx_1 + xd[1] * dlyap_dx_2
        dv = [dlyap_dx_1, dlyap_dx_2]
    return dv, v_dot
