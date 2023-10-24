import numpy as np

def f_x(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    f = x1**2 + x2**2 + x3**2
    return f

def h_x(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    h1 = x1**2/4 + x2**2/5 + x3**2/25 - 1
    h2 = x1 + x2 - x3
    h = np.array([[h1], [h2]])
    return h

def df_dd(d):
    x3 = d[0]
    df_d = np.array([2*x3])
    return df_d

def Df_Dd(df_d, df_s, dh_s, dh_d):
    Df_d = df_d - df_s @ np.linalg.inv(dh_s) @ dh_d
    return Df_d

def df_ds(s):
    x1 = s[0]
    x2 = s[1]
    df_s = np.array([2*x1, 2*x2])
    return df_s

def dh_dd(d):
    x3 = d[0]
    dh_d = np.array([[2*x3/25], [-1]])
    return dh_d

def dh_ds(s):
    x1 = float(s[0])
    x2 = float(s[1])
    dh_s = np.array([[x1/2, 2*x2/5], [1, 1]])
    return dh_s

def phi(alpha, t, d, s):
    x = np.array([s[0], s[1], d[0]])
    f = f_x(x)
    phi = f - alpha * t * np.linalg.norm(Df_Dd(df_dd(d), df_ds(s), dh_ds(s), dh_dd(d)))
    return phi
def Inexact_Line_Search(d, s, max_iter):
    iter = 0
    alpha = 1
    b = 0.5
    t = 0.3

    d_i = d - alpha*Df_Dd(df_dd(d), df_ds(s), dh_ds(s), dh_dd(d))
    s_i = s + alpha*np.transpose(np.linalg.inv(dh_ds(s))@dh_dd(d)@np.transpose(Df_Dd(df_dd(d), df_ds(s), dh_ds(s), dh_dd(d))))
    XN = np.array([s_i[0], s_i[1], d_i[0]])
    f_alpha = f_x(XN)
    phi_alpha = phi(alpha, t, d, s)

    while f_alpha > phi_alpha and iter <= max_iter:
        if iter == max_iter:
            print('Max line search iterations hit')

        alpha = b * alpha

        d_i = d_i - alpha * Df_Dd(df_dd(d_i), df_ds(s_i), dh_ds(s_i), dh_dd(d_i))
        s_i = s_i + alpha * np.transpose(np.linalg.inv(dh_ds(s_i)) @ dh_dd(d_i) @ np.transpose(Df_Dd(df_dd(d_i), df_ds(s_i), dh_ds(s_i), dh_dd(d_i))))
        XN = np.array([s_i[0], s_i[1], d_i[0]])
        f_alpha = f_x(XN)
        phi_alpha = phi(alpha, t, d, s)

        iter += 1

    return alpha

def solve(d, s, max_j):
    h_p1 = h_x(np.array([s[0], s[1], d[0]]))
    j = 0

    while np.linalg.norm(h_p1) > eps and j <= max_j:
        if j == max_j:
            print('Max Newton-Ralphson iterations hit')

        q = np.linalg.inv(dh_ds(s))@h_x(np.array([float(s[0]), float(s[1]), float(d[0])]))
        s_p1_1 = s[0] - q[0]
        s_p1_2 = s[1] - q[1]
        s_p1 = np.array([float(s_p1_1), float(s_p1_2)])
        h_p1 = h_x(np.array([float(s[0]), float(s[1]), float(d[0])])) + dh_ds(s)@np.transpose(s_p1 - s)
        s = s_p1

        j += 1

    return s


# let s = [x1, x2]
# let d = [x3]
d = np.array([0])
s = np.array([-1, -1])
k = 0
iter = 0
max_k = 2000
max_iter = 100
max_j = 100
eps = 1e-3
s = solve(d, s, max_j) # return feasible state variables
x0 = np.array([float(s[0]), float(s[1]), float(d[0])]) # satisfies h(x0) = 0
print('\ninitial h1 =', float(h_x(x0)[0]), '\ninitial h2 =', float(h_x(x0)[1]), '\n')
s = [x0[0], x0[1]] # grab state vars
d = [x0[2]] # grab decision var(s)
Df_Dd_kp1 = Df_Dd(df_dd(d), df_ds(s), dh_ds(s), dh_dd(d))
while np.linalg.norm(Df_Dd_kp1) > eps and k <= max_k:
    if k == max_k:
        print('Max reduced gradient iterations hit -', max_k, 'iterations \n')
    alpha = Inexact_Line_Search(d, s, max_iter)
    # alpha = 0.1
    d = d - alpha * Df_Dd(df_dd(d), df_ds(s), dh_ds(s), dh_dd(d))
    s_o = s + alpha * np.transpose(np.linalg.inv(dh_ds(s)) @ dh_dd(d) @ np.transpose(Df_Dd(df_dd(d), df_ds(s), dh_ds(s), dh_dd(d))))
    s = solve(d, s_o, max_j)
    Df_Dd_kp1 = Df_Dd(df_dd(d), df_ds(s), dh_ds(s), dh_dd(d))
    k += 1
x = np.array([float(s[0]), float(s[1]), float(d[0])])
print('x1 =', x[0], '\nx2 =', x[1], '\nx3 =', x[2], '\n')
print('reduced gradient =', float(Df_Dd_kp1))
print('h1 =', float(h_x(x)[0]), '\nh2 =', float(h_x(x)[1]), '\n')
print('Value of f:', f_x(x))