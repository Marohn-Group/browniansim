import numpy as np
from scipy.optimize import least_squares
import scipy.signal

def makey_rd(p,t):
    a0,tau,bias = p
    return a0*np.exp(-t/tau)+bias



def makeJ_rd(p,t):
    a0,tau,bias = p
    E = np.exp(-t/tau)
    J = np.empty((t.size, 3), dtype=float)  # d yhat / d params
    J[:, 0] = E
    J[:, 1] = a0/tau**2*E*t
    J[:, 2] = 1

    return J

def make_fun_jac_rd(t):
    t = np.asarray(t, float)

    def fun_rd(p, y):
        yhat = makey_rd(p,t)
        return y - yhat

    def jac_rd(p, y):
        J = makeJ_rd(p,t)
        return -J  # residual 的导数

    return fun_rd, jac_rd

def makey_rdsq(p,t):
    '''
    This function is used to approx our envelope by sqrt(a0*exp(-2t/tau)+bias), this model is more accurate than the ring down model.
    Consider the x(t) = x0*cos(wt+phi0)*E(t)+x_noise(t),y(t) = x0*cos(wt+phi0)*E(t)+y_noise(t),the envelope is <sqrt(x^2+y^2)> = sqrt(a0*exp(-2t/tau)+bias),
    '''

    a0,tau,bias = p
    return np.sqrt(a0*np.exp(-2*t/tau)+bias)



def makeJ_rdsq(p,t):
    a0,tau,bias = p
    E = np.exp(-2*t/tau)
    Y = np.sqrt(a0*np.exp(-2*t/tau)+bias)
    J = np.empty((t.size, 3), dtype=float)  # d yhat / d params
    J[:, 0] = E/Y/2
    J[:, 1] = a0/tau**2*E*t/Y
    J[:, 2] = 1/Y/2

    return J

def make_fun_jac_rdsq(t):
    t = np.asarray(t, float)

    def fun_rdsq(p, y):
        yhat = makey_rdsq(p,t)
        return y - yhat

    def jac_rdsq(p, y):
        J = makeJ_rdsq(p,t)
        return -J  # residual 的导数

    return fun_rdsq, jac_rdsq


def fit_many_warmstart_rd(Y, t, p0, make_fun_jac, bounds=None, x_scale=None,loss="huber",method = 'dogbox'):
    '''
    This functon is used to approx a list of data with same function form, and the initial parameters for j+1th data is theresult of jth data's fitting result
    
    Parameters:
    Y(list[np.array]): list of 1D data
    t(np.array): time array
    p0(list): the initial guess
    make_fun_jac(function): function that genrate loss function and jacobian function
    bounds(tuple(list,list)): lower bounds and upper bounds of data.
    x_scale(list): scaling factor,usually decided by
    loss(str): the type of loss func for optimizer
    method(str): the type of method for optimizer
    
    Returns:
    P(list[list]):list of fitting results
    costs(list):list of fitting costs
    nfevs(list):list of fitting steps
    '''
    #E 预先计算好，避免exp函数拖慢进度
    fun, jac = make_fun_jac(t)
    n_seg = len(Y)
    P = np.empty((n_seg, 3), float)
    costs = np.empty(n_seg, float)   # 每段残差代价 (linear loss 下 = 0.5*sum(r^2))
    nfevs = np.empty(n_seg, int)     # 每段函数评估次数
    p = np.array(p0, float)

    # 参数量纲不一：用 x_scale 让 xtol 对各分量“相对”一致。
    # 不传则用初值幅度，避免 0 用 1e-12 兜底
    if x_scale is None:
        x_scale = np.maximum(np.abs(p0), 1e-12)

    for i, y in enumerate(Y):
        res = least_squares(
            lambda pp: fun(pp, y),
            x0=p,
            jac=lambda pp: jac(pp, y),
            method=method ,
            bounds=bounds if bounds is not None else (-np.inf, np.inf),
            loss=loss,
            x_scale=x_scale,
            max_nfev=50,    # 小问题别让它跑太久
            xtol=1e-15, ftol=1e-15, gtol=1e-15
        )
        p = res.x
        P[i] = p
        costs[i] = res.cost
        nfevs[i] = res.nfev
    return P, costs, nfevs


def design_lowpass_filter(cutoff, sample_rate, order=5):
    sos = scipy.signal.butter(order, cutoff, btype='low', fs=sample_rate, output='sos')
    return sos

def apply_filter(data, sos):
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def Get_envelope(y_list,dt,cutoff_frequency=500,filter_order=5):
    '''
    This functon is used to approx the envelope for ring down data
    
    Parameters:
    y_list(list[np.array]): list of 1D data
    dt(float): time step
    cutoff_frequency(float): the cutoff frequency of the filter
    filter_order(int): the order of the filter
    
    Returns:
    avg_y(np.array): envelope
    '''    
    # Filter design
    sample_rate = 1/dt  # 200 kHz
    sos = design_lowpass_filter(cutoff_frequency, sample_rate, filter_order)

    filtered_y = []
    for i in range(len(y_list)):
        y_tmp = np.square(y_list[i])
        # Apply filter
        filtered_y.append(apply_filter(y_tmp, sos))

    avg_y = np.mean(filtered_y, axis=0)
    avg_y = np.sqrt(avg_y)
    # cut off first 500 data points (as in LabVIEW fit)

    return avg_y