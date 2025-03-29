from scipy import optimize
import numpy as np
import pandas as pd

def best_response_household_x_z_given_y(y,r_d,pi,i,alpha,beta,gamma,theta):
    x = alpha * ( (theta/(1-theta))*(r_d+pi)+beta/y )**(-1)
    z = gamma * ( (theta/(1-theta))*(r_d-i)+beta/y )**(-1)
    
    return {'x':x, 'y':y, 'z':z}


def best_response_function_to_solve_for_zero(y,r_d,pi,i,alpha,beta,gamma,theta):
    

    x = alpha * ( (theta/(1-theta))*(r_d+pi)+beta/y )**(-1)
    z = gamma * ( (theta/(1-theta))*(r_d-i)+beta/y )**(-1)
    
    return x+y+z-1



def best_response_household(r_d,pi,i,alpha,beta,gamma,theta):
    y_star = optimize.root(best_response_function_to_solve_for_zero, x0=0.5,args=(r_d,pi,i,alpha,beta,gamma,theta)).x[0]
    
    return best_response_household_x_z_given_y(y_star,r_d,pi,i,alpha,beta,gamma,theta)


def derivative_wrt_rd_household(r_d,pi,i,alpha,beta,gamma,theta, h_step=.0001):
    # gradient
    f_plus = pd.Series(best_response_household(r_d+h_step,pi,i,alpha,beta,gamma,theta))
    f_minus = pd.Series(best_response_household(r_d-h_step,pi,i,alpha,beta,gamma,theta))
    
    return (f_plus - f_minus)/(2*h_step)
    


def utility_commercial_bank(r_d,pi,i,alpha,beta,gamma,theta,r_L,sign=-1):
    y_star = optimize.root(best_response_function_to_solve_for_zero, x0=0.5,args=(r_d,pi,i,alpha,beta,gamma,theta)).x[0]

    return sign*(r_L-r_d)*y_star


def best_response_commercial_bank(pi,i,alpha,beta,gamma,theta,r_L=1):
    return optimize.minimize(utility_commercial_bank, 2*pi, args=(pi,i,alpha,beta,gamma,theta,r_L), method='Nelder-Mead', tol=1e-8).x.item()


def commercial_bank_function_to_solve_for_zero(r_d,pi,i,alpha,beta,gamma,theta,r_L=1):
    
    y_star = optimize.root(best_response_function_to_solve_for_zero, x0=0.5,args=(r_d,pi,i,alpha,beta,gamma,theta)).x[0]
    
    partial_y = derivative_wrt_rd_household(r_d,pi,i,alpha,beta,gamma,theta).loc['y']
    
    if abs(partial_y) < 1e-4: partial_y=0

    output = partial_y*(r_L-r_d) - y_star
    
    
    return np.round(output,3)


def optimal_response_commercial_bank(pi,i,alpha,beta,gamma,theta,r_L=1):

    r_d_star = optimize.bisect(commercial_bank_function_to_solve_for_zero, -pi, r_L, args=(pi,i,alpha,beta,gamma,theta,r_L))    
    
    return np.maximum(np.minimum(r_d_star,r_L),-pi)


def utility_central_bank(i,pi,alpha,beta,gamma,theta,r_L_CB=1, r_L=1,sign=-1):
    
    # NC: sign, -1 to minimize the -Utility
    try:
        r_d_star = best_response_commercial_bank(pi=pi,i=i,alpha=alpha,beta=beta,gamma=gamma,theta=theta,r_L=r_L)
        z_star = best_response_household(r_d_star,pi,i,alpha,beta,gamma,theta)['z'].squeeze()        
        result =  sign*(r_L_CB-i)*z_star
        return result
    except Exception as e:
        print(f'{e} with i {i} and pi {pi}')
        
        return 1e6
    


def best_response_central_bank(pi,alpha,beta,gamma,theta,r_L_CB=1,r_L=1):
    return optimize.minimize(utility_central_bank, 2*pi, args=(pi,alpha,beta,gamma,theta,r_L_CB,r_L), method='Nelder-Mead',tol=1e-8).x.item()

def vector_objective_household(x,pi,r_d_observed,S_cash,r_L):
    
    
    alpha = x[0]
    theta = x[1]

    S_deposit = 1-S_cash
    print(f"alpha {alpha}")
    print(f"theta {theta}")
    print(f"pi {pi}")
    print(f"r_d_observed {r_d_observed}")
    print(f"S_cash {S_cash}")
    print(f"S_deposit {S_deposit}")

    r_d = best_response_commercial_bank(pi=pi,i=99,alpha=alpha,beta=1-alpha,gamma=0,theta=theta,r_L=r_L)

    print(f"r_d is {r_d}")
    household_response = best_response_household(r_d=r_d,pi=pi,i=.99,alpha=alpha,beta=1-alpha,gamma=0,theta=theta)
    
    print(f"household_response {household_response}")
    errors = np.array([household_response['x']-S_cash, household_response['y']-S_deposit,r_d-r_d_observed])
    return np.dot(errors,errors)