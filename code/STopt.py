
"""
Created on Wed Oct  23 15:00:00 2024

@author: lauralund

Short term model
"""

import pandas as pd
import numpy as np
import random
import cplex
from docplex.mp.model import Model
import utilsPPA

def get_var_value_from_sol(x, sol, tolerance=1e-10):
    
    y = {}

    for key, var in x.items():
        value = sol.get_var_value(var)
        # Apply tolerance to treat very small values as zero
        y[key] = 0 if abs(value) < tolerance else value

    y = pd.DataFrame.from_dict(y, orient='index', columns=["Value"])
    
    return y

def DAopt(parameter_dict, month, season, S_PPA, PPA_price_t, profiletype, DAscenario, PwMax, EBESS, current_SoC, PbMax, SoCmin, SoCmax, Emax, eta_dis, eta_cha, P_grid_limit, mu, ad, setT):
    """
    Solves model for DA with PPA FHP as input
    """

    # Forecasts based on DAscenario
    DA_gen_forecast =  DAscenario.iloc[:,1] * parameter_dict['wind_capacity']
    DA_SP_forecast = DAscenario.iloc[:,5]
    
    T = 24

    # First stage SoC set
    setSoCT = setT + [setT[-1] + 1]
    T0 = setT[0]

    # PPA violation penalty and preferred SoC for slack variables
    penalty = 1e4
    SoC_penalty = 1e3
    preferred_SoCmin = 0.3 * Emax
    preferred_SoCmax = 0.7 * Emax
    
    DAmodel = Model('Day-Ahead Model')

    # First stage variables 
    P_total_t = DAmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='Total hourly bid')
    P_SM_t = DAmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='SM hourly bid') 
    P_PPA_t = DAmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='PPA hourly delivery')
    
    P_total_dis_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='Total hourly discharge')
    P_SM_dis_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='SM hourly discharge')
    P_PPA_dis_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='PPA hourly discharge')
    P_cha_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='SM hourly charge') # lb and ub? 

    P_SM_w_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PwMax, name='SM wind subhourly')
    P_PPA_w_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PwMax, name='PPA wind subhourly')

    E_t   = DAmodel.continuous_var_dict(setSoCT, lb=SoCmin*Emax, ub=SoCmax*Emax, name='Bat Energy Content')
    z_t   = DAmodel.binary_var_dict(setT, name='Cha or Discha')

    y_t   = DAmodel.binary_var_dict(setT, name='Big M for penalty')

    # battery slack variables
    slack_above = DAmodel.continuous_var_dict(setT, lb=0, name='Slack Above Preferred SoC')
    slack_below = DAmodel.continuous_var_dict(setT, lb=0, name='Slack Below Preferred SoC')

    # constraints
    DAmodel.add_constraints(P_total_t[t] == P_SM_t[t] + P_PPA_t[t] for t in setT)
    DAmodel.add_constraints(P_PPA_t[t] == (P_PPA_dis_t[t] + P_PPA_w_t[t]) for t in setT) # no sm to ppa
    DAmodel.add_constraints(P_SM_t[t] == (P_SM_dis_t[t] + P_SM_w_t[t]) for t in setT) # no sm to ppa
    DAmodel.add_constraints((P_SM_w_t[t] + P_PPA_w_t[t] + P_cha_t[t]) <= DA_gen_forecast[t] for t in setT) # no strategic imbalance bidding

    DAmodel.add_constraints((P_SM_dis_t[t] + P_PPA_dis_t[t]) <= PbMax * z_t[t] for t in setT)
    DAmodel.add_constraints(P_cha_t[t] <= PbMax * (1-z_t[t]) for t in setT)
    
    DAmodel.add_constraint(E_t[T0] == current_SoC*Emax)
    DAmodel.add_constraints(E_t[t+1] ==  E_t[t] + P_cha_t[t] * eta_cha - (P_SM_dis_t[t] +  P_PPA_dis_t[t])/eta_dis for t in setT)
    DAmodel.add_constraints(E_t[t+1] <=  SoCmax * Emax for t in setT)
    DAmodel.add_constraints(E_t[t+1] >= SoCmin * Emax for t in setT)

    # battery slack constraints
    DAmodel.add_constraints(E_t[t+1] >= preferred_SoCmin * Emax - slack_below[t] for t in setT)
    DAmodel.add_constraints(E_t[t+1] <= preferred_SoCmax * Emax + slack_above[t] for t in setT)

    DAmodel.add_constraints(P_total_t[t] <= P_grid_limit for t in setT)

    DAmodel.add_constraints(P_total_dis_t[t] == P_SM_dis_t[t] + P_PPA_dis_t[t] for t in setT)

    # Fixed hourly profile constraint depending on profile type 
    profiletype.get_constraints_ST(DAmodel, P_PPA_t, S_PPA, y_t, month, season, setT)
        
    # Objective function 
    DAmodel.maximize(DAmodel.sum(DA_SP_forecast[t] * P_SM_t[t] 
                    + PPA_price_t * P_PPA_t[t] 
                    - y_t[t] * penalty 
                    - ad * EBESS * mu * (P_SM_dis_t[t] + P_cha_t[t] + P_PPA_dis_t[t]) 
                    - SoC_penalty * (slack_above[t] + slack_below[t]) for t in setT))
    
    # Solve model
    DAmodel.solve()

    DAmodel.print_information()
    sol = DAmodel.solve(log_output=False)
    sol_details = DAmodel.get_solve_details()

    if sol:    
        P_SM_t_opt = get_var_value_from_sol(P_SM_t, sol)   
        # P_SM_t_opt_np = P_SM_t_opt_dict.to_numpy()
        # P_SM_B_t_opt = get_var_value_from_sol(P_SM_B_t, sol)
        P_PPA_t_opt = get_var_value_from_sol(P_PPA_t, sol)
        #P_PPA_t_opt_np = P_PPA_t_opt_dict.to_numpy()

        P_total_dis_t_opt = get_var_value_from_sol(P_total_dis_t, sol)
        P_SM_dis_t_opt = get_var_value_from_sol(P_SM_dis_t, sol)
        P_PPA_dis_t_opt = get_var_value_from_sol(P_PPA_dis_t, sol)
        P_cha_t_opt = get_var_value_from_sol(P_cha_t, sol)

        P_SM_w_t_opt = get_var_value_from_sol(P_SM_w_t, sol)
        P_PPA_w_t_opt = get_var_value_from_sol(P_PPA_w_t, sol)

        E_t_opt = get_var_value_from_sol(E_t, sol)
        z_t_opt = get_var_value_from_sol(z_t, sol)

        y_t_opt = get_var_value_from_sol(y_t, sol)
        y_t_opt_np = y_t_opt.to_numpy()

        obj = sol.get_objective_value() 


    else:
       print('No solution found')
       print('Model status:', DAmodel.get_solve_status())
    
    # print(month)
    # print(f'PPA Split:{S_PPA[month]}')
    # print(f'PPA delivery:{P_PPA_t_opt}')
    # print(f'big M: {M_t_opt_np}')

    return P_SM_t_opt, P_PPA_t_opt, P_total_dis_t_opt, P_SM_dis_t_opt, P_PPA_dis_t_opt, P_cha_t_opt, P_SM_w_t_opt, P_PPA_w_t_opt, E_t_opt, y_t_opt_np
 


def robustDAopt(parameter_dict, month, season, S_PPA, PPA_price_t, profiletype, DAscenario, PwMax, EBESS, current_SoC, PbMax, SoCmin, SoCmax, Emax, eta_dis, eta_cha, P_grid_limit, mu, ad, setT):
    """
    Solves model for DA with PPA FHP as input
    """

    # Forecasts based on DAscenario
    DA_gen_forecast_1 =  DAscenario.iloc[:,1] * parameter_dict['wind_capacity']
    DA_gen_forecast_2 =  DAscenario.iloc[:,2] * parameter_dict['wind_capacity']
    DA_gen_forecast_3 =  DAscenario.iloc[:,3] * parameter_dict['wind_capacity']
    DA_SP_forecast = DAscenario.iloc[:,5]
    
    T = 24

    # Sets
    setSoCT = setT + [setT[-1] + 1]
    T0 = setT[0]

    # PPA violation penalty and preferred SoC for slack variables
    penalty = 1e4
    SoC_penalty = 1e3
    preferred_SoCmin = 0.3 * Emax
    preferred_SoCmax = 0.7 * Emax
    
    DAmodel = Model('Day-Ahead Model')

    # First stage variables 
    P_total_t = DAmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='Total hourly bid')
    P_SM_t = DAmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='SM hourly bid') 
    P_PPA_t = DAmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='PPA hourly delivery')
    
    P_total_dis_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='Total hourly discharge')
    P_SM_dis_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='SM hourly discharge')
    P_PPA_dis_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='PPA hourly discharge')
    P_cha_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='SM hourly charge') # lb and ub? 

    P_SM_w_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PwMax, name='SM wind subhourly')
    P_PPA_w_t = DAmodel.continuous_var_dict(setT, lb=0, ub=PwMax, name='PPA wind subhourly')

    worst_case_forecast = DAmodel.continuous_var_dict(setT, lb=0, ub=PwMax, name='Worst case forecast')
    expected_forecast = DAmodel.continuous_var_dict(setT, lb=0, ub=PwMax, name='Expected forecast')

    E_t   = DAmodel.continuous_var_dict(setSoCT, lb=SoCmin*Emax, ub=SoCmax*Emax, name='Bat Energy Content')
    z_t   = DAmodel.binary_var_dict(setT, name='Cha or Discha')

    y_t   = DAmodel.binary_var_dict(setT, name='Big M for penalty')

    # # battery slack variables
    # slack_above = DAmodel.continuous_var_dict(setT, lb=0, name='Slack Above Preferred SoC')
    # slack_below = DAmodel.continuous_var_dict(setT, lb=0, name='Slack Below Preferred SoC')

    # constraints
    DAmodel.add_constraints(P_total_t[t] == P_SM_t[t] + P_PPA_t[t] for t in setT)
    DAmodel.add_constraints(P_PPA_t[t] == (P_PPA_dis_t[t] + P_PPA_w_t[t]) for t in setT) # no sm to ppa
    DAmodel.add_constraints(P_SM_t[t] == (P_SM_dis_t[t] + P_SM_w_t[t]) for t in setT) # no sm to ppa
    #DAmodel.add_constraints((P_SM_w_t[t] + P_PPA_w_t[t] + P_cha_t[t]) <= DA_gen_forecast[t] for t in setT) # no strategic imbalance bidding
    DAmodel.add_constraints(worst_case_forecast[t] == min(DA_gen_forecast_1[t], DA_gen_forecast_2[t], DA_gen_forecast_3[t]) for t in setT) # worst case forecast
    DAmodel.add_constraints((P_SM_w_t[t] + P_PPA_w_t[t] + P_cha_t[t]) <= worst_case_forecast[t] for t in setT) # no strategic imbalance bidding - worst case
    # DAmodel.add_constraints(expected_forecast[t] == (DA_gen_forecast_1[t] + DA_gen_forecast_2[t] + DA_gen_forecast_3[t])/3 for t in setT) # expected forecast
    # DAmodel.add_constraints((P_SM_w_t[t] + P_PPA_w_t[t] + P_cha_t[t]) <= expected_forecast[t] for t in setT) # no strategic imbalance bidding - expected

    DAmodel.add_constraints((P_SM_dis_t[t] + P_PPA_dis_t[t]) <= PbMax * z_t[t] for t in setT)
    DAmodel.add_constraints(P_cha_t[t] <= PbMax * (1-z_t[t]) for t in setT)
    
    DAmodel.add_constraint(E_t[T0] == current_SoC*Emax)
    DAmodel.add_constraints(E_t[t+1] ==  E_t[t] + P_cha_t[t] * eta_cha - (P_SM_dis_t[t] +  P_PPA_dis_t[t])/eta_dis for t in setT)
    DAmodel.add_constraints(E_t[t+1] <=  SoCmax * Emax for t in setT)
    DAmodel.add_constraints(E_t[t+1] >= SoCmin * Emax for t in setT)

    # battery slack constraints
    # DAmodel.add_constraints(E_t[t+1] >= preferred_SoCmin * Emax - slack_below[t] for t in setT)
    # DAmodel.add_constraints(E_t[t+1] <= preferred_SoCmax * Emax + slack_above[t] for t in setT)

    DAmodel.add_constraints(P_total_t[t] <= P_grid_limit for t in setT)

    DAmodel.add_constraints(P_total_dis_t[t] == P_SM_dis_t[t] + P_PPA_dis_t[t] for t in setT)

    # Fixed hourly profile constraint depending on profile type 
    profiletype.get_constraints_ST(DAmodel, P_PPA_t, S_PPA, y_t, month, season, setT)

    # Objective function 
    DAmodel.maximize(DAmodel.sum(DA_SP_forecast[t] * P_SM_t[t] 
                    + PPA_price_t * P_PPA_t[t] 
                    - y_t[t] * penalty 
                    - ad * EBESS * mu * (P_SM_dis_t[t] + P_cha_t[t] + P_PPA_dis_t[t]) for t in setT))

    # # Objective function 
    # DAmodel.maximize(DAmodel.sum(DA_SP_forecast[t] * P_SM_t[t] 
    #                 + PPA_price_t * P_PPA_t[t] 
    #                 - y_t[t] * penalty 
    #                 - ad * EBESS * mu * (P_SM_dis_t[t] + P_cha_t[t] + P_PPA_dis_t[t]) 
    #                 - SoC_penalty * (slack_above[t] + slack_below[t]) for t in setT))
    
    # Solve model
    DAmodel.solve()

    DAmodel.print_information()
    sol = DAmodel.solve(log_output=False)
    sol_details = DAmodel.get_solve_details()

    if sol:    
        P_SM_t_opt = get_var_value_from_sol(P_SM_t, sol)   
        # P_SM_t_opt_np = P_SM_t_opt_dict.to_numpy()
        # P_SM_B_t_opt = get_var_value_from_sol(P_SM_B_t, sol)
        P_PPA_t_opt = get_var_value_from_sol(P_PPA_t, sol)
        #P_PPA_t_opt_np = P_PPA_t_opt_dict.to_numpy()

        P_total_dis_t_opt = get_var_value_from_sol(P_total_dis_t, sol)
        P_SM_dis_t_opt = get_var_value_from_sol(P_SM_dis_t, sol)
        P_PPA_dis_t_opt = get_var_value_from_sol(P_PPA_dis_t, sol)
        P_cha_t_opt = get_var_value_from_sol(P_cha_t, sol)

        P_SM_w_t_opt = get_var_value_from_sol(P_SM_w_t, sol)
        P_PPA_w_t_opt = get_var_value_from_sol(P_PPA_w_t, sol)

        E_t_opt = get_var_value_from_sol(E_t, sol)
        z_t_opt = get_var_value_from_sol(z_t, sol)

        y_t_opt = get_var_value_from_sol(y_t, sol)
        y_t_opt_np = y_t_opt.to_numpy()

        obj = sol.get_objective_value() 


    else:
       print('No solution found')
       print('Model status:', DAmodel.get_solve_status())
    
    # print(month)
    # print(f'PPA Split:{S_PPA[month]}')
    # print(f'PPA delivery:{P_PPA_t_opt}')
    # print(f'big M: {M_t_opt_np}')

    return P_SM_t_opt, P_PPA_t_opt, P_total_dis_t_opt, P_SM_dis_t_opt, P_PPA_dis_t_opt, P_cha_t_opt, P_SM_w_t_opt, P_PPA_w_t_opt, E_t_opt, y_t_opt_np
 