
"""
Created on Wed Oct  23 15:00:00 2024

@author: lauralund

Post processing RT model
"""

import pandas as pd
import numpy as np
import random
import cplex
from docplex.mp.model import Model
import utilsPPA
from SEMS import get_var_value_from_sol

def RTopt(parameter_dict, hour_of_day, month, season, P_SM_t_opt, S_PPA, profiletype, gen_real, current_SoC, PwMax, EBESS, PbMax, SoCmin, SoCmax, Emax, eta_dis, eta_cha, P_grid_limit, mu, ad):
    """
    Post-processing of deliveries and imbalances 
    """
    T = 1

    # First stage sets
    setT = [i for i in range(T)]
    setSoCT = [i for i in range(T+1)] 
    
    RTmodel = Model('Real time model')


    # First stage variables 
    P_total_RT_t = RTmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='Total hourly bid')
    P_SM_RT_t = RTmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='SM hourly bid') 

    P_PPA_RT_t = RTmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='PPA hourly delivery')
    P_PPA_promised_t = RTmodel.continuous_var_dict(setT, lb=0, ub=(PbMax + PwMax), name='PPA hourly promised delivery')

    P_total_RT_dis_t = RTmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='Total hourly discharge')
    P_SM_RT_dis_t = RTmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='SM hourly discharge')
    P_PPA_RT_dis_t = RTmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='PPA hourly discharge')
    P_cha_RT_t = RTmodel.continuous_var_dict(setT, lb=0, ub=PbMax, name='Hourly charge')

    P_SM_RT_w_t = RTmodel.continuous_var_dict(setT, lb=0, ub=PwMax, name='SM wind hourly')
    P_PPA_RT_w_t = RTmodel.continuous_var_dict(setT, lb=0, ub=PwMax, name='PPA wind hourly')

    E_RT_t   = RTmodel.continuous_var_dict(setSoCT, lb=SoCmin*Emax, ub=SoCmax*Emax, name='Battery energy content')
    SoC_RT_t = RTmodel.continuous_var_dict(setSoCT, lb=SoCmin, ub=SoCmax, name='SoC')
    z_RT_t   = RTmodel.binary_var_dict(setT, name='Cha or Discha')

    SM_imbalance_t = RTmodel.continuous_var_dict(setT, lb=-cplex.infinity, ub=cplex.infinity, name='SM Imbalance')
    PPA_imbalance_t = RTmodel.continuous_var_dict(setT, lb=-cplex.infinity, ub=cplex.infinity, name='PPA Imbalance')
    wind_curtailment_t = RTmodel.continuous_var_dict(setT, lb=0, ub=cplex.infinity, name='Curtailment')

    abs_SM_imbalance_t = RTmodel.continuous_var_dict(setT, lb=0, name="Absolute value of SM imbalance")
    abs_PPA_imbalance_t = RTmodel.continuous_var_dict(setT, lb=0, name="Absolute value of PPA imbalance")

    # First stage constraints
    # PPA delivery
    profiletype.get_constraints_RT(RTmodel, P_PPA_promised_t, S_PPA, hour_of_day, month, season, setT)

    # real time PPA deliveries
    RTmodel.add_constraints(P_PPA_RT_t[t] == (P_PPA_RT_dis_t[t] + P_PPA_RT_w_t[t]) for t in setT) 
    RTmodel.add_constraints(P_PPA_RT_t[t] <= P_PPA_promised_t[t] for t in setT) # cannot overdeliver to PPA

    # real time spot market deliveries
    RTmodel.add_constraints(P_SM_RT_t[t] == (P_SM_RT_dis_t[t] + P_SM_RT_w_t[t]) for t in setT) 

    RTmodel.add_constraints((P_SM_RT_w_t[t] + P_PPA_RT_w_t[t] + P_cha_RT_t[t]) <= gen_real for t in setT) # no strategic imbalance bidding
    RTmodel.add_constraints(P_total_RT_t[t] == P_SM_RT_t[t] + P_PPA_RT_t[t] for t in setT)
   
    RTmodel.add_constraints(P_total_RT_dis_t[t] == P_SM_RT_dis_t[t] + P_PPA_RT_dis_t[t] for t in setT)
    RTmodel.add_constraints(P_total_RT_dis_t[t] <= PbMax * z_RT_t[t] for t in setT)
    RTmodel.add_constraints(P_cha_RT_t[t] <= PbMax * (1-z_RT_t[t]) for t in setT)
    
    # Current battery energy
    RTmodel.add_constraint(E_RT_t[0] == current_SoC * Emax) 
    #RTmodel.add_constraint(RTmodel.sum(P_SM_RT_dis_t[t] + P_PPA_RT_dis_t[t] for t in setT) <= current_EBESS)

    # Battery energy balance for next hour 
    RTmodel.add_constraints(E_RT_t[t+1] == E_RT_t[t] - (P_SM_RT_dis_t[t] +  P_PPA_RT_dis_t[t])/eta_dis + (P_cha_RT_t[t]) * eta_cha for t in setT)
    RTmodel.add_constraints(E_RT_t[t+1] <= SoCmax * Emax for t in setT)
    RTmodel.add_constraints(E_RT_t[t+1] >= SoCmin * Emax for t in setT)


    RTmodel.add_constraints(P_total_RT_t[t] <= P_grid_limit for t in setT)

    # Compute imbalances
    RTmodel.add_constraints(SM_imbalance_t[t] == P_SM_RT_t[t] - P_SM_t_opt for t in setT) # negative is underdelivery
    RTmodel.add_constraints(PPA_imbalance_t[t] == P_PPA_promised_t[t] - P_PPA_RT_t[t] for t in setT) # always positive, cannot overdeliver

    # Curtailment 
    RTmodel.add_constraints(wind_curtailment_t[t] == gen_real - P_SM_RT_w_t[t] - P_PPA_RT_w_t[t] - P_cha_RT_t[t] for t in setT)

    # Constraints for absolute values
    RTmodel.add_constraints(abs_SM_imbalance_t[t] >= SM_imbalance_t[t] for t in setT)
    RTmodel.add_constraints(abs_SM_imbalance_t[t] >= -SM_imbalance_t[t] for t in setT)

    RTmodel.add_constraints(abs_PPA_imbalance_t[t] >= PPA_imbalance_t[t] for t in setT)
    RTmodel.add_constraints(abs_PPA_imbalance_t[t] >= -PPA_imbalance_t[t] for t in setT)

    RTmodel.add_constraints(SoC_RT_t[t] == E_RT_t[t] / Emax for t in setSoCT)

    # Objective function 
    RTmodel.minimize(RTmodel.sum(4 * abs_SM_imbalance_t[t] +  6 * abs_PPA_imbalance_t[t] + wind_curtailment_t[t] for t in setT))

    # Solve model
    RTmodel.solve()

    RTmodel.print_information()
    sol = RTmodel.solve(log_output=False)
    sol_details = RTmodel.get_solve_details()

    if sol:    
        SM_imbalance_t_opt = [sol.get_value(SM_imbalance_t[t]) for t in SM_imbalance_t]
        PPA_imbalance_t_opt = [sol.get_value(PPA_imbalance_t[t]) for t in PPA_imbalance_t]
        abs_SM_imbalance_t_opt = [sol.get_value(abs_SM_imbalance_t[t]) for t in abs_SM_imbalance_t]
        abs_PPA_imbalance_t_opt = [sol.get_value(abs_PPA_imbalance_t[t]) for t in abs_PPA_imbalance_t]
        curtailment_t_opt = [sol.get_value(wind_curtailment_t[t]) for t in wind_curtailment_t]

        P_SM_RT_t_opt = [sol.get_value(P_SM_RT_t[t]) for t in P_SM_RT_t]
        P_PPA_RT_t_opt = [sol.get_value(P_PPA_RT_t[t]) for t in P_PPA_RT_t]
        P_cha_RT_t_opt = [sol.get_value(P_cha_RT_t[t]) for t in P_cha_RT_t]
        P_total_RT_dis_t_opt = [sol.get_value(P_total_RT_dis_t[t]) for t in P_total_RT_dis_t]
        P_SM_RT_dis_t_opt = [sol.get_value(P_SM_RT_dis_t[t]) for t in P_SM_RT_dis_t]
        P_SM_RT_w_t_opt = [sol.get_value(P_SM_RT_w_t[t]) for t in P_SM_RT_w_t]
        P_PPA_RT_w_t_opt = [sol.get_value(P_PPA_RT_w_t[t]) for t in P_PPA_RT_w_t]
        P_PPA_RT_dis_t_opt = [sol.get_value(P_PPA_RT_dis_t[t]) for t in P_PPA_RT_dis_t]
        P_PPA_promised_t_opt = [sol.get_value(P_PPA_promised_t[t]) for t in P_PPA_promised_t]
        
        z_RT_t_opt = [sol.get_value(z_RT_t[t]) for t in z_RT_t]

        # SM_imbalance_t_opt_list = [sol.get_value(SM_imbalance_t[t]) for t in SM_imbalance_t]
        # PPA_imbalance_t_opt_list = [sol.get_value(PPA_imbalance_t[t]) for t in PPA_imbalance_t]
        # curtailment_t_opt_list = [sol.get_value(wind_curtailment_t[t]) for t in wind_curtailment_t]
        # P_PPA_RT_t_opt_list = [sol.get_value(P_PPA_RT_t[t]) for t in P_PPA_RT_t]

        SoC_RT_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(SoC_RT_t), orient='index')
        
        # Update SoC for next hour
        next_SoC = [sol.get_value(SoC_RT_t[t+1]) for t in setT][0]

        obj = sol.get_objective_value() 
    
    else:
       print('No solution found')
       print('Model status:', RTmodel.get_solve_status())

    return next_SoC, SoC_RT_t_opt, obj, SM_imbalance_t_opt, PPA_imbalance_t_opt, abs_SM_imbalance_t_opt, abs_PPA_imbalance_t_opt, curtailment_t_opt, P_SM_RT_t_opt, P_PPA_RT_t_opt, P_cha_RT_t_opt, P_total_RT_dis_t_opt, P_SM_RT_dis_t_opt, P_SM_RT_w_t_opt, P_PPA_RT_w_t_opt, P_PPA_RT_dis_t_opt, P_PPA_promised_t_opt