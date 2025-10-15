
"""
Created on Fri Nov  1 15:00:00 2024

@author: lauralund

Long term model
"""

import pandas as pd
import numpy as np
import random
import cplex
from docplex.mp.model import Model
#from SEMS import get_var_value_from_sol
import utilsPPA
import deliveryprofiles

def LTopt_FHP(PPA_price, scenariosLT, equiprobIn, profiletype, parameter_dict, simulation_dict, PwMax, EBESS, PbMax, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, P_grid_limit, mu, beta, alpha=0.8):
    """
    Solves model for PPA optimization
    """

    # Scenarios
    T = scenariosLT[0].shape[0]
    n_scens = len(scenariosLT) # check if this is correct

    gen_scenarios =  np.array([scenario['Realised Generation'].values for scenario in scenariosLT]) * parameter_dict['wind_capacity']
    SP_scenarios = np.array([scenario['Realised Spot Price'].values for scenario in scenariosLT])

    # Sets
    setT = [i for i in range(T)] # hours
    setSoCT = [i for i in range(T+1)] # hours + 1 for initial SoC
    setS = [i for i in range(n_scens)] # scenarios
    setTS = [(i,j) for i in setS for j in setT] # hours and scenarios
    setSoCTS = [(i,j) for i in setS for j in setSoCT] # hours and scenarios

    # Monthly settlement period sets 
    hours_in_a_year = 8736
    hours_in_a_day = 24
    months_in_a_year = 12
    days_in_a_month = [30, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # january set to 30 days to equal 8736 hours

    monthly_hours = [[] for _ in range(months_in_a_year)]  # set of sets of monthly hours

    hour_index = 0  

    for month, days in enumerate(days_in_a_month):
        for day in range(days):
            for hour in range(hours_in_a_day):
                monthly_hours[month].append(hour_index)
                hour_index += 1

    # Seasonal settlement period sets 
    seasons = [
        monthly_hours[11] + monthly_hours[0] + monthly_hours[1],  # winter
        monthly_hours[2] + monthly_hours[3] + monthly_hours[4],   # spring
        monthly_hours[5] + monthly_hours[6] + monthly_hours[7],   # summer
        monthly_hours[8] + monthly_hours[9] + monthly_hours[10]   # fall
    ] 

    # Battery degradation parameters
    degradation_cost_per_mwh = parameter_dict['degradation_cost_per_mwh']

    # PPA price adjustment for pricing scheme
    if len(PPA_price) == 12:
        # Monthly settlement: Repeat PPA_price for each month
        PPA_price_t = np.concatenate([
            np.full(len(monthly_hours[month]), PPA_price[month]) for month in range(len(PPA_price))])

    elif len(PPA_price) == 4:
        # Seasonal settlement: Repeat PPA_price for each season
        PPA_price_t = np.concatenate([
            np.full(len(seasons[season]), PPA_price[season]) for season in range(len(PPA_price))])

    elif len(PPA_price) == 1:
        # Annual settlement: Repeat the same price for the whole year
        PPA_price_t = np.full(hours_in_a_year, PPA_price[0])


    LT_model = Model('LT Model')

    # CVaR variables 
    zeta = LT_model.continuous_var(lb=-cplex.infinity, ub=cplex.infinity, name='zeta') 
    eta_s = LT_model.continuous_var_dict(setS, lb=0, ub=cplex.infinity, name='eta') 

    # Variables 
    P_total_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=(PbMax + PwMax), name='Total hourly bid')
    P_SM_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=(PbMax + PwMax), name='SM hourly bid') # upper bound !!
    P_PPA_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=(PbMax + PwMax), name='PPA hourly delivery')

    P_SM_dis_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PbMax, name='SM hourly discharge')
    P_PPA_dis_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PbMax, name='PPA hourly discharge')
    P_cha_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PbMax, name='SM hourly charge') # lb and ub? 

    P_SM_w_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PwMax, name='SM wind subhourly')
    P_PPA_w_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PwMax, name='PPA wind subhourly')

    E_t_s   = LT_model.continuous_var_dict(setSoCTS, lb=0, ub=Emax, name='SoC') # :)
    z_t_s   = LT_model.binary_var_dict(setTS, name='Cha or Discha')

    # Battery variables
    #SoH_t = LT_model.continuous_var_dict(setSoCT, lb=0.7, ub=1, name='Battery State of Health')
    cycles_s = LT_model.continuous_var_dict(setS, lb=0, ub=cplex.infinity, name='Battery Cycles')
    throughput_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=cplex.infinity, name='Battery energy throughput')
    #battery_replacement_t = LT_model.binary_var_dict(setT, name='Battery Replacement')

    # Constraints
    LT_model.add_constraints(P_total_t_s[s,t] == P_SM_t_s[s,t] + P_PPA_t_s[s,t] for s in setS for t in setT)
    LT_model.add_constraints(P_PPA_t_s[s,t] == (P_PPA_dis_t_s[s,t] + P_PPA_w_t_s[s,t]) for s in setS for t in setT) # no sm to ppa 
    LT_model.add_constraints(P_SM_t_s[s,t] == (P_SM_dis_t_s[s,t] + P_SM_w_t_s[s,t]) for s in setS for t in setT) # no sm to ppa     
    LT_model.add_constraints(P_SM_w_t_s[s,t] + P_PPA_w_t_s[s,t] + P_cha_t_s[s,t] <= gen_scenarios[s,t] for s in setS for t in setT) 
    LT_model.add_constraints((P_SM_dis_t_s[s,t] + P_PPA_dis_t_s[s,t]) <= PbMax * z_t_s[s,t] for s in setS for t in setT)
    LT_model.add_constraints(P_cha_t_s[s,t] <= PbMax * (1-z_t_s[s,t]) for s in setS for t in setT)

    LT_model.add_constraints(P_total_t_s[s,t] <= P_grid_limit for s in setS for t in setT)

    # Battery constraints
    LT_model.add_constraints(E_t_s[s,0] == 0.5 * Emax for s in setS)
    LT_model.add_constraints(E_t_s[s,t+1] == E_t_s[s,t] - (P_SM_dis_t_s[s,t] +  P_PPA_dis_t_s[s,t])/eta_dis + (P_cha_t_s[s,t]) * eta_cha for s in setS for t in setT)
    LT_model.add_constraints(E_t_s[s,t+1] <= SoCmax * Emax for s in setS for t in setT)
    LT_model.add_constraints(E_t_s[s,t+1] >= SoCmin * Emax for s in setS for t in setT)

    LT_model.add_constraints(throughput_t_s[s,t] == P_SM_dis_t_s[s,t] + P_PPA_dis_t_s[s,t] + P_cha_t_s[s,t] for s in setS for t in setT)
    LT_model.add_constraints(cycles_s[s] == LT_model.sum(throughput_t_s[s,t] / (2 * EBESS) for s in setS for t in setT) for s in setS)
    #LT_model.add_constraints(SoH_t[t+1] == SoH_t[t] - degradation_per_cycle * (throughput_t[t]) / EBESS + LT_model.sum(0.3 * battery_replacement_t[t] for t in setT) for t in setT)
    #LT_model.add_constraints(SoH_t[t+1] == SoH_t[t] - degradation_per_cycle * (throughput_t[t]) / EBESS + LT_model.sum(0.3 * battery_replacement_t[t_replace] for t_replace in setT if t_replace >= t) for t in setT)
    #LT_model.add_constraint(cycles <= total_cycles_per_bat + total_cycles_per_bat * LT_model.sum(battery_replacement_t[t] for t in setT))

    # Delivery profile dependent constraints 
    S_PPA = profiletype.get_constraints_LT(LT_model, P_PPA_t_s, n_scens, seasons, monthly_hours, hours_in_a_day, months_in_a_year)

    # Risk adversity constraint
    # LT_model.add_constraints(- LT_model.sum(SP_scenarios[s,t] * P_SM_t_s[s,t] + PPA_price_t[t] * P_PPA_t_s[s,t] 
    #                         -  degradation_cost_per_mwh * throughput_t_s[s,t] for t in setT)
    #                         + zeta - eta_s[s] <= 0 for s in setS)
    
    LT_model.add_constraints(LT_model.sum(SP_scenarios[s,t] * P_SM_t_s[s,t] + PPA_price_t[t] * P_PPA_t_s[s,t] 
                            - degradation_cost_per_mwh * throughput_t_s[s,t] for t in setT)
                            - zeta + eta_s[s] <= 0 for s in setS)

    # Objective function 
    LT_model.maximize(
        (1-beta)*(LT_model.sum(equiprobIn * SP_scenarios[s,t] * P_SM_t_s[s,t] for s in setS for t in setT)
        + LT_model.sum(PPA_price_t[t] * P_PPA_t_s[s,t] for s in setS for t in setT)
        -  LT_model.sum(degradation_cost_per_mwh * throughput_t_s[s,t] for s in setS for t in setT))
        + beta * (zeta - (1/(1-alpha)) * LT_model.sum(equiprobIn * eta_s[s] for s in setS))
        )
   
    # Solve model
    LT_model.parameters.mip.tolerances.mipgap = 0.01

    LT_model.solve(log_output=True)

    LT_model.print_information()
    sol = LT_model.solution
    sol_details = LT_model.get_solve_details()
    
    if sol:    
        P_SM_t_opt = get_var_value_from_sol(P_SM_t_s, sol)   
        P_PPA_t_opt = get_var_value_from_sol(P_PPA_t_s, sol)

        P_SM_dis_t_opt = get_var_value_from_sol(P_SM_dis_t_s, sol)
        P_PPA_dis_t_opt = get_var_value_from_sol(P_PPA_dis_t_s, sol)
        P_cha_t_opt = get_var_value_from_sol(P_cha_t_s, sol)

        P_SM_w_t_opt = get_var_value_from_sol(P_SM_w_t_s, sol)
        P_PPA_w_t_opt = get_var_value_from_sol(P_PPA_w_t_s, sol)

        E_t_opt = get_var_value_from_sol(E_t_s, sol)
        z_t_opt = get_var_value_from_sol(z_t_s, sol)

        if isinstance(S_PPA, dict):  # If S_PPA is a dictionary of variables
            S_PPA_opt = get_var_value_from_sol(S_PPA, sol)
        else:  # If S_PPA is a single variable
            S_PPA_opt = pd.DataFrame([sol.get_var_value(S_PPA)], columns=['PPA share'])

        cycles_opt = get_var_value_from_sol(cycles_s,sol)

        zeta_opt = sol.get_var_value(zeta)
        eta_s_opt = {s: sol.get_var_value(eta_s[s]) for s in setS}  # need to keep it as a dict, as eta_s is defined as a dict
        
        obj = sol.get_objective_value() 
        
    else:
       print('No solution found')
       print('Model status:', LT_model.get_solve_status())

    print(f"zeta_opt: {zeta_opt}")
    print(f"eta_s_opt: {eta_s_opt}")

    CVaR = zeta_opt - (1/(1-alpha)) * LT_model.sum(equiprobIn * eta_s_opt[s] for s in setS)     

    return cycles_opt, S_PPA_opt, P_SM_t_opt, P_PPA_t_opt, P_cha_t_opt, P_SM_dis_t_opt, P_PPA_dis_t_opt, P_SM_w_t_opt, P_PPA_w_t_opt, E_t_opt, z_t_opt, obj, CVaR


# def LTopt_PAP(PPA_price, scenariosLT, equiprobIn, parameter_dict, simulation_dict, PwMax, EBESS, PbMax, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, P_grid_limit, beta, alpha=0.6):
#     """
#     Solves model for PPA optimization
#     """

#     # Scenarios
#     #scenarios, scenariosOut, equiprob, equiprobOut = utilsPPA.generate_scenarios_LT(8, 42)
#     T = scenariosLT[0].shape[0]
#     n_scens = len(scenariosLT) # check if this is correct

#     realised_gen =  np.array([scenario['Realised Generation'].values for scenario in scenariosLT]) * parameter_dict['wind_capacity']
#     gen_scenarios =  np.array([scenario['Generation Forecast'].values for scenario in scenariosLT]) * parameter_dict['wind_capacity']
#     SP_scenarios = np.array([scenario['Spot Price Forecast'].values for scenario in scenariosLT])
#     #system_signal_scenarios = np.array([scenario['System Signal'].values for scenario in scenariosLT])

#     # if all weather years must bid same SM amount
#     scenario_groups = {
#         1: [0,1,2],
#         2: [3,4,5],
#         3: [6,7,8],
#         4: [9,10,11],
#         5: [12,13,14],
#         6: [15,16,17],
#         7: [18,19,20],
#         8: [21,22,23],
#     }

#     # Fetch PPA price
#     #_, capture_factor = profiletype.getPricePremiums(SP_scenarios)
#     #GO_per_MWh = 3
#     #PPA_price = np.mean(SP_scenarios) * capture_factor + GO_per_MWh
#     #PPA_price = 55

#     # Sets
#     setT = [i for i in range(T)] # hours
#     setSoCT = [i for i in range(T+1)] # hours + 1 for initial SoC
#     setS = [i for i in range(n_scens)] # scenarios
#     setTS = [(i,j) for i in setS for j in setT] # hours and scenarios
#     setSoCTS = [(i,j) for i in setS for j in setSoCT] # hours and scenarios

#     # Monthly settlement period sets 
#     hours_in_a_day = 24
#     months_in_a_year = 12
#     days_in_a_month = [30, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # january set to 30 days to equal 8736 hours

#     monthly_hours = [[] for _ in range(months_in_a_year)]  # set of sets of monthly hours

#     hour_index = 0  

#     for month, days in enumerate(days_in_a_month):
#         for day in range(days):
#             for hour in range(hours_in_a_day):
#                 monthly_hours[month].append(hour_index)
#                 hour_index += 1

#     # Seasonal settlement period sets 
#     seasons = [
#         monthly_hours[11] + monthly_hours[0] + monthly_hours[1],  # winter
#         monthly_hours[2] + monthly_hours[3] + monthly_hours[4],   # spring
#         monthly_hours[5] + monthly_hours[6] + monthly_hours[7],   # summer
#         monthly_hours[8] + monthly_hours[9] + monthly_hours[10]   # fall
#     ] 

#     # Battery degradation parameters
#     degradation_per_cycle = 0.02 / 1000
#     degradation_cost_per_mwh = parameter_dict['degradation_cost_per_mwh']
#     #total_cycles_per_bat = parameter_dict['Total cycles per battery']

#     LT_model = Model('LT Model')

#     # CVaR variables 
#     zeta = LT_model.continuous_var(lb=-cplex.infinity, ub=cplex.infinity, name='zeta') 
#     eta_s = LT_model.continuous_var_dict(setS, lb=0, ub=cplex.infinity, name='eta') 

#     # Variables 
#     P_total_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=(PbMax + PwMax), name='Total hourly bid')
#     P_SM_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=(PbMax + PwMax), name='SM hourly bid') # upper bound !!
#     #P_PPA_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=(PbMax + PwMax), name='PPA hourly delivery')

#     P_SM_dis_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PbMax, name='SM hourly discharge')
#     #P_PPA_dis_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PbMax, name='PPA hourly discharge')
#     P_cha_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PbMax, name='SM hourly charge') # lb and ub? 

#     P_SM_w_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PwMax, name='SM wind subhourly')
#     P_PPA_w_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=PwMax, name='PPA wind subhourly')

#     E_t_s   = LT_model.continuous_var_dict(setSoCTS, lb=-cplex.infinity, ub=cplex.infinity, name='SoC')
#     z_t_s   = LT_model.binary_var_dict(setTS, name='Cha or Discha')

#     S_PPA = LT_model.continuous_var(lb=0, ub=1, name='PAP share for PPA')
    
#     # Battery variables
#     #SoH_t = LT_model.continuous_var_dict(setSoCT, lb=0.7, ub=1, name='Battery State of Health')
#     cycles_s = LT_model.continuous_var_dict(setS, lb=0, ub=cplex.infinity, name='Battery Cycles')
#     throughput_t_s = LT_model.continuous_var_dict(setTS, lb=0, ub=cplex.infinity, name='Battery energy throughput')
#     #battery_replacement_t = LT_model.binary_var_dict(setT, name='Battery Replacement')

#     # Constraints
#     LT_model.add_constraints(P_total_t_s[s,t] == P_SM_t_s[s,t] + P_PPA_w_t_s[s,t] for s in setS for t in setT)
#     #LT_model.add_constraints(P_PPA_w_t_s[s,t] == (P_PPA_dis_t_s[s,t] + P_PPA_w_t_s[s,t]) for s in setS for t in setT) # no sm to ppa 
#     LT_model.add_constraints(P_SM_t_s[s,t] == (P_SM_dis_t_s[s,t] + P_SM_w_t_s[s,t]) for s in setS for t in setT) # no sm to ppa 
#     #LT_model.add_constraints(P_SM_B_t[t] == P_SM_dis_t[t] for t in setT)
    
#     # Max delivery  constraints 
#     LT_model.add_constraints(P_PPA_w_t_s[s,t] == gen_scenarios[s,t] * S_PPA for t in setT for s in setS)
#     #below should be redundant 
#     LT_model.add_constraints((P_SM_w_t_s[s,t] + P_cha_t_s[s,t] <= gen_scenarios[s,t] * (1-S_PPA)) for t in setT for s in setS)
#     LT_model.add_constraints(P_SM_w_t_s[s,t] + P_PPA_w_t_s[s,t] + P_cha_t_s[s,t] <= gen_scenarios[s,t] for s in setS for t in setT) 

#     LT_model.add_constraints(P_SM_dis_t_s[s,t] <= PbMax * z_t_s[s,t] for s in setS for t in setT)
#     LT_model.add_constraints(P_cha_t_s[s,t] <= PbMax * (1-z_t_s[s,t]) for s in setS for t in setT)

#     LT_model.add_constraints(P_total_t_s[s,t] <= P_grid_limit for s in setS for t in setT)

#     # Battery constraints
#     LT_model.add_constraints(E_t_s[s,t+1] == E_t_s[s,t] - (P_SM_dis_t_s[s,t])/eta_dis + (P_cha_t_s[s,t]) * eta_cha for s in setS for t in setT)
#     LT_model.add_constraints(E_t_s[s,t+1] <= SoCmax * Emax for s in setS for t in setT)
#     LT_model.add_constraints(E_t_s[s,t+1] >= SoCmin * Emax for s in setS for t in setT)

#     LT_model.add_constraints(throughput_t_s[s,t] == P_SM_dis_t_s[s,t] + P_cha_t_s[s,t] for s in setS for t in setT)
#     LT_model.add_constraints(cycles_s[s] == LT_model.sum(throughput_t_s[s,t] / (2 * EBESS) for s in setS for t in setT) for s in setS)

#     # Non-anticipativity constraints - same weather years must bid same amount in spot market
#     # for group_id, scenario_list in scenario_groups.items():
#     #     reference_scenario = scenario_list[0]  # Reference scenario in the group
#     #     for scenario in scenario_list[1:]:
#     #         LT_model.add_constraints(
#     #             P_SM_t_s[reference_scenario, t] == P_SM_t_s[scenario, t]
#     #             for t in setT)
            
#     # Risk adversity constraint 
#     # LT_model.add_constraints(- LT_model.sum(SP_scenarios[s,t] * P_SM_t_s[s,t] + PPA_price * P_PPA_w_t_s[s,t] 
#     #                          -  degradation_cost_per_mwh * throughput_t_s[s,t] for t in setT)
#     #                         + zeta - eta_s[s] <= 0 for s in setS)

#     # Gain constraint 
#     LT_model.add_constraints(LT_model.sum(SP_scenarios[s,t] * P_SM_t_s[s,t] + PPA_price * P_PPA_w_t_s[s,t] 
#                              - degradation_cost_per_mwh * throughput_t_s[s,t] for t in setT)
#                              - zeta + eta_s[s] >= 0 for s in setS
#     )

#     # Objective function 
#     LT_model.maximize(
#         (1-beta)*(LT_model.sum(equiprobIn * SP_scenarios[s,t] * P_SM_t_s[s,t] for s in setS for t in setT)
#         + LT_model.sum(PPA_price * P_PPA_w_t_s[s,t] for s in setS for t in setT)
#         #-  LT_model.sum(degradation_cost_per_mwh * throughput_t_s[s,t] for s in setS for t in setT)
#         + beta * (zeta - (1/(1-alpha)) * LT_model.sum(equiprobIn * eta_s[s] for s in setS))
#         ))

#     # Solve model
#     LT_model.parameters.mip.tolerances.mipgap = 0.02

#     LT_model.solve(log_output=True)

#     LT_model.print_information()
#     sol = LT_model.solution
#     sol_details = LT_model.get_solve_details()
    
#     if sol:    
#         P_SM_t_opt = get_var_value_from_sol(P_SM_t_s, sol)   
#         #P_SM_B_t_opt = get_var_value_from_sol(P_SM_B_t, sol)
#         P_PPA_w_t_opt = get_var_value_from_sol(P_PPA_w_t_s, sol)

#         P_SM_dis_t_opt = get_var_value_from_sol(P_SM_dis_t_s, sol)
#         P_cha_t_opt = get_var_value_from_sol(P_cha_t_s, sol)

#         P_SM_w_t_opt = get_var_value_from_sol(P_SM_w_t_s, sol)

#         E_t_opt = get_var_value_from_sol(E_t_s, sol)
#         z_t_opt = get_var_value_from_sol(z_t_s, sol)

#         S_PPA_opt = sol.get_var_value(S_PPA)

#         cycles_opt = get_var_value_from_sol(cycles_s,sol)

#         # battery_replacement_t_opt = get_var_value_from_sol(battery_replacement_t, sol)
#         # battery_replacement_t_opt_np = battery_replacement_t_opt.to_numpy().flatten()
#         # batteries_used = 1 + sum(battery_replacement_t_opt_np) 

#         zeta_opt = sol.get_var_value(zeta)
#         eta_s_opt = {s: sol.get_var_value(eta_s[s]) for s in setS}  # need to keep it as a dict, as eta_s is defined as a dict
        
#         obj = sol.get_objective_value() 
        
#     else:
#        print('No solution found')
#        print('Model status:', LT_model.get_solve_status())

#     print(f"zeta_opt: {zeta_opt}")
#     print(f"eta_s_opt: {eta_s_opt}")

#     CVaR = zeta_opt - (1/(1-alpha)) * LT_model.sum(equiprobIn * eta_s_opt[s] for s in setS)

#     # exp_profit = (sum(equiprob * SP_scenarios[t,s] * P_SM_t_opt[t] for s in setS for t in setT) 
#     #     + sum(PPA_price * P_PPA_t_opt[t] for t in setT) 
#     #     - sum(ad * EBESS * mu * (P_SM_dis_t_opt[t] + P_SM_cha_t_opt[t] + P_PPA_dis_t_opt[t]) for t in setT)
#     # )            

#     return cycles_opt, S_PPA_opt, P_SM_t_opt, P_PPA_w_t_opt, P_cha_t_opt, P_SM_dis_t_opt, P_SM_w_t_opt, E_t_opt, z_t_opt, obj, CVaR


def get_var_value_from_sol(x, sol, tolerance=1e-10):
    
    y = {}

    for key, var in x.items():
        value = sol.get_var_value(var)
        # Apply tolerance to treat very small values as zero
        y[key] = 0 if abs(value) < tolerance else value

    y = pd.DataFrame.from_dict(y, orient='index', columns=["Value"])
    
    return y





