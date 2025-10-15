import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import os
import math
from docplex.mp.model import Model
import tqdm
import random 
import cplex
import numpy_financial as npf
from tqdm import tqdm
import rainflow
import scipy.stats as stats
import ast

import STopt 
import LTopt
import RTopt
import Deg_Calculation_appro as DegCal

def generate_scenarios(seed):
    # Load data
    nineteen_eightyfive = pd.read_csv('../data/1985data.csv')
    twenty_twelve = pd.read_csv('../data/2012data_forstochastic.csv')
    twenty_fourteen = pd.read_csv('../data/2014data.csv')

    # Generate DA scenario using only 2012 data
    np.random.seed(seed)
    spotprice_scenarios_2012 = twenty_twelve.iloc[:, 1:4]  # Assuming columns 1-3 are spot price scenarios
    forecasts_2012 = twenty_twelve.columns[4:7]  # Generation forecast
    prod_2012 = twenty_twelve.columns[7]  # Realized generation

    scenario_for_DA_model_2025 = pd.concat(
        [spotprice_scenarios_2012.iloc[:, 0], twenty_twelve[forecasts_2012], twenty_twelve[prod_2012]], 
        axis=1
    )

    scenario_for_DA_model_2035 = pd.concat(
        [spotprice_scenarios_2012.iloc[:, 1], twenty_twelve[forecasts_2012], twenty_twelve[prod_2012]], 
        axis=1
    )

    scenario_for_DA_model_2045 = pd.concat(
        [spotprice_scenarios_2012.iloc[:, 2], twenty_twelve[forecasts_2012], twenty_twelve[prod_2012]], 
        axis=1
    )

    # Extend each scenario to 10 years (assuming yearly data is provided)
    scenario_for_DA_model_2025 = pd.concat([scenario_for_DA_model_2025] * 10, ignore_index=True)
    scenario_for_DA_model_2035 = pd.concat([scenario_for_DA_model_2035] * 10, ignore_index=True)
    scenario_for_DA_model_2045 = pd.concat([scenario_for_DA_model_2045] * 10, ignore_index=True)

    # Stack all scenarios
    scenario_for_DA_model = pd.DataFrame(
        np.vstack([scenario_for_DA_model_2025.values, scenario_for_DA_model_2035.values, scenario_for_DA_model_2045.values]),
        columns=['Realised Spot Price', 'DA Generation Forecast 1', 'DA Generation Forecast 2', 'DA Generation Forecast 3', 'Realised Generation']
    )

    # Add "Spot Price Forecast" column with Gaussian errors
    normalised_errors_2012 = np.random.normal(0.02, 0.3, size=len(scenario_for_DA_model))
    errors_2012 = normalised_errors_2012 * scenario_for_DA_model['Realised Spot Price']
    scenario_for_DA_model['Spot Price Forecast'] = np.maximum(
        scenario_for_DA_model['Realised Spot Price'] + errors_2012, 0
    )

    # Add HA price and generation forecasts to DA scenario
    twenty_twelve_HA_data = pd.read_csv('../data/2012_HAdata.csv')
    HA_data_repeated = pd.concat([twenty_twelve_HA_data] * 30, ignore_index=True)
    DAscenario = pd.concat([scenario_for_DA_model, HA_data_repeated], axis=1)
    DAscenario.columns = [
        'Realised Spot Price', 'DA Generation Forecast 1',  'DA Generation Forecast 2', 'DA Generation Forecast 3', 'Realised Generation', 'Spot Price Forecast',
        'HA Generation Forecast', 'Realised Balancing Price', 'System Signal'
    ]

    # Generate LT scenarios using only 1985 and 2014 data
    weather_years_LT = [nineteen_eightyfive, twenty_fourteen]
    scenariosLT = []

    for y in weather_years_LT:
        spotprice_scenarios = y.iloc[:, 1:4]  # Assuming columns 1-3 are spot price scenarios
        forecast = y.columns[4]  # Generation forecast
        prod = y.columns[5]  # Realized generation

        for spotprices in spotprice_scenarios.columns:
            scenario = pd.concat([spotprice_scenarios[spotprices], y[forecast], y[prod]], axis=1)
            scenario.columns = ['Realised Spot Price', 'Generation Forecast', 'Realised Generation']

            # Add "Spot Price Forecast" column with Gaussian errors
            normalised_errors = np.random.normal(0.01, 0.58, size=len(scenario))
            errors = normalised_errors * scenario['Realised Spot Price']
            scenario['Spot Price Forecast'] = np.maximum(
                scenario['Realised Spot Price'] + errors, 0
            )

            scenariosLT.append(scenario)

    # Assign probabilities
    equiprobLT = 1 / len(scenariosLT)

    return scenariosLT, equiprobLT, DAscenario

def create_PPA_price_lifetime_array(PPA_price, WTG_lifetime):
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

    lifetime_PPA_price_array = np.tile((PPA_price_t), WTG_lifetime)

    return lifetime_PPA_price_array


def captureRateWind():
    scenariosLT, equiprobLT, DAscenario  = generate_scenarios(42)

    realised_gen =  np.array([scenario['Realised Generation'].values for scenario in scenariosLT]) * 38
    SP_scenarios = np.array([scenario['Spot Price Forecast'].values for scenario in scenariosLT])

    capture_prices = []

    for i in range(len(scenariosLT)):
        hourly_revenues = realised_gen[i] * SP_scenarios[i]
        capture_prices.append(sum(hourly_revenues) / np.sum(np.array(realised_gen[i])))

    average_capture_price = np.mean(capture_prices)
    average_capture_rate = average_capture_price / np.mean(SP_scenarios)

    GO_price = 3
    return average_capture_price, average_capture_rate, GO_price


def runLT(parameter_dict, simulation_dict, scenariosLT, equiprobLT, cap_price, profiletype, out_dir):

    # Retrieve parameter values
    PwMax = parameter_dict["wind_capacity"]
    EBESS = parameter_dict["battery_energy_capacity"]
    PbMax = parameter_dict["battery_power_capacity"]
    SoCmin = parameter_dict["battery_minimum_SoC"]
    SoCmax = parameter_dict["battery_maximum_SoC"]
    SoCini = parameter_dict["battery_initial_SoC"]
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"]
    P_grid_limit = parameter_dict["hpp_grid_connection"]
    mu = parameter_dict["battery_marginal_degradation_cost"]
    Emax = EBESS
    GO_price = parameter_dict["GO price"]

    # Create output directory
    out_dir_LT = os.path.join(out_dir, 'LT/')
    os.makedirs(out_dir_LT, exist_ok=True)

    # Initialize outputs
    combined_output_obj_vals = pd.DataFrame()
    S_PPA_opt_all = []

    #betas = np.array([0, 0.25, 0.5, 0.75])  # Beta values
    #PPA_discounts = np.array([0.16, 0.175, 0.2, 0.4, 0.6, 0.65, 0.7, 0.8, 1])  # PPA discount values

    betas = np.array([0.2])  # Beta values
    PPA_discounts = np.array([0.5])  # PPA discount values

    for PPA_discount in PPA_discounts:  # Outer loop for PPA prices
        PPA_price = list(PPA_discount * np.array(cap_price) + GO_price)
        for beta in betas:  # Inner loop for beta values
            total_cycles, S_PPA_opt, P_SM_t_opt, P_PPA_t_opt, P_cha_t_opt, P_SM_dis_t_opt, P_PPA_dis_t_opt, \
            P_SM_w_t_opt, P_PPA_w_t_opt, E_t_opt, z_t_opt, obj, CVaR = LTopt.LTopt_FHP(
                PPA_price, scenariosLT, equiprobLT, profiletype, parameter_dict, simulation_dict, PwMax,
                EBESS, PbMax, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, P_grid_limit, mu, beta, alpha=0.8)

            # Include beta and PPA price in S_PPA_opt output
            beta_row = np.array([beta])
            PPA_discount_row = np.array([PPA_discount])
            S_PPA_opt_with_metadata = np.hstack((beta_row, PPA_discount_row, S_PPA_opt.values.flatten()))
            S_PPA_opt_all.append(S_PPA_opt_with_metadata)

            # Objective Value output with PPA price and beta
            output_obj_val = pd.DataFrame([[PPA_price, beta, obj, CVaR]],
                                          columns=['PPA Price', 'Beta', 'Objective Value', 'CVaR'])
            combined_output_obj_vals = pd.concat([combined_output_obj_vals, output_obj_val], ignore_index=True)

    # Save combined objective values to CSV
    obj_vals_file = os.path.join(out_dir_LT, 'Objective_Values.csv')
    combined_output_obj_vals.to_csv(obj_vals_file, index=False)

    # Save S_PPA_opt_all to CSV
    S_PPA_opt_all = np.vstack(S_PPA_opt_all)
    no_of_SPs = S_PPA_opt_all.shape[1] - 2  # minus two for the beta and PPA price columns
    header = ['Beta'] + ['PPA discount'] + [f"S_PPA in Month/Season {i + 1}" for i in range(no_of_SPs)]
    S_PPA_opt_file = os.path.join(out_dir_LT, 'S_PPA_opt.csv')
    np.savetxt(S_PPA_opt_file, np.array(S_PPA_opt_all), delimiter=",", header=",".join(header), comments='')

    # Save PPA price(s) to a CSV
    PPA_price_file = os.path.join(out_dir_LT, 'PPA_prices.csv')
    no_of_price_periods = len(PPA_price)
    header = [f"PPA price in Month/Season/Year {j + 1}" for j in range(no_of_price_periods)]
    np.savetxt(PPA_price_file, np.array([PPA_price]), delimiter=",", header=",".join(header), comments='')

    # Save battery parameters to a CSV
    battery_parameters = [[EBESS, PbMax]]
    header = ['Energy (MWh)', 'Power (MW)'] 
    battery_parameters_file = os.path.join(out_dir_LT, 'battery_parameters.csv')
    np.savetxt(battery_parameters_file, np.array(battery_parameters), delimiter=",", header=",".join(header), comments='')

    return S_PPA_opt_all


def runST(beta, parameter_dict, simulation_dict, S_PPA_beta, PPA_price, PPA_discount, EBESS, PbMax, DAscenario, profiletype, ST_out_dir):
        
    # Initialize parameters
    DI = parameter_dict["dispatch_interval"]
    T = int(1/DI*24)    
    PwMax = parameter_dict["wind_capacity"] 
    SoCmin = parameter_dict["battery_minimum_SoC"] 
    SoCmax = parameter_dict["battery_maximum_SoC"] 
    SoCini = parameter_dict["battery_initial_SoC"] 
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    P_grid_limit = parameter_dict["hpp_grid_connection"]
    Emax = EBESS*(1-parameter_dict["battery_initial_degradation"])
    Ini_nld = parameter_dict["battery_initial_degradation"]
    pre_nld = Ini_nld
    SoC0 = SoCini
    ld1 = 0
    nld1 = Ini_nld
    ad = 1e-7   # slope   
    #capital_cost = parameter_dict["battery_capital_cost"] # €/MWh 
    mu = parameter_dict["battery_marginal_degradation_cost"]
    replace_percent = 0.3   # maybe change to 0.25
    #total_cycles = 3500

    # Initialize outputs
    out_dir = ST_out_dir
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
    
    shc = pd.DataFrame(list(), columns=['SM bid','cha','dis_total','dis_SM','w_SM','dis_PPA','w_PPA',
                                        'PPA_del_RT', 'SM_RT','cha_RT','dis_total_RT','dis_SM_RT','w_SM_RT',
                                        'dis_PPA_RT','w_PPA_RT','PPA_promised'])
    imb = pd.DataFrame(list(), columns=['SM_imbalance','PPA_imbalance', 'PPA_promised', 'curtailment'])
    slo = pd.DataFrame([ad], columns=['slope'])
    soc = pd.DataFrame(list(), columns=['SoC'])
    de  = pd.DataFrame(list(), columns=['nld','ld','cycles'])

    shc.to_csv(out_dir+f'schedule.csv',index=False)
    imb.to_csv(out_dir+f'imbalance.csv',index=False)
    slo.to_csv(out_dir+f'slope.csv',index=False)
    soc.to_csv(out_dir+f'SoC.csv',index=False)
    de.to_csv(out_dir+f'Degradation.csv',index=False)

    # Define months and seasons
    days_in_months = [30, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 30 days in January to match 8736 hours
    hours_in_months = [days * 24 for days in days_in_months]  
    seasons = {0: [12, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8], 3: [9, 10, 11]}
    hours_in_a_year = 8736

    PPA_violations_beta = 0
    battery_count = 1 
    current_hour = 0    
    hour_of_battery_replacement = 0
    hour_of_battery_replacement_list = []
    current_day = 1
    battery_days = 1
    current_SoC = SoC0
    accumulated_hours = 0
    month = 0

    # Realized wind generation
    all_realised_gen = DAscenario.iloc[:, 2] * parameter_dict['wind_capacity']  

    if len(PPA_price) == 12:
        # Monthly settlement: Repeat PPA_price for each month
        PPA_price = np.concatenate([
            np.full(hours_in_months[month], PPA_price[month]) for month in range(len(PPA_price))])
        
    elif len(PPA_price) == 4:
        # Create arrays for each season using the predefined hours_in_months
        seasonal_arrays = []
        for season in range(len(PPA_price)):
            # Sum the hours for each month in the season
            season_hours = sum(hours_in_months[month-1] for month in seasons[season])
            seasonal_price_array = np.full(season_hours, PPA_price[season])
            seasonal_arrays.append(seasonal_price_array)
        
        # Concatenate the arrays
        PPA_price = np.concatenate(seasonal_arrays)

    elif len(PPA_price) == 1:
        # Annual settlement: Repeat the same price for the whole year
        PPA_price = np.full(hours_in_a_year, PPA_price[0])

    # Run DAopt for one day at a time
    for i in tqdm(range(0, DAscenario.shape[0], 24), desc="DAopt"):
        if battery_days == 1:
            current_SoC = SoC0

        Emax = EBESS*(1-pre_nld)

        accumulated_hours += 24

        while accumulated_hours >= hours_in_months[month]:
            accumulated_hours -= hours_in_months[month]
            month += 1
            
            if month == 12: 
                accumulated_hours = 0 
                month = 0 

        season = next(season for season, months in seasons.items() if month + 1 in months)
        next_DA_forecast = DAscenario[i:i+24]
        setT = list(next_DA_forecast.index)

        # Get current PPA price 
        PPA_price_t = PPA_price[accumulated_hours]

        # Call DA optimization model
        P_SM_t_opt, P_PPA_t_opt, P_total_dis_t_opt, P_SM_dis_t_opt, P_PPA_dis_t_opt, P_cha_t_opt, P_SM_w_t_opt, P_PPA_w_t_opt, E_t_opt, M_t_opt_np = STopt.robustDAopt(
            parameter_dict, month, season, S_PPA_beta, PPA_price_t, 
            profiletype, next_DA_forecast, PwMax, EBESS, current_SoC, PbMax, SoCmin, 
            SoCmax, Emax, eta_dis, eta_cha, P_grid_limit, mu, ad, setT
        )

        SoC_ts = []
        SoC_ts.append({'SoC': current_SoC}) # initial for hver DAopt (ikke den helt initial)


        # Check for PPA constraint violations
        violations = (M_t_opt_np != 0).sum()
        PPA_violations_beta += violations

        P_SM_RT_t_opt_24h = []
        P_PPA_RT_t_opt_24h = []
        P_cha_RT_t_opt_24h = []
        P_total_dis_t_opt_24h = []

        P_total_RT_dis_t_opt_24h = []
        P_SM_RT_dis_t_opt_24h = []
        P_PPA_RT_dis_t_opt_24h = []
        P_SM_RT_w_t_opt_24h = []
        P_PPA_RT_w_t_opt_24h = []
        P_PPA_promised_t_opt_24h = []

        P_SM_imbalance = []
        P_PPA_imbalance = []
        P_PPA_promised = []
        curtailment = []

        next_gen_real = all_realised_gen[i:i+24].values

        # Run RTopt for the respective day
        for j in range(0,24):
            # Data for current iteration
            gen_real = next_gen_real[j]
            SM_bid = P_SM_t_opt.iloc[j,0]
            hour_of_day = j % 24

            # Run RTopt for the current hour
            next_SoC, SoC_RT_t_opt, obj, SM_imbalance_t_opt, PPA_imbalance_t_opt, abs_SM_imbalance_t_opt, abs_PPA_imbalance_t_opt, curtailment_t_opt, P_SM_RT_t_opt, P_PPA_RT_t_opt, P_cha_RT_t_opt, P_total_RT_dis_t_opt, P_SM_RT_dis_t_opt, P_SM_RT_w_t_opt, P_PPA_RT_w_t_opt, P_PPA_RT_dis_t_opt, P_PPA_promised_t_opt = RTopt.RTopt(
                    parameter_dict, hour_of_day, month, season, SM_bid, S_PPA_beta, profiletype, gen_real, current_SoC,
                    PwMax, EBESS, PbMax, SoCmin, SoCmax, Emax, eta_dis, eta_cha,
                    P_grid_limit, mu, ad)
        
            # Update current_hour and battery state
            current_SoC = next_SoC
            current_hour += 1

            SoC_ts.append({'SoC': current_SoC})  

            # for schedule file
            P_SM_RT_t_opt_24h.extend(P_SM_RT_t_opt) 
            P_PPA_RT_t_opt_24h.extend(P_PPA_RT_t_opt)
            P_cha_RT_t_opt_24h.extend(P_cha_RT_t_opt)
            P_total_RT_dis_t_opt_24h.extend(P_total_RT_dis_t_opt)
            P_SM_RT_dis_t_opt_24h.extend(P_SM_RT_dis_t_opt)
            P_SM_RT_w_t_opt_24h.extend(P_SM_RT_w_t_opt)
            P_PPA_RT_w_t_opt_24h.extend(P_PPA_RT_w_t_opt)
            P_PPA_RT_dis_t_opt_24h.extend(P_PPA_RT_dis_t_opt) 
            P_PPA_promised_t_opt_24h.extend(P_PPA_promised_t_opt)

            # for imbalance file
            P_SM_imbalance.extend(SM_imbalance_t_opt)
            P_PPA_imbalance.extend(PPA_imbalance_t_opt)
            P_PPA_promised.extend(P_PPA_promised_t_opt)
            curtailment.extend(curtailment_t_opt)

            current_SoC = next_SoC

        P_SM_RT_t_opt_24h = pd.DataFrame(P_SM_RT_t_opt_24h)
        P_PPA_RT_t_opt_24h = pd.DataFrame(P_PPA_RT_t_opt_24h)
        P_cha_RT_t_opt_24h = pd.DataFrame(P_cha_RT_t_opt_24h)
        P_total_dis_t_opt_24h = pd.DataFrame(P_total_dis_t_opt_24h)
        P_total_RT_dis_t_opt_24h = pd.DataFrame(P_total_RT_dis_t_opt_24h)
        P_SM_RT_dis_t_opt_24h = pd.DataFrame(P_SM_RT_dis_t_opt_24h)
        P_SM_RT_w_t_opt_24h = pd.DataFrame(P_SM_RT_w_t_opt_24h)
        P_PPA_RT_w_t_opt_24h = pd.DataFrame(P_PPA_RT_w_t_opt_24h)
        P_PPA_RT_dis_t_opt_24h = pd.DataFrame(P_PPA_RT_dis_t_opt_24h)
        P_PPA_promised_t_opt_24h = pd.DataFrame(P_PPA_promised_t_opt_24h)

        P_SM_imbalance = pd.DataFrame(P_SM_imbalance)
        P_PPA_imbalance = pd.DataFrame(P_PPA_imbalance)
        P_PPA_promised = pd.DataFrame(P_PPA_promised)
        curtailment = pd.DataFrame(curtailment)

        SoC_all = pd.read_csv(out_dir+'SoC.csv')
        SoC_ts = pd.DataFrame(SoC_ts[:T])
        if SoC_all.empty:
            SoC_all = SoC_ts
        else:
            SoC_all = pd.concat([SoC_all, SoC_ts]) 
        
        SoC_all = SoC_all.values.flatten().tolist()
        SoC_for_rainflow = SoC_all

        if battery_days == 1:  
            SoC_for_rainflow = SoC_ts.values.flatten().tolist()  
        else:
            SoC_for_rainflow = SoC_for_rainflow[hour_of_battery_replacement:]

        ld, nld, ld1, nld1, rf_DoD, rf_SoC, rf_count, nld_t, cycles = DegCal.Deg_Model(SoC_for_rainflow, Ini_nld, pre_nld, ld1, nld1, battery_days)

        #Deg_cost = (nld - pre_nld)/replace_percent * EBESS * capital_cost
    
        #if current_day==1:
        #    Deg_cost_by_cycle = cycles.iloc[0,0]/total_cycles * EBESS * capital_cost  
        #else:                
        #    Deg = pd.read_csv(out_dir+f'Degradation_{beta_row}.csv') 
        #    cycle_of_day = Deg.iloc[-1,2] - Deg.iloc[-2,2] 
        #    Deg_cost_by_cycle = cycle_of_day/total_cycles * EBESS * capital_cost   

        # output_schedule = pd.concat([P_SM_t_opt, P_cha_t_opt, P_total_dis_t_opt, P_SM_dis_t_opt, P_SM_w_t_opt, 
        #                             P_PPA_dis_t_opt, P_PPA_w_t_opt, P_PPA_RT_t_opt_24h, P_SM_RT_t_opt_24h, P_cha_RT_t_opt_24h, P_total_RT_dis_t_opt_24h,
        #                             P_SM_RT_dis_t_opt_24h, P_SM_RT_w_t_opt_24h, P_PPA_RT_dis_t_opt_24h, P_PPA_RT_w_t_opt_24h, P_PPA_promised_t_opt_24h], axis=1)
        

        output_schedule_DA = pd.concat([P_SM_t_opt, P_cha_t_opt, P_total_dis_t_opt, P_SM_dis_t_opt, P_SM_w_t_opt, 
                            P_PPA_dis_t_opt, P_PPA_w_t_opt], axis=1)
        output_schedule_RT = pd.concat([P_PPA_RT_t_opt_24h, P_SM_RT_t_opt_24h, P_cha_RT_t_opt_24h, P_total_RT_dis_t_opt_24h,
                                    P_SM_RT_dis_t_opt_24h, P_SM_RT_w_t_opt_24h, P_PPA_RT_dis_t_opt_24h, P_PPA_RT_w_t_opt_24h, P_PPA_promised_t_opt_24h], axis=1)
        output_schedule = pd.concat([output_schedule_DA, output_schedule_RT.set_index(output_schedule_DA.index)], axis=1)
        output_schedule.to_csv(out_dir+f'schedule.csv', mode='a', index=False, header=False)

        output_imbalance = pd.concat([P_SM_imbalance, P_PPA_imbalance, P_PPA_promised, curtailment], axis=1)
        output_imbalance.to_csv(out_dir+f'imbalance.csv', mode='a', index=False, header=False)

        if current_day == 1:
            output_deg = pd.concat([pd.DataFrame([Ini_nld, nld], columns=['nld']), pd.DataFrame([0, ld], columns=['ld']), pd.DataFrame([0, cycles.iloc[0,0]], columns=['cycles'])], axis=1)
        else:
            output_deg = pd.concat([pd.DataFrame([nld], columns=['nld']), pd.DataFrame([ld], columns=['ld']), cycles], axis=1)
            
        # Update slope for battery degradation
        Pdis_all = pd.read_csv(out_dir+'schedule.csv', usecols=[2]) 
        Pcha_all = pd.read_csv(out_dir+'schedule.csv', usecols=[1])
        nld_all = pd.read_csv(out_dir+'Degradation.csv', usecols=[0])
        ad_all =pd.read_csv(out_dir+'slope.csv', usecols=[0])
        ad = DegCal.slope_update(Pdis_all, Pcha_all, nld_all, current_day, 7, T, DI, ad_all)

        pd.DataFrame([ad], columns=['slope']).to_csv(out_dir+'slope.csv', mode='a', index=False, header=False)
  
        output_deg.to_csv(out_dir+'Degradation.csv', mode='a', index=False, header=False)
        SoC_ts.to_csv(out_dir+'SoC.csv', mode='a', index=False, header=False)

        current_day += 1
        battery_days += 1       

        # Counter for battery degradation
        if nld > replace_percent:
            battery_count += 1
            nld = 0
            pre_nld = nld
            hour_of_battery_replacement = current_day*T
            hour_of_battery_replacement_list.append(hour_of_battery_replacement)
            battery_days = 1

        # Update degradation for next DAopt run 
        pre_nld = nld
            
    # Save the number of batteries used and replacement hours to a CSV
    battery_count_df = pd.DataFrame({
        "batteries_used": [battery_count],
        "replacement_hours": [hour_of_battery_replacement_list]  # Store as a list in the CSV
    })
    battery_count_df.to_csv(f"{ST_out_dir}battery_count.csv", index=False)

def runPostProcessing(beta, parameter_dict, DAscenario, PPA_price, PPA_discount, PP_out_dir, ST_out_dir):
    
    # Extract data 
    real_spotprices = DAscenario['Realised Spot Price']
    balancing_prices = DAscenario['Realised Balancing Price'] 
    system_signal = DAscenario['System Signal']
    WTG_lifetime = parameter_dict["WTG lifetime"]
    grid_cap = parameter_dict["hpp_grid_connection"]

    imbalance_fee = parameter_dict["imbalance fee"]
    GO_price = parameter_dict["GO price"]
    PPA_price_lifetime_array = create_PPA_price_lifetime_array((PPA_price) * 1.8, WTG_lifetime)
    

    # Ensure SM_imbalances, PPA_imbalances, and PPA_deliveries are NumPy arrays and concatenate dimensions if necessary
    SM_bids = np.array(pd.read_csv(f'{ST_out_dir}schedule.csv', usecols=[0])).flatten()
    PPA_deliveries = np.array(pd.read_csv(f'{ST_out_dir}schedule.csv', usecols=[7])).flatten()
    SM_imbalances = np.array(pd.read_csv(f'{ST_out_dir}imbalance.csv', usecols=[0])).flatten()
    PPA_imbalances = np.array(pd.read_csv(f'{ST_out_dir}imbalance.csv', usecols=[1])).flatten()
    PPA_promised = np.array(pd.read_csv(f'{ST_out_dir}imbalance.csv', usecols=[2])).flatten()
    curtailment = np.array(pd.read_csv(f'{ST_out_dir}imbalance.csv', usecols=[3])).flatten()

    # Initialize list to store hourly profits
    num_hours = len(SM_bids)
    hourly_profits = []
    hourly_SM_profits = []
    hourly_PPA_profits = []
    SM_imbalance_revenues = [0] * num_hours
    PPA_imbalance_penalties = [0] * num_hours
    spotmarket_revenues = [0] * num_hours
    PPA_revenues = [0] * num_hours
    GO_revenues = [0] * num_hours
    remaining_cap = [0] * num_hours
    upregulation = [0] * num_hours

    # Calculate spot market revenue and imbalance revenue for each hour
    for t, bid in enumerate(SM_bids): 
        spotmarket_revenues[t] = bid * real_spotprices[t] # check that SM_bid is being updated above
        PPA_revenues[t] = PPA_deliveries[t] * PPA_price_lifetime_array[t] 
        
        # One price scheme
        SM_imbalance_revenues[t] = SM_imbalances[t] * balancing_prices[t] - imbalance_fee * abs(SM_imbalances[t])
            
        if curtailment[t] > 0:
            remaining_cap[t] = max(0, grid_cap - (bid + PPA_deliveries[t]))  
            upregulation[t] = min(remaining_cap[t], curtailment[t])  
            SM_imbalance_revenues[t] += upregulation[t] * balancing_prices[t] - imbalance_fee * abs(SM_imbalances[t])
            
        # PPA imbalance penalty for the hour
        PPA_imbalance_penalties[t] = - PPA_imbalances[t] * real_spotprices[t]

        # Guarantees of origin 
        GO_revenues[t] = PPA_deliveries[t] * GO_price 

        # Calculate total revenue for the hour
        total_hourly_profit = spotmarket_revenues[t] + PPA_revenues[t] + SM_imbalance_revenues[t] + PPA_imbalance_penalties[t] + GO_revenues[t]
        total_hourly_SM_profit = spotmarket_revenues[t] + SM_imbalance_revenues[t]
        total_hourly_PPA_profit = PPA_revenues[t] + PPA_imbalance_penalties[t] + GO_revenues[t]

        # Append the hourly profit to the list
        hourly_profits.append(total_hourly_profit)
        hourly_SM_profits.append(total_hourly_SM_profit)
        hourly_PPA_profits.append(total_hourly_PPA_profit)

    # PPA violations
    PPA_violations = np.count_nonzero(PPA_imbalances > 0) # account for numerical instability ? 

    # compute yearly profits
    yearly_profits = []
    yearly_SM_imbalance_costs = []
    yearly_PPA_imbalance_costs = []
    yearly_SM_profits = []
    yearly_PPA_profits = []
    for i in range(0, len(hourly_profits), 8736):
        yearly_profits.append(sum(hourly_profits[i:i+8736])) 
        yearly_SM_profits.append(sum(hourly_SM_profits[i:i+8736]))
        yearly_PPA_profits.append(sum(hourly_PPA_profits[i:i+8736]))
        yearly_SM_imbalance_costs.append(sum(SM_imbalance_revenues[i:i+8736])) 
        yearly_PPA_imbalance_costs.append(sum(PPA_imbalance_penalties[i:i+8736])) 

    # Save the yearly profits to a CSV
    yearly_profits_df = pd.DataFrame({
        'Yearly total profits': yearly_profits,
        'Yearly SM profits': yearly_SM_profits,
        'Yearly PPA profits': yearly_PPA_profits
        })
    
    out_dir = PP_out_dir
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
    yearly_profits_df.to_csv(f"{out_dir}/yearly_profits.csv", index=False)

    # yearly SM and PPA imbalance costs
    imbalance_costs_df = pd.DataFrame({
    "Yearly SM_imbalance_revenue": yearly_SM_imbalance_costs,
    "Yearly PPA_imbalance_penalty": yearly_PPA_imbalance_costs})
    imbalance_costs_df.to_csv(f"{out_dir}/imbalance_costs.csv", index=False)

    # Save the number of PPA violations to a CSV
    PPA_violations_df = pd.DataFrame({
        "PPA_violations": [PPA_violations]})
    PPA_violations_df.to_csv(f"{out_dir}/PPA_violations.csv", index=False)
    
    # Create DataFrame with hourly data
    hourly_profits_df = pd.DataFrame({
        'Hour': range(len(hourly_profits)),
        'Total_Profit': hourly_profits,
        'SM_Profit': hourly_SM_profits,
        'PPA_Profit': hourly_PPA_profits,
        'SM_Imbalance_Revenue': SM_imbalance_revenues,
        'PPA_Imbalance_Penalty': PPA_imbalance_penalties,
        'GO_Revenue': GO_revenues,
        'SM_Bid': SM_bids,
        'PPA_Delivery': PPA_deliveries,
        'SM_Imbalance': SM_imbalances,
        'PPA_Imbalance': PPA_imbalances,
        'Curtailment': curtailment,
        'Upregulation': upregulation
    })
    
    # Save hourly data
    hourly_profits_df.to_csv(f"{out_dir}/hourly_profits.csv", index=False)


def calculateNPVandIRR_w_CAPEX_phasing(beta, parameter_dict, DAscenario, PPA_price, windcap, battery_energycap, battery_powercap, ST_out_dir, PP_out_dir, NPV_out_dir):
    # Battery replacement results
    battery_count = pd.read_csv(f"{ST_out_dir}battery_count.csv", usecols=[0]).iloc[0, 0]
    battery_replacement_hours = pd.read_csv(f"{ST_out_dir}battery_count.csv", usecols=[1]).iloc[0, 0]
    battery_replacement_hours = ast.literal_eval(battery_replacement_hours)[0]
    battery_replacement_years = [battery_replacement_hours // 8736]  # Convert hours to years

    # WTG lifetime
    WTG_lifetime = parameter_dict["WTG lifetime"]

    # Battery CAPEX adjustment/discount for sensitivity analysis
    battery_CAPEX_adjustment = parameter_dict["Battery CAPEX adjustment"]

    # CAPEX and OPEX
    WACC = parameter_dict["WACC"]
    Wind_yearly_OPEX = parameter_dict["Wind yearly OPEX per MW"] * windcap
    Wind_CAPEX = parameter_dict["WTG CAPEX per MW"] * windcap
    Wind_DEVEX = parameter_dict["WTG DEVEX per MW"] * windcap
    WTG_variable_OM = parameter_dict['WTG Variable OPEX per MWh']
    wind_generation = DAscenario['Realised Generation'] * windcap
    Battery_yearly_OPEX = parameter_dict["Battery yearly OPEX per MW"]

    # Inflation factors for different base years
    inflation_rate = 0.02

    # Battery CAPEX inflation factor (costs are in real2015 in DEA catalog)
    inflation_2015_2025 = (1 + inflation_rate) ** (2025 - 2015)
    inflation_2015_2040 = (1 + inflation_rate) ** (2040 - 2015)

    # Profit inflation factors
    inflation_factors_SM_profits = [(1 + inflation_rate) ** (year - 2025) for year in range(2025, 2035)] # inflation factors for spot prices (need to be inflated for 10 years until next price update)

    # OPEX inflation factors
    inflation_factors_2025_OPEX = [(1 + inflation_rate) ** (year - 2024) for year in range(2025, 2025 + WTG_lifetime)] # inflation factors for prices in real2025

    # Inflation adjustment for battery costs to nominal values
    Battery_CAPEX_2025 = (
        parameter_dict['2025 Battery Energy CAPEX per MWh'] * battery_energycap +
        parameter_dict['2025 Battery Power CAPEX per MW'] * battery_powercap +
        parameter_dict['2025 Battery DEVEX per MW'] * battery_powercap
    ) * inflation_2015_2025 * battery_CAPEX_adjustment

    Battery_CAPEX_2040 = (
        parameter_dict['2040 Battery Energy CAPEX per MWh'] * battery_energycap +
        parameter_dict['2040 Battery Power CAPEX per MW'] * battery_powercap +
        parameter_dict['2040 Battery DEVEX per MW'] * battery_powercap
    ) * inflation_2015_2040 * battery_CAPEX_adjustment

    yearly_prod = np.sum(wind_generation) / WTG_lifetime

    # Inflated yearly fixed and variable OPEX
    yearly_fixed_OPEX = np.array([(Wind_yearly_OPEX + Battery_yearly_OPEX) * inflation_factors_2025_OPEX[year - 2025] for year in range(2025, 2025 + WTG_lifetime)])
    yearly_variable_OM = np.array([(WTG_variable_OM * yearly_prod) * inflation_factors_2025_OPEX[year - 2025] for year in range(2025, 2025 + WTG_lifetime)])

    # Yearly profits - inflation adjusted
    PPA_profits = np.array(pd.read_csv(f"{PP_out_dir}yearly_profits.csv", usecols=[2]).values).flatten()
    SM_profits = np.array(pd.read_csv(f"{PP_out_dir}yearly_profits.csv", usecols=[1]).values).flatten()
    
    # Power profits in 2025-2034
    SM_profits_2025_2034 = SM_profits[:10]
    SM_profits_2025_2034 = SM_profits_2025_2034 * inflation_factors_SM_profits 

    # Power profits in 2035-2044
    SM_profits_2035_2044 = SM_profits[10:20]
    SM_profits_2035_2044 = SM_profits_2035_2044 * inflation_factors_SM_profits

    # Power profits in 2045-2054
    SM_profits_2045_2054 = SM_profits[20:]
    SM_profits_2045_2054 = SM_profits_2045_2054 * inflation_factors_SM_profits

    SM_profits_nominal = np.concatenate([SM_profits_2025_2034, SM_profits_2035_2044, SM_profits_2045_2054])

    # Corporate tax and depreciation
    corporate_tax_rate = 0.22
    profits = PPA_profits + SM_profits_nominal - yearly_fixed_OPEX - yearly_variable_OM
    initial_CAPEX = Wind_CAPEX + Wind_DEVEX + Battery_CAPEX_2025
    yearly_depreciation = initial_CAPEX / WTG_lifetime  # linear depreciation for initial CAPEX
    depreciation_schedule = np.full(WTG_lifetime, yearly_depreciation)  # initialize depreciation schedule

    # Adjust depreciation for extra batteries in the replacement years
    for year in range(1, WTG_lifetime + 1):  # 1-based year index
        if year in battery_replacement_years:
            remaining_years = WTG_lifetime - year + 1  # years remaining from replacement to the end of WTG lifetime
            extra_depreciation = Battery_CAPEX_2025 / remaining_years  # depreciation of the new battery
            for remaining_year in range(year - 1, WTG_lifetime):  # adjust future years
                depreciation_schedule[remaining_year] += extra_depreciation

    # Calculate taxable profits and taxes
    taxable_profits = profits - depreciation_schedule
    taxes = taxable_profits * corporate_tax_rate

    # Calculate yearly cash flows
    yearly_cash_flows = profits - taxes
    yearly_cash_flows = np.insert(yearly_cash_flows, 0, -initial_CAPEX)  # subtract initial CAPEX

    # Subtract the cost of the extra battery in the replacement years
    for year in range(1, WTG_lifetime + 1):  # 1-based year index
        if year in battery_replacement_years:
            yearly_cash_flows[year] -= Battery_CAPEX_2040
 

    # Calculate NPV and IRR
    total_CAPEX = Wind_CAPEX + Wind_DEVEX + Battery_CAPEX_2025 + Battery_CAPEX_2040 * (battery_count - 1)
    NPV = np.sum(yearly_cash_flows / (1 + WACC) ** np.arange(WTG_lifetime + 1))
    NPV_over_CAPEX = NPV / total_CAPEX

    IRR = npf.irr(yearly_cash_flows)

    result_data = {
        'Battery CAPEX': [Battery_CAPEX_2025 + Battery_CAPEX_2040 * (battery_count - 1)],
        'WACC': [WACC],
        'Battery CAPEX adjustment': [battery_CAPEX_adjustment],
        'NPV': [NPV],
        'NPV_over_CAPEX': [NPV_over_CAPEX],
        'IRR': [IRR],
    }

    # Save results
    os.makedirs(NPV_out_dir, exist_ok=True)
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(f"{NPV_out_dir}/NPV_Bat_CAPEX_adj_{battery_CAPEX_adjustment}_WACC_{WACC}.csv", index=False)


def runAllST(parameter_dict, simulation_dict, DAscenario, profiletype, out_dir):
    # Define output directories
    out_dir_LT = out_dir + 'LT/'
    ST_out_dir = out_dir + 'ST/'
    PP_out_dir = out_dir + 'PP/'
    NPV_out_dir = out_dir + 'NPV_and_IRR/'

    # Extract parameters
    windcap = parameter_dict['wind_capacity']
    EBESS = pd.read_csv(out_dir_LT + 'battery_parameters.csv')['Energy (MWh)'].iloc[0]
    PbMax = pd.read_csv(out_dir_LT + 'battery_parameters.csv')['Power (MW)'].iloc[0]

    # Read data
    S_PPA_opt_all_betas_np = np.array(pd.read_csv(out_dir_LT + 'S_PPA_opt.csv'))
    PPA_prices_np = np.array(pd.read_csv(out_dir_LT + 'PPA_prices.csv').values.flatten())

    # Iterate through beta rows
    for beta_row in range(S_PPA_opt_all_betas_np.shape[0]):
        # Extract the beta-specific PPA price and S_PPA values
        beta = S_PPA_opt_all_betas_np[beta_row, 0]
        PPA_discount = S_PPA_opt_all_betas_np[beta_row, 1]
        PPA_price = PPA_prices_np
        S_PPA_beta = S_PPA_opt_all_betas_np[beta_row,2:]   # Get the row for the specific beta

        #Run ST optimization with PPA split from LT
        # runST(
        #     beta,
        #     parameter_dict,
        #     simulation_dict,
        #     S_PPA_beta,
        #     PPA_price,
        #     PPA_discount,
        #     EBESS,
        #     PbMax,
        #     DAscenario,
        #     profiletype,
        #     ST_out_dir = ST_out_dir + f'beta_{beta}_PPA_discount_{PPA_discount}/'
        # )

        # Run post-processing - still needs variable PPA pricing
        runPostProcessing(
            beta_row, 
            parameter_dict,
            DAscenario, 
            PPA_price, 
            PPA_discount,
            PP_out_dir = PP_out_dir + f'beta_{beta}_PPA_discount_{PPA_discount}/',
            ST_out_dir = ST_out_dir + f'beta_{beta}_PPA_discount_{PPA_discount}/'
        )

        #Calculate NPV and IRR for the current beta row
        calculateNPVandIRR_w_CAPEX_phasing(
            beta_row,
            parameter_dict,
            DAscenario,
            PPA_price,
            windcap, 
            EBESS,
            PbMax,
            ST_out_dir = ST_out_dir + f'beta_{beta}_PPA_discount_{PPA_discount}/',
            PP_out_dir = PP_out_dir + f'beta_{beta}_PPA_discount_{PPA_discount}/',
            NPV_out_dir= NPV_out_dir + f'beta_{beta}_PPA_discount_{PPA_discount}/'
        )



