# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:05:20 2022

@author: ruzhu
"""
from deliveryprofiles import DeliveryProfile
import utilsPPA
import numpy as np 
import pandas as pd

parameter_dict = {
        # # country of the HPP
        # 'country': 'DK',  # DK, NO, SE, FL, others
        # hpp parameters
        'hpp_grid_connection': 52,  # in MW  https://www.energy-supply.dk/article/view/857982/eurowind_energy_gar_efter_at_udvide_stor_hybridpark   https://ens.dk/sites/ens.dk/files/Vindenergi/13-80211-44_nettilslutning_af_kystnaere_havmolleparker_endelig_udgave_-_eng.pdf

        # hpp wind parameters
        'wind_capacity': 37.8, #in MW


        # # hpp solar parameters
        # 'solar_capacity': 0,  # in MW
       
        'dispatch_interval': 1,

        # hpp battery parameters
        'battery_energy_capacity': 120,  # in MWh
        'battery_power_capacity': 20,  # in MW
        'battery_minimum_SoC': 0.05,
        'battery_maximum_SoC': 0.95,
        'battery_initial_SoC': 0.5,
        'battery_hour_discharge_efficiency': 0.985,  #
        'battery_hour_charge_efficiency': 0.975,
        'battery_self_discharge_efficiency': 0,

        # hpp battery degradation parameters
        'battery_initial_degradation': 0,  
        'battery_marginal_degradation_cost': 142000, # in /MWh 
        'battery_capital_cost': 142000, # in /MWh
        'degradation_in_optimization': 1, # 1:yes 0:no
        'degradation_cost_per_mwh': 3.13, # for LTopt, in /MWh 3.13 if energy/power ratio is 3:1


        # Laura parameters
        '2025 Battery Energy CAPEX per MWh' : 62000, # DEA energy catalogue
        '2025 Battery Power CAPEX per MW' : 16000, # DEA energy catalogue  
        '2025 Battery DEVEX per MW' : 80000, 
        '2040 Battery Energy CAPEX per MWh' : 44000, # DEA energy catalogue
        '2040 Battery Power CAPEX per MW' : 16000, 
        '2040 Battery DEVEX per MW' : 50000, 
        'WTG CAPEX per MW' : 640000,  #https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/-/blob/main/hydesign/examples/Europe/hpp_pars.yml?ref_type=heads
        'WTG DEVEX per MW' : 260000, #https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/-/blob/main/hydesign/examples/Europe/hpp_pars.yml?ref_type=heads
        'Wind yearly OPEX per MW' : 12600, #EUR/MW/year, https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/-/blob/main/hydesign/examples/Europe/hpp_pars.yml?ref_type=heads
        'WTG Variable OPEX per MWh' : 1.35, #EUR/MWh_e, https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/-/blob/main/hydesign/examples/Europe/hpp_pars.yml?ref_type=heads
        'Battery yearly OPEX per MW' : 0, # https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/-/blob/main/hydesign/examples/Europe/hpp_pars.yml?ref_type=heads
        'WACC' : 0.06, 
        "WTG lifetime": 30, # years, do not change, functions are not flexible to it 
        "Battery CAPEX adjustment": 1, # for sensitivity analysis
        "GO price": 3, # in EUR/MWh
        "imbalance fee": 0.133 # in EUR/MWh
    }

# simulation_dict = {
#         'wind_as_component': 1,
#         'solar_as_component': 1,  
#         'battery_as_component': 1,
#         'start_date': '1/1/21',
#         'number_of_run_day': 1,   # 
#         'out_dir':"./results/",

#         'DA_wind': "DA",   #DA, Measurement
#         'HA_wind': "HA" ,  #HA, Measurement
#         'FMA_wind':"RT",#5min_ahead, Measurement
#         'DA_solar': "Measurement",
#         'HA_solar': "Measurement",
#         'FMA_solar': "Measurement",
#         'SP': "SM_forecast",  # SM_forecast;SM_cleared
#         'RP': "reg_forecast_DNN", #reg_cleared;reg_forecast_pre
#         'BP': 1, #1:forecast value 2: perfect value
        
#         # Data
#         'wind_dir': "../Data/Winddata2021_15min.csv",
#         'solar_dir': "../Data/Solardata2021_15min.csv",
#         'market_dir': "../Data/Market2021.csv",
    
#         # for DDEMS (spot market) -- Historical data
#         'history_wind_dir': "../Data/Winddata2022_15min.csv",
#         'history_market_dir': "../Data/Market2021.csv",
        
#         # for REMS (balancing market)
#         'HA_wind_error_ub': "5%_fc_error",
#         'HA_wind_error_lb': "95%_fc_error",
        
#         # for SEMS
#         'price_scenario_dir': "../Data/Market2021.csv",  # "../Data/xxx.csv", if None then use the build in method to generate price scenarios
#         'number_of_wind_scenario': 3,
#         'number_of_solar_scenario': 3, 
#         'number_of_price_scenario': 2, 
#     }


deliveryprofile_dict = {
    'Seasonally Free': DeliveryProfile('Seasonally Free', 0, None), #ignore this one - i never used it
    'Monthly Free': DeliveryProfile('Monthly Free', 1, None), #ignore this one - i never used it
    'Monthly Continuous Baseload': DeliveryProfile('Monthly Continuous Baseload', 2, [1] * 24),
    'Seasonal Continuous Baseload': DeliveryProfile('Seasonal Continuous Baseload', 3, [1] * 24),
    'Yearly Continuous Baseload': DeliveryProfile('Yearly Continuous Baseload', 4, [1] * 24),
    'Generic Industrial Monthly': DeliveryProfile('Generic Industrial Monthly', 5, [0.593030735, 0.554395406, 0.546245849, 0.53915524, 0.552433083, 0.620155643, 0.732682151, 0.84368243, 0.927410869, 0.975923914, 0.990102265, 0.987748587,0.988789471, 1, 0.996437111, 0.98324188, 0.960533801, 0.946952824, 0.911729005, 0.878990223, 0.811109466, 0.750608622, 0.682631749, 0.632859476]),
    'Generic Industrial Seasonal': DeliveryProfile('Generic Industrial Seasonal', 6, [0.593030735, 0.554395406, 0.546245849, 0.53915524, 0.552433083, 0.620155643, 0.732682151, 0.84368243, 0.927410869, 0.975923914, 0.990102265, 0.987748587,0.988789471, 1, 0.996437111, 0.98324188, 0.960533801, 0.946952824, 0.911729005, 0.878990223, 0.811109466, 0.750608622, 0.682631749, 0.632859476]),
    'Generic Industrial Yearly': DeliveryProfile('Generic Industrial Yearly', 7, [0.593030735, 0.554395406, 0.546245849, 0.53915524, 0.552433083, 0.620155643, 0.732682151, 0.84368243, 0.927410869, 0.975923914, 0.990102265, 0.987748587,0.988789471, 1, 0.996437111, 0.98324188, 0.960533801, 0.946952824, 0.911729005, 0.878990223, 0.811109466, 0.750608622, 0.682631749, 0.632859476]),
    'Generic Private Monthly': DeliveryProfile('Generic Private Monthly', 8, [0.405392308, 0.363868092, 0.331129724, 0.302547855, 0.285679627, 0.299305534, 0.411206526, 0.52205873, 0.507150756, 0.490946791, 0.486622381, 0.495561923, 0.51057742, 0.511541268, 0.521249321, 0.551870621, 0.655318008, 0.892684358, 1, 0.938593833, 0.829571565, 0.732613459, 0.604970492, 0.477711519]),
    'Generic Private Seasonal': DeliveryProfile('Generic Private Seasonal', 9, [0.405392308, 0.363868092, 0.331129724, 0.302547855, 0.285679627, 0.299305534, 0.411206526, 0.52205873, 0.507150756, 0.490946791, 0.486622381, 0.495561923, 0.51057742, 0.511541268, 0.521249321, 0.551870621, 0.655318008, 0.892684358, 1, 0.938593833, 0.829571565, 0.732613459, 0.604970492, 0.477711519]),
    'Generic Private Yearly': DeliveryProfile('Generic Private Yearly', 10, [0.405392308, 0.363868092, 0.331129724, 0.302547855, 0.285679627, 0.299305534, 0.411206526, 0.52205873, 0.507150756, 0.490946791, 0.486622381, 0.495561923, 0.51057742, 0.511541268, 0.521249321, 0.551870621, 0.655318008, 0.892684358, 1, 0.938593833, 0.829571565, 0.732613459, 0.604970492, 0.477711519]),
    'Inverse Solar Monthly': DeliveryProfile('Inverse Solar Monthly', 11, [1, 1, 1, 0.997772592, 0.919229953, 0.757901994, 0.558071701, 0.373355961, 0.201633432, 0.092013152, 0.010288502, 0, 0.069633008, 0.145736105, 0.277047094, 0.460065762, 0.650774289, 0.848589308, 0.97300594, 0.999363598, 1, 1, 1, 1]),
    'Inverse Solar Seasonal': DeliveryProfile('Inverse Solar Seasonal', 12, [1, 1, 1, 0.997772592, 0.919229953, 0.757901994, 0.558071701, 0.373355961, 0.201633432, 0.092013152, 0.010288502, 0, 0.069633008, 0.145736105, 0.277047094, 0.460065762, 0.650774289, 0.848589308, 0.97300594, 0.999363598, 1, 1, 1, 1]),
    'Inverse Solar Yearly': DeliveryProfile('Inverse Solar Yearly', 13, [1, 1, 1, 0.997772592, 0.919229953, 0.757901994, 0.558071701, 0.373355961, 0.201633432, 0.092013152, 0.010288502, 0, 0.069633008, 0.145736105, 0.277047094, 0.460065762, 0.650774289, 0.848589308, 0.97300594, 0.999363598, 1, 1, 1, 1]),
    'Peak Monthly': DeliveryProfile('Peak Monthly', 14, [0.653050827, 0.625423729, 0.62006779, 0.663525441, 0.781966078, 0.87440682, 0.882271149, 0.789559315, 0.779661017, 0.778881369, 0.728813559, 0.703389831, 0.682305081, 0.68572879, 0.714915234, 0.765084736, 0.818169495, 0.986915305, 1, 0.903254207, 0.826033875, 0.771186441, 0.681050861, 0.620338983]),
    'Peak Seasonal': DeliveryProfile('Peak Seasonal', 15, [0.653050827, 0.625423729, 0.62006779, 0.663525441, 0.781966078, 0.87440682, 0.882271149, 0.789559315, 0.779661017, 0.778881369, 0.728813559, 0.703389831, 0.682305081, 0.68572879, 0.714915234, 0.765084736, 0.818169495, 0.986915305, 1, 0.903254207, 0.826033875, 0.771186441, 0.681050861, 0.620338983]),
    'Peak Yearly': DeliveryProfile('Peak Yearly', 16, [0.653050827, 0.625423729, 0.62006779, 0.663525441, 0.781966078, 0.87440682, 0.882271149, 0.789559315, 0.779661017, 0.778881369, 0.728813559, 0.703389831, 0.682305081, 0.68572879, 0.714915234, 0.765084736, 0.818169495, 0.986915305, 1, 0.903254207, 0.826033875, 0.771186441, 0.681050861, 0.620338983]),
    'Binary Peak Monthly': DeliveryProfile('Binary Peak Monthly', 17, [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]), # this is what i actually call peak in the report, the other peak is the "system peak"
    'Binary Peak Seasonal': DeliveryProfile('Binary Peak Seasonal', 18, [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]), # this is what i actually call peak in the report, the other peak is the "system peak"
    'Binary Peak Yearly': DeliveryProfile('Binary Peak Yearly', 19, [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]) # this is what i actually call peak in the report, the other peak is the "system peak"
}

# Battery parameters for sensitivity analysis
EBESS = parameter_dict['battery_energy_capacity']
PbMax = parameter_dict['battery_power_capacity'] 

# Choose profile type 
profile_name = 'Inverse Solar Monthly'
pricing_scheme = 'Yearly' # choose between 'Monthly', 'Seasonal', 'Yearly' (yearly is same as fixed)

# Generate scenarios - DAscenario is 30 years long 
scenariosLT, equiprobLT, DAscenario  = utilsPPA.generate_scenarios(42)

# Calculate PPA price based on profiletype and spot price LT scenarios
profiletype = deliveryprofile_dict[profile_name]
SP_scenarios = np.array([scenario['Realised Spot Price'].values for scenario in scenariosLT])
monthly_capture_prices, monthly_capture_factors, seasonal_capture_prices, seasonal_capture_factors, yearly_capture_price, yearly_capture_factor = profiletype.getPricePremiums(SP_scenarios)

if pricing_scheme == 'Monthly':
    PPA_price = monthly_capture_prices  
elif pricing_scheme == 'Seasonal':
    PPA_price = seasonal_capture_prices
elif pricing_scheme == 'Yearly':    
    PPA_price = [yearly_capture_price]
else:
    print('Pricing scheme not recognized')

print(PPA_price)

out_dir = f"{simulation_dict['out_dir']}{profile_name}_PS_{pricing_scheme}_Bat_{EBESS}_{PbMax}/"

S_PPA_opt_all = utilsPPA.runLT(
        parameter_dict, 
        simulation_dict,
        scenariosLT, 
        equiprobLT,
        PPA_price,
        profiletype = deliveryprofile_dict[profile_name],
        out_dir = out_dir
        )

# Commented out so it does not start the long optimization
# utilsPPA.runAllST(
#         parameter_dict,
#         simulation_dict, 
#         DAscenario,
#         profiletype = deliveryprofile_dict[profile_name], 
#         out_dir = out_dir
#         )


