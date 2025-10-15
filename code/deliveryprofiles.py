import cplex
import numpy as np

class DeliveryProfile:
        
    def __init__(self, name, profile_id, profile_matrix):
        self.name = name
        self.profile_id = profile_id
        self.profile_matrix = profile_matrix

    def __repr__(self):
        return (f"DeliveryProfile({self.name}, ID: {self.profile_id}, Profile: {self.profile_matrix}")

    def get_constraints_LT_old(self, mdl, P_PPA_t, seasons, monthly_hours, hours_in_a_day, months_in_a_year):        
        
        if self.name == 'Seasonally Free':
            for i, season in enumerate(seasons):
                for remainder in range(hours_in_a_day):
                    indices = [t for t in season if t % 24 == remainder]
                    mdl.add_constraints(P_PPA_t[indices[0]] == P_PPA_t[t] for t in indices[1:])
                    
        elif self.name == 'Monthly Free':
            for month in range(months_in_a_year):
                for remainder in range(hours_in_a_day):
                    indices = [t for t in monthly_hours[month] if t % 24 == remainder]
                    mdl.add_constraints(P_PPA_t[indices[0]] == P_PPA_t[t] for t in indices[1:])
                    
        elif self.name == 'Monthly Continuous Baseload':
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split monthly continuous baseload')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month])

        # elif self.name == 'Seasonal Continuous Baseload':
        #     for i, season in enumerate(seasons):
        #         indices = seasons[i]
        #         for t in indices[1:]:
        #             constraints.append(mdl.add_constraint(P_PPA_t[indices[0]] == P_PPA_t[t]))

        elif self.name == 'Seasonal Continuous Baseload':
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, lb=0, ub=cplex.infinity, name='Split seasonal continuous baseload')
            for i, season in enumerate(seasons):
                indices = seasons[i]  # Get the hours for the current season
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[i])

        elif self.name == 'Generic Industrial Monthly':
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split generic industrial monthly')
            for month in range(months_in_a_year):
                indices = monthly_hours[month]  # Get the hours for the current month
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month])
        
        elif self.name == 'Generic Industrial Seasonal':
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, lb=0, ub=cplex.infinity, name='Split industrial seasonal')
            for i, season in enumerate(seasons):
                indices = seasons[i]  # Get the hours for the current season
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[i])
    
        elif self.name == 'Generic Private Monthly':
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split generic private monthly')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month])
        
        elif self.name == 'Generic Private Seasonal':
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, lb=0, ub=cplex.infinity, name='Split generic private seasonal')
            for i, season in enumerate(seasons):
                indices = seasons[i]  # Get the hours for the current season
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[i])
    
        elif self.name == 'Inverse Solar Monthly':
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split inverse solar monthly')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month] )
        
        elif self.name == 'Inverse Solar Seasonal':
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, lb=0, ub=cplex.infinity, name='Split inverse solar seasonal')
            for i, season in enumerate(seasons):
                indices = seasons[i]  # Get the hours for the current season
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[i])
    
        elif self.name == 'Peak Monthly':
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split peak monthly')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month])
        
        return S_PPA
    

    def get_constraints_LT(self, mdl, P_PPA_t_s, n_scens, seasons, monthly_hours, hours_in_a_day, months_in_a_year):        
        
        # if self.name == 'Seasonally Free':
        #     for i, season in enumerate(seasons):
        #         for remainder in range(hours_in_a_day):
        #             indices = [t for t in season if t % 24 == remainder]
        #             mdl.add_constraints(P_PPA_t_s[indices[0]] == P_PPA_t_s[t] for t in indices[1:])
                    
        # elif self.name == 'Monthly Free':
        #     for month in range(months_in_a_year):
        #         for remainder in range(hours_in_a_day):
        #             indices = [t for t in monthly_hours[month] if t % 24 == remainder]
        #             mdl.add_constraints(P_PPA_t_s[indices[0]] == P_PPA_t_s[t] for t in indices[1:])

        if self.name == 'Monthly Continuous Baseload':
            setS = [i for i in range(n_scens)] # scenarios
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split monthly continuous baseload')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[month] for s in setS)

        elif self.name == 'Seasonal Continuous Baseload':
            setS = [i for i in range(n_scens)] # scenarios
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, lb=0, ub=cplex.infinity, name='Split seasonal continuous baseload')
            for i, season in enumerate(seasons):
                indices = seasons[i]  # Get the hours for the current season
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[i] for s in setS)
    
        elif self.name == 'Generic Industrial Monthly':
            setS = [i for i in range(n_scens)] # scenarios
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split monthly industrial')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[month]  for s in setS)

        elif self.name == 'Generic Industrial Seasonal':
            setS = [i for i in range(n_scens)] # scenarios
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, lb=0, ub=cplex.infinity, name='Split industrial seasonal')
            for i, season in enumerate(seasons):
                indices = seasons[i]  # Get the hours for the current season
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[i] for s in setS)
    
        elif self.name == 'Generic Private Monthly':
            setS = [i for i in range(n_scens)] # scenarios
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split monthly private')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[month] for s in setS)

        elif self.name == 'Generic Private Seasonal':     
            setS = [i for i in range(n_scens)] # scenarios       
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, lb=0, ub=cplex.infinity, name='Split generic private seasonal')
            for i, season in enumerate(seasons):
                indices = seasons[i]  # Get the hours for the current season
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[i] for s in setS)
    
        elif self.name == 'Inverse Solar Monthly':
            setS = [i for i in range(n_scens)] # scenarios
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split monthly inverse solar')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[month] for s in setS)

        elif self.name == 'Inverse Solar Seasonal':
            setS = [i for i in range(n_scens)] # scenarios
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, lb=0, ub=cplex.infinity, name='Split inverse solar seasonal')
            for i, season in enumerate(seasons):
                indices = seasons[i]  # Get the hours for the current season
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[i]  for s in setS)
    
        elif self.name == 'Peak Monthly':
            setS = [i for i in range(n_scens)] # scenarios
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, lb=0, ub=cplex.infinity, name='Split peak monthly')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[month]  for s in setS)

        elif self.name == 'Peak Seasonal':
            setS = [i for i in range(n_scens)] # scenarios
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, lb=0, ub=cplex.infinity, name='Split peak seasonal')
            for i, season in enumerate(seasons):
                indices = seasons[i]  # Get the hours for the current season
                for t in indices:
                    hour_of_day = t % 24  # Get corresponding hour index
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[i]  for s in setS)

        elif self.name == 'Binary Peak Monthly':
            setS = [i for i in range(n_scens)]
            setM = [i for i in range(months_in_a_year)]
            S_PPA = mdl.continuous_var_dict(setM, name='Split binary peak monthly')
            for month in range(months_in_a_year):
                indices = monthly_hours[month] 
                for t in indices:
                    hour_of_day = t % 24  
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[month] for s in setS)

        elif self.name == 'Binary Peak Seasonal':
            setS = [i for i in range(n_scens)]
            setSeasons = [i for i in range(4)]
            S_PPA = mdl.continuous_var_dict(setSeasons, name='Split binary peak seasonal')
            for i, season in enumerate(seasons):
                indices = seasons[i]
                for t in indices:
                    hour_of_day = t % 24
                    mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA[i] for s in setS)
    
        ### Yearly settlement periods 
        elif self.name == 'Yearly Continuous Baseload':
            setS = [i for i in range(n_scens)]
            S_PPA = mdl.continuous_var(lb=0, ub=cplex.infinity, name='Yearly continuous baseload')
            for t in range(8736):
                hour_of_day = t % 24
                mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA for s in setS)

        elif self.name == 'Generic Industrial Yearly':
            setS = [i for i in range(n_scens)]
            S_PPA = mdl.continuous_var(lb=0, ub=cplex.infinity, name='Yearly industrial')
            for t in range(8736):
                hour_of_day = t % 24
                mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA for s in setS)
        
        elif self.name == 'Generic Private Yearly':
            setS = [i for i in range(n_scens)]
            S_PPA = mdl.continuous_var(lb=0, ub=cplex.infinity, name='Yearly private')
            for t in range(8736):
                hour_of_day = t % 24
                mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA for s in setS)
        
        elif self.name == 'Inverse Solar Yearly':
            setS = [i for i in range(n_scens)]
            S_PPA = mdl.continuous_var(lb=0, ub=cplex.infinity, name='Yearly inverse solar')
            for t in range(8736):
                hour_of_day = t % 24
                mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA for s in setS)
        
        elif self.name == 'Peak Yearly':
            setS = [i for i in range(n_scens)]
            S_PPA = mdl.continuous_var(lb=0, ub=cplex.infinity, name='Yearly peak')
            for t in range(8736):
                hour_of_day = t % 24
                mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA for s in setS)
        
        elif self.name == 'Binary Peak Yearly':
            setS = [i for i in range(n_scens)]
            S_PPA = mdl.continuous_var(lb=0, ub=cplex.infinity, name='Yearly binary peak')
            for t in range(8736):
                hour_of_day = t % 24
                mdl.add_constraints(P_PPA_t_s[s,t] == self.profile_matrix[hour_of_day] * S_PPA for s in setS)

        return S_PPA


    def get_constraints_ST(self, mdl, P_PPA_t, S_PPA, y_t, month, season, setT):
    
        if self.name == 'Monthly Continuous Baseload':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month] * (1-y_t[t]))

        elif self.name == 'Seasonal Continuous Baseload':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season] * (1-y_t[t]))

        elif self.name == 'Generic Industrial Monthly':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month] * (1-y_t[t]))

        elif self.name == 'Generic Industrial Seasonal':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season] * (1-y_t[t]))

        elif self.name == 'Generic Private Monthly':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == (self.profile_matrix[hour_of_day] * S_PPA[month]) * (1-y_t[t]))

        elif self.name == 'Generic Private Seasonal':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season] * (1-y_t[t]))

        elif self.name == 'Inverse Solar Monthly':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month] * (1-y_t[t]))

        elif self.name == 'Inverse Solar Seasonal':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season] * (1-y_t[t]))

        elif self.name == 'Peak Monthly':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month] * (1-y_t[t]))

        elif self.name == 'Peak Seasonal':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season] * (1-y_t[t]))
        
        elif self.name == 'Binary Peak Monthly':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month] * (1-y_t[t]))
        
        elif self.name == 'Binary Peak Seasonal':
            for t in setT: 
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season] * (1-y_t[t]))

        elif self.name == 'Yearly Continuous Baseload':
            for t in setT:
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA * (1-y_t[t]))
        
        elif self.name == 'Generic Industrial Yearly':
            for t in setT:
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA * (1-y_t[t]))
        
        elif self.name == 'Generic Private Yearly':
            for t in setT:
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA * (1-y_t[t]))
        
        elif self.name == 'Inverse Solar Yearly':
            for t in setT:
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA * (1-y_t[t]))
        
        elif self.name == 'Peak Yearly':
            for t in setT:
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * S_PPA * (1-y_t[t]))
        
        elif self.name == 'Binary Peak Yearly':
            for t in setT:
                hour_of_day = t % 24
                mdl.add_constraint(P_PPA_t[t] == self.profile_matrix[hour_of_day] * float(S_PPA) * (1-y_t[t]))


    

    def get_constraints_RT(self, mdl, P_PPA_promised_t, S_PPA, hour_of_day, month, season, setT):

        if self.name == 'Monthly Continuous Baseload':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month])

        elif self.name == 'Seasonal Continuous Baseload':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season])

        elif self.name == 'Generic Industrial Monthly':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month])

        elif self.name == 'Generic Industrial Seasonal':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season])

        elif self.name == 'Generic Private Monthly':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == (self.profile_matrix[hour_of_day] * S_PPA[month]))
                print(t,month, hour_of_day, self.profile_matrix[hour_of_day], S_PPA[month])
        elif self.name == 'Generic Private Seasonal':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season])

        elif self.name == 'Inverse Solar Monthly':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month])

        elif self.name == 'Inverse Solar Seasonal':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season])

        elif self.name == 'Peak Monthly':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month])

        elif self.name == 'Peak Seasonal':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season])

        elif self.name == 'Binary Peak Monthly':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[month])

        elif self.name == 'Binary Peak Seasonal':
            for t in setT: 
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA[season])

        elif self.name == 'Yearly Continuous Baseload':
            for t in setT:
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA)
        
        elif self.name == 'Generic Industrial Yearly':
            for t in setT:
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA)
        
        elif self.name == 'Generic Private Yearly':
            for t in setT:
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA)
        
        elif self.name == 'Inverse Solar Yearly':
            for t in setT:
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA)
        
        elif self.name == 'Peak Yearly':
            for t in setT:
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * S_PPA)
        
        elif self.name == 'Binary Peak Yearly':
            for t in setT:
                mdl.add_constraint(P_PPA_promised_t[t] == self.profile_matrix[hour_of_day] * float(S_PPA))


    def getPricePremiums(self, spotprices):
        hourly_generation = np.array(self.profile_matrix)

        # Define days in each month to sum to 8736 hours
        days_in_a_month = [30, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # 30 days in Jan for 8736 hours
        
        # Create the monthly hours list
        monthly_hours = [[] for _ in range(len(days_in_a_month))]
        hour_index = 0
        for month, days in enumerate(days_in_a_month):
            for day in range(days):
                for hour in range(24):
                    monthly_hours[month].append(hour_index)
                    hour_index += 1

        # Define seasons
        seasons = [
            monthly_hours[11] + monthly_hours[0] + monthly_hours[1],  # Winter is Dec, Jan, Feb
            monthly_hours[2] + monthly_hours[3] + monthly_hours[4],   # Spring
            monthly_hours[5] + monthly_hours[6] + monthly_hours[7],   # Summer
            monthly_hours[8] + monthly_hours[9] + monthly_hours[10],  # Fall
        ]

        # Calculate monthly capture prices and factors
        monthly_capture_prices = []
        monthly_capture_factors = []
        start_hour = 0

        for days in days_in_a_month:
            # Calculate the end hour for the current month
            end_hour = start_hour + days * 24
            
            # Extract spot prices for the current month
            monthly_spot_prices = spotprices[:, start_hour:end_hour]
            
            # Calculate average revenue for each hour of the day in the month
            average_revenue_hour_of_day = [
                hourly_generation[i] * np.mean(monthly_spot_prices[:, i::24])
                for i in range(24)
            ]
            
            # Compute the capture price for the month
            capture_price = sum(average_revenue_hour_of_day) / sum(hourly_generation)
            capture_factor = capture_price / np.mean(monthly_spot_prices)
            
            # Store results for the month
            monthly_capture_prices.append(capture_price)
            monthly_capture_factors.append(capture_factor)
            
            # Update start hour for the next month
            start_hour = end_hour

        # Calculate seasonal capture prices and factors
        seasonal_capture_prices = []
        seasonal_capture_factors = []

        for season_hours in seasons:
            seasonal_spot_prices = spotprices[:, season_hours]
            
            # Calculate average revenue for each hour of the day in the season
            average_revenue_hour_of_day = [
                hourly_generation[i] * np.mean(seasonal_spot_prices[:, i::24])
                for i in range(24)
            ]
            
            # Compute the capture price and factor for the season
            capture_price = sum(average_revenue_hour_of_day) / sum(hourly_generation)
            capture_factor = capture_price / np.mean(seasonal_spot_prices)
            
            seasonal_capture_prices.append(capture_price)
            seasonal_capture_factors.append(capture_factor)

        # Yearly averages
        yearly_capture_price = sum(monthly_capture_prices) / len(monthly_capture_prices)
        yearly_capture_factor = sum(monthly_capture_factors) / len(monthly_capture_factors)

        return monthly_capture_prices, monthly_capture_factors, seasonal_capture_prices, seasonal_capture_factors, yearly_capture_price, yearly_capture_factor

deliveryprofile_names = [
    DeliveryProfile('Seasonally Free', 0, None),
    DeliveryProfile('Monthly Free', 1, None),
    DeliveryProfile('Monthly Continuous Baseload', 2, [1] * 24),
    DeliveryProfile('Seasonal Continuous Baseload', 3, [1] * 24),
    DeliveryProfile('Yearly Continuous Baseload', 3, [1] * 24),
    DeliveryProfile('Generic Industrial Monthly', 4, [0.593030735, 0.554395406, 0.546245849, 0.53915524, 0.552433083, 0.620155643, 0.732682151, 0.84368243, 0.927410869, 0.975923914, 0.990102265, 0.987748587,0.988789471, 1, 0.996437111, 0.98324188, 0.960533801, 0.946952824, 0.911729005, 0.878990223, 0.811109466, 0.750608622, 0.682631749, 0.632859476]),
    DeliveryProfile('Generic Industrial Seasonal', 5, [0.593030735, 0.554395406, 0.546245849, 0.53915524, 0.552433083, 0.620155643, 0.732682151, 0.84368243, 0.927410869, 0.975923914, 0.990102265, 0.987748587,0.988789471, 1, 0.996437111, 0.98324188, 0.960533801, 0.946952824, 0.911729005, 0.878990223, 0.811109466, 0.750608622, 0.682631749, 0.632859476]),
    DeliveryProfile('Generic Industrial Yearly', 5, [0.593030735, 0.554395406, 0.546245849, 0.53915524, 0.552433083, 0.620155643, 0.732682151, 0.84368243, 0.927410869, 0.975923914, 0.990102265, 0.987748587,0.988789471, 1, 0.996437111, 0.98324188, 0.960533801, 0.946952824, 0.911729005, 0.878990223, 0.811109466, 0.750608622, 0.682631749, 0.632859476]),
    DeliveryProfile('Generic Private Monthly', 6, [0.405392308, 0.363868092, 0.331129724, 0.302547855, 0.285679627, 0.299305534, 0.411206526, 0.52205873, 0.507150756, 0.490946791, 0.486622381, 0.495561923, 0.51057742, 0.511541268, 0.521249321, 0.551870621, 0.655318008, 0.892684358, 1, 0.938593833, 0.829571565, 0.732613459, 0.604970492, 0.477711519]),
    DeliveryProfile('Generic Private Seasonal', 7, [0.405392308, 0.363868092, 0.331129724, 0.302547855, 0.285679627, 0.299305534, 0.411206526, 0.52205873, 0.507150756, 0.490946791, 0.486622381, 0.495561923, 0.51057742, 0.511541268, 0.521249321, 0.551870621, 0.655318008, 0.892684358, 1, 0.938593833, 0.829571565, 0.732613459, 0.604970492, 0.477711519]),
    DeliveryProfile('Generic Private Yearly', 7, [0.405392308, 0.363868092, 0.331129724, 0.302547855, 0.285679627, 0.299305534, 0.411206526, 0.52205873, 0.507150756, 0.490946791, 0.486622381, 0.495561923, 0.51057742, 0.511541268, 0.521249321, 0.551870621, 0.655318008, 0.892684358, 1, 0.938593833, 0.829571565, 0.732613459, 0.604970492, 0.477711519]),
    DeliveryProfile('Inverse Solar Monthly', 8, [1, 1, 1, 0.997772592, 0.919229953, 0.757901994, 0.558071701, 0.373355961, 0.201633432, 0.092013152, 0.010288502, 0, 0.069633008, 0.145736105, 0.277047094, 0.460065762, 0.650774289, 0.848589308, 0.97300594, 0.999363598, 1, 1, 1, 1]),
    DeliveryProfile('Inverse Solar Seasonal', 9, [1, 1, 1, 0.997772592, 0.919229953, 0.757901994, 0.558071701, 0.373355961, 0.201633432, 0.092013152, 0.010288502, 0, 0.069633008, 0.145736105, 0.277047094, 0.460065762, 0.650774289, 0.848589308, 0.97300594, 0.999363598, 1, 1, 1, 1]),
    DeliveryProfile('Inverse Solar Yearly', 9, [1, 1, 1, 0.997772592, 0.919229953, 0.757901994, 0.558071701, 0.373355961, 0.201633432, 0.092013152, 0.010288502, 0, 0.069633008, 0.145736105, 0.277047094, 0.460065762, 0.650774289, 0.848589308, 0.97300594, 0.999363598, 1, 1, 1, 1]),
    DeliveryProfile('Peak Monthly', 10, [0.653050827, 0.625423729, 0.62006779, 0.663525441, 0.781966078, 0.87440682, 0.882271149, 0.789559315, 0.779661017, 0.778881369, 0.728813559, 0.703389831, 0.682305081, 0.68572879, 0.714915234, 0.765084736, 0.818169495, 0.986915305, 1, 0.903254207, 0.826033875, 0.771186441, 0.681050861, 0.620338983]),
    DeliveryProfile('Peak Seasonal', 11, [0.653050827, 0.625423729, 0.62006779, 0.663525441, 0.781966078, 0.87440682, 0.882271149, 0.789559315, 0.779661017, 0.778881369, 0.728813559, 0.703389831, 0.682305081, 0.68572879, 0.714915234, 0.765084736, 0.818169495, 0.986915305, 1, 0.903254207, 0.826033875, 0.771186441, 0.681050861, 0.620338983]),
    DeliveryProfile('Peak Yearly', 11, [0.653050827, 0.625423729, 0.62006779, 0.663525441, 0.781966078, 0.87440682, 0.882271149, 0.789559315, 0.779661017, 0.778881369, 0.728813559, 0.703389831, 0.682305081, 0.68572879, 0.714915234, 0.765084736, 0.818169495, 0.986915305, 1, 0.903254207, 0.826033875, 0.771186441, 0.681050861, 0.620338983]),
    DeliveryProfile('Binary Peak Monthly', 12, [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]),
    DeliveryProfile('Binary Peak Seasonal', 13, [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]),
    DeliveryProfile('Binary Peak Yearly', 13, [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]),
    ]