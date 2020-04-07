# +
import random

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

# # !pip install pulp
from pulp import *
# -

# # Constants

# +
teams = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
UNIT_COST = 320

skill_filename = 'all_teams_skills.xlsx'
attendance_filename = 'all_team_att.xlsx'
preference_filename = 'all_teams_prefer.xlsx'


# -

# # Functions

# ### Read data

def read_team_data(team, skill_filename, attendance_filename, preference_filename):
    skill_sheetname = f'Team{team}_Skills'
    attendance_sheetname = f'Team{team}_Att'
    preference_sheetname = f'Team{team}'
    
    team_dict = dict()
    team_dict['skills'] = pd.read_excel(skill_filename, sheet_name=skill_sheetname, index_col=0)
    team_dict['attendance'] = pd.read_excel(attendance_filename, sheet_name=attendance_sheetname, index_col=0)
    team_dict['preference'] = pd.read_excel(preference_filename, sheet_name=preference_sheetname, index_col=0)
    return team_dict


def read_data(skill_filename=skill_filename, attendance_filename=attendance_filename, preference_filename=preference_filename):
    data = dict()
    for team in teams:
        data[team] = read_team_data(team, skill_filename, attendance_filename, preference_filename)
    return data


# ### Employee attendance

def generate_attendance(att):
    att_prob = att.mean(axis=1) 
    present = att_prob > np.random.rand(*att_prob.shape)
    return present[present == True].index.to_list()


# ### Skill based assignment

# +
# Function for skill based optimization

def skill_based_optimize(skills, people_present):
    skills_present = skills[skills.index.isin(people_present)]
    skills_present = skills_present.applymap(lambda x: 0.5 if x == 0 else x)
    
    workstations = skills_present.columns.to_list()
    employees = skills_present.index.to_list()
    
    prob = LpProblem("assignment", LpMinimize)  # Minimization problem
    assignment = LpVariable.dicts('Assign', (employees, workstations), cat='Binary')  # Each value of matrix is a binary variable
    prob += lpSum([assignment[emp][wrk] for emp in employees for wrk in workstations])  # Objective, which is sum of assignment matrix
    
    for workstation in workstations:
        # for each workstation, sum(assignment * skill) over all employees = 1
        prob += lpSum([assignment[employee][workstation] * skills_present.loc[employee, workstation] for employee in employees]) == 1
        
    for employee in employees:
        # Each employee can be assigned at max 1 workstation
        prob += lpSum([assignment[employee][workstation] for workstation in workstations]) <= 1
        
    prob.solve()
    solve_status = LpStatus[prob.status]
    
    if solve_status == 'Optimal':
        # Do something
        result = pd.DataFrame(assignment).applymap(lambda x: x.varValue).T.astype(int)
        people_assigned = value(prob.objective)
        workstation_people_count = result.sum()
        excess_people = result.sum(axis=1).loc[lambda x: x == 0].index.to_list()
        return True, result, excess_people  # First element tells if solution was found
    else:
        # Do something else
        return False, None, None


# -

# ### Preference based assignment

# +
# Function for preference based optimization

def preference_maximize(skill_assignment, skills, preference, unit_cost=UNIT_COST):
    people_present = skill_assignment.index.to_list()
    skill_assigned_count = skill_assignment.sum().sum()
    
    skills_present = skills[skills.index.isin(people_present)]
    skills_present = skills_present.applymap(lambda x: 0.5 if x == 0 else x)
    
    workstations = skills_present.columns.to_list()
    employees = skills_present.index.to_list()
    
    prob = LpProblem("preference", LpMaximize)
    assignment = LpVariable.dicts('Prefer', (employees, workstations), cat='Binary')

    prob += lpSum([preference.loc[emp, wrk] * assignment[emp][wrk] for emp in employees for wrk in workstations])
    
    for workstation in workstations:
        # for each workstation, sum(assignment * skill) over all employees = 1
        prob += lpSum([assignment[employee][workstation] * skills_present.loc[employee, workstation] for employee in employees]) == 1
        
    for employee in employees:
        # Each employee can be assigned at max 1 workstation
        prob += lpSum([assignment[employee][workstation] for workstation in workstations]) <= 1
        
    prob += lpSum([assignment[emp][wrk] for emp in employees for wrk in workstations]) == skill_assigned_count
    
    prob.solve()
    
    solve_status = LpStatus[prob.status]
    
    if solve_status == 'Optimal':
        # Do something
        result = pd.DataFrame(assignment).applymap(lambda x: x.varValue).T.astype(int)
        preference_matched_count = int(value(prob.objective))
#         worsktation_people_count = result.sum()
        excess_people = result.sum(axis=1).loc[lambda x: x == 0].index.to_list()
        assigned_people_cost = result.sum().sum() * unit_cost

#         print(f'Preferences matched for {preference_matched_count} out of {skill_assigned_count} people.\n')
        return result, excess_people, assigned_people_cost
    else:
        # Do something else
        return None, None, None


# -

#  ### Cross training

crosstrain_daycount = dict()


# +
# Run only if optimal solution is found

def crosstraining_update(excess_people, skills, preference_assignment, crosstrain_daycount, unit_cost=320):
    
    all_workstations = preference_assignment.columns.to_list()
    occupied_workstations = [list(v.keys())[0] for k,v in crosstrain_daycount.items()]
    available_workstations = list(set(all_workstations) - set(occupied_workstations))
    total_cost = 0

    # Cross-training operations
    for emp in excess_people:
        emp_unskilled_workstations = skills.loc[emp].loc[lambda x: x == 0].index.to_list()
        
        if len(emp_unskilled_workstations) == 0:
            continue
        
        if emp in crosstrain_daycount.keys():
            this_workstation = list(crosstrain_daycount[emp].keys())[0]
            current_daycount = crosstrain_daycount[emp][this_workstation]

            # Update skills of employees with 2 days of cross-training
            if current_daycount >= 2:
                # Skills update
                skills.loc[emp, this_workstation] = 1

                # Remove employee from daycount and free up the workstation
                crosstrain_daycount.pop(emp)

            # Ask employee for one more day of training if daycount < 2
            elif current_daycount < 2:
                crosstrain_daycount[emp][this_workstation] += 1
                total_cost += unit_cost

        # If employee is cross-training for first time, 
        # assign new empty workstation where employee is unskilled randomly
        else:
            workstation_candidates = list(set(emp_unskilled_workstations).intersection(set(available_workstations)))
            if workstation_candidates:
                this_workstation = random.choice(workstation_candidates)
                this_dict = {this_workstation: 0}
                crosstrain_daycount[emp] = this_dict
                available_workstations.remove(this_workstation)
                total_cost += unit_cost
            else:
                # If no workstation remains unskilled for this employee, pass
                pass
            
    return skills, crosstrain_daycount, total_cost


# -

print(crosstrain_daycount)

skills

# # Main Pipeline

# +
# Read in all the team data into a dictionary

data = read_data()
# -

data['A']['skills']

# +
skill_data = dict()
excess_from_skill_assignment = list()

for team in teams:
    people_present = generate_attendance(data[team]['attendance'])
    
    team_skill_results = dict()
    team_skill_results['status'], team_skill_results['assignment'], excess = \
    skill_based_optimize(data[team]['skills'], people_present)
    if excess:
        excess_from_skill_assignment.extend(excess)
    skill_data[team] = team_skill_results
# -

excess_from_skill_assignment

failed_teams = [k for k, v in skill_data.items() if v['status'] == False]

failed_teams

preference_assignment, excess_people = preference_maximize(skill_assignment, skills, preference)
preference_assignment

# # Single team pipeline

# +
team = 'A'
crosstrain_daycount = dict()

data = read_data()
skills = data[team]['skills']
attendance = data[team]['attendance']
preference = data[team]['preference']

days = []
skill_sum = []
total_cost = []

for i in range(500):
    days.append(i)
    skill_sum.append(skills.sum().sum())
    people_present = generate_attendance(attendance)
    
    skill_assignment_status, skill_assignment, skill_excess = skill_based_optimize(skills, people_present)
    
    if not skill_assignment_status:
        total_cost.append(None)
        continue
    
    preference_assignment, preference_excess, assigned_people_cost = preference_maximize(skill_assignment, skills, preference)
    skills, crosstrain_daycount, crosstrain_cost = crosstraining_update(preference_excess, skills, preference_assignment, crosstrain_daycount)
    net_cost = assigned_people_cost + crosstrain_cost
    total_cost.append(net_cost)
# -

tdf = pd.DataFrame(data={'days': days, 'total_cost': total_cost, 'skill_sum': skill_sum})

fig, ax = plt.subplots(figsize=(15, 5))
tdf.plot(x='days', y='skill_sum', kind='line', ax=ax)
# tdf.plot(x='days', y='')

preference

skills


