from pathlib import Path
from citylearn import CityLearn

def create_environment(n_steps):
    if n_steps >= 8760*4:
        raise ValueError('n_steps must be <= 8760*4-1 (4 years)')

    climate_zone = 5
    params = {'data_path':Path("data/Climate_Zone_"+str(climate_zone)),
            'building_attributes':'building_attributes.json',
            'weather_file':'weather_data.csv',
            'solar_profile':'solar_generation_1kW.csv',
            'carbon_intensity':'carbon_intensity.csv',
            'building_ids':["Building_"+str(i) for i in [5,6,7,8,9]],
            'buildings_states_actions':'buildings_state_action_space.json',
            'simulation_period': (0, n_steps),
            'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'],
            'central_agent': False,
            'save_memory': False }
    return CityLearn(**params)