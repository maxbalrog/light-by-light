'''
Simulation script for launching on cluster 

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
from light_by_light.vacem_simulation import run_simulation_postprocess
from light_by_light.utils import read_yaml

default_name = 'default_params.yml'


def main(save_path):
    default_yaml = f'{save_path}{default_name}'
    yaml_data = read_yaml(default_yaml)
    params = {}
    lasers = yaml_data['lasers']
    params['laser_params'] = [lasers[f'laser_{i}'] for i in range(len(lasers))]
    params['save_path'] = save_path
    keys = ['simbox_params', 'geometry', 'low_memory_mode', 'n_threads', 'pol_idx']
    for key in keys:
        params[key] = yaml_data[key]
    
    run_simulation_postprocess(**params)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulation parameters')
    parser.add_argument('--save_path', metavar='path', required=True,
                        help='the path to folder with simulation parameters')
    args = parser.parse_args()
    
    main(save_path=args.save_path)


