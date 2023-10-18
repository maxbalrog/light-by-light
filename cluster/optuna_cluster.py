'''
Optuna script for launching on cluster 

Author: Maksim Valialshchikov, @maxbalrog (github)
'''
from light_by_light.optuna_optimization import run_optuna_optimization

default_name = 'default_params.yml'
optuna_name = 'optuna_params.yml'


def main(save_path):
    default_yaml = f'{save_path}{default_name}'
    vary_yaml = f'{save_path}{optuna_name}'
    
    run_optuna_optimization(default_yaml, vary_yaml, save_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna specifications')
    parser.add_argument('--save_path', metavar='path', required=True,
                        help='the path to folder with simulation specifications')
    args = parser.parse_args()
    
    main(save_path=args.save_path)


