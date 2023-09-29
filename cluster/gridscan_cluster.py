'''
Gridscan script for launching on cluster 

Author: Maksim Valialshchikov, @maxbalrog (github)
'''

default_name = 'default_params.yml'
vary_name = 'vary_params.yml'


def main(save_path):
    default_yaml = f'{save_path}{default_name}'
    vary_yaml = f'{save_path}{vary_name}'
    
    run_gridscan(default_yaml, vary_yaml, save_path)


if __name__ == '__main__':
    from scripts.vacem_simulation import run_gridscan
    import argparse
    
    parser = argparse.ArgumentParser(description='Gridscan specifications')
    parser.add_argument('--save_path', metavar='path', required=True,
                        help='the path to folder with simulation specifications')
    args = parser.parse_args()
    
    main(save_path=args.save_path)


