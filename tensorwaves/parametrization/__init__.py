from tensorwaves.parametrization import lobato

potentials = {'lobato': {'potential': lobato.potential,
                         'soft_potential': lobato.soft_potential,
                         'projected_potential': lobato.projected_potential,
                         'default_parameters': 'tensorwaves/parametrization/data/lobato.txt'}
              }
