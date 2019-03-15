from z0mgs_dust.fitting import template_generator

ks = {'SE': 10.10, 'FB': 25.83, 'BE': 20.73}

for model_name in ['SE', 'FB', 'BE']:
    for instr in ['spire', 'pacs']:
        template_generator(template_name='default',
                           model_name=model_name,
                           beta_f=2.0,
                           lambdac_f=300.0,
                           instr=instr,
                           kappa_160=ks[model_name],
                           parallel_mode=True,
                           num_proc=5)

# Need calibration code for beta 1.8 to work
