import random
import os
template="""
command: |
    . ~/.bashrc
    source activate nsim
    python test.py {ntrial} {seed} {fwhm_setting}
mode: bynode
"""

random.seed(3141)

ntrial=100000
fwhms=[None] + [0.3, 0.5, 0.65, 0.9, 1.25, 1.4, 2.0]
for fwhm in fwhms:
    seed=random.randint(0,2**15)
    if fwhm is None:
        fwhm_setting=""
        fname='run-star-ntrial%06d.yaml' % ntrial
    else:
        fwhm_setting='--fwhm-gal=%g' % fwhm
        fname='run-%.2f-ntrial%06d.yaml' % (fwhm, ntrial)

    text = template.format(
        seed=seed,
        fwhm_setting=fwhm_setting,
        ntrial=ntrial,
    )

    print fname
    with open(fname,'w') as fobj:
        fobj.write(text)
