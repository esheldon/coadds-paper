import os
import fitsio
import numpy as np
import ngmix
import galsim
import esutil as eu
import biggles
import images
from multiprocessing import Pool

from argparse import ArgumentParser

parser=ArgumentParser()

scale=0.263
DEFAULT_FWHM=0.9
DEFAULT_FRAC=0.1

parser.add_argument(
    'ntrial',
    type=int,
    help='number of trials',
)
parser.add_argument(
    'seed',
    type=int,
    help='seed for rng',
)


parser.add_argument(
    '--nepoch',
    default=10,
    type=int,
    help='number of images per observation',
)
#parser.add_argument(
#    '--noise',
#    type=float,
#    default=0.01,
#    help='noise for images',
#)
parser.add_argument(
    '--fwhm',
    type=float,
    default=DEFAULT_FWHM,
    help='fwhm for the psf',
)
parser.add_argument(
    '--frac',
    type=float,
    default=DEFAULT_FRAC,
    help='fractional variation for fwhm',
)
parser.add_argument(
    '--fwhm-gal',
    type=float,
    default=None,
    help='fwhm for galaxies',
)


def get_R(args):
    if args.fwhm_gal is not None:
        Rfac = args.fwhm**2/(args.fwhm**2 + args.fwhm_gal**2)
        Rfac = Rfac**2
        R = args.fwhm_gal**2/(args.fwhm**2 + args.fwhm_gal**2)
        print("R:",R,"Rfac:",Rfac)
    else:
        Rfac = 1.0
        R = 0.0

    return R, Rfac

def predict_var_fac(args, s2n):

    R, Rfac = get_R(args)
    fac=(1.0 + 1.0/args.nepoch * Rfac * (args.frac * s2n)**2 )
    return fac

class ObsMaker(object):
    def __init__(self, args, noise, seed):

        self.args=args
        self._noise=noise

        sigma, sigma_tot, sigma_gal = get_sigmas(args)
        self._sigma=sigma
        self._sigma_gal=sigma_gal
        self._sigma_tot=sigma_tot

        self._sigma_frac=args.frac

        dim=int(round(2*self._sigma_tot*5))
        if (dim % 2) == 0:
            dim += 1
        self._dim=dim
        self._cen = (dim-1.0)/2.0

        self.rng = np.random.RandomState(seed)
        gs_seed = self.rng.randint(low=0, high=2**15)
        gs_rng=galsim.BaseDeviate(int(gs_seed))
        self._noiseobj=galsim.GaussianNoise(rng=gs_rng, sigma=noise)

    def get_obs_list(self):
        obslist=ngmix.ObsList()

        for i in range(self.args.nepoch):
            obs = self._make_obs()
            obslist.append( obs )

        return obslist

    def _make_obs(self):

        im, im_nonoise, wt, jac, gm = self._make_epoch()

        obs = ngmix.Observation(
            im,
            weight=wt,
            jacobian=jac,
            gmix=gm,
        )

        obs.im_nonoise = im_nonoise

        return obs

    def _make_epoch(self):
        frac = self.rng.normal(scale=self._sigma_frac)
        sigma = self._sigma*(1.0 + frac)

        obj = galsim.Gaussian(sigma=sigma)

        if self._sigma_gal is not None:
            gal = galsim.Gaussian(sigma=self._sigma_gal)
            obj = galsim.Convolve(obj, gal)
            sigma_tot2 = sigma**2 + self._sigma_gal**2
        else:
            sigma_tot2 = sigma**2

        T = 2*sigma_tot2
        pars=[0.0, 0.0, 0.0, 0.0, T, 1.0]
        gm = ngmix.GMixModel(pars, 'gauss')

        imgs=obj.drawImage(
            nx=self._dim,
            ny=self._dim,
            scale=1.0,
            method='no_pixel',
        )

        im_nonoise=imgs.array.copy()

        imgs.addNoise( self._noiseobj )

        im=imgs.array
        wt = im*0 + 1.0/self._noise**2

        jacobian=ngmix.UnitJacobian(
            row=self._cen,
            col=self._cen,
        )


        return im, im_nonoise, wt, jacobian, gm

def dofit(args, obs):
    fitter=ngmix.fitting.TemplateFluxFitter(obs)
    fitter.go()

    res=fitter.get_result()
    return res

def get_sigmas(args):
    sigma=ngmix.moments.fwhm_to_sigma(args.fwhm/scale)
    if args.fwhm_gal is not None:
        sigma_gal = ngmix.moments.fwhm_to_sigma(args.fwhm_gal/scale)
        sigma_tot = np.sqrt(sigma**2 + sigma_gal**2)
    else:
        sigma_gal = None
        sigma_tot = sigma

    return sigma, sigma_tot, sigma_gal

def fit_gauss(obs, gm):

    obs.im_noisy = obs.image
    obs.image = obs.im_nonoise

    Tguess=gm.get_T()
    am = ngmix.admom.Admom(obs)
    am.go(gm)

    res=am.get_result()

    if res['flags'] == 0:
        gm = am.get_gmix()
    else:
        raise ngmix.gexceptions.BootPSFFailure("failed to fit")

    obs.image = obs.im_noisy
    return gm

def make_coadd(args, obs_list):
    im = obs_list[0].image*0
    im_nonoise = obs_list[0].image*0
    wt = obs_list[0].weight*0
    jac = obs_list[0].jacobian.copy()

    nobs = len(obs_list)
    for obs in obs_list:

        im += obs.image
        im_nonoise += obs.im_nonoise
        wt += obs.weight

    im *= 1.0/nobs
    im_nonoise *= 1.0/nobs

    coadd_obs = ngmix.Observation(
        im,
        weight=wt,
        jacobian=jac,
    )
    """
    coadd_obs.im_nonoise = im_nonoise

    sigma, sigma_tot, sigma_gal = get_sigmas(args)
    pars=[0.0, 0.0, 0.0, 0.0, 2*sigma_tot**2, 1.0]
    gm = ngmix.GMixModel(pars, 'gauss')

    gm = fit_gauss(coadd_obs, gm)
    coadd_obs.set_gmix(gm)
    """
    coadd_obs.template = im_nonoise


    return coadd_obs

def make_fname(args):
    R,Rfac = get_R(args)
    fname='cnoise-fwhm%.2f-frac%.2f-R%.2f-ntrial%d' % (
        args.fwhm,
        args.frac,
        R,
        args.ntrial,
    )

    fname += '.fits'
    fname=os.path.join('output',fname)
    return fname

def make_epsname(args):
    fname=make_fname(args)
    fname = fname.replace('.fits','.eps')
    return fname

def write(args, data):
    s2n = data['s2n'].mean()
    fname=make_fname(args, s2n)

    print("writing:",fname)
    fitsio.write(fname, data, clobber=True)

def do_one(allargs):
    args, noise, seed, ntrial = allargs

    edtype=[
        ('flux','f8'),
        ('flux_err','f8'),
        ('s2n','f8'),
        ('coadd_flux','f8'),
        ('coadd_flux_err','f8'),
        ('coadd_s2n','f8'),
    ]

    sdtype=[
        ('s2n','f8'),
        ('coadd_s2n','f8'),
        ('mean_err','f8'),
        ('mean_coadd_err','f8'),
        ('fac','f8'),
        ('fac_meas','f8'),
        ('err_ratio','f8'),
    ]
    sdata = np.zeros(1,dtype=sdtype)


    maker=ObsMaker(args, noise, seed)
    d = np.zeros(ntrial,dtype=edtype)

    i=0
    while i < ntrial:
    #for i in range(ntrial):
        obs_list = maker.get_obs_list()

        try:
            coadd_obs = make_coadd(args, obs_list)
        except ngmix.gexceptions.BootPSFFailure:
            continue


        coadd_res=dofit(args, coadd_obs)
        res=dofit(args, obs_list)

        #print(res)
        td = d[i]
        td['flux'] = res['flux']
        td['flux_err'] = res['flux_err']
        td['s2n'] = res['flux']/res['flux_err']

        td['coadd_flux'] = coadd_res['flux']
        td['coadd_flux_err'] = coadd_res['flux_err']
        td['coadd_s2n'] = coadd_res['flux']/coadd_res['flux_err']

        i += 1

    eu.stat.print_stats(d['s2n'])

    sdata['s2n'] = d['s2n'].mean()
    sdata['coadd_s2n'] = d['coadd_s2n'].mean()

    mean_err = np.sqrt( (d['flux_err']**2).mean())
    mean_coadd_err = np.sqrt( (d['coadd_flux_err']**2).mean())

    err_meas = d['flux'].std()
    coadd_err_meas = d['coadd_flux'].std()

    fac_meas = coadd_err_meas/mean_coadd_err

    fac2 = predict_var_fac(args, sdata['coadd_s2n'])
    #fac2 = predict_var_fac(args, sdata['s2n'])
    fac = np.sqrt(fac2)

    sdata['mean_err'] = mean_err
    sdata['mean_coadd_err'] = mean_coadd_err
    sdata['fac'] = fac
    sdata['fac_meas'] = fac_meas

    err_rat = coadd_err_meas/err_meas
    sdata['err_ratio'] = err_rat
    #print(err_meas, coadd_err_meas, err_rat)

    #images.multiview(coadd_obs.image)
    #stop
    return sdata

def get_noise_range(args):
    min_noise = 0.01
    max_noise = 0.13

    res = do_one( (args, min_noise, 31141, 100) )

    """
    maker=ObsMaker(args, max_noise, 31141)
    obs_list = maker.get_obs_list()
    coadd_obs = make_coadd(args, obs_list)

    res=dofit(args, obs_list)

    s2n = res['flux']/res['flux_err']
    """
    fac = res['s2n']/60.0
    print("fac:",fac)
    max_noise = max_noise*fac
    min_noise = min_noise*fac

    return [min_noise, max_noise]

def main():
    args=parser.parse_args()

    noise_range = get_noise_range(args)
    nnoise=12
    inoises=np.linspace(1.0/noise_range[1], 1.0/noise_range[0], nnoise)
    noises = 1.0/inoises

    #pmap = Pool(4).map
    pmap = Pool().map
    #pmap=map

    np.random.seed(args.seed)
    seeds=np.random.randint(low=0, high=2**15, size=nnoise)
    allargs = [(args, ns[0], ns[1], args.ntrial) for ns in zip(noises,seeds)]
    results = pmap(do_one, allargs)

    sdata = eu.numpy_util.combine_arrlist(results)

    key=biggles.PlotKey(
        0.1, 0.9,
        halign='left',
    )
    plt=biggles.FramedPlot(
        xlabel='S/N',
        ylabel=r'$\sigma_m/\sigma',
        key=key,
    )
    #pc = biggles.Curve(
    #    sdata['s2n'],
    #    sdata['fac'],
    #    label='predicted',
    #)


    mpts = biggles.Points(
        sdata['s2n'],
        #sdata['fac_meas'],
        sdata['err_ratio'],
        color='red',
        label='measured',
        type='filled circle',
    )

    mean_predict = (sdata['mean_coadd_err']/sdata['mean_err']).mean()
    pc = biggles.LineY(
        mean_predict,
        color='blue',
        label='predicted',
    )

    plt.add(mpts)
    plt.add(pc)

    #plt.show()

    R,Rfac = get_R(args)
    Rdata = np.zeros(1, dtype=[('R','f8')])
    Rdata['R'] = R
    fname = make_fname(args)
    eps_name = make_epsname(args)
    print('writing:',eps_name)
    plt.write_eps(eps_name)
    print('writing:',fname)
    with fitsio.FITS(fname,'rw',clobber=True) as fits:
        fits.write(sdata)
        fits.write(Rdata)

    #write(args, d)

main()
