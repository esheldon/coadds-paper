from __future__ import print_function
import fitsio
import ngmix
import nsim
import numpy
import esutil as eu


def get_std(sqsum, ssum, n):
    return numpy.sqrt( sqsum/n - ssum**2/n**2 )
    #return sqsum/n - ssum**2/n**2

def get_ratio(sqsum1, sum1, sqsum2, sum2, n):
    return get_std(sqsum1, sum1, n)/get_std(sqsum2, sum2, n)

def jackknife_std_ratio(d1, d2, chunksize=1):
    assert d1.size==d2.size
    ntot = d1.size

    nchunks = ntot//chunksize

    g1 = d1['mcal_g'][:,0]
    g1sq = g1**2
    g2 = d2['mcal_g'][:,0]
    g2sq = g2**2

    g1sum = g1.sum()
    g2sum = g2.sum()
    g1sqsum = g1sq.sum()
    g2sqsum = g2sq.sum()

    #ratio = (g1sqsum/ntot - g1sum**2/ntot**2) / (g2sqsum - g2sum**2)

    ratio = get_ratio(g1sqsum, g1sum, g2sqsum, g2sum, ntot)


    ratios = numpy.zeros(nchunks)
    for i in xrange(nchunks):

        beg = i*chunksize
        end = (i+1)*chunksize

        tn = ntot - (end-beg+1)

        tg1sum  = g1[beg:end].sum()
        tg2sum  = g2[beg:end].sum()
        tg1sqsum  = g1sq[beg:end].sum()
        tg2sqsum  = g2sq[beg:end].sum()

        j_g1sum = g1sum - tg1sum
        j_g2sum = g2sum - tg2sum
        j_g1sqsum = g1sqsum - tg1sqsum
        j_g2sqsum = g2sqsum - tg2sqsum

        #ratios[i] = (j_g1sqsum - j_g1sum**2) / (j_g2sqsum - j_g2sum**2)
        ratios[i] = get_ratio(
            j_g1sqsum, j_g1sum, j_g2sqsum, j_g2sum, tn,
        )

    fac = (nchunks-1)/float(nchunks)

    ratio_cov = fac*( ((ratio-ratios)**2).sum() )

    ratio_err = numpy.sqrt(ratio_cov)
    return ratio, ratio_err


def read_data(runlist):
    #n=10000
    #n=100000
    n=17600000
    #n=18000000
    rows = numpy.arange(n)
    columns=[
        'mcal_s2n_r',
        'mcal_g',
        'mcal_g_1p',
        'mcal_g_1m',
        'mcal_g_2p',
        'mcal_g_2m',
    ]
    print("reading",n,"from run",runlist)

    flist = [nsim.files.get_output_url(r) for r in runlist]
    #data = nsim.files.read_output(run, rows=rows,columns=columns)
    data = eu.io.read(flist, columns=columns)

    data = data[0:n]

    assert data.size == n
    return data

def read_data_old(run):
    #n=10000
    n=100000
    #n=17600000
    #n=18000000
    rows = numpy.arange(n)
    columns=[
        'mcal_s2n_r',
        'mcal_g',
        'mcal_g_1p',
        'mcal_g_1m',
        'mcal_g_2p',
        'mcal_g_2m',
    ]
    print("reading",n,"from run",run)

    data = nsim.files.read_output(run, rows=rows,columns=columns)

    data = data[0:n]

    assert data.size == n
    return data


def get_stats(runlist):
    #d = read_data_old(runlist[0])
    d = read_data(runlist)

    print("getting subset")

    print("jackknifing")
    res = ngmix.metacal.jackknife_shear(d)

    print("getting std")
    gstd = d['mcal_g'].std(axis=0)
    return res, gstd, d

def main():
    # this one coadded
    r1, gstd1, d1 = get_stats(
        ['run-e30-mcal-01b',
         'run-e30-mcal-02b',
         'run-e30-mcal-03b'],
    )
    r2, gstd2, d2 = get_stats(
        ['run-e31-mcal-01b',
         'run-e31-mcal-02b',
         'run-e31-mcal-03b'],
    )

    print("shear err ratio")
    print(r1['shear_err'])
    print(r2['shear_err'])
    print(r1['shear_err']/r2['shear_err'])

    print("std ratio")
    print(gstd1/gstd2)

    print("std ratio jackknifed")
    ratio, ratio_err = jackknife_std_ratio(d1, d2, chunksize=1)
    print("%g +/- %g" % (ratio, ratio_err))

main()
