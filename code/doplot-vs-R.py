import fitsio
import numpy as np
import biggles
import pcolors

from argparse import ArgumentParser

parser=ArgumentParser()

parser.add_argument(
    'files',
    nargs='+',
    help='files to read',
)

def read_data(fname):
    with fitsio.FITS(fname) as fits:
        data = fits[1].read()
        R = fits[2]['R'][:]

    return data, R[0]

def main():
    args = parser.parse_args()

    fs = args.files[0].split('-')
    pdfname = '-'.join(fs[0:3] + fs[4:])
    pdfname = pdfname.replace('.fits','.pdf')

    num=len(args.files)
    Rvals=np.zeros(num)
    means=np.zeros(num)
    means_predict=np.zeros(num)
    for i,f in enumerate(args.files):

        data,R = read_data(f)
        print(R)

        # means over s/n
        Rvals[i] = R
        means[i] = data['err_ratio'].mean()
        means_predict[i] = (data['mean_coadd_err']/data['mean_err']).mean()

    key=biggles.PlotKey(
        0.9, 0.9,
        halign='right',
    )
    plt=biggles.FramedPlot(
        xlabel='R',
        ylabel=r'$noise/optimal noise$',
        key=key,
        aspect_ratio=1.0/1.618,
    )


    curve = biggles.Curve(
        Rvals,
        means_predict,
        color='blue',
        type='solid',
        label='predicted',
    )

    frac = 0.1
    means_predict_sq = 1.0 + 2*(1.0 - Rvals)**2 *frac**2
    means_predict_v2 = np.sqrt(means_predict_sq)
    c2 = biggles.Curve(
        Rvals,
        means_predict_v2,
        color='magenta',
        type='dashed',
        label='toy model',
    )


    pts = biggles.Points(
        Rvals,
        means,
        color='red',
        type='filled circle',
        label='measured',
    )

    plt.add(pts, curve, c2)

    #plt.show()
    print("writing:",pdfname)
    plt.write(pdfname)


main()
