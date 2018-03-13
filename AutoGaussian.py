# Auto Gaussian fit
# Allen Scheie
# May, 2017

import numpy as np
from scipy.optimize import curve_fit

def gaus(x, a, x0, sig):
    return a/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - x0)/sig, 2.)/2)

def lorentzian(x, a, x0, gamma):
	return a/np.pi * (0.5*gamma)/((x-x0)**2 + (0.5*gamma)**2)

#def fitfun(x, a, x0, sig, bg, bgs):
#	return gaus(x,a,x0,sig) + bg + bgs*x
def fitfun(x, a, x0, sig, bg):
	return gaus(x,a,x0,sig) + bg #+ bgs*x

def autoGaussianFit(datax, datay, **kwargs):
	# Step I: auto-find starting parameters
	#Step 1: find maximum in datay
	maxy = np.amax(datay)
	miny = np.amin(datay)
	argmaxy = np.argmax(datay)
	maxxpt = datax[argmaxy]
	# step 2: Find fwhm in peak of datay
	try:
		for i in range(len(datay)):
			if (datay[argmaxy + i]-miny) < ((maxy-miny)/2.):
				hwhm = np.abs(datax[argmaxy + i] - maxxpt)
				break
	except IndexError:
		hwhm = np.abs(maxxpt - datax[0])/2.
	# Step 3: Estimate area under peak
	estarea = hwhm*(maxy-miny)

	# Step II: Fit
	try:
		popt,pcov = curve_fit(fitfun, datax, datay,p0=[estarea,maxxpt,hwhm, miny], **kwargs)

		fitresults = {'a':popt[0], 'x0':popt[1], 'sigma':popt[2], 'bg':popt[3]}
		fituncertainty = {'a':pcov[0,0]**0.5, 'x0':pcov[1,1]**0.5, 'sigma':pcov[2,2]**0.5, 
							'bg':pcov[3,3]**0.5}
		# fitresults = {'a':popt[0], 'x0':popt[1], 'sigma':popt[2], 'bg':popt[3], 'bgs':popt[4]}
		# fituncertainty = {'a':pcov[0,0]**0.5, 'x0':pcov[1,1]**0.5, 'sigma':pcov[2,2]**0.5, 
		# 					'bg':pcov[3,3]**0.5, 'bgs':pcov[4,4]**0.5}

		# Create strings of fit results
		gausfitParamStr = {}
		for prm in fitresults:
			dist = int(np.log10(np.abs(fituncertainty[prm])))-1
			gausfitParamStr[prm] = '$'+str(round(fitresults[prm],-dist))+' \pm '+\
					str(round(fituncertainty[prm],-dist))+'$'

		return fitresults, fituncertainty, gausfitParamStr

	except RuntimeError:
		print("Failed to fit Gaussian function")
		return {'a':0, 'x0':np.nan, 'sigma':np.nan,'bg':np.nan},{'a':np.nan, 'x0':np.nan, 'sigma':np.nan,'bg':np.nan},{'a':'', 'x0':'', 'sigma':'','bg':''}


def fitfunLorentz(x, a, x0, gamma, bg):
	return gaus(x,a,x0,gamma) + bg #+ bgs*x

def autoLorentzianFit(datax, datay, **kwargs):
	# Step I: auto-find starting parameters
	#Step 1: find maximum in datay
	maxy = np.amax(datay)
	miny = np.amin(datay)
	argmaxy = np.argmax(datay)
	maxxpt = datax[argmaxy]
	# step 2: Find fwhm in peak of datay
	try:
		for i in range(len(datay)):
			if (datay[argmaxy + i]-miny) < ((maxy-miny)/2.):
				hwhm = np.abs(datax[argmaxy + i] - maxxpt)
				break
	except IndexError:
		hwhm = np.abs(maxxpt - datax[0])/2.
	# Step 3: Estimate area under peak
	estarea = hwhm*(maxy-miny)

	# Step II: Fit
	try:
		popt,pcov = curve_fit(fitfunLorentz, datax, datay,p0=[estarea,maxxpt,hwhm, miny], **kwargs)

		fitresults = {'a':popt[0], 'x0':popt[1], 'gamma':popt[2], 'bg':popt[3]}
		fituncertainty = {'a':pcov[0,0]**0.5, 'x0':pcov[1,1]**0.5, 'gamma':pcov[2,2]**0.5, 
							'bg':pcov[3,3]**0.5}
		# fitresults = {'a':popt[0], 'x0':popt[1], 'sigma':popt[2], 'bg':popt[3], 'bgs':popt[4]}
		# fituncertainty = {'a':pcov[0,0]**0.5, 'x0':pcov[1,1]**0.5, 'sigma':pcov[2,2]**0.5, 
		# 					'bg':pcov[3,3]**0.5, 'bgs':pcov[4,4]**0.5}

		# Create strings of fit results
		gausfitParamStr = {}
		for prm in fitresults:
			dist = int(np.log10(np.abs(fituncertainty[prm])))-1
			gausfitParamStr[prm] = '$'+str(round(fitresults[prm],-dist))+' \pm '+str(round(fituncertainty[prm],-dist))+'$'

		return fitresults, fituncertainty, gausfitParamStr

	except RuntimeError:
		print("Failed to fit Gaussian function")
		return {'a':0, 'x0':np.nan, 'gamma':np.nan,'bg':np.nan},{'a':np.nan, 'x0':np.nan, 'gamma':np.nan,'bg':np.nan},{'a':'', 'x0':'', 'sigma':'','bg':''}
