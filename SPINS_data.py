# Code to read SPINS files

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from AutoGaussian import autoGaussianFit
import copy

class SpinsFile:
	"""for importing .ng5 files into python"""
	def __init__(self,infile=None):

		if isinstance(infile,str):
			self._readng5file(infile)
		elif isinstance(infile,SpinsFile):
			self.scantype = 'Copied Scan'
		elif infile==None:
			self.scantype = 'Empty Object'

	def _readng5file(self,infile):
		f = open(infile)
		lines = f.readlines()
		#f.seek(0)
		#self.originaltext = f.read()
		f.close()

		if lines[0].startswith(' Motor'):
			firstline = lines[0].split('   ')
			self.datalabels = []
			for fl in firstline:
				if 'no.  3' in fl: self.datalabels.append('A3')
				elif 'no.  4' in fl: self.datalabels.append('A4')
			self.datalabels.append('Intensity')
			# Import the data
			self.data = np.array([j.split() for j in lines[1:len(lines)]]).astype(np.float)
			self.data = self.data.transpose()
			self.DATA = {}
			for k in range(len(self.datalabels)):
				self.DATA[self.datalabels[k]] = self.data[k]
			# Record counts and uncertainty
			self.DATA["NormCounts"] = self.DATA['Intensity']
			self.DATA["NormUnc"] = np.sqrt(self.DATA['Intensity'])
			self.DATA["NormUnc"][np.where(self.DATA["NormCounts"]==0)] += 1

			return

		# Define the important parts of the scan
		firstline = lines[0].split()
		#if firstline[0] != '\''+infile+'\'':
		#	print "not ordinary ng5 file"
		self.time = float(firstline[2]) + self._timetoday(firstline[4][:-1])
		filename = firstline[0]
		self.scantype = firstline[5]
		self.mon = float(firstline[6]) * float(firstline[7])  # monitor times pref
		numpts = int(firstline[9])

		self.Ei = float(lines[7].split()[2])
		self.deltaE = float(lines[7].split()[1])

		self.Qposition = [int(float(j)) for j in lines[9].split()[0:3]]

		if self.scantype == "'B'":
			#get data
			self._importB(numpts,lines)
		elif self.scantype == "'Q'":
			self._importQ(numpts,lines)

		# make monitor into an array
		self.mon = np.full(len(self.data[0]), self.mon)

		# Build dictionary
		self.DATA = {}
		for k in range(len(self.datalabels)):
		 	self.DATA[self.datalabels[k]] = self.data[k]
		 	if self.datalabels[k]== 'Counts':
		 		self.DATA['COUNTS'] = self.data[k]
		 	if self.datalabels[k]== 'H-Field':
		 		self.DATA['H'] = self.data[k]
		 		self.DATA['Field'] = self.data[k]
		 		self.DATA['B'] = self.data[k]
		 	elif self.datalabels[k] in ['TEMP','T-act']:
		 		self.DATA['T'] = self.data[k]
		 	elif self.datalabels[k] in ['MIN','min']:
		 		self.DATA["time"] = np.arange(1,len(self.data[k])+1)*self.data[k]
		self.avgtime = self.DATA['time'][-1]*60/len(self.DATA['time'])  #in seconds
		self.DATA["NormCounts"] = self.DATA['COUNTS']/self.mon
		self.DATA["NormUnc"] = np.sqrt(self.DATA['COUNTS']/(self.mon)**2 + (self.DATA['COUNTS'])**2/(self.mon)**3)
		self.DATA["NormCountsTime"] = self.DATA['COUNTS']/self.avgtime
		self.DATA["NormUncTime"] = self.DATA["NormUnc"]*self.mon/self.avgtime

		#get average temperature
		self.Tavg = np.average(self.DATA['T'])

		print("Imported file "+filename+", with data ",self.datalabels)

	def _importB(self,numpts,lines):
		"""Imports Bragg buffer data"""
		#get data
		i=9
		while i<len(lines):
			if ('COUNTS' in lines[i]) or ('Counts' in lines[i]):
				self.datalabels = lines[i].split()
				self.data = np.array([j.split() for j in lines[i+1:i+1+numpts]]).astype(np.float)
				break
			i+=1
		self.data = self.data.transpose()


	def _importQ(self,numpts,lines):
		"""Imports Q buffer data"""
		i=9
		while i<len(lines):
			if ('COUNTS' in lines[i]) or ('Counts' in lines[i]):
				self.datalabels = lines[i].split()
				self.data = np.array([j.split() for j in lines[i+1:i+1+numpts]]).astype(np.float)
				break
			i+=1
		self.data = self.data.transpose()
		self.NormCounts = self.data[-1]/self.mon
		self.Unc = np.sqrt(self.data[-1]/(self.mon)**2 + (self.data[-1])**2/(self.mon)**3)

	def _timetoday(self,tstring):
		t = tstring.split(':')
		hr = float(t[0])/24
		mn = float(t[1])/(60*24)
		return hr + mn

	def writefile(self,dr):
		"""use for finding all scans at a given bragg point"""
		if self.Qposition == [2,2,2]:
			f = open(dr+'testfile.txt', 'w')
			f.write(self.originaltext)

	#**********User called functions

	def rebin(self, x_new, axis):
		"""This function modified from Kemp Plumb's SliceTools library"""
		d = np.diff(x_new)/2.
		edges = np.hstack([x_new[0]-d[0],x_new[0:-1]+d,x_new[-1]+d[-1]])

		bin_idx = np.digitize(self.DATA[axis],edges)
		bin_count, b = np.histogram(self.DATA[axis], edges)

		y_new = np.zeros(np.size(x_new))
		unc_new = np.zeros(np.size(x_new))

		mask_idx = bin_count < 1. 
		mask_ct = np.ma.array(bin_count,mask = mask_idx)

		for ii, idx in enumerate(bin_idx):
			try:
				#Take weighted average of all points in a bin
				if unc_new[idx-1] == 0:
					y_new[idx-1] += self.DATA['NormCounts'][ii]
					unc_new[idx-1] += self.DATA['NormUnc'][ii]
				else:
					y_new[idx-1] = (y_new[idx-1]*unc_new[idx-1]**-2 +\
						self.DATA['NormCounts'][ii] * self.DATA['NormUnc'][ii]**-2 ) /\
						(unc_new[idx-1]**-2 + self.DATA['NormUnc'][ii]**-2)
					unc_new[idx-1] = np.sqrt(1/(unc_new[idx-1]**-2 + self.DATA['NormUnc'][ii]**-2))

			except IndexError: continue    # if there is an index error, the given value is not within the array
		# mask zeros
		y_new = np.ma.masked_where(y_new==0 , y_new)
		unc_new = np.ma.masked_where(unc_new==0 , unc_new)

		return y_new, unc_new

	def A3toQ(self):
		"""Convert A3 to Q, based off incident wavelength."""

		# A) Find final wavelength
		Ef = self.Ei - self.deltaE
		mn = 939.56536e9 # mass of neutron in meV/c^2
		PlanckConst = 4.135667662e-12  #meV*s
		c = 2.99792e18 # \AA/s
		lambda_f = PlanckConst/np.sqrt(2*Ef*mn)*c  # in \AA
		try: 
			self.DATA['Q'] = 4*np.pi/lambda_f * np.sin(np.pi/180*self.DATA['A3']/2)
		except KeyError: print("A3 not in DATA!")


	def fitGaussian(self, variable):
		fitvars, fitunc, gfparamstr = autoGaussianFit(datax=self.DATA[variable],
			datay=self.DATA['NormCounts'], sigma=self.DATA['NormUnc'])
		x = np.linspace(np.amin(self.DATA[variable]), np.amax(self.DATA[variable]), 300)
		self.gausfitx = x
		self.gausfitParamStr = gfparamstr
		try:
			self.gausfit = gaus(x, fitvars['a'],fitvars['x0'],fitvars['sigma']) +\
						 fitvars['bg'] #+ x*fitvars['bgs']
		except TypeError: #The fit must have failed
			self.gausfit = x*np.nan
		return fitvars, fitunc



	def __getitem__(self,key):
		"""for indexing (truncating) dictionary values"""
		newobject = copy.deepcopy(self)
		# newobject.__dict__
		for ddd in self.DATA:
			newobject.DATA[ddd] = self.DATA[ddd][key]
		newobject.data = self.data[:,key]
		newobject.mon = self.mon[key]
		return newobject


	def __add__(self,other):
		"""returns a new object with DATA concatenated from two input objects"""
		newobject = SpinsFile()
		newobject.scantype = 'Added Object'
		newobject.DATA = {}
		for ddd in self.DATA:
			newobject.DATA[ddd] = np.hstack((self.DATA[ddd], other.DATA[ddd]))
		newobject.mon = np.hstack((self.mon, other.mon))
		newobject.Tavg = np.average(newobject.DATA['T'])
		return newobject

	# def __mul__(self,other):
	# 	"""returns a new object with DATA concatenated from two input objects"""
	# 	newobject = SpinsFile()
	# 	newobject.scantype = 'Multiplied Object'
	# 	newobject.DATA = {}
	# 	for ddd in self.DATA:
	# 		newobject.DATA[ddd] = copy.deepcopy(self.DATA[ddd])
	# 	return newobject


def gaus(x, a, x0, sig):
    return a/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - x0)/sig, 2.)/2.)

#*********************
# Plotting Utilities
#*********************
def addtoplot(axes, scan, indep, scale=1, **kwargs):
	axes.errorbar(scan.DATA[indep],scale*scan.DATA['NormCounts'],scale*scan.DATA['NormUnc'], 
		**kwargs)

