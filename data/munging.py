"""
Data munging script for NIPS and Twitter data sets.
"""

import math
import json
import urllib2
import scipy.io as io
import numpy as np

API_KEY = "70d252a9d1254b8d8c088b1e847f77f0"
SUNLIGHT_FIELDS = "first_name,last_name,chamber,party,state,twitter_id"

def _get_congress_on_twitter():
	"""
	Retrieving tweets for Congresspeople on Twitter.
	"""
	congress = []

	for i in xrange(int(math.ceil(540/50.0))):
		endpoint = ("http://congress.api.sunlightfoundation.com/legislators?"
				"apikey={0}"
				"&fields={1}"
				"&per_page=50&page={2}".format(API_KEY, SUNLIGHT_FIELDS, (i+1)))
		resp = urllib2.urlopen(endpoint)
		payload = json.load(resp)
		results = payload['results']
		congress.append(results)

	return congress

def nips_munging():
	"""
	Format the NIPS data as in Miller et al (2009). Steps:
	
	1. Retrieve information for the 234 authors with the most links to other
	   authors.
	   
	See p. 8 for details.
	"""
	nips = io.loadmat('nips_1-17.mat')
	authors = nips['authors_names']

	# Build the coauthorship link matrix. We only want to know whether 
	# two authors have ever worked together, so set all nonzero values == 1.
	link_mat = np.zeros(authors.shape[0], authors.shape[0])
	for row in nips['docs_authors']:
		link_mat += row.T * row
	np.fill_diagonal(link_mat, 0)
	link_mat[np.nonzero(link_mat)] = 1		# only want to indicate links...

	# Get #LINKS for each author. Miller uses 234 authors, which is equivalent
	# to keeping authors with 8+ links.
	coauthor_counts = np.squeeze(np.asarray(np.sum(link_mat, axis=1)))
	sorted_count_idxs = np.argsort(coauthor_counts)[::-1][:234]
