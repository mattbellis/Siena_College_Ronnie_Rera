import numpy as np
import pdfs 

################################################################################
# Precalculate the probabilities for the fast lognormal distributions.
################################################################################
def rise_time_prob_fast_exp_dist(rise_time,energy,mu0,sigma0,murel,sigmarel,numrel,xlo,xhi):

    expfunc = lambda p, x: p[1]*np.exp(-p[0]*x) + p[2]

    # Pull out the constants for the polynomials.
    fast_mean0 = expfunc(mu0,energy)
    fast_sigma0 = expfunc(sigma0,energy)
    if type(rise_time)==np.ndarray:
        fast_num0 = np.ones(len(rise_time)).astype('float')
    else:
        fast_num0 = 1.0

    # The entries for the relationship between the broad and narrow peak.
    fast_mean_rel = expfunc(murel,energy)
    fast_sigma_rel = expfunc(sigmarel,energy)
    fast_logn_num_rel = expfunc(numrel,energy)

    fast_mean1 = fast_mean0 - fast_mean_rel
    fast_sigma1 = fast_sigma0 - fast_sigma_rel
    fast_num1 = fast_num0 / fast_logn_num_rel

    tempnorm = (fast_num0+fast_num1)

    fast_num0 /= tempnorm
    fast_num1 /= tempnorm

    #print "Fast NUMS 0 and 1: ",fast_num0[10],fast_num1[10]

    if type(rise_time)==np.ndarray:
        ret = np.zeros(len(rise_time))
        for i in xrange(len(ret)):
            ##print rise_time[i],allmu[i],allsigma[i]
            pdf0 = pdfs.lognormal(rise_time[i],fast_mean0[i],fast_sigma0[i],xlo,xhi)
            pdf1 = pdfs.lognormal(rise_time[i],fast_mean1[i],fast_sigma1[i],xlo,xhi)
            #ret[i] = fast_num0[i]*pdf0 + fast_num1[i]*pdf1
            ######### BELLIS IS THIS HOW WE NORMALIZE THE SUM?????
            ret[i] = (fast_num0[i]*pdf0 + fast_num1[i]*pdf1)/(fast_num0[i]+fast_num1[i])
            #ret[i] = (fast_num0[i]*pdf0 + fast_num1[i]*pdf1)
            #print "\t",ret[i]
    else:
        pdf0 = pdfs.lognormal(rise_time,fast_mean0,fast_sigma0,xlo,xhi)
        pdf1 = pdfs.lognormal(rise_time,fast_mean1,fast_sigma1,xlo,xhi)
        #ret = fast_num0*pdf0 + fast_num1*pdf1
        ######### BELLIS IS THIS HOW WE NORMALIZE THE SUM?????
        ret = (fast_num0*pdf0 + fast_num1*pdf1)/(fast_num0+fast_num1)
        #ret = (fast_num0*pdf0 + fast_num1*pdf1)
        #print "\t",ret

    return ret

################################################################################
