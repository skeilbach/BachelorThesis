
## total number of ttbar events expected at FCC-ee
N_tot = 1.9E6

## Branching ratio of different processes
BRs = {
       "tlepTlep" : 0.106, 
       "tlepThad" : 0.220, 
       "thadTlep" : 0.220,
       "thadThad" : 0.454
      }


## impact on the overall cross section from each variation
xsec_variation = {}

## standard model case
xsec_variation[''] = 1.0

## ta_ttA = 0.424237 (up) or -0.424237 (down)
## (ta_ttZ = -0.140487 or 0.140487 forced by gauge invariance)
## corresponds to 0.8 shift in D_gam in Patrick's framework
## D_gam = - 4 * sw * ta_ttA
## impacts are same size and same direction for up and down variations
xsec_variation['ta_ttAup_']   = 1.060
xsec_variation['ta_ttAdown_'] = 1.060

## tv_ttA = 0.010606 (up) or -0.010606 (down)
## (tv_ttZ = -0.003512 or 0.003512 forced by gauge invariance)
## correspondes to 0.02 shift in C_gam in Patrick's framework
## C_gam = 4 i sw * tv_ttA
## impacts are same size but opposite directions for up and down variations
xsec_variation['tv_ttAup_']   = 0.951
xsec_variation['tv_ttAdown_'] = 1.051

## vt_ttZ = 0.17638 (up) or -0.17638 (down)
## corresponds to 0.1 shift in A_Z or B_Z in Patrick's framework
## A_Z = -2 i sw (-1 /(4 sw cw) * (vl_ttZ + vr_ttZ - 4 sw^2 2/3) )
## B_Z = -2 i sw ( 1 /(4 sw cw) * (vl_ttZ - vr_ttZ))
## impacts are asymmetric for up and down variations
xsec_variation['vr_ttZup_']   = 0.954
xsec_variation['vr_ttZdown_'] = 1.065

## number of events expected from each process under given parameters
N_expect = {}
for process in BRs:
    for variation in xsec_variation:
        sample = 'wzp6_ee_SM_tt_' + process + '_noCKMmix_keepPolInfo_' + variation + 'ecm365' 
        N_expect[sample] = N_tot * xsec_variation[variation] * BRs[process]

#print(N_expect)

