def l20(n, j, alb, anc, ir, sd, amean, almax):
    if(j > ir):
        l60()
    else:
        nj = n[j-1]
        alj = alb[j-1]
        ancj = anc[j-1]
        if (not ((nj < 0) or (ancj < 0))):
            l30(alj=alj, nj=nj, ancj=ancj, sd=sd, amean=amean, almax=almax)
        else:
            ifault = 3
            l260_finish()

def l30(alj, nj, anjc, sd, amean, almax):
    pass

def l60():
    pass

def l260_finish():
    pass