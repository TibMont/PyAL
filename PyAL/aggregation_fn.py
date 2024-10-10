def conductivity_aggregation_fn(x, delta_beta, uncert=False):

    if len(x.shape) == 1:
        x = x.reshape(1,-1).T

    #print(x.shape)
    #print(x)
    if uncert == False:
        conductivity = x[0,:] - delta_beta*x[1,:] - x[2,:]*delta_beta**2
        return conductivity

    else:
        uncertainty = x[0,:] + delta_beta*x[1,:] + x[2,:]*delta_beta**2
        return uncertainty
