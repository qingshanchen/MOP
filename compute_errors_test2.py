import netCDF4 as nc
import numpy as np

out = nc.Dataset('output.nc', 'r+')

out.createVariable('errThickness', 'f8', ('Time', 'nCells', 'nVertLevels'))
out.createVariable('errVorticity', 'f8', ('Time', 'nCells', 'nVertLevels'))
out.createVariable('errDivergence', 'f8', ('Time', 'nCells', 'nVertLevels'))

thickness = out.variables['thickness'][:]
vorticity = out.variables['vorticity_cell'][:]
divergence = out.variables['divergence'][:]
nSteps = np.size(thickness, 0)

for k in xrange(nSteps):
    out.variables['errThickness'][k,:,:] = thickness[k,:,:] - thickness[0,:,:]
    out.variables['errVorticity'][k,:,:] = vorticity[k,:,:] - vorticity[0,:,:]
    out.variables['errDivergence'][k,:,:] = divergence[k,:,:] - divergence[0,:,:]

out.close()

