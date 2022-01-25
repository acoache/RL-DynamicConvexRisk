"""
Misc functions
Creation of colors and directories

"""
# misc
import os.path
from matplotlib.colors import LinearSegmentedColormap


# define colors and colormaps
mblue = (0.098,0.18,0.357)
mred = (0.902,0.4157,0.0196)
mgreen = (0.,0.455,0.247)
mpurple = (0.5804,0.2157,0.9412)
mgray = (0.5012,0.5012,0.5012)
myellow = (0.8,0.8,0)
mwhite = (1.,1.,1.)
cmap = LinearSegmentedColormap.from_list('beamer_cmap', [mred, mwhite, mblue])
colors = [mblue, mred, mgreen, myellow, mpurple, mgray]


# define directory function
def directory(file):
    if os.path.exists(file):
        return
    else:
        os.mkdir(file)
    return