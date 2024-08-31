''' Creation: 2021.09.26
    Last update: 2021.10.04
    
    Electroluminecence greateyes raw images reading, croping and plotting 
    
    '''

# Standard library import
import os
from pathlib import Path

# 3rd party imports
import matplotlib.pyplot as plt

#Internal import 
import pl 


NCOLS_SUPPRESS = 10 # Number of columns supressed for the image plotting

#Reads, crops and stitches the set of electroluminesence images acquired with greateyes camera
#file_names = ["SERAPHIM-EM-0640_Isc_ap500hXDH.dat",
#              "JINERGY3272023326035_Isc_T2.dat",
#              "JINERGY3272023326035_Isc_T1.dat",
#              "EL_Komma_Problem.dat"
#              ]
#file_name = file_names[3]

#file = pv.DEFAULT_DIR / Path("PVcharacterization_files") / Path(file_name)
file = "C:/users/franc/Temp/CHIC815.dat"
croped_image = pl.crop_image(file)

# Plots the image throwing away the NCOLS_SUPPRESS first columns
fig,axe = plt.subplots(1,1,figsize=(15,15))
axe.imshow(croped_image[:,NCOLS_SUPPRESS:],
           cmap='gray', 
           interpolation=None)
for axis in ['top','bottom','left','right']:
            axe.spines[axis].set_linewidth(0)
axe.set_xticklabels([])
axe.set_yticklabels([])
axe.set_xticks([])
axe.set_yticks([])
#plt.title(file_name)
plt.title(file)
plt.show()

# Dumps the image in Gwyddion Simple Field Files format
#file_gsf = os.path.splitext(file_name)[0] + '_full.gsf'
#file = pv.DEFAULT_DIR / Path("PVcharacterization_files") / Path(file_gsf)
#file =  "C:/Pati/ELDATA/3SUN-2202_T0.dat" / Path(file_gsf)
