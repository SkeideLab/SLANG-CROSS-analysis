import matplotlib.pyplot as plt
import numpy as np

# %% Example data
groups    = ['1', '2', '4']  # x-axis
visual_w  = [500, 730, 1060]   # y-values in mm
spoken_w  = [300, 350, 320]

# Plot lines
plt.plot(groups, visual_w, marker='o', label='Visual word')
plt.plot(groups, spoken_w, marker='s', label='Auditory word')

# Add labels and title
plt.xlabel('Grades')
plt.ylabel(r'Total volume (mm$^3$)')
plt.title('Left hemisphere (VWFA)')
plt.ylim(0,2000)
plt.legend()
plt.grid(False)
plt.savefig('left.png', dpi=300, bbox_inches='tight')  # PNG format, high resolution
plt.show()

# %%
