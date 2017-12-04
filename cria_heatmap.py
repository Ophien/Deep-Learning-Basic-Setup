import matplotlib.pyplot as plt
import numpy as np

filename = '5subset_bagging.csv'
data = np.loadtxt(filename, delimiter=';')

desc = ['other', 'animal', 'cartoon', 'chevron', 'floral', 'geometry', 'houndstooth', 'ikat', 'numb', 'plain', 
              'polka dot', 'scales', 'skull', 'squares', 'stars', 'stripes', 'tribal']

# 
strategy_path = []
# 
# # Computing strategies
#for s in stat:
for d in desc:
 	strategy_path.append(d)	
 #		strategy_path.append(s + ': ' + d)

				
##############################################################################

#  Finishing Touches
fig,ax=plt.subplots()
# using the ax subplot object, we use the same
# syntax as above, but it allows us a little
# bit more advanced control
#ax.pcolor(data,cmap=plt.cm.Reds,edgecolors='k')
#ax.pcolor(data,cmap=plt.cm.Greens)
#ax.pcolor(data,cmap=plt.cm.gnuplot)
ax.set_xticks(np.arange(0,len(strategy_path)))
ax.set_yticks(np.arange(0,len(strategy_path)))
 
#cmap = plt.get_cmap('BlueRed2')
plt.imshow(data, cmap=plt.cm.gnuplot, interpolation='nearest')
#plt.imshow(data, cmap=plt.cm.gnuplot)
#plt.clim(-0.05,0.25)
plt.colorbar()


# Here we put the x-axis tick labels
# on the top of the plot.  The y-axis
# command is redundant, but inocuous.
ax.xaxis.tick_top()
ax.yaxis.tick_left()
# similar syntax as previous examples
ax.set_xticklabels(strategy_path,minor=False,fontsize=12,rotation=90)
ax.set_yticklabels(strategy_path,minor=False,fontsize=12)
 
# Here we use a text command instead of the title
# to avoid collision between the x-axis tick labels
# and the normal title position
#plt.text(0.5,1.08,'Main Plot Title',
#         fontsize=25,
#         horizontalalignment='center',
#         transform=ax.transAxes
#         ) 
 
# standard axis elements
#plt.ylabel('Y Axis Label',fontsize=10)
#plt.xlabel('X Axis Label',fontsize=10)

plt.savefig(filename + '.eps', bbox_inches='tight')
plt.show()
