#Matplotlib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#to pass the datapoints
plt.plot([],[],label="", marker='',s=, linewidth=)
#arg1= all y datapoints 
#arg2= all x datapoints
#label these datapoints
#linewidth is the width of the line
#marker is the symbol used for plotting datapoints
#s is the size if the marker

plt.figure(1)
#arg1 flagsize=(w,h)

plt.subplot(nrows=x,ncols=y,figsize=(x,y)).set_axis_off()
#2 is the number of row to divide figure into
#1 is the number of col to divide figure into
#1 is the current active figure
plt.grid()
plt.legand()
plt.ylabel("")
plt.xlabel("")
plt.title("")
plt.show()
plt.axis(off)
plt.close()
#........................................................................
plt.imshow()
#arg:-
#X [m,n,3],[m,n,4],[m,n]
#cmap (matplotlib.colors.Colormap) (ignored if RBGA/RGB values are given)
#aspect (equals, auto) maintains the aspect ration (ie square pixels)
#interpolation

plt.imread()
plt.imread().shape

plt.hist(x,y,histtype='', rwidth=)
plt.bar(x,y,label='',color='',)
plt.scatter(x,y,label='',color='',marker='',s=)
plt.staclplot(x,[y1,y2,...,yn],colors=['y1color','y2color',...,'yncolor'])
plt.pie(x,[y1,y2,...,yn],colors=['y1color','y2color',...,'yncolor'],startangle=,shadow=True,explode=(0,0.1,0...),autopct='%1.1f%%')
plt.




