import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
from matplotlib import cm 
import itertools 
from sklearn.metrics import roc_curve, auc 

plt.rcParams.update({'font.size': 28,'legend.fontsize':28})



class PlotUtil(object):
    
    def __init__(self):
        pass 

    @staticmethod
    def plotMeanWithCI(means,lbs,ubs,xticks,legends):
        """
        means : The list of list means to plot (multiple plots can be plotted) 
        lbs : The list of list lower bounds of confidence intervals 
        ubs : The list of list upper bounds of confidence intervals
        xticks : x-axis labels 
        color_mean : color of the mean plot 
        color_shading : color of the shading (for confidence intervals)
        """

        xticks = [str(x) for x in xticks]



        n = len(means)
        assert len(means) == len(lbs)
        assert len(means) == len(ubs)

        colors=cm.jet(np.linspace(0,1,n))
        bg = np.array([1, 1, 1]) # background of the legend is white 

        colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]

        handler_map = {} 

        for i,color in enumerate(colors):

            plt.fill_between(range(means[i].shape[0]), ubs[i], lbs[i],
                            color=color, alpha=.5)


            plt.plot(means[i])

            handler_map[i] = LegendObject(colors[i], colors_faded[i])


        plt.legend(range(n), legends,handler_map=handler_map)
        plt.xticks(np.arange(len(means[0])),xticks)

        plt.tight_layout()
        return plt 


    @staticmethod
    def plotMeanWithErrorBars(means,lbs,ubs,xticks,legends,xlabel,ylabel,vmin=None,vmax=None):
        plt.rcParams.update({'font.size': 40,'legend.fontsize':40})

        fig = plt.figure(figsize=(15,15))
        xticks = [str(x) for x in xticks]

        n = len(means)
        assert len(means) == len(lbs)
        assert len(means) == len(ubs)

        colors=cm.jet(np.linspace(0,1,n))
        marker = itertools.cycle(('^', 'o', '*', 's', 'X')) 

        for i in range(len(means)):

            yerr = np.array(lbs[i])[None]
            yerr = np.vstack((yerr, np.array(ubs[i])[None]))

            yerr = np.abs(yerr - means[i])

            markers, caps, bars = plt.errorbar(xticks,means[i],yerr=yerr,color = colors[i],
                                                ecolor=colors[i],linestyle='-',marker=next(marker),
                                                markersize = 20,elinewidth = 5,linewidth = 10,
                                                capsize=5, capthick=1)
            bars[0].set_linestyle('--')

            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]

        if legends is not None:
            plt.legend(legends)

        plt.xticks(np.arange(len(means[0])),xticks)

        if vmin is not None and vmax is not None:
            plt.ylim(vmin,vmax)


        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return plt 


    @staticmethod
    def _plotCrossValidationROCurve(truelist,predlist,name,plt,color='b'):

        i = 0
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i in range(len(truelist)):
            fpr, tpr, thresholds = roc_curve(truelist[i], predlist[i]) 
            tprs.append(np.interp(mean_fpr, fpr, tpr)) 
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, lw=1, alpha=0.3,
            #         label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=color,
                label=fr'Mean ROC, {name} (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=5, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", prop={'size': 20})
        
        return plt 



    @staticmethod
    def _pltROCurve(true,pred,name,color='b'):
        fpr, tpr, thresholds = roc_curve(true, pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color,
        label=fr'ROC, {name} (AUC = %0.2f)' % (roc_auc),
        lw=5, alpha=.8)

        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", prop={'size': 20})

        return plt 

    @staticmethod
    def plotCrossValidationROCurve(truelist,predlist,name):
        import matplotlib.pyplot as plt 

        plt.figure(figsize=(10,10))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        plt = _plotCrossValidationROCurve(truelist,predlist,name,plt)

        return plt 


    @staticmethod
    def plotMultipleCrossValidationROCurves(truelistlists,predlistlists,names,pvalue=None):
        import matplotlib.pyplot as plt 

        plt.figure(figsize=(10,10))
        
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
                label='Chance', alpha=.8)


        n = len(truelistlists)
        colors=cm.jet(np.linspace(0,1,n))

        for i in range(len(truelistlists)):
            plt = PlotUtil._plotCrossValidationROCurve(truelistlists[i],predlistlists[i],names[i],plt,color=colors[i])
            
        if pvalue is not None:
            plt.text(.8,.3,fr"p={pvalue}",fontsize=22)

        return plt 

    @staticmethod
    def plotROCurve(truelist,predlist,names,pvalue=None):
        import matplotlib.pyplot as plt 

        plt.figure(figsize=(10,10))
        
        n = len(truelist)
        colors=cm.jet(np.linspace(0,1,n))


        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
                label='Chance', alpha=.8)

        for i in range(len(truelist)):
            plt = PlotUtil._pltROCurve(truelist[i],predlist[i],names[i],color=colors[i]) 

        if pvalue is not None:
            plt.text(.8,.3,fr"p={pvalue}",fontsize=22)
    

        return plt 

    # @staticmethod
    # def plotBarGraph(values,names):
    #     fig = plt.figure(figsize = (10,5))
    #     plt.bar(names,values,color ='gray') 
    #     xlocs, xlabs = plt.xticks()
    #     xlocs=[i+1.17 for i in range(0,10)]
    #     xlabs=[i/2 for i in range(0,10)]

    #     for i, v in enumerate(values):
    #         plt.text(xlocs[i] - 0.25, v + 0.3, str(v))

    #     plt.xlabel("Gleason Grade Group (GGG)")
    #     plt.ylabel("Number of Lesions")
    #     plt.show()






class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed
 
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)
 
        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)
 
        return patch




if __name__ == "__main__":
    
    names = [1,2,3,4,5]
    values = [55,88,36,8,37]


    PlotUtil.plotBarGraph(values,names)

    import pdb 
    pdb.set_trace()

