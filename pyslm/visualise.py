from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.collections as mc

import numpy as np

from .geometry import Layer

def plotPolygon(polygons, zPos=0.0, lineColor='k', lineWidth=0.7, fillColor='r',
                plot3D=False, plotFilled=False,
                handle: Tuple[plt.Figure, plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Helper method for plotting polygons (numpy coordinates) and those composed of Python lists.

    :param polygons:  A list of polygons
    :param zPos: The z position of the polygons if plot3D is enabled
    :param lineColor: Line color used for matplotlib (optional)
    :param lineWidth: Line width used for matplotlib (optional)
    :param fillColor:  Fill color for the polygon if plotFilled is enabled (optional)
    :param plot3D: Plot the polygons in 3D
    :param plotFilled: Plot filled
    :param handle: A previous matplotlib (Figure, Axis) object

    :return: A tuple with the matplotlib (Figure, Axis)
    """

    if handle:
        fig = handle[0]
        ax = handle[1]

    else:
        if plot3D:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = plt.axes(projection='3d')
        else:
            fig, ax = plt.subplots()

    ax.axis('equal')
    plotNormalize = matplotlib.colors.Normalize()

    patchList = []

    for contour in polygons:
        if plot3D:
            ax.plot(contour[:, 0], contour[:, 1], zs=zPos, color=lineColor, linewidth=lineWidth)
        else:
            if plotFilled:
                polygon = matplotlib.patches.Polygon(contour, fill=True, linewidth=lineWidth, edgecolor=lineColor, color=fillColor, facecolor=fillColor)
                ax.add_patch(polygon)
                patchList.append(polygon)

            else:
                ax.plot(contour[:, 0], contour[:, 1], color=lineColor, linewidth=lineWidth)

    #p = mc.PatchCollection(patchList, alpha=1)
    #ax.add_collection(p)

    return fig, ax

def plotLayers(layers: List[Layer], plotContours=True, plotHatches=True, plotPoints=True, plotOrderLine=False, handle=None) -> Tuple[plt.Figure, plt.Axes]:

    if handle:
        fig = handle[0]
        ax = handle[1]
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

    for layer in layers:
        fig, ax = plot(layer, layer.z/1000, plot3D=True, plotContours=plotContours, plotHatches=plotHatches, plotPoints=plotPoints, handle = (fig,ax))

def plot(layer: Layer, zPos=0, plotContours=True, plotHatches=True, plotPoints=True, plot3D=True, plotArrows=False,
         plotOrderLine=False, handle=None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the all the scan vectors (contours and hatches) and point exposures for each Layer Geometry in a Layer
    using `Matplotlib`. The Layer may be plotted in 3D by setting the plot3D parameter.

    :param layer: The Layer containing the Layer Geometry
    :param zPos: The position of the layer when using the 3D plot (optional)
    :param plotContours: Plots the inner hatch scan vectors. Defaults to `True`
    :param plotHatches: Plots the hatch scan vectors
    :param plotPoints: Plots point exposures
    :param plot3D: Plots the layer in 3D
    :param plotArrows: Plot the direction of each scan vector. This reduces the plotting performance due to use of
                       matplotlib annotations, should be disabled for large datasets
    :param plotOrderLine: Plots an additional line showing the order of vector scanning
    :param handle: Matplotlib handle to re-use
    """

    if handle:
        fig = handle[0]
        ax = handle[1]

    else:
        if plot3D:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = plt.axes(projection='3d', aspect='equal')
        else:
            fig, ax = plt.subplots()
            ax.axis('equal')


    plotNormalize = matplotlib.colors.Normalize()

    if plotHatches:
        hatches = layer.getHatchGeometry()

        if len(hatches) > 0:

            hatches = np.vstack([hatchGeom.coords.reshape(-1, 2, 2) for hatchGeom in layer.getHatchGeometry()])

            lc = mc.LineCollection(hatches, colors=plt.cm.rainbow(plotNormalize(np.arange(len(hatches)))),
                                            linewidths=0.5)

            if plotArrows and not plot3D:
                for hatch in hatches:
                    midPoint = np.mean(hatch, axis=0)
                    delta = hatch[1, :] - hatch[0, :]

                    plt.annotate('', xytext = midPoint - delta * 1e-4,
                                     xy = midPoint,
                                     arrowprops={'arrowstyle': "->", 'facecolor': 'black'})

            if plot3D:
                ax.add_collection3d(lc, zs=zPos)

            if not plot3D and plotOrderLine:
                ax.add_collection(lc)
                midPoints = np.mean(hatches, axis=1)
                idx6 = np.arange(len(hatches))
                ax.plot(midPoints[idx6][:, 0], midPoints[idx6][:, 1])

            ax.add_collection(lc)

    if plotContours:

        for contourGeom in layer.getContourGeometry():

            if contourGeom.subType == 'inner':
                lineColor = '#f57900'
                lineWidth = 1
            elif contourGeom.subType == 'outer':
                lineColor = '#204a87'
                lineWidth = 1.4
            else:
                lineColor = 'k'
                lineWidth = 0.7

            if plotArrows and not plot3D:
                for i in range(contourGeom.coords.shape[0] - 1):
                    midPoint = np.mean(contourGeom.coords[i:i + 2], axis=0)
                    delta = contourGeom.coords[i + 1, :] - contourGeom.coords[i, :]

                    plt.annotate('',
                                 xytext=midPoint - delta * 1e-4,
                                 xy=midPoint,
                                 arrowprops={'arrowstyle': "->", 'facecolor': 'black'})

            if plot3D:
                ax.plot(contourGeom.coords[:, 0], contourGeom.coords[:, 1], zs=zPos, color=lineColor,
                        linewidth=lineWidth)
            else:
                ax.plot(contourGeom.coords[:, 0], contourGeom.coords[:, 1], color=lineColor,
                        linewidth=lineWidth)

    if plotPoints:
        for pointsGeom in layer.getPointsGeometry():
            ax.scatter(pointsGeom.coords[:, 0], pointsGeom.coords[:, 1], 'x')

    return fig, ax
