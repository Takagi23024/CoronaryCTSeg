import os

import numpy as np
import pydicom
import vtk
from vtk.util import numpy_support


def numpy_to_vtk_image_data(numpy_array, spacing):
    """
    NumPy配列をvtkImageDataに変換する関数

    Parameters:
    numpy_array (np.ndarray): NumPy配列
    spacing (tuple): ピクセル間の間隔

    Returns:
    vtk.vtkImageData: vtkImageDataオブジェクト
    """
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(numpy_array.shape)
    vtk_image_data.SetSpacing(spacing)

    flat_numpy_array = numpy_array.flatten(order="F")
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_numpy_array, deep=True, array_type=vtk.VTK_FLOAT)

    vtk_image_data.GetPointData().SetScalars(vtk_array)

    return vtk_image_data
