import os

import numpy as np
import pydicom
import vtk
from vtk.util import numpy_support


def read_angio_dicom(angio_dicom_dir):
    first_file = [f for f in os.listdir(angio_dicom_dir) if f.endswith(".dcm")][0]
    dicom_path = os.path.join(angio_dicom_dir, first_file)
    dicom_data = pydicom.dcmread(dicom_path)

    # アーム角度の取得
    rao_lao_angle = float(dicom_data[(0x0018, 0x1510)].value)
    cra_cau_angle = float(dicom_data[(0x0018, 0x1511)].value)
    return rao_lao_angle, cra_cau_angle


def read_ct_dicom(ct_dicom_dir):
    """
    DICOMファイルを読み込み、関連する情報を取得する関数

    Parameters:
    ct_dicom_dir (str): DICOMファイルが格納されているディレクトリのパス

    Returns:
    tuple: (ArrayDicom, ConstPixelDims, ConstPixelSpacing)
        - ArrayDicom (np.ndarray): DICOMデータのNumPy配列
        - ConstPixelDims (list): 画像の次元
        - ConstPixelSpacing (tuple): ピクセル間の間隔
    """
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(ct_dicom_dir)
    reader.Update()

    _extent = reader.GetDataExtent()
    ConstPixelDims = [
        _extent[1] - _extent[0] + 1,
        _extent[3] - _extent[2] + 1,
        _extent[5] - _extent[4] + 1,
    ]

    ConstPixelSpacing = reader.GetPixelSpacing()

    rescale_slope = reader.GetRescaleSlope() if reader.GetRescaleSlope() != 0 else 1
    rescale_intercept = reader.GetRescaleOffset()

    imageData = reader.GetOutput()
    pointData = imageData.GetPointData()
    assert pointData.GetNumberOfArrays() == 1
    arrayData = pointData.GetArray(0)

    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)

    if ConstPixelDims[2] == 1:
        ArrayDicom = ArrayDicom.reshape(ConstPixelDims[:2], order="F")
    else:
        ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order="F")
        ArrayDicom = ArrayDicom[:, :, ::-1]

    ArrayDicom = ArrayDicom * rescale_slope + rescale_intercept

    return ArrayDicom, ConstPixelDims, ConstPixelSpacing
