import numpy as np
import pydicom
import vtk
from vtk.util import numpy_support


def read_angio_dicom(angio_dicom_dir):

    import glob
    import os

    files = glob.glob(os.path.join(angio_dicom_dir, "*.dcm"))
    datasets = [pydicom.dcmread(f) for f in files]

    angio_dicom_infos = []
    for i in range(len(datasets)):
        dicom_data = datasets[i]

        ArrayDicom = dicom_data.pixel_array  # (Frame_num, Height, Width)
        ArrayDicom = np.transpose(ArrayDicom, (2, 1, 0))  # (Width, Height, Frame_num)
        ArrayDicom = ArrayDicom[:, ::-1, :]

        # カメラと患者の距離
        distance_source_to_patient = float(dicom_data[(0x0018, 0x1111)].value)

        # アーム角度の取得
        rao_lao_angle = float(dicom_data[(0x0018, 0x1510)].value)
        cra_cau_angle = float(dicom_data[(0x0018, 0x1511)].value)

        angio_dicom_infos.append((ArrayDicom, distance_source_to_patient, rao_lao_angle, cra_cau_angle))

    return angio_dicom_infos


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
