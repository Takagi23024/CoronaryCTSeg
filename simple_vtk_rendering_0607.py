# %%


import os

import numpy as np
import pydicom
import vtk
from vtk.util import numpy_support
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor, vtkAxesActor
from vtkmodules.vtkRenderingCore import vtkPropAssembly


def load_dicom_data(dicom_dir):
    """
    DICOMファイルを読み込み、関連する情報を取得する関数

    Parameters:
    dicom_dir (str): DICOMファイルが格納されているディレクトリのパス

    Returns:
    tuple: (ArrayDicom, ConstPixelDims, ConstPixelSpacing)
        - ArrayDicom (np.ndarray): DICOMデータのNumPy配列
        - ConstPixelDims (list): 画像の次元
        - ConstPixelSpacing (tuple): ピクセル間の間隔
    """
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicom_dir)
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

    ArrayDicom = ArrayDicom * rescale_slope + rescale_intercept

    return ArrayDicom, ConstPixelDims, ConstPixelSpacing


def load_dicom_data_pydicom(dicom_dir):
    """
    DICOMファイルを読み込み、関連する情報を取得する関数

    Parameters:
    dicom_dir (str): DICOMファイルが格納されているディレクトリのパス

    Returns:
    tuple: (ArrayDicom, ConstPixelDims, ConstPixelSpacing, ImageOrientation)
        - ArrayDicom (np.ndarray): DICOMデータのNumPy配列
        - ConstPixelDims (list): 画像の次元
        - ConstPixelSpacing (tuple): ピクセル間の間隔
        - ImageOrientation (list): 患者の向き
    """
    # DICOMファイルのリストを取得
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith(".dcm")]

    # 最初のDICOMファイルを読み込んでメタデータを取得
    ds = pydicom.dcmread(dicom_files[0])
    ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
    ImageOrientation = ds.ImageOrientationPatient

    # DICOMデータをNumPy配列に変換
    ArrayDicom = np.zeros((len(dicom_files), ds.Rows, ds.Columns), dtype=ds.pixel_array.dtype)

    for i, dicom_file in enumerate(dicom_files):
        ds = pydicom.dcmread(dicom_file)
        ArrayDicom[i, :, :] = ds.pixel_array

    ConstPixelDims = ArrayDicom.shape

    return ArrayDicom, ConstPixelDims, ConstPixelSpacing, ImageOrientation


def setup_volume_rendering(vtk_image_data):
    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputData(vtk_image_data)

    volume_color = vtk.vtkColorTransferFunction()
    volume_color.AddRGBPoint(-3024, 0.0, 0.0, 0.0)
    volume_color.AddRGBPoint(-77, 0.5, 0.2, 0.2)
    volume_color.AddRGBPoint(94, 0.9, 0.6, 0.3)
    volume_color.AddRGBPoint(179, 1.0, 0.9, 0.9)
    volume_color.AddRGBPoint(260, 1.0, 0.9, 0.9)
    volume_color.AddRGBPoint(3071, 0.9, 0.9, 0.9)

    volume_scalar_opacity = vtk.vtkPiecewiseFunction()
    volume_scalar_opacity.AddPoint(-3024, 0.0)
    volume_scalar_opacity.AddPoint(-77, 0.0)
    volume_scalar_opacity.AddPoint(94, 0.29)
    volume_scalar_opacity.AddPoint(179, 0.55)
    volume_scalar_opacity.AddPoint(260, 0.84)

    volume_gradient_opacity = vtk.vtkPiecewiseFunction()
    volume_gradient_opacity.AddPoint(0, 0.0)
    volume_gradient_opacity.AddPoint(90, 0.5)
    volume_gradient_opacity.AddPoint(100, 1.0)

    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_scalar_opacity)
    volume_property.SetGradientOpacity(volume_gradient_opacity)
    volume_property.SetInterpolationTypeToLinear()
    volume_property.ShadeOn()
    volume_property.SetAmbient(0.4)
    volume_property.SetDiffuse(0.6)
    volume_property.SetSpecular(0.2)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume


def MakeAxesActor(scale, xyzLabels):
    axes = vtkAxesActor()
    axes.SetScale(scale)
    axes.SetShaftTypeToCylinder()
    axes.SetXAxisLabelText(xyzLabels[0])
    axes.SetYAxisLabelText(xyzLabels[1])
    axes.SetZAxisLabelText(xyzLabels[2])
    axes.SetCylinderRadius(0.5 * axes.GetCylinderRadius())
    axes.SetConeRadius(1.025 * axes.GetConeRadius())
    axes.SetSphereRadius(1.5 * axes.GetSphereRadius())
    tprop = axes.GetXAxisCaptionActor2D().GetCaptionTextProperty()
    tprop.ItalicOn()
    tprop.ShadowOn()
    tprop.SetFontFamilyToTimes()
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tprop)
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tprop)
    return axes


def MakeAnnotatedCubeActor(colors):
    cube = vtkAnnotatedCubeActor()
    cube.SetXPlusFaceText("R")
    cube.SetXMinusFaceText("L")
    cube.SetYPlusFaceText("A")
    cube.SetYMinusFaceText("P")
    cube.SetZPlusFaceText("I")
    cube.SetZMinusFaceText("S")
    cube.SetFaceTextScale(0.5)
    cube.GetCubeProperty().SetColor(colors.GetColor3d("Gainsboro"))

    cube.GetTextEdgesProperty().SetColor(colors.GetColor3d("LightSlateGray"))

    cube.GetXPlusFaceProperty().SetColor(colors.GetColor3d("Tomato"))
    cube.GetXMinusFaceProperty().SetColor(colors.GetColor3d("Tomato"))
    cube.GetYPlusFaceProperty().SetColor(colors.GetColor3d("DeepSkyBlue"))
    cube.GetYMinusFaceProperty().SetColor(colors.GetColor3d("DeepSkyBlue"))
    cube.GetZPlusFaceProperty().SetColor(colors.GetColor3d("SeaGreen"))
    cube.GetZMinusFaceProperty().SetColor(colors.GetColor3d("SeaGreen"))
    return cube


def MakeCubeActor(scale, xyzLabels, colors):
    cube = MakeAnnotatedCubeActor(colors)
    axes = MakeAxesActor(scale, xyzLabels)

    assembly = vtkPropAssembly()
    assembly.AddPart(axes)
    assembly.AddPart(cube)
    return assembly


def render_volume(volume):
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    render_window_interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

    renderer.SetBackground(1, 1, 1)
    render_window.SetSize(800, 800)

    colors = vtkNamedColors()

    # ボリュームを中心に配置
    renderer.AddVolume(volume)
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(volume.GetCenter())
    camera.SetPosition(volume.GetCenter()[0], volume.GetCenter()[1], volume.GetCenter()[2] + 500)
    renderer.ResetCameraClippingRange()

    xyzLabels = ["X", "Y", "Z"]
    scale = [1.5, 1.5, 1.5]
    axes = MakeCubeActor(scale, xyzLabels, colors)

    om = vtkOrientationMarkerWidget()
    om.SetOrientationMarker(axes)
    om.SetViewport(0.8, 0.8, 1.0, 1.0)
    om.SetInteractor(render_window_interactor)
    om.EnabledOn()
    om.InteractiveOn()

    render_window.Render()
    render_window_interactor.Start()


def main():
    PathDicom = "./data/MIE022_CT/"
    ArrayDicom, ConstPixelDims, ConstPixelSpacing, ImageOrientation = load_dicom_data(PathDicom)
    vtk_image_data = numpy_to_vtk_image_data(ArrayDicom, ConstPixelSpacing)
    volume = setup_volume_rendering(vtk_image_data)
    volume = setup_volume_rendering(vtk_image_data)
    render_volume(volume)


if __name__ == "__main__":
    main()


# %%
