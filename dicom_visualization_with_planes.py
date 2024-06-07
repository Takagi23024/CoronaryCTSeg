# %%
import numpy as np
import vtk
from vtk.util import numpy_support
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor, vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkPropAssembly,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
)


def load_dicom(dicom_dir):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicom_dir)
    reader.Update()
    imageData = reader.GetOutput()
    return imageData


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


def setup_volume_rendering(vtk_image_data):
    """
    ボリュームレンダリングの設定を行う関数

    Parameters:
    vtk_image_data (vtk.vtkImageData): vtkImageDataオブジェクト

    Returns:
    vtk.vtkVolume: ボリュームレンダリングの設定がされたvtkVolumeオブジェクト
    """
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


def render_volume(volume):
    """
    ボリュームをレンダリングする関数

    Parameters:
    volume (vtk.vtkVolume): ボリュームレンダリングの設定がされたvtkVolumeオブジェクト
    """
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    renderer.SetBackground(1, 1, 1)
    render_window.SetSize(800, 800)

    colors = vtkNamedColors()

    # ボリュームを中心に配置
    renderer.AddVolume(volume)
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(volume.GetCenter())
    print(volume.GetCenter())
    camera.SetPosition(volume.GetCenter()[0], volume.GetCenter()[1], volume.GetCenter()[2] + 500)
    camera.SetViewUp(1, 0, 0)
    renderer.ResetCameraClippingRange()

    # Create an annotated cube actor adding it into an orientation marker widget.
    xyzLabels = ["X", "Y", "Z"]
    scale = [1.5, 1.5, 1.5]
    axes = MakeCubeActor(scale, xyzLabels, colors)
    # axes = MakeAnnotatedCubeActor(colors)
    om = vtkOrientationMarkerWidget()
    om.SetOrientationMarker(axes)
    om.SetViewport(0.8, 0.8, 1.0, 1.0)  # Position upper right in the viewport.
    om.SetInteractor(render_window_interactor)
    om.EnabledOn()
    om.InteractiveOn()

    render_window.Render()
    render_window_interactor.Start()


def MakeAxesActor(scale, xyzLabels):
    """
    :param scale: Sets the scale and direction of the axes.
    :param xyzLabels: Labels for the axes.
    :return: The axes actor.
    """
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
    # Use the same text properties on the other two axes.
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tprop)
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShallowCopy(tprop)
    return axes


def MakeAnnotatedCubeActor(colors):
    """
    :param colors: Used to determine the cube color.
    :return: The annotated cube actor.
    """
    # A cube with labeled faces.
    cube = vtkAnnotatedCubeActor()
    cube.SetXPlusFaceText("R")  # Right
    cube.SetXMinusFaceText("L")  # Left
    cube.SetYPlusFaceText("A")  # Anterior
    cube.SetYMinusFaceText("P")  # Posterior
    cube.SetZPlusFaceText("S")  # Superior/Cranial
    cube.SetZMinusFaceText("I")  # Inferior/Caudal
    cube.SetFaceTextScale(0.5)
    cube.GetCubeProperty().SetColor(colors.GetColor3d("Gainsboro"))

    cube.GetTextEdgesProperty().SetColor(colors.GetColor3d("LightSlateGray"))

    # Change the vector text colors.
    cube.GetXPlusFaceProperty().SetColor(colors.GetColor3d("Tomato"))
    cube.GetXMinusFaceProperty().SetColor(colors.GetColor3d("Tomato"))
    cube.GetYPlusFaceProperty().SetColor(colors.GetColor3d("DeepSkyBlue"))
    cube.GetYMinusFaceProperty().SetColor(colors.GetColor3d("DeepSkyBlue"))
    cube.GetZPlusFaceProperty().SetColor(colors.GetColor3d("SeaGreen"))
    cube.GetZMinusFaceProperty().SetColor(colors.GetColor3d("SeaGreen"))
    return cube


def MakeCubeActor(scale, xyzLabels, colors):
    """
    :param scale: Sets the scale and direction of the axes.
    :param xyzLabels: Labels for the axes.
    :param colors: Used to set the colors of the cube faces.
    :return: The combined axes and annotated cube prop.
    """
    # We are combining a vtkAxesActor and a vtkAnnotatedCubeActor
    # into a vtkPropAssembly
    cube = MakeAnnotatedCubeActor(colors)
    axes = MakeAxesActor(scale, xyzLabels)

    # Combine orientation markers into one with an assembly.
    assembly = vtkPropAssembly()
    assembly.AddPart(axes)
    assembly.AddPart(cube)
    return assembly


def main():
    PathDicom = "./data/MIE022_CT/"
    # ArrayDicom, ConstPixelDims, ConstPixelSpacing = load_dicom_data(PathDicom)
    # vtk_image_data = numpy_to_vtk_image_data(ArrayDicom, ConstPixelSpacing)
    vtk_image_data = load_dicom(PathDicom)

    volume = setup_volume_rendering(vtk_image_data)
    render_volume(volume)


if __name__ == "__main__":
    main()

# %%
