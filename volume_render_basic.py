# %%
import numpy as np
import vtk
from vtk.util import numpy_support

# DICOMファイルが格納されているディレクトリのパスを設定
PathDicom = "./data/MIE022_CT/"

# vtkDICOMImageReaderオブジェクトを作成し、DICOMファイルのディレクトリを設定してデータを読み込む
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(PathDicom)
reader.Update()

# 画像の次元を取得
_extent = reader.GetDataExtent()
ConstPixelDims = [
    _extent[1] - _extent[0] + 1,
    _extent[3] - _extent[2] + 1,
    _extent[5] - _extent[4] + 1,
]

# ピクセル間の間隔（スライス間の距離）を取得
ConstPixelSpacing = reader.GetPixelSpacing()

# Rescale SlopeとRescale Interceptを取得
rescale_slope = reader.GetRescaleSlope() if reader.GetRescaleSlope() != 0 else 1
rescale_intercept = reader.GetRescaleOffset()

# vtkImageDataオブジェクトをreaderから取得
imageData = reader.GetOutput()
# vtkImageDataオブジェクトからvtkPointDataオブジェクトを取得
pointData = imageData.GetPointData()
# vtkPointDataオブジェクトに1つの配列しか含まれていないことを確認
assert pointData.GetNumberOfArrays() == 1
# numpy_support.vtk_to_numpy関数に必要なvtkArray（またはその派生型）を取得
arrayData = pointData.GetArray(0)

# vtkArrayをNumPy配列に変換
ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
# NumPy配列を3次元にリシェイプ（ConstPixelDimsをshapeとして使用）
ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order="F")

# Hounsfield Unitsに変換
ArrayDicom = ArrayDicom * rescale_slope + rescale_intercept


# NumPy配列をvtkImageDataに変換する関数
def numpy_to_vtk_image_data(numpy_array, spacing):
    # vtkImageDataオブジェクトの作成
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(numpy_array.shape)
    vtk_image_data.SetSpacing(spacing)

    # NumPy配列をvtkArrayに変換
    flat_numpy_array = numpy_array.flatten(order="F")
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_numpy_array, deep=True, array_type=vtk.VTK_FLOAT)

    # vtkImageDataに配列をセット
    vtk_image_data.GetPointData().SetScalars(vtk_array)

    return vtk_image_data


# vtkImageDataに変換
vtk_image_data = numpy_to_vtk_image_data(ArrayDicom, ConstPixelSpacing)

# ボリュームレンダリングの設定
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
volume_scalar_opacity.AddPoint(-3024, 0.00)
volume_scalar_opacity.AddPoint(-77, 0.00)
volume_scalar_opacity.AddPoint(94, 0.29)
volume_scalar_opacity.AddPoint(179, 0.55)
volume_scalar_opacity.AddPoint(260, 0.84)
volume_scalar_opacity.AddPoint(3071, 0.875)

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

renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

renderer.AddVolume(volume)
renderer.SetBackground(1, 1, 1)
render_window.SetSize(800, 800)
render_window.Render()
render_window_interactor.Start()
