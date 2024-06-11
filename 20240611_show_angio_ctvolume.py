# %%
import os

import numpy as np
import pydicom
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget

from utils.converter import numpy_to_vtk_image_data
from utils.reader import read_ct_dicom
from utils.vtk_tools import MakeCubeActor


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


def read_angio_dicom(directory):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(directory)
    reader.Update()
    return reader.GetOutput()


def create_vtk_image_actor(image_data):
    actor = vtk.vtkImageActor()
    actor.GetMapper().SetInputData(image_data)
    return actor


def get_angio_angles(dicom_dir):
    # アンギオDICOMディレクトリから最初のファイルを読み込み
    first_file = [f for f in os.listdir(dicom_dir) if f.endswith(".dcm")][0]
    dicom_path = os.path.join(dicom_dir, first_file)
    dicom_data = pydicom.dcmread(dicom_path)

    # アーム角度の取得
    rao_lao_angle = float(dicom_data[(0x0018, 0x1510)].value)
    cra_cau_angle = float(dicom_data[(0x0018, 0x1511)].value)
    return rao_lao_angle, cra_cau_angle


# DICOMデータのディレクトリを指定
angio_dicom_dir = "./data/MIE022_Angio/"
ct_dicom_dir = "./data/MIE022_CT/"

# アンギオのアーム角度を取得
rao_lao_angle, cra_cau_angle = get_angio_angles(angio_dicom_dir)

# DICOMデータの読み込み
angio_image_data = read_angio_dicom(angio_dicom_dir)

# VTKウィンドウの設定
render_window = vtk.vtkRenderWindow()
render_window.SetSize(1200, 600)

# 左側のアンギオ画像を表示するレンダラー
angio_renderer = vtk.vtkRenderer()
render_window.AddRenderer(angio_renderer)
angio_renderer.SetViewport(0.0, 0.0, 0.5, 1.0)

angio_actor = create_vtk_image_actor(angio_image_data)
angio_renderer.AddActor(angio_actor)
angio_renderer.ResetCamera()

# 右側のCTボリュームレンダリングを表示するレンダラー
ct_renderer = vtk.vtkRenderer()
render_window.AddRenderer(ct_renderer)
ct_renderer.SetViewport(0.5, 0.0, 1.0, 1.0)

ArrayDicom, ConstPixelDims, ConstPixelSpacing = read_ct_dicom(ct_dicom_dir)
vtk_image_data = numpy_to_vtk_image_data(ArrayDicom, ConstPixelSpacing)
ct_volume = setup_volume_rendering(vtk_image_data)
ct_renderer.AddVolume(ct_volume)

# アンギオ画像の角度をCTレンダラーに反映
camera = ct_renderer.GetActiveCamera()
camera.SetViewUp(0, 1, 0)
camera.SetPosition(np.sin(np.radians(rao_lao_angle)), 0, np.cos(np.radians(cra_cau_angle)))
camera.SetFocalPoint(0, 0, 0)

ct_renderer.ResetCamera()


# レンダリングの開始
render_window.Render()
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

xyzLabels = ["X", "Y", "Z"]
scale = [1.5, 1.5, 1.5]
colors = vtkNamedColors()
axes = MakeCubeActor(scale, xyzLabels, colors)
om = vtkOrientationMarkerWidget()
om.SetOrientationMarker(axes)
om.SetViewport(0.8, 0.8, 1.0, 1.0)
om.SetInteractor(render_window_interactor)
om.EnabledOn()
om.InteractiveOn()


render_window_interactor.Start()

# %%
