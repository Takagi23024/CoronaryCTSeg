import numpy as np
import vtk
from vtk.util import numpy_support
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionWidgets import (
    vtkButtonWidget,
    vtkOrientationMarkerWidget,
    vtkTexturedButtonRepresentation2D,
)
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor, vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkPropAssembly,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
)


def load_dicom_and_extract_points(dicom_dir):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicom_dir)
    reader.Update()
    vtk_image_data = reader.GetOutput()

    extent = reader.GetDataExtent()
    pixel_dims = [
        extent[1] - extent[0] + 1,
        extent[3] - extent[2] + 1,
        extent[5] - extent[4] + 1,
    ]
    pixel_spacing = reader.GetPixelSpacing()

    rescale_slope = reader.GetRescaleSlope() if reader.GetRescaleSlope() != 0 else 1
    rescale_intercept = reader.GetRescaleOffset()

    point_data = vtk_image_data.GetPointData()
    assert point_data.GetNumberOfArrays() == 1
    array_data = point_data.GetArray(0)
    array_dicom = numpy_support.vtk_to_numpy(array_data)

    if pixel_dims[2] == 1:
        array_dicom = array_dicom.reshape(pixel_dims[:2], order="F")
    else:
        array_dicom = array_dicom.reshape(pixel_dims, order="F")

    array_dicom = array_dicom * rescale_slope + rescale_intercept
    array_dicom = array_dicom[:, :, ::-1]

    origin = reader.GetImagePositionPatient()
    x = np.arange(pixel_dims[0]) * pixel_spacing[0] + origin[0]
    y = np.arange(pixel_dims[1]) * pixel_spacing[1] + origin[1]
    z = np.arange(pixel_dims[2]) * pixel_spacing[2] + origin[2]
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T
    scalars = array_dicom.ravel()

    return array_dicom, points, scalars, pixel_dims, pixel_spacing, origin


def numpy_to_vtk_image_data(numpy_array, spacing):
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(numpy_array.shape)
    vtk_image_data.SetSpacing(spacing)

    flat_numpy_array = numpy_array.flatten(order="F")
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_numpy_array, deep=True, array_type=vtk.VTK_FLOAT)

    vtk_image_data.GetPointData().SetScalars(vtk_array)

    return vtk_image_data


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
    volume_scalar_opacity.AddPoint(-3024, 0.00)
    volume_scalar_opacity.AddPoint(-77, 0.0)
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

    return volume


PathDicom = "./data/MIE022_CT/"
array_dicom, points, scalars, pixel_dims, pixel_spacing, origin = load_dicom_and_extract_points(PathDicom)

vtk_image_data = numpy_to_vtk_image_data(array_dicom, pixel_spacing)


def create_vtk_image_data_from_points(points, scalars, pixel_dims, spacing, origin):
    array_dicom = np.zeros(pixel_dims, dtype=scalars.dtype)

    x_indices = ((points[:, 0] - origin[0]) / pixel_spacing[0]).astype(int)
    y_indices = ((points[:, 1] - origin[1]) / pixel_spacing[1]).astype(int)
    z_indices = ((points[:, 2] - origin[2]) / pixel_spacing[2]).astype(int)

    array_dicom[x_indices, y_indices, z_indices] = scalars

    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(array_dicom.shape)
    vtk_image_data.SetSpacing(spacing)

    flat_array_dicom = array_dicom.flatten(order="F")
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_array_dicom, deep=True, array_type=vtk.VTK_FLOAT)

    vtk_image_data.GetPointData().SetScalars(vtk_array)

    return vtk_image_data


volume = setup_volume_rendering(vtk_image_data)

renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

renderer.SetBackground(1, 1, 1)
render_window.SetSize(800, 800)

renderer.AddVolume(volume)
renderer.ResetCamera()
camera = renderer.GetActiveCamera()
camera.SetFocalPoint(volume.GetCenter())
camera.SetPosition(volume.GetCenter()[0], volume.GetCenter()[1] + 500, volume.GetCenter()[2])
camera.SetViewUp(0, 0, 1)
renderer.ResetCameraClippingRange()


lay_info = None
ray_actor = None
intersection_point = None
mesh_mapper = None


def calculate_distances_to_ray(points, ray_start, ray_dir):
    ray_vector_norm = ray_dir / np.linalg.norm(ray_dir)
    point_vectors = points - ray_start
    projection_lengths = np.dot(point_vectors, ray_vector_norm)
    projections = ray_start + np.outer(projection_lengths, ray_vector_norm)
    distances_to_ray = np.linalg.norm(points - projections, axis=1)
    return distances_to_ray


import os

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from vtkmodules.vtkFiltersCore import vtkProbeFilter

# 赤い点のリストを保持するためのリスト
red_points = []


def on_right_button_down(obj, event):
    global volume, renderer, points, scalars, pixel_dims, pixel_spacing, origin, vtk_image_data, red_points

    click_pos = render_window_interactor.GetEventPosition()
    camera = renderer.GetActiveCamera()
    cam_pos = np.array(camera.GetPosition())
    picker = vtk.vtkWorldPointPicker()
    picker.Pick(click_pos[0], click_pos[1], 0, renderer)
    click_pos_3d = np.array(picker.GetPickPosition())

    ray_dir = click_pos_3d - cam_pos
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    cylinder = vtk.vtkCylinder()
    cylinder.SetCenter(click_pos_3d)
    cylinder.SetRadius(0.5)
    cylinder.SetAxis(ray_dir)

    extract_geometry = vtk.vtkExtractGeometry()
    extract_geometry.SetInputData(vtk_image_data)
    extract_geometry.SetImplicitFunction(cylinder)
    extract_geometry.Update()

    extracted_data = extract_geometry.GetOutput()

    points = extracted_data.GetPoints()
    scalars = extracted_data.GetPointData().GetScalars()

    if points and scalars:
        num_points = points.GetNumberOfPoints()
        if num_points:
            distances = []
            scalar_values = []
            for i in range(num_points):
                point = np.array(points.GetPoint(i))
                distance = np.linalg.norm(point - cam_pos)
                scalar_value = scalars.GetTuple1(i)
                distances.append(distance)
                scalar_values.append(scalar_value)

            # 距離に基づいてソート
            sorted_indices = np.argsort(distances)
            sorted_distances = np.array(distances)[sorted_indices]
            sorted_scalar_values = np.array(scalar_values)[sorted_indices]

            # ソートされたスカラー値でピークを探す
            peaks, _ = find_peaks(sorted_scalar_values, height=300)
            if peaks.size > 0:
                peak_index = sorted_indices[peaks[0]]
                peak_point = points.GetPoint(peak_index)
                print(f"peak_point: {peak_point}")

                # 赤いポイントを描画
                sphere_source = vtk.vtkSphereSource()
                sphere_source.SetCenter(peak_point)
                sphere_source.SetRadius(1.0)
                sphere_source.Update()

                sphere_mapper = vtk.vtkPolyDataMapper()
                sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

                sphere_actor = vtk.vtkActor()
                sphere_actor.SetMapper(sphere_mapper)
                sphere_actor.GetProperty().SetColor(1, 0, 0)  # 赤色

                renderer.AddActor(sphere_actor)
                red_points.append(peak_point)

                # 二つ以上の赤い点がある場合、線分を描画
                if len(red_points) > 1:
                    line_source = vtk.vtkLineSource()
                    line_source.SetPoint1(red_points[-2])
                    line_source.SetPoint2(red_points[-1])
                    line_source.Update()

                    line_mapper = vtk.vtkPolyDataMapper()
                    line_mapper.SetInputConnection(line_source.GetOutputPort())

                    line_actor = vtk.vtkActor()
                    line_actor.SetMapper(line_mapper)
                    line_actor.GetProperty().SetColor(0, 0, 1)  # 青色

                    renderer.AddActor(line_actor)

    render_window.Render()


interactor_style = vtk.vtkInteractorStyleTrackballCamera()
render_window_interactor.SetInteractorStyle(interactor_style)
render_window_interactor.AddObserver("RightButtonPressEvent", on_right_button_down)


def create_button(text, callback):
    button_rep = vtkTexturedButtonRepresentation2D()
    button_rep.SetNumberOfStates(1)

    # テキストを画像に変換
    text_property = vtk.vtkTextProperty()
    text_property.SetFontSize(24)
    text_property.SetColor(1, 1, 1)

    text_actor = vtk.vtkTextActor()
    text_actor.SetTextProperty(text_property)
    text_actor.SetInput(text)

    text_actor.GetTextProperty().SetJustificationToCentered()
    text_actor.GetTextProperty().SetVerticalJustificationToCentered()

    text_actor.SetPosition(50, 50)

    # レンダラーとレンダリングウィンドウを作成
    text_renderer = vtk.vtkRenderer()
    text_render_window = vtk.vtkRenderWindow()
    text_render_window.AddRenderer(text_renderer)

    text_renderer.AddActor2D(text_actor)
    text_render_window.Render()

    # テキストを画像として取得
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(text_render_window)
    window_to_image_filter.Update()

    button_rep.SetButtonTexture(0, window_to_image_filter.GetOutput())

    button_widget = vtkButtonWidget()
    button_widget.SetInteractor(render_window_interactor)
    button_widget.SetRepresentation(button_rep)
    button_widget.AddObserver(vtk.vtkCommand.StateChangedEvent, callback)

    return button_widget


def on_finish_button_pressed(obj, event):
    global red_points, renderer, vtk_image_data

    if len(red_points) < 2:
        print("赤い点が二点以上必要です。")
        return

    # 複数の円柱を作成してボリュームデータを抽出
    append_filter = vtk.vtkAppendPolyData()
    radius = 5  # 円柱の半径

    # スプラインを作成
    spline = vtk.vtkParametricSpline()
    points = vtk.vtkPoints()
    for point in red_points:
        points.InsertNextPoint(point)
    spline.SetPoints(points)

    # スプラインをサンプル
    num_samples = 100
    spline_points = vtk.vtkPoints()
    for i in range(num_samples):
        t = i / (num_samples - 1)
        pt = [0.0, 0.0, 0.0]
        spline.Evaluate([t, 0.0, 0.0], pt, [0] * 9)
        spline_points.InsertNextPoint(pt)

    # スプラインの線を作成
    spline_polyline = vtk.vtkPolyLine()
    spline_polyline.GetPointIds().SetNumberOfIds(num_samples)
    for i in range(num_samples):
        spline_polyline.GetPointIds().SetId(i, i)

    # ポリラインをポリデータに追加
    spline_polydata = vtk.vtkPolyData()
    spline_polydata.SetPoints(spline_points)
    spline_polydata.Allocate(1, 1)
    spline_polydata.InsertNextCell(spline_polyline.GetCellType(), spline_polyline.GetPointIds())

    # スプラインの線を描画
    spline_mapper = vtk.vtkPolyDataMapper()
    spline_mapper.SetInputData(spline_polydata)

    spline_actor = vtk.vtkActor()
    spline_actor.SetMapper(spline_mapper)
    spline_actor.GetProperty().SetColor(1, 0, 0)  # 赤色

    renderer.AddActor(spline_actor)

    render_window.Render()


finish_button = create_button("Finish", on_finish_button_pressed)
finish_button.SetInteractor(render_window_interactor)
finish_button.On()


render_window.Render()
finish_button.On()
render_window_interactor.Start()
