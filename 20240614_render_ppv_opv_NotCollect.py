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

# def load_dicom_data(dicom_dir):
#     reader = vtk.vtkDICOMImageReader()
#     reader.SetDirectoryName(dicom_dir)
#     reader.Update()

#     _extent = reader.GetDataExtent()
#     ConstPixelDims = [
#         _extent[1] - _extent[0] + 1,
#         _extent[3] - _extent[2] + 1,
#         _extent[5] - _extent[4] + 1,
#     ]

#     ConstPixelSpacing = reader.GetPixelSpacing()

#     rescale_slope = reader.GetRescaleSlope() if reader.GetRescaleSlope() != 0 else 1
#     rescale_intercept = reader.GetRescaleOffset()

#     imageData = reader.GetOutput()
#     pointData = imageData.GetPointData()
#     assert pointData.GetNumberOfArrays() == 1
#     arrayData = pointData.GetArray(0)

#     ArrayDicom = numpy_support.vtk_to_numpy(arrayData)

#     if ConstPixelDims[2] == 1:
#         ArrayDicom = ArrayDicom.reshape(ConstPixelDims[:2], order="F")
#     else:
#         ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order="F")

#     ArrayDicom = ArrayDicom * rescale_slope + rescale_intercept

#     ArrayDicom = ArrayDicom[:, :, ::-1]

#     return ArrayDicom, ConstPixelDims, ConstPixelSpacing, reader


def load_dicom_and_extract_points(dicom_dir):
    # DICOMデータを読み込む
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicom_dir)
    reader.Update()
    vtk_image_data = reader.GetOutput()

    # データの寸法とピクセル間隔を取得
    extent = reader.GetDataExtent()
    pixel_dims = [
        extent[1] - extent[0] + 1,
        extent[3] - extent[2] + 1,
        extent[5] - extent[4] + 1,
    ]
    pixel_spacing = reader.GetPixelSpacing()

    # スカラー値のリスケーリング
    rescale_slope = reader.GetRescaleSlope() if reader.GetRescaleSlope() != 0 else 1
    rescale_intercept = reader.GetRescaleOffset()

    # VTKのスカラー値をNumPy配列に変換
    point_data = vtk_image_data.GetPointData()
    assert point_data.GetNumberOfArrays() == 1
    array_data = point_data.GetArray(0)
    array_dicom = numpy_support.vtk_to_numpy(array_data)

    if pixel_dims[2] == 1:
        array_dicom = array_dicom.reshape(pixel_dims[:2], order="F")
    else:
        array_dicom = array_dicom.reshape(pixel_dims, order="F")

    array_dicom = array_dicom * rescale_slope + rescale_intercept
    # array_dicom = array_dicom[:, :, ::-1]

    # 各点の座標とスカラー値を計算
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
# ArrayDicom, ConstPixelDims, ConstPixelSpacing, reader = load_dicom_data(PathDicom)

array_dicom, points, scalars, pixel_dims, pixel_spacing, origin = load_dicom_and_extract_points(PathDicom)

vtk_image_data = numpy_to_vtk_image_data(array_dicom, pixel_spacing)
print(f"points.shape: {points.shape}")
print(f"scalars.shape: {scalars.shape}")
print(f"pixel_dims: {pixel_dims}")
print(f"pixel_spacing: {pixel_spacing}")


def create_vtk_image_data_from_points(points, scalars, pixel_dims, spacing, origin):

    # points, scalars, pixel_dims, pixel_spacing, originからarray_dicomを求める
    array_dicom = np.zeros(pixel_dims, dtype=scalars.dtype)

    # 各点の座標をインデックスに変換
    x_indices = ((points[:, 0] - origin[0]) / pixel_spacing[0]).astype(int)
    y_indices = ((points[:, 1] - origin[1]) / pixel_spacing[1]).astype(int)
    z_indices = ((points[:, 2] - origin[2]) / pixel_spacing[2]).astype(int)

    # スカラー値をarray_dicomに設定
    array_dicom[x_indices, y_indices, z_indices] = scalars

    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(array_dicom.shape)
    vtk_image_data.SetSpacing(spacing)

    flat_array_dicom = array_dicom.flatten(order="F")
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_array_dicom, deep=True, array_type=vtk.VTK_FLOAT)

    vtk_image_data.GetPointData().SetScalars(vtk_array)

    return vtk_image_data


# vtk_image_data = create_vtk_image_data_from_points(points, scalars, pixel_dims, pixel_spacing, origin)

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
    cube.SetZPlusFaceText("S")
    cube.SetZMinusFaceText("I")
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

text_actor = vtk.vtkTextActor()
text_actor.GetTextProperty().SetFontSize(24)
text_actor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
text_actor.SetPosition(render_window.GetSize()[0] - 350, 10)
renderer.AddActor2D(text_actor)

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


def on_right_button_down(obj, event):
    global volume, renderer, points, scalars, pixel_dims, pixel_spacing, origin, vtk_image_data

    click_pos = render_window_interactor.GetEventPosition()
    camera = renderer.GetActiveCamera()
    cam_pos = np.array(camera.GetPosition())
    picker = vtk.vtkWorldPointPicker()
    picker.Pick(click_pos[0], click_pos[1], 0, renderer)
    click_pos_3d = np.array(picker.GetPickPosition())

    ray_dir = click_pos_3d - cam_pos
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    ray_start = cam_pos
    distances_to_ray = calculate_distances_to_ray(points, ray_start, ray_dir)

    threshold = 500
    mask = distances_to_ray <= threshold

    true_count = np.sum(mask)
    print(f"Number of True values in mask: {true_count}")

    # print(f"mask.shape:{mask.shape}")
    # print(f"mask:{mask}")

    # マスク部分のスカラー値を0に設定
    updated_scalars = np.copy(scalars)
    updated_scalars[~mask] = 0
    print(f"updated_scalars.shape: {updated_scalars.shape}")
    print(f"updated_scalars: {updated_scalars}")

    # # フィルタリングされたデータからvtkImageDataを作成
    # vtk_image_data = create_vtk_image_data_from_points(points, updated_scalars, pixel_dims, pixel_spacing, origin)

    # # ボリュームレンダリングの設定
    # filtered_volume = setup_volume_rendering(vtk_image_data)

    # # 元のボリュームを削除して新しいボリュームを追加
    # renderer.RemoveVolume(volume)
    # renderer.AddVolume(filtered_volume)
    # render_window.Render()

    # # 新しいボリュームをグローバル変数に設定
    # volume = filtered_volume


interactor_style = vtk.vtkInteractorStyleTrackballCamera()
render_window_interactor.SetInteractorStyle(interactor_style)
render_window_interactor.AddObserver("RightButtonPressEvent", on_right_button_down)


# %% Keyboard, Mouse interactions


def on_mouse_move(obj, event):
    mouse_pos = render_window_interactor.GetEventPosition()
    picker = vtk.vtkWorldPointPicker()
    picker.Pick(mouse_pos[0], mouse_pos[1], 0, renderer)
    world_pos = picker.GetPickPosition()
    text_actor.SetInput(f"Mouse: ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})")
    text_actor.SetPosition(render_window.GetSize()[0] - 350, 10)
    render_window.Render()


render_window_interactor.AddObserver("MouseMoveEvent", on_mouse_move)


# カメラリセット関数の定義
def reset_camera():
    # ボリュームの中心を取得
    volume_center = volume.GetCenter()
    center_x = volume_center[0]
    center_y = volume_center[1] + 500
    center_z = volume_center[2]

    camera = renderer.GetActiveCamera()
    camera.SetPosition(center_x, center_y, center_z)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCameraClippingRange()
    render_window.Render()


# リセットボタンの作成
reset_button = vtk.vtkTextActor()
reset_button.SetInput("Reset Camera")
reset_button.GetTextProperty().SetFontSize(24)
reset_button.GetTextProperty().SetColor(1.0, 0.0, 0.0)  # 赤色
reset_button.SetPosition(render_window.GetSize()[0] - 170, 40)  # 右下に位置を設定
renderer.AddActor2D(reset_button)


# クリックイベントのコールバック関数
def on_left_button_down(obj, event):
    click_pos = render_window_interactor.GetEventPosition()
    picker = vtk.vtkPropPicker()
    picker.Pick(click_pos[0], click_pos[1], 0, renderer)
    picked_actor = picker.GetActor2D()
    if picked_actor == reset_button:
        reset_camera()


# 左クリックイベントをカスタムコールバックにバインド
render_window_interactor.AddObserver("LeftButtonPressEvent", on_left_button_down)


# ウィンドウのリサイズイベントのコールバック関数
def on_resize(obj, event):
    render_window.Render()
    reset_button.SetPosition(render_window.GetSize()[0] - 160, 40)  # リサイズ後も右下に位置を設定


# リサイズイベントをカスタムコールバックにバインド
render_window_interactor.AddObserver("ConfigureEvent", on_resize)

render_window.Render()
render_window_interactor.Start()

# %%

# print(vtk_image_data)


# # ボリュームレンダリングの設定
# volume = setup_volume_rendering(vtk_image_data)

# renderer = vtk.vtkRenderer()
# render_window = vtk.vtkRenderWindow()
# render_window.AddRenderer(renderer)
# render_window_interactor = vtk.vtkRenderWindowInteractor()
# render_window_interactor.SetRenderWindow(render_window)

# renderer.SetBackground(1, 1, 1)
# render_window.SetSize(800, 800)

# renderer.AddVolume(volume)
# renderer.ResetCamera()
# camera = renderer.GetActiveCamera()
# camera.SetFocalPoint(volume.GetCenter())
# camera.SetPosition(volume.GetCenter()[0], volume.GetCenter()[1] + 500, volume.GetCenter()[2])
# camera.SetViewUp(0, 0, 1)
# renderer.ResetCameraClippingRange()

# xyzLabels = ["X", "Y", "Z"]
# scale = [1.5, 1.5, 1.5]
# colors = vtkNamedColors()
# axes = MakeCubeActor(scale, xyzLabels, colors)
# om = vtkOrientationMarkerWidget()
# om.SetOrientationMarker(axes)
# om.SetViewport(0.8, 0.8, 1.0, 1.0)
# om.SetInteractor(render_window_interactor)
# om.EnabledOn()
# om.InteractiveOn()

# text_actor = vtk.vtkTextActor()
# text_actor.GetTextProperty().SetFontSize(24)
# text_actor.GetTextProperty().SetColor(1.0, 0.0, 0.0)
# text_actor.SetPosition(render_window.GetSize()[0] - 350, 10)
# renderer.AddActor2D(text_actor)

# lay_info = None
# ray_actor = None
# intersection_point = None
# mesh_mapper = None

# interactor_style = vtk.vtkInteractorStyleTrackballCamera()
# render_window_interactor.SetInteractorStyle(interactor_style)
# render_window_interactor.AddObserver("RightButtonPressEvent", on_right_button_down)
