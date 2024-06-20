# %%
# %%
import numpy as np
import pydicom
import vtk
from vtk.util import numpy_support
from vtkmodules.vtkCommonColor import vtkNamedColors

"""--------------↓reader.py-----------------"""


def read_angio_dicom(angio_dicom_dir):

    import glob
    import os

    files = glob.glob(os.path.join(angio_dicom_dir, "*.dcm"))
    datasets = [pydicom.dcmread(f) for f in files]

    angio_dicom_infos = []
    for i in range(len(datasets)):
        dicom_data = datasets[i]

        ArrayDicom = dicom_data.pixel_array  # (Slice, Height, Width)
        ArrayDicom = np.transpose(ArrayDicom, (2, 1, 0))  # ->(Width, Height, Slice)
        ArrayDicom = ArrayDicom[:, ::-1, :]
        total_slices = ArrayDicom.shape[2]  # Correctly calculate total_slices

        # カメラと患者の距離
        distance_source_to_patient = float(dicom_data[(0x0018, 0x1111)].value)

        # アーム角度の取得
        rao_lao_angle = float(dicom_data[(0x0018, 0x1510)].value)
        cra_cau_angle = float(dicom_data[(0x0018, 0x1511)].value)

        angio_dicom_infos.append((ArrayDicom, total_slices, distance_source_to_patient, rao_lao_angle, cra_cau_angle))

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


"""--------------↑reader.py-----------------"""
"""--------------↓converter.py-----------------"""


def ct_numpy_to_vtk_image(numpy_array, spacing):
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


def angio_numpy_to_vtk_image(numpy_array):
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(numpy_array.shape)
    flat_numpy_array = numpy_array.flatten(order="F")
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_numpy_array, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image_data.GetPointData().SetScalars(vtk_array)
    return vtk_image_data


"""--------------↑converter.py-----------------"""


def rotation_matrix(axis, theta):
    """
    回転軸と回転角度から回転行列を生成する。

    パラメータ:
    axis (numpy.ndarray): 回転軸ベクトル。
    theta (float): 回転角度（ラジアン）。

    戻り値:
    numpy.ndarray: 3x3の回転行列。
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta)
    b, c, d = axis * np.sin(theta)
    return np.array(
        [
            [a + b * b * (1 - a), b * c * (1 - a) - d, b * d * (1 - a) + c],
            [c * b * (1 - a) + d, a + c * c * (1 - a), c * d * (1 - a) - b],
            [d * b * (1 - a) - c, d * c * (1 - a) + b, a + d * d * (1 - a)],
        ]
    )


def rotate_camera_around_volume_center(camera, volume_center, horizontal_angle, vertical_angle):
    """
    カメラを volume_center を回転中心として Y軸を基準に回転させる。

    パラメータ:
    camera (vtk.vtkCamera): VTKカメラオブジェクト。
    volume_center (tuple): ボリュームの中心点 (x, y, z)。
    horizontal_angle (float): 水平角度（度）。
    vertical_angle (float): 垂直角度（度）。
    """
    # 度をラジアンに変換
    horizontal_radian = np.radians(horizontal_angle)
    vertical_radian = np.radians(vertical_angle)

    # カメラの位置を取得
    position = np.array(camera.GetPosition())

    # カメラの位置を volume_center を基準に移動
    position_relative = position - np.array(volume_center)

    # 水平回転行列を生成 (Y軸周りの回転)
    horizontal_axis = np.array([0, 1, 0])
    horizontal_rotation_matrix = rotation_matrix(horizontal_axis, horizontal_radian)

    # 垂直回転行列を生成 (X軸周りの回転)
    vertical_axis = np.array([1, 0, 0])
    vertical_rotation_matrix = rotation_matrix(vertical_axis, vertical_radian)

    # 回転行列を組み合わせる
    combined_matrix = np.dot(vertical_rotation_matrix, horizontal_rotation_matrix)

    # カメラの位置を回転
    new_position_relative = np.dot(combined_matrix, position_relative)
    new_position = new_position_relative + np.array(volume_center)

    # カメラの位置を更新
    camera.SetPosition(new_position)

    # カメラのビューアップベクトルを更新
    view_up = np.array(camera.GetViewUp())
    new_view_up = np.dot(combined_matrix, view_up)
    camera.SetViewUp(new_view_up)


"""--------------↑vector_calculator.py-----------------"""


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


angio_dicom_dir = "./data/MIE022_Angio/"
ct_dicom_dir = "./data/MIE022_CT/"

# read angio dicom and make vtk_image
angio_dicom_infos = read_angio_dicom(angio_dicom_dir)
rao_lao_angles = [angle for info in angio_dicom_infos for angle in [info[3]] * info[1]]
cra_cau_angles = [angle for info in angio_dicom_infos for angle in [info[4]] * info[1]]
combined_array = np.concatenate([info[0] for info in angio_dicom_infos], axis=-1)
angio_vtk_image = angio_numpy_to_vtk_image(combined_array)

# read ct dicom and make vtk_image
ArrayDicom, ConstPixelDims, ConstPixelSpacing = read_ct_dicom(ct_dicom_dir)
ct_vtk_image = ct_numpy_to_vtk_image(ArrayDicom, ConstPixelSpacing)
ct_volume = setup_volume_rendering(ct_vtk_image)

"""-------------------Visuallization↓---------------------------"""
# Colors
colors = vtkNamedColors()

render_window = vtk.vtkRenderWindow()
render_window.SetSize(2000, 1000)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

renderer = vtk.vtkRenderer()
render_window.AddRenderer(renderer)

"""-------------------Angio image viewer↓---------------------------"""
# Create an image viewer
image_viewer = vtk.vtkImageViewer2()
image_viewer.SetInputData(angio_vtk_image)
image_viewer.SetupInteractor(render_window_interactor)
image_viewer.SetRenderWindow(render_window)
image_viewer.GetRenderer().SetViewport(0, 0, 0.5, 1)

# Set window and level for the image viewer
min_val = np.min(combined_array)
max_val = np.max(combined_array)
image_viewer.SetColorWindow(max_val - min_val)
image_viewer.SetColorLevel((max_val + min_val) / 2)


# Initialize the slice index
slice_index = 0
image_viewer.SetSlice(slice_index)


def update_slice_text(actor, slice_index, total_slices, rao_lao_angles, cra_cau_angles):
    slice_text = f"Slice: {slice_index + 1}/{total_slices}\nRAO/LAO: {rao_lao_angles[slice_index]:.2f}\nCRA/CAU: {cra_cau_angles[slice_index]:.2f}"
    actor.SetInput(slice_text)


# Create a text actor to display the slice number and angles
total_slices = angio_vtk_image.GetDimensions()[2]
slice_text_actor = vtk.vtkTextActor()
update_slice_text(slice_text_actor, slice_index, total_slices, rao_lao_angles, cra_cau_angles)
slice_text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White color
slice_text_actor.GetTextProperty().SetFontSize(24)
slice_text_actor.SetPosition(10, 10)  # Position at the bottom left corner
image_viewer.GetRenderer().AddActor2D(slice_text_actor)


# Define a callback function to handle key press events
def KeyPressCallback(obj, event):
    global slice_index
    key = obj.GetKeySym()
    if key == "Up":
        slice_index += 1
    elif key == "Down":
        slice_index -= 1
    slice_index = max(0, min(slice_index, total_slices - 1))
    image_viewer.SetSlice(slice_index)
    update_slice_text(
        slice_text_actor, slice_index, total_slices, rao_lao_angles, cra_cau_angles
    )  # Update the slice number text

    # Update the camera angles based on the current slice
    horizontal_angle = rao_lao_angles[slice_index]
    vertical_angle = cra_cau_angles[slice_index]

    camera.SetFocalPoint(volume_center)
    camera.SetPosition(volume_center[0], volume_center[1] + 500, volume_center[2])
    camera.SetViewUp(0, 0, 1)

    # カメラを回転
    rotate_camera_around_volume_center(camera, volume_center, horizontal_angle, vertical_angle)

    # レンダラーを更新
    ct_renderer.ResetCameraClippingRange()

    image_viewer.Render()


# Add the callback function to the interactor
render_window_interactor.AddObserver("KeyPressEvent", KeyPressCallback)


# Define a callback function to handle mouse press events
def MousePressCallback(obj, event):
    x, y = render_window_interactor.GetEventPosition()
    width, height = render_window.GetSize()
    if x < width / 2:
        # Left side (Angio image viewer)
        interactor_style = vtk.vtkInteractorStyleImage()
        render_window_interactor.SetInteractorStyle(interactor_style)
    else:
        # Right side (CT volume rendering)
        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        render_window_interactor.SetInteractorStyle(interactor_style)


# Add the mouse press callback function to the interactor
render_window_interactor.AddObserver("RightButtonPressEvent", MousePressCallback)

image_viewer.Render()


"""-------------------CT volume rendering ↓---------------------------"""
ct_renderer = vtk.vtkRenderer()
render_window.AddRenderer(ct_renderer)
ct_renderer.SetViewport(0.5, 0, 1, 1)
ct_renderer.AddVolume(ct_volume)

# Set up the camera for the CT renderer
ct_renderer.ResetCamera()
ct_camera = ct_renderer.GetActiveCamera()

# アンギオ画像の角度をCTレンダラーに反映
camera = ct_renderer.GetActiveCamera()
volume_center = ct_volume.GetCenter()
camera.SetFocalPoint(volume_center)
camera.SetPosition(volume_center[0], volume_center[1] + 500, volume_center[2])
camera.SetViewUp(0, 0, 1)

# 回転角度を設定
horizontal_angle = rao_lao_angles[0]
vertical_angle = cra_cau_angles[0]

# カメラを回転
rotate_camera_around_volume_center(camera, volume_center, horizontal_angle, vertical_angle)

# レンダラーを更新
ct_renderer.ResetCameraClippingRange()

# Render and start interaction
render_window.Render()
render_window_interactor.Start()

# %%
