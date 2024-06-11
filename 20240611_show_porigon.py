# %%
import numpy as np
import vtk
from skimage import filters
from vtk.util import numpy_support


def closest_midpoint_between_lines(p1, d1, p2, d2):
    """
    2本の直線の最短距離の中点を求める関数

    Parameters:
    p1 (np.array): 1本目の直線上の点
    d1 (np.array): 1本目の直線の方向ベクトル
    p2 (np.array): 2本目の直線上の点
    d2 (np.array): 2本目の直線の方向ベクトル

    Returns:
    np.array: 最短距離の中点
    """

    def closest_points_on_lines(p1, d1, p2, d2):
        w0 = p1 - p2
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, w0)
        e = np.dot(d2, w0)

        denom = a * c - b * b
        if denom == 0:
            # 直線が平行な場合
            return None, None

        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom

        closest_point_on_line1 = p1 + s * d1
        closest_point_on_line2 = p2 + t * d2

        return closest_point_on_line1, closest_point_on_line2

    # 最も近い点を求める
    point1, point2 = closest_points_on_lines(p1, d1, p2, d2)

    if point1 is not None and point2 is not None:
        # 最も近い点の中点を求める
        midpoint = (point1 + point2) / 2
        return midpoint
    else:
        return None


# DICOMファイルが格納されているディレクトリのパスを設定
PathDicom = "./data/MIE022_CT/"

PathObj = "lumen.obj"
# PathObj = PathBase + "whole.obj"
# PathObj = PathBase + "calcium.obj"

# vtkDICOMImageReaderオブジェクトを作成し、DICOMファイルのディレクトリを設定してデータを読み込む
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(PathDicom)
# reader.SetFileName("./data/diao_0.nii")
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

# 大津閾値でフィルタリング
# thresholds = filters.threshold_multiotsu(ArrayDicom)
# ArrayDicom[ArrayDicom < thresholds[1]] = 0

ArrayDicom[ArrayDicom < 100] = 0
ArrayDicom[ArrayDicom > 500] = 0
# ArrayDicom[ArrayDicom < 700] = 0


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
volume_color.AddRGBPoint(-1000, 0, 0, 0)
volume_color.AddRGBPoint(400, 1.0, 0.0, 0.0)
volume_color.AddRGBPoint(600, 0.8, 0.8, 0.8)
volume_color.AddRGBPoint(1000, 1.0, 1.0, 1.0)

volume_scalar_opacity = vtk.vtkPiecewiseFunction()
volume_scalar_opacity.AddPoint(-1000, 0.0)
volume_scalar_opacity.AddPoint(400, 0.5)
volume_scalar_opacity.AddPoint(600, 1.0)
volume_scalar_opacity.AddPoint(1000, 1.0)

volume_gradient_opacity = vtk.vtkPiecewiseFunction()
volume_gradient_opacity.AddPoint(0, 0.00)
volume_gradient_opacity.AddPoint(100, 0.5)

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


# ボリュームデータの取得
volume_data = volume.GetMapper().GetInput()

# Marching Cubesアルゴリズムを使用してポリゴンデータを生成
marching_cubes = vtk.vtkMarchingCubes()
marching_cubes.SetInputData(volume_data)
marching_cubes.SetValue(0, 128)  # しきい値を設定

# ポリゴンデータの取得
marching_cubes.Update()
poly_data = marching_cubes.GetOutput()

# ポリゴンデータをスムージング
smooth_filter = vtk.vtkSmoothPolyDataFilter()
smooth_filter.SetInputData(poly_data)
smooth_filter.SetNumberOfIterations(50)  # スムージングの反復回数を設定
smooth_filter.SetRelaxationFactor(0.2)  # リラクゼーション係数を設定
smooth_filter.FeatureEdgeSmoothingOff()  # エッジのスムージングを無効化
smooth_filter.BoundarySmoothingOn()  # 境界のスムージングを有効化
smooth_filter.Update()

# スムージングされたポリゴンデータを取得
smoothed_poly_data = smooth_filter.GetOutput()

# OBJファイルにポリゴンデータを保存
obj_writer = vtk.vtkOBJWriter()
obj_writer.SetFileName(PathObj)
obj_writer.SetInputData(smoothed_poly_data)
obj_writer.Write()


renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

renderer.AddVolume(volume)
renderer.SetBackground(0, 0, 0)
render_window.SetSize(800, 800)

# レイの情報を格納するリスト
ray_info_list = []


def on_right_button_down(obj, event):
    click_pos = render_window_interactor.GetEventPosition()
    print(f"クリック位置の2Dスクリーン座標: {click_pos}")

    # カメラの位置と方向を取得
    camera = renderer.GetActiveCamera()
    cam_pos = np.array(camera.GetPosition())
    focal_point = np.array(camera.GetFocalPoint())
    view_up = np.array(camera.GetViewUp())

    print(f"カメラ位置: {cam_pos}")
    print(f"カメラの焦点位置: {focal_point}")
    print(f"カメラの上方向ベクトル: {view_up}")

    # クリック位置の2Dスクリーン座標を3Dワールド座標に変換
    picker = vtk.vtkWorldPointPicker()
    picker.Pick(click_pos[0], click_pos[1], 0, renderer)
    click_pos_3d = np.array(picker.GetPickPosition())
    print(f"クリック位置の3Dワールド座標: {click_pos_3d}")

    # レイの方向を計算
    ray_dir = click_pos_3d - cam_pos
    ray_dir = ray_dir / np.linalg.norm(ray_dir)  # 正規化

    # レイの長さを指定
    ray_length = 1000.0
    ray_end = cam_pos + ray_dir * ray_length

    # レイキャスティングで交差点を取得
    ray = vtk.vtkLineSource()
    ray.SetPoint1(cam_pos)
    ray.SetPoint2(ray_end)
    ray.Update()

    ray_mapper = vtk.vtkPolyDataMapper()
    ray_mapper.SetInputData(ray.GetOutput())

    ray_actor = vtk.vtkActor()
    ray_actor.SetMapper(ray_mapper)

    # レイの色を赤と青で交互に設定
    ray_color = [1, 0, 0] if len(ray_info_list) % 2 == 0 else [0, 0, 1]
    ray_actor.GetProperty().SetColor(ray_color)

    renderer.AddActor(ray_actor)
    render_window.Render()

    # レイの情報をリストに追加
    ray_info_list.append({"click_pos_3d": click_pos_3d.tolist(), "ray_dir": ray_dir.tolist(), "ray_color": ray_color})

    print("----------------------------------------------------------------")
    print(f"レイの情報: {ray_info_list[-1]}")
    print("----------------------------------------------------------------")

    # 2本のレイが描画されたら交点を計算
    if len(ray_info_list) == 2:
        p1, d1 = np.array(ray_info_list[0]["click_pos_3d"]), np.array(ray_info_list[0]["ray_dir"])
        p2, d2 = np.array(ray_info_list[1]["click_pos_3d"]), np.array(ray_info_list[1]["ray_dir"])

        # 交点または最も近い点を計算
        closest_point = closest_midpoint_between_lines(p1, d1, p2, d2)

        # 交点を描画
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(closest_point)
        sphere.SetRadius(2.0)
        sphere.Update()

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputData(sphere.GetOutput())

        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(0, 1, 0)  # 緑色

        renderer.AddActor(sphere_actor)
        render_window.Render()


# 左クリックはカメラ操作のために既定のイベントハンドラを設定
interactor_style = vtk.vtkInteractorStyleTrackballCamera()
render_window_interactor.SetInteractorStyle(interactor_style)

# 右クリックイベントをカスタムコールバックにバインド
render_window_interactor.AddObserver("RightButtonPressEvent", on_right_button_down)

render_window.Render()
render_window_interactor.Start()

# %%
