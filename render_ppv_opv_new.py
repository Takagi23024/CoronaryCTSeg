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


def load_dicom_data(dicom_dir):
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

    ArrayDicom = ArrayDicom[:, :, ::-1]

    return ArrayDicom, ConstPixelDims, ConstPixelSpacing, reader


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
ArrayDicom, ConstPixelDims, ConstPixelSpacing, reader = load_dicom_data(PathDicom)

vtk_image_data = numpy_to_vtk_image_data(ArrayDicom, ConstPixelSpacing)
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


def on_right_button_down(obj, event):
    global lay_info, ray_actor, intersection_point, mesh_mapper, reader

    if ray_actor is None:
        click_pos = render_window_interactor.GetEventPosition()
        camera = renderer.GetActiveCamera()
        cam_pos = np.array(camera.GetPosition())
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        click_pos_3d = np.array(picker.GetPickPosition())

        ray_dir = click_pos_3d - cam_pos
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        ray_length = 1000.0
        ray_end = cam_pos + ray_dir * ray_length

        ray = vtk.vtkLineSource()
        ray.SetPoint1(cam_pos)
        ray.SetPoint2(ray_end)
        ray.Update()

        ray_mapper = vtk.vtkPolyDataMapper()
        ray_mapper.SetInputData(ray.GetOutput())

        ray_actor = vtk.vtkActor()
        ray_actor.SetMapper(ray_mapper)
        ray_actor.GetProperty().SetColor(1, 0, 0)

        renderer.AddActor(ray_actor)
        render_window.Render()

        lay_info = (click_pos_3d, ray_dir)
    else:
        click_pos = render_window_interactor.GetEventPosition()
        camera = renderer.GetActiveCamera()
        cam_pos = np.array(camera.GetPosition())
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, renderer)
        click_pos_3d = np.array(picker.GetPickPosition())
        ray_dir = click_pos_3d - cam_pos
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        p1, d1 = lay_info
        p2, d2 = click_pos_3d, ray_dir

        def closest_midpoint_between_lines(p1, d1, p2, d2):
            def closest_points_on_lines(p1, d1, p2, d2):
                w0 = p1 - p2
                a = np.dot(d1, d1)
                b = np.dot(d1, d2)
                c = np.dot(d2, d2)
                d = np.dot(d1, w0)
                e = np.dot(d2, w0)

                denom = a * c - b * b
                if denom == 0:
                    return None, None

                s = (b * e - c * d) / denom
                t = (a * e - b * d) / denom

                closest_point_on_line1 = p1 + s * d1
                closest_point_on_line2 = p2 + t * d2

                return closest_point_on_line1, closest_point_on_line2

            point1, point2 = closest_points_on_lines(p1, d1, p2, d2)

            if point1 is not None and point2 is not None:
                midpoint = (point1 + point2) / 2
                return midpoint
            else:
                return None

        intersection_point = closest_midpoint_between_lines(p1, d1, p2, d2)

        renderer.RemoveActor(ray_actor)
        render_window.Render()

        radius = 5
        sphere = vtk.vtkSphere()
        sphere.SetCenter(intersection_point)
        sphere.SetRadius(radius)

        extract_sphere = vtk.vtkExtractGeometry()
        extract_sphere.SetInputData(vtk_image_data)
        extract_sphere.SetImplicitFunction(sphere)
        extract_sphere.Update()

        extracted_data = extract_sphere.GetOutput()
        points = extracted_data.GetPoints()
        point_data = extracted_data.GetPointData().GetScalars()

        if points and point_data:
            coordinates = []
            values = []
            for i in range(points.GetNumberOfPoints()):
                coordinate = points.GetPoint(i)
                value = point_data.GetTuple1(i)
                coordinates.append(coordinate)
                values.append(value)

            coordinates_np = np.array(coordinates)
            values_np = np.array(values)

            center_value = ArrayDicom[
                int((intersection_point[0] - reader.GetDataExtent()[0]) / ConstPixelSpacing[0]),
                int((intersection_point[1] - reader.GetDataExtent()[2]) / ConstPixelSpacing[1]),
                int((intersection_point[2] - reader.GetDataExtent()[4]) / ConstPixelSpacing[2]),
            ]

            mask = (values_np >= center_value - 10) & (values_np <= center_value + 10)
            filtered_coordinates = coordinates_np[mask]

            from sklearn.decomposition import PCA

            pca = PCA(n_components=3)
            pca.fit(filtered_coordinates)

            center_of_mass = np.mean(filtered_coordinates, axis=0)

            principal_direction_1 = pca.components_[0]
            principal_direction_2 = pca.components_[1]
            principal_direction_3 = pca.components_[2]

            diameter = 2 * radius

            line_start_1 = center_of_mass - principal_direction_1 * (diameter / 2)
            line_end_1 = center_of_mass + principal_direction_1 * (diameter / 2)

            arrow_source_1 = vtk.vtkArrowSource()
            arrow_source_1.SetTipLength(0.3)
            arrow_source_1.SetTipRadius(0.2)
            arrow_source_1.SetShaftRadius(0.1)

            direction_1 = line_end_1 - line_start_1
            direction_1 = direction_1 / np.linalg.norm(direction_1)

            transform_1 = vtk.vtkTransform()
            transform_1.Translate(line_start_1)

            v1 = np.array([1, 0, 0])
            v2 = direction_1

            axis = np.cross(v1, v2)
            angle_1 = np.arccos(np.dot(v1, v2))

            if np.linalg.norm(axis) < 1e-6:
                axis = np.array([0, 0, 1])
                if np.dot(v1, v2) < 0:
                    angle_1 = np.pi

            axis = axis / np.linalg.norm(axis)
            transform_1.RotateWXYZ(np.degrees(angle_1), axis[0], axis[1], axis[2])
            transform_1.Scale(np.linalg.norm(line_end_1 - line_start_1), 1, 1)

            transform_filter_1 = vtk.vtkTransformPolyDataFilter()
            transform_filter_1.SetTransform(transform_1)
            transform_filter_1.SetInputConnection(arrow_source_1.GetOutputPort())
            transform_filter_1.Update()

            arrow_mapper_1 = vtk.vtkPolyDataMapper()
            arrow_mapper_1.SetInputData(transform_filter_1.GetOutput())

            arrow_actor_1 = vtk.vtkActor()
            arrow_actor_1.SetMapper(arrow_mapper_1)
            arrow_actor_1.GetProperty().SetColor(1, 1, 0)

            renderer.AddActor(arrow_actor_1)

            line_start_2 = center_of_mass - principal_direction_2 * (diameter / 4)
            line_end_2 = center_of_mass + principal_direction_2 * (diameter / 4)

            arrow_source_2 = vtk.vtkArrowSource()
            arrow_source_2.SetTipLength(0.3)
            arrow_source_2.SetTipRadius(0.2)
            arrow_source_2.SetShaftRadius(0.1)

            direction_2 = line_end_2 - line_start_2
            direction_2 = direction_2 / np.linalg.norm(direction_2)

            transform_2 = vtk.vtkTransform()
            transform_2.Translate(line_start_2)

            v1 = np.array([1, 0, 0])
            v2 = direction_2

            axis = np.cross(v1, v2)
            angle_2 = np.arccos(np.dot(v1, v2))

            if np.linalg.norm(axis) < 1e-6:
                axis = np.array([0, 0, 1])
                if np.dot(v1, v2) < 0:
                    angle_2 = np.pi

            axis = axis / np.linalg.norm(axis)
            transform_2.RotateWXYZ(np.degrees(angle_2), axis[0], axis[1], axis[2])
            transform_2.Scale(np.linalg.norm(line_end_2 - line_start_2), 1, 1)

            transform_filter_2 = vtk.vtkTransformPolyDataFilter()
            transform_filter_2.SetTransform(transform_2)
            transform_filter_2.SetInputConnection(arrow_source_2.GetOutputPort())
            transform_filter_2.Update()

            arrow_mapper_2 = vtk.vtkPolyDataMapper()
            arrow_mapper_2.SetInputData(transform_filter_2.GetOutput())

            arrow_actor_2 = vtk.vtkActor()
            arrow_actor_2.SetMapper(arrow_mapper_2)
            arrow_actor_2.GetProperty().SetColor(0, 1, 0)

            renderer.AddActor(arrow_actor_2)

            line_start_3 = center_of_mass - principal_direction_3 * (diameter / 4)
            line_end_3 = center_of_mass + principal_direction_3 * (diameter / 4)

            arrow_source_3 = vtk.vtkArrowSource()
            arrow_source_3.SetTipLength(0.3)
            arrow_source_3.SetTipRadius(0.2)
            arrow_source_3.SetShaftRadius(0.1)

            direction_3 = line_end_3 - line_start_3
            direction_3 = direction_3 / np.linalg.norm(direction_3)

            transform_3 = vtk.vtkTransform()
            transform_3.Translate(line_start_3)

            v1 = np.array([1, 0, 0])
            v2 = direction_3

            axis = np.cross(v1, v2)
            angle_3 = np.arccos(np.dot(v1, v2))

            if np.linalg.norm(axis) < 1e-6:
                axis = np.array([0, 0, 1])
                if np.dot(v1, v2) < 0:
                    angle_3 = np.pi

            axis = axis / np.linalg.norm(axis)
            transform_3.RotateWXYZ(np.degrees(angle_3), axis[0], axis[1], axis[2])
            transform_3.Scale(np.linalg.norm(line_end_3 - line_start_3), 1, 1)

            transform_filter_3 = vtk.vtkTransformPolyDataFilter()
            transform_filter_3.SetTransform(transform_3)
            transform_filter_3.SetInputConnection(arrow_source_3.GetOutputPort())
            transform_filter_3.Update()

            arrow_mapper_3 = vtk.vtkPolyDataMapper()
            arrow_mapper_3.SetInputData(transform_filter_3.GetOutput())

            arrow_actor_3 = vtk.vtkActor()
            arrow_actor_3.SetMapper(arrow_mapper_3)
            arrow_actor_3.GetProperty().SetColor(0, 0, 1)

            renderer.AddActor(arrow_actor_3)

            def angle_to_label(angle_y_axis):
                if angle_y_axis[0] >= 0:
                    vertical_label = "CRA"
                    vertical_degree = np.degrees(angle_y_axis[0])
                else:
                    vertical_label = "CAU"
                    vertical_degree = -np.degrees(angle_y_axis[0])

                if angle_y_axis[2] >= 0:
                    horizontal_label = "LAO"
                    horizontal_degree = np.degrees(angle_y_axis[2])
                else:
                    horizontal_label = "RAO"
                    horizontal_degree = -np.degrees(angle_y_axis[2])

                return f"{vertical_label}:{vertical_degree:.1f}°, {horizontal_label}:{horizontal_degree:.1f}°"

            angle_1_y_axis = np.cross(np.array([0, 1, 0]), direction_1)
            angle_2_y_axis = np.cross(np.array([0, 1, 0]), direction_2)
            angle_3_y_axis = np.cross(np.array([0, 1, 0]), direction_3)

            label_1 = angle_to_label(angle_1_y_axis)
            label_2 = angle_to_label(angle_2_y_axis)
            label_3 = angle_to_label(angle_3_y_axis)

            text_actor_1 = vtk.vtkTextActor()

            text_actor_1.SetInput(f"Ve: {label_1}")
            text_actor_1.GetTextProperty().SetFontSize(24)
            text_actor_1.GetTextProperty().SetColor(1, 1, 0)
            text_actor_1.SetPosition(10, 70)

            text_actor_2 = vtk.vtkTextActor()
            text_actor_2.SetInput(f"PPV: {label_2}")
            text_actor_2.GetTextProperty().SetFontSize(24)
            text_actor_2.GetTextProperty().SetColor(0, 1, 0)
            text_actor_2.SetPosition(10, 40)

            text_actor_3 = vtk.vtkTextActor()
            text_actor_3.SetInput(f"OPV: {label_3}")
            text_actor_3.GetTextProperty().SetFontSize(24)
            text_actor_3.GetTextProperty().SetColor(0, 0, 1)
            text_actor_3.SetPosition(10, 10)

            renderer.AddActor2D(text_actor_1)
            renderer.AddActor2D(text_actor_2)
            renderer.AddActor2D(text_actor_3)
            render_window.Render()


interactor_style = vtk.vtkInteractorStyleTrackballCamera()
render_window_interactor.SetInteractorStyle(interactor_style)
render_window_interactor.AddObserver("RightButtonPressEvent", on_right_button_down)


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
reset_button.SetPosition(render_window.GetSize()[0] - 160, 40)  # 右下に位置を設定
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
