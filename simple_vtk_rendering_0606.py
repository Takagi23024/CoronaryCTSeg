# %%


import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor, vtkAxesActor
from vtkmodules.vtkRenderingCore import vtkPropAssembly


def load_dicom(dicom_dir):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dicom_dir)
    reader.Update()
    imageData = reader.GetOutput()
    return imageData


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
    vtk_image_data = load_dicom(PathDicom)
    volume = setup_volume_rendering(vtk_image_data)
    volume = setup_volume_rendering(vtk_image_data)
    render_volume(volume)


if __name__ == "__main__":
    main()


# %%
