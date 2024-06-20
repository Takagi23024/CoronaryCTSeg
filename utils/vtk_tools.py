import vtk
from vtkmodules.vtkRenderingAnnotation import vtkAnnotatedCubeActor, vtkAxesActor
from vtkmodules.vtkRenderingCore import vtkPropAssembly


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


class myVtkInteractorStyleImage(vtk.vtkInteractorStyleImage):
    def __init__(self, imageViewer=None, statusMapper=None):
        self._ImageViewer = imageViewer
        self._StatusMapper = statusMapper
        self._Slice = 0
        self._MinSlice = 0
        self._MaxSlice = 0
        if imageViewer:
            self.SetImageViewer(imageViewer)
        if statusMapper:
            self.SetStatusMapper(statusMapper)

    def SetImageViewer(self, imageViewer):
        self._ImageViewer = imageViewer
        self._MinSlice = imageViewer.GetSliceMin()
        self._MaxSlice = imageViewer.GetSliceMax()
        self._Slice = self._MinSlice
        print(f"Slicer: Min = {self._MinSlice}, Max = {self._MaxSlice}")

    def SetStatusMapper(self, statusMapper):
        self._StatusMapper = statusMapper

    def MoveSliceForward(self):
        if self._Slice < self._MaxSlice:
            self._Slice += 1
            self._ImageViewer.SetSlice(self._Slice)
            self._ImageViewer.Render()

    def MoveSliceBackward(self):
        if self._Slice > self._MinSlice:
            self._Slice -= 1
            self._ImageViewer.SetSlice(self._Slice)
            self._ImageViewer.Render()

    def OnKeyDown(self):
        key = self.GetInteractor().GetKeySym()
        if key == "Up":
            self.MoveSliceForward()
        elif key == "Down":
            self.MoveSliceBackward()
        vtk.vtkInteractorStyleImage.OnKeyDown(self)

    def OnMouseWheelForward(self):
        self.MoveSliceForward()

    def OnMouseWheelBackward(self):
        self.MoveSliceBackward()


def create_slider_widget(imageViewer, interactor, min_slice, max_slice):
    sliderRep = vtk.vtkSliderRepresentation2D()
    sliderRep.SetMinimumValue(min_slice)
    sliderRep.SetMaximumValue(max_slice)
    sliderRep.SetValue(min_slice)
    sliderRep.SetTitleText("Slice")
    sliderRep.GetSliderProperty().SetColor(1, 0, 0)
    sliderRep.GetTitleProperty().SetColor(1, 0, 0)
    sliderRep.GetLabelProperty().SetColor(1, 0, 0)
    sliderRep.GetSelectedProperty().SetColor(0, 1, 0)
    sliderRep.GetTubeProperty().SetColor(1, 1, 0)
    sliderRep.GetCapProperty().SetColor(1, 1, 0)
    sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint1Coordinate().SetValue(0.3, 0.1)
    sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    sliderRep.GetPoint2Coordinate().SetValue(0.7, 0.1)
    sliderRep.SetSliderLength(0.02)
    sliderRep.SetSliderWidth(0.03)
    sliderRep.SetEndCapLength(0.01)
    sliderRep.SetEndCapWidth(0.03)
    sliderRep.SetTubeWidth(0.005)
    sliderRep.SetLabelFormat("%0.0f")

    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(interactor)
    sliderWidget.SetRepresentation(sliderRep)
    sliderWidget.SetAnimationModeToAnimate()
    sliderWidget.EnabledOn()

    def slider_callback(obj, event):
        value = int(obj.GetRepresentation().GetValue())
        imageViewer.SetSlice(value)
        imageViewer.Render()

    sliderWidget.AddObserver("InteractionEvent", slider_callback)

    return sliderWidget
