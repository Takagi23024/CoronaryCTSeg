# %%
import numpy as np
import pydicom
import vtkmodules.all as vtk
from vtk.util import numpy_support


def read_angio_dicom(angio_dicom_dir):

    import glob
    import os

    files = glob.glob(os.path.join(angio_dicom_dir, "*.dcm"))
    datasets = [pydicom.dcmread(f) for f in files]

    angio_dicom_infos = []
    for i in range(len(datasets)):
        dicom_data = datasets[i]

        ArrayDicom = dicom_data.pixel_array  # (Frame_num, Height, Width)
        ArrayDicom = np.transpose(ArrayDicom, (2, 1, 0))  # (Width, Height, Frame_num)
        ArrayDicom = ArrayDicom[:, ::-1, :]

        # カメラと患者の距離
        distance_source_to_patient = float(dicom_data[(0x0018, 0x1111)].value)

        # アーム角度の取得
        rao_lao_angle = float(dicom_data[(0x0018, 0x1510)].value)
        cra_cau_angle = float(dicom_data[(0x0018, 0x1511)].value)

        angio_dicom_infos.append((ArrayDicom, distance_source_to_patient, rao_lao_angle, cra_cau_angle))

    return angio_dicom_infos


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


def numpy_to_vtk_image(numpy_array):
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(numpy_array.shape)
    flat_numpy_array = numpy_array.flatten(order="F")
    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_numpy_array, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_image_data.GetPointData().SetScalars(vtk_array)
    return vtk_image_data


def main():
    angio_dicom_dir = "./data/MIE022_Angio/"

    angio_dicom_infos = read_angio_dicom(angio_dicom_dir)
    combined_array = np.concatenate([info[0] for info in angio_dicom_infos], axis=-1)

    vtk_image = numpy_to_vtk_image(combined_array)

    imageViewer = vtk.vtkImageViewer2()
    imageViewer.SetInputData(vtk_image)

    # スカラー範囲を設定
    min_val = np.min(combined_array)
    max_val = np.max(combined_array)
    imageViewer.GetWindowLevel().SetWindow(max_val - min_val)
    imageViewer.GetWindowLevel().SetLevel((max_val + min_val) / 2)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    imageViewer.SetupInteractor(renderWindowInteractor)

    statusMapper = vtk.vtkTextMapper()
    statusActor = vtk.vtkActor2D()
    statusActor.SetMapper(statusMapper)
    statusActor.SetPosition(15, 10)

    imageViewer.GetRenderer().AddActor2D(statusActor)

    # スライドバーを作成して追加
    sliderWidget = create_slider_widget(
        imageViewer, renderWindowInteractor, imageViewer.GetSliceMin(), imageViewer.GetSliceMax()
    )

    imageViewer.Render()
    imageViewer.GetRenderer().ResetCamera()
    imageViewer.Render()

    renderWindowInteractor.Start()


if __name__ == "__main__":
    main()

# %%
