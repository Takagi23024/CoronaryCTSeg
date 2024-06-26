import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


class DICOMVolumeRenderer:
    def __init__(self, dicom_dir):
        self.dicom_dir = dicom_dir
        self.vtk_image_data, self.scalar_array, self.point_cloud, self.pixel_dims, self.pixel_spacing, self.origin = (
            self.load_dicom_and_extract_points()
        )

    def load_dicom_and_extract_points(self):
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(self.dicom_dir)
        reader.Update()

        vtk_image_data = reader.GetOutput()
        pixel_spacing = reader.GetPixelSpacing()
        pixel_dims = vtk_image_data.GetDimensions()
        rescale_slope = reader.GetRescaleSlope() if reader.GetRescaleSlope() != 0 else 1
        rescale_intercept = reader.GetRescaleOffset()
        scalars = vtk_to_numpy(vtk_image_data.GetPointData().GetScalars())
        scalar_array = scalars.reshape(pixel_dims, order="F")
        scalar_array = scalar_array * rescale_slope + rescale_intercept
        last_image_first_pixel_coords = reader.GetImagePositionPatient()

        x = np.arange(pixel_dims[0]) * pixel_spacing[0]
        y = np.arange(pixel_dims[1]) * pixel_spacing[1]
        z = np.arange(pixel_dims[2]) * pixel_spacing[2]
        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        point_cloud = np.vstack(
            (
                xv.ravel(),
                yv.ravel(),
                zv.ravel(),
                scalar_array.ravel(),
            )
        ).T

        return vtk_image_data, scalar_array, point_cloud, pixel_dims, pixel_spacing, last_image_first_pixel_coords

    def point_cloud_to_scalar_array(self, point_cloud, pixel_dims, pixel_spacing):
        scalar_array = np.zeros(pixel_dims, dtype=np.int16)

        x_indices = (point_cloud[:, 0] / pixel_spacing[0]).astype(int)
        y_indices = (point_cloud[:, 1] / pixel_spacing[1]).astype(int)
        z_indices = (point_cloud[:, 2] / pixel_spacing[2]).astype(int)

        scalar_array[x_indices, y_indices, z_indices] = point_cloud[:, 3]

        return scalar_array

    def scalar_array_to_vtk_image(self, scalar_array, pixel_spacing):
        if not isinstance(scalar_array, np.ndarray):
            raise ValueError("scalar_array must be a numpy ndarray")
        if len(pixel_spacing) != 3:
            raise ValueError("pixel_spacing must be a tuple or list of three elements")

        vtk_image_data = vtk.vtkImageData()
        vtk_image_data.SetDimensions(scalar_array.shape)
        vtk_image_data.SetSpacing(pixel_spacing)

        flat_array_dicom = scalar_array.flatten(order="F")
        vtk_array = numpy_to_vtk(flat_array_dicom, deep=True, array_type=vtk.VTK_INT)

        vtk_image_data.GetPointData().SetScalars(vtk_array)

        return vtk_image_data

    def create_volume_property(self):
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

        return volume_property

    def create_volume_actor(self, vtk_image_data):
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_image_data)
        volume_property = self.create_volume_property()
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)

        return volume

    def render_volume(self, volume, view_orientation="CORONAL"):
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.SetSize(1000, 1000)
        render_window.AddRenderer(renderer)
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        renderer.SetBackground(1, 1, 1)
        renderer.AddVolume(volume)

        # カメラの設定
        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(0, 0, 0)

        if view_orientation == "AXIAL":
            camera.SetPosition(0, 0, 1)
            camera.SetViewUp(0, -1, 0)
        elif view_orientation == "SAGITTAL":
            camera.SetPosition(1, 0, 0)
            camera.SetViewUp(0, 0, 1)
        elif view_orientation == "CORONAL":
            camera.SetPosition(0, 1, 0)
            camera.SetViewUp(0, 0, -1)
        else:
            raise ValueError("Invalid view orientation. Choose from 'AXIAL', 'SAGITTAL', 'CORONAL', 'OBLIQUE'.")

        renderer.ResetCamera()
        render_window.Render()
        render_window_interactor.Start()


dicom_dir = "./data/MIE022_CT/"
renderer = DICOMVolumeRenderer(dicom_dir)
volume = renderer.create_volume_actor(renderer.vtk_image_data)
renderer.render_volume(volume, view_orientation="AXIAL")
renderer.render_volume(volume, view_orientation="SAGITTAL")
renderer.render_volume(volume, view_orientation="CORONAL")
